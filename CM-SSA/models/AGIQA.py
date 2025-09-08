import torch
import torch.nn as nn
import os
import  clip.clip as clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import copy

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg['MODEL']['BACKBONE']['NAME']
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root=os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg['TRAINER']['COOP']['N_CTX']}

    model = clip.build_model_maple(state_dict or model.state_dict(), design_details) # arch changed

    return model



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer # modified arch !!!
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        
    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg['TRAINER']['COOP']['N_CTX'] #cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg['TRAINER']['COOP']['CTX_INIT'] #cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224 #cfg['INPUT'].SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if cfg['TRAINER']['COOP']['CSC']:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        ## MaPLe parts
        self.compound_prompts_depth = 12 #cfg.TRAINER.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim+768, 768) # 512 768
        # self.proj.half()

        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, ctx_dim)) #768
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)

        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim+768, 768) # ctx_dim
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg['TRAINER']['COOP']['CLASS_TOKEN_POSITION']

        single_layer = nn.Linear(ctx_dim*2, ctx_dim) # ctx_dim
        self.compound_prompt_projections2 = _get_clones(single_layer, self.compound_prompts_depth - 1)

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def get_text(self):
        return self.ctx, self.compound_prompts_text

    def forward(self, share, vis):
        ctx = self.ctx 
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            cc = torch.cat((self.compound_prompts_text[index], vis[index]), dim=1)
            visual_deep_prompts.append(layer(cc))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        cc = torch.cat((self.ctx, share), dim=1)
        return prompts, self.proj(cc), self.compound_prompts_text, visual_deep_prompts #self.compound_prompts_vis # pass here original, as for visual 768 is required

class PromptLearner1(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg['TRAINER']['COOP']['N_CTX'] #cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg['TRAINER']['COOP']['CTX_INIT'] #cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224 #cfg['INPUT'].SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if cfg['TRAINER']['COOP']['CSC']:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        #self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        ## MaPLe parts
        self.compound_prompts_depth = 12 #cfg.TRAINER.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        #self.proj = nn.Linear(ctx_dim, 768) # 512 768
        # self.proj.half()

        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        #self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512)) #768
        #                                              for _ in range(self.compound_prompts_depth-1)])
        # for single_para in self.compound_prompts_text:
        #     nn.init.normal_(single_para, std=0.02)

        self.compound_prompts_vis = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 768)) #768
                                                      for _ in range(self.compound_prompts_depth)])
        for single_para in self.compound_prompts_vis:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        # single_layer = nn.Linear(ctx_dim, 768) # ctx_dim
        # self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth-1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg['TRAINER']['COOP']['CLASS_TOKEN_POSITION']

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts


    def forward(self):
        # Before returning, need to transform
        # prompts to 768 for the visual side
        # visual_deep_prompts = []
        # for index, layer in enumerate(self.compound_prompt_projections):
        #     visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        # ctx = self.ctx 
        # if ctx.dim() == 2:
        #     ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        # prefix = self.token_prefix
        # suffix = self.token_suffix
        # prompts = self.construct_prompts(ctx, prefix, suffix)

        # visual_deep_prompts = []
        # for index, layer in enumerate(self.compound_prompt_projections):
        #     cc = self.compound_prompts_text[index] #torch.cat((self.compound_prompts_text[index], vis[index]), dim=1)
        #     visual_deep_prompts.append(layer(cc))

        return 0, self.compound_prompts_vis[0], 0, self.compound_prompts_vis[1:] #self.compound_promp

def load_clip_to_cpu2(cfg):
    backbone_name = cfg['MODEL']['BACKBONE']['NAME']
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root=os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class TextEncoder2(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = copy.deepcopy(clip_model.transformer)
        self.positional_embedding = copy.deepcopy(clip_model.positional_embedding)
        self.ln_final = copy.deepcopy(clip_model.ln_final)
        self.text_projection = copy.deepcopy(clip_model.text_projection)
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        #self.prompt_learner2 = PromptLearner1(cfg, ['aligned_photo', 'misaligned_photo'], clip_model)
        self.prompt_learner2 = PromptLearner1(cfg, ['simple_complexity', 'littel_complexity', 'moderate_complexity',
               'very_complexity', 'high_complexity'], clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.vv = TextEncoder2(load_clip_to_cpu2(cfg).float())
        self.clip_model = clip_model

        # self.pos_ = ["This is a good photo because it is %s."%i for i in self.pos]
        # self.neg_ = ["This is a bad photo because it is %s."%i for i in self.neg]

        # self.pos_idx =  torch.cat([clip.tokenize(p) for p in self.pos_]).cuda()
        # self.neg_idx =  torch.cat([clip.tokenize(p) for p in self.neg_]).cuda() # 4, 77
        
    def forward(self, image, wordid):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        with torch.no_grad():
            pos_embedding = self.clip_model.token_embedding(wordid).type(self.dtype) # b, 77
        text_features = self.vv(pos_embedding, wordid) # b, 512

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner2()
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        #text_features = self.text_encoder(prompts, self.prompt_learner2.tokenized_prompts, deep_compound_prompts_text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)        
        logits = (image_features * text_features).sum(1).unsqueeze(1)
        score2 = logits #logits.softmax(dim=-1)[:, :1] # (image_features * text_features).sum(1).unsqueeze(1) #logits.softmax(dim=-1)[:, :1]

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(shared_ctx, deep_compound_prompts_vision)

        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        score = logits.softmax(dim=-1)[:, :1]
        # s = 0
        # for t in text:
        #     logits = logit_scale * image_features @ t.t() #text_features.permute(0,2,1)) 
        #     score = logits.softmax(dim=-1)[:, :1]
        #     s += score 
        # score = s / len(text) 
        #s = torch.cat(s, dim=1)
        #score = self.prompt_learner.proj2(s)
        return score, score2 #, aligned_score

class CoOp(object):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.build_model()

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = cfg['DATASET']['CLASS_NAME']

        print(f"Loading CLIP (backbone: {cfg['MODEL']['BACKBONE']['NAME']})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg['TRAINER']['COOP']['PREC'] == "fp32" or cfg['TRAINER']['COOP']['PREC'] == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg['MODEL']['INIT_WEIGHTS']:
            pass # load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        # self.optim = torch.optim.Adam(
        #     self.model.prompt_learner.parameters(), lr= cfg['OPTIM']['LR'])

        self.sched = None #build_lr_scheduler(self.optim, cfg.OPTIM)

        assert cfg['TRAINER']['COOP']['PREC'] != "amp"
        self.scaler = None #GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            pass
            # print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            # self.model = nn.DataParallel(self.model)


    def train(self):
        pass

    def predict(self, x):
        output = self.model(x)
        return output




    # def parse_batch_train(self, batch):
    #     input = batch["img"]
    #     label = batch["label"]
    #     input = input.to(self.device)
    #     label = label.to(self.device)
    #     return input, label
    #
    def load_model(self, model_path, epoch=None):

        checkpoint = torch.load(model_path)
        #print(checkpoint)
        state_dict = checkpoint

        # Ignore fixed token vectors
        if "token_prefix" in state_dict:
            del state_dict["token_prefix"]

        if "token_suffix" in state_dict:
            del state_dict["token_suffix"]

        # print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
        # set strict=False
        self.model.prompt_learner.load_state_dict(state_dict, strict=False)
    # def load_model(self, directory, epoch=None):
    #     if not directory:
    #         print("Note that load_model() is skipped as no pretrained model is given")
    #         return
    #
    #     names = self.get_model_names()
    #
    #     # By default, the best model is loaded
    #     model_file = "model-best.pth.tar"
    #
    #     if epoch is not None:
    #         model_file = "model.pth.tar-" + str(epoch)
    #
    #     for name in names:
    #         model_path = osp.join(directory, name, model_file)
    #
    #         if not osp.exists(model_path):
    #             raise FileNotFoundError('Model not found at "{}"'.format(model_path))
    #
    #         checkpoint = load_checkpoint(model_path)
    #         state_dict = checkpoint["state_dict"]
    #         epoch = checkpoint["epoch"]
    #
    #         # Ignore fixed token vectors
    #         if "token_prefix" in state_dict:
    #             del state_dict["token_prefix"]
    #
    #         if "token_suffix" in state_dict:
    #             del state_dict["token_suffix"]
    #
    #         print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
    #         # set strict=False
    #         self._models[name].load_state_dict(state_dict, strict=False)

if __name__ == '__main__':
    import yaml
    with open('coop.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # print(cfg)
    # print(cfg.TRAINER.OPTIM.LR)
    # print(cfg.DATASET.CLASS_NAME)
    coop = CoOp(cfg)

    x = torch.randn(1,3,224,224)
    print(coop.predict(x.cuda()))


