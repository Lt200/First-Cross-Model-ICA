import sys
from tkinter import image_names
sys.path.append("..")

import os 
import torch.utils.data as data
import pandas as pd 
import numpy as np 

import xlrd 
import matplotlib.pyplot as plt 

import  clip 
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

from scipy import stats
import scipy.io as sio

from PIL import Image
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


import os
import pandas as pd
import numpy as np
import torch.utils.data as data
import clip
from PIL import Image

def pil_loader(path):
    # 用 Pillow 打开图像并转成 RGB
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class IC9600(data.Dataset):
    def __init__(self, root, index, transform, patch_num):
        self.root = root
        csv_file = os.path.join(root, 'dataV3.csv')
        df = pd.read_csv(csv_file)

        imgnames = df['name'].tolist()
        labels = np.array(df['mos_quality']).astype(np.float32)
        align_labels = np.array(df['mos_align']).astype(np.float32)

        # 只获取 prompt 列（不再读取 level）
        prompts = df['prompt2'].tolist()

        # 确保同一类的图片在同一集合中
        object_dict = {}
        idx = 0
        for p in prompts:
            if object_dict.get(p) is None:
                object_dict[p] = [idx]
            else:
                object_dict[p] += [idx]
            idx += 1

        keys = list(object_dict.keys())
        keys.sort()
        choose_index = []
        for idx in index:
            if idx < len(keys):
                choose_index += object_dict[keys[idx]]
            else:
                print(f"索引 {idx} 超出 keys 的长度范围 {len(keys)}")

        # 生成数据集
        na = []
        nb = []
        sample = []
        for idx in choose_index:
            # 去掉 level，只用 prompt
            p = prompts[idx].strip()
            new_p = p

            # 将生成的描述与其他信息一起加入样本
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'img', imgnames[idx]),
                               labels[idx],
                               align_labels[idx],
                               new_p))
                na.append(labels[idx])
                nb.append(align_labels[idx])

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, align, prompt = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        word_idx = clip.tokenize(prompt, truncate=True)[0]
        return sample, target / 5.0, word_idx, path, align / 5.0

    def __len__(self):
        return len(self.samples)




'''
class IC9600(data.Dataset):
    def __init__(self, root, index, transform, patch_num):
        self.root = root
        csv_file = os.path.join(root, 'Blip12IC9600captions.csv')
        df = pd.read_csv(csv_file)

        imgnames = df['name'].tolist()
        labels = np.array(df['mos_quality']).astype(np.float32)
        align_labels = np.array(df['mos_align']).astype(np.float32)

        # 获取 level 和 prompt 列
        levels = df['level'].tolist()
        prompts = df['blipprompt1'].tolist()

        # 确保同一类的图片在同一集合中
        object_dict = {}
        idx = 0
        for p in prompts:
            if object_dict.get(p) is None:
                object_dict[p] = [idx]
            else:
                object_dict[p] += [idx]
            idx += 1

        keys = list(object_dict.keys())
        keys.sort()
        choose_index = []
        for idx in index:
            if idx < len(keys):
                choose_index += object_dict[keys[idx]]
            else:
                print(f"索引 {idx} 超出 keys 的长度范围 {len(keys)}")

        # 生成数据集
        na = []
        nb = []

        sample = []
        for idx in choose_index:
            # 结合 level 和 prompt 列生成新的描述
            level_desc = levels[idx].strip()
            p = prompts[idx].strip()
            new_p = '%s %s' % (level_desc, p)

            # 将生成的描述与其他信息一起加入样本
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'img', imgnames[idx]), labels[idx], align_labels[idx], new_p))
                na.append(labels[idx])
                nb.append(align_labels[idx])

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, align, prompt = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        word_idx = clip.tokenize(prompt, truncate=True)[0]
        #word_idx = clip.tokenize(prompt)[0]
        return sample, target / 5.0, word_idx, path, align / 5.0

    def __len__(self):
        length = len(self.samples)
        return length
'''
'''
class IC9600(data.Dataset):
    def __init__(self, root, index, transform, patch_num):
        self.root = root
        csv_file = os.path.join(root, 'data.csv')
        df = pd.read_csv(csv_file)

        imgnames = df['name'].tolist()
        labels = np.array(df['mos_quality']).astype(np.float32)
        align_labels = np.array(df['mos_align']).astype(np.float32)

        # 获取 level 和 prompt 列
        levels = df['level'].tolist()
        prompts = df['prompt'].tolist()

        # 确保同一类的图片在同一集合中
        object_dict = {}
        idx = 0
        for p in prompts:
            if object_dict.get(p) is None:
                object_dict[p] = [idx]
            else:
                object_dict[p] += [idx]
            idx += 1

        keys = list(object_dict.keys())
        keys.sort()
        choose_index = []
        for idx in index:
            if idx < len(keys):
                choose_index += object_dict[keys[idx]]
            else:
                print(f"索引 {idx} 超出 keys 的长度范围 {len(keys)}")

        # 生成数据集
        na = []
        nb = []

        sample = []
        for idx in choose_index:
            # 结合 level 和 prompt 列生成新的描述
            level_desc = levels[idx].strip()
            p = prompts[idx].strip()
            new_p = '%s %s' % (level_desc, p)

            # 将生成的描述与其他信息一起加入样本
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'img', imgnames[idx]), labels[idx], align_labels[idx], new_p))
                na.append(labels[idx])
                nb.append(align_labels[idx])

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, align, prompt = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        word_idx = clip.tokenize(prompt)[0]
        return sample, target / 5.0, word_idx, path, align / 5.0

    def __len__(self):
        length = len(self.samples)
        return length
'''



'''  class IC9600(data.Dataset):
    def __init__(self, root, index, transform, patch_num):
        self.root = root
        csv_file = os.path.join(root, 'data.csv')
        df = pd.read_csv(csv_file)

        imgnames = df['name'].tolist()
        labels = np.array(df['mos_quality']).astype(np.float32)
        align_labels = np.array(df['mos_align']).astype(np.float32)


        prompts = df['prompt'].tolist()

        # here we make sure the image with the same object label falls into the same set.
        object_dict = {} # 300 in total
        idx = 0
        for p in prompts:
            if object_dict.get(p) is None:
                object_dict[p] = [idx]
            else:
                object_dict[p] += [idx]
            idx += 1

        keys = list(object_dict.keys())
        keys.sort()
        choose_index = []
        for idx in index:
            if idx < len(keys):
                choose_index += object_dict[keys[idx]]
            else:
                print(f"索引 {idx} 超出 keys 的长度范围 {len(keys)}")

        # make datasets
        na = []
        nb = []

        sample = []
        for idx in choose_index:
            p = prompts[idx]
            # print(p)
            p = p.split(',')
            p = [item.strip() for item in p]

            #newp = '%s.' % p[0] #The prompt of this image is that
            if len(p) == 1:
                newp = 'The prompt of this image is that %s.' % p[0]
            elif len(p) == 2:
                if 'style' in p[1]:
                    newp = 'The prompt of this image is that %s, and its style is %s.' % (p[0], p[1])
                else:
                    newp = 'The prompt of this image is that %s, and its detail is %s.' % (p[0], p[1])
            else:
                newp = 'The prompt of this image is that %s, ' % p[0]
                news = 'its detail is '

                flag = 0
                for i in range(1, len(p)):
                    if 'style'  in p[i]:
                        flag = 1 # always in the end
                        break
                    #     news += '%s, ' % p[i]
                    # else:
                    #     news += 'and its style is %s.' % p[i]
                if flag:
                    news += ', '.join(p[1:-1])
                    news += ', and its style is %s.' % p[-1]
                else:
                    news += ', '.join(p[1:-1])
                    news += '.'
                newp += news
            # new_p = newp.rstrip(' ')
            # if newp[-1] == ',':
            #     newp = newp[:-1] + '.'
            #print(newp, newp[-1])
            new_p = 'This is a photo aligned with the prompt.'#newp
            # for pp in p:
            #     pp = pp.split(' ')
            #     pp = [item.replace('-', '') for item in pp if item]
            #     pp = ' '.join(pp)
            #     new_p += [pp]
            # new_p = ' '.join(new_p)
            # print(new_p)
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'img', imgnames[idx]), labels[idx], align_labels[idx], new_p))
                na.append(labels[idx])
                nb.append(align_labels[idx])
        self.samples = sample
        self.transform = transform
        # plt.scatter(na, nb)
        # srcc, _ = stats.spearmanr(na, nb)
        # plt.title("%.2f"%srcc)
        # plt.show()
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, align, prompt = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        #print(prompt)
        word_idx = clip.tokenize(prompt)[0]
        #word_idx = word_idx[0].numpy().tolist() # [77]
        # tokenize prompt
        return sample, target/5.0, word_idx, path , align / 5.0 # for 0 - 1

    def __len__(self):
        length = len(self.samples)
        return length '''


class IC6720(data.Dataset):
    def __init__(self, root, index, transform, patch_num):
        self.root = root
        csv_file = os.path.join(root, 'label_gene_with_complexity_train.csv')
        df = pd.read_csv(csv_file)

        imgnames = df['name'].tolist()
        labels = np.array(df['mos_quality']).astype(np.float32)
        align_labels = np.array(df['mos_align']).astype(np.float32)


        prompts = df['prompt'].tolist()

        # here we make sure the image with the same object label falls into the same set.
        object_dict = {} # 300 in total
        idx = 0
        for p in prompts:
            if object_dict.get(p) is None:
                object_dict[p] = [idx]
            else:
                object_dict[p] += [idx]
            idx += 1

        keys = list(object_dict.keys())
        keys.sort()
        choose_index = []
        for idx in index:
            if idx < len(keys):
                choose_index += object_dict[keys[idx]]
            else:
                print(f"索引 {idx} 超出 keys 的长度范围 {len(keys)}")

        # make datasets
        na = []
        nb = []

        sample = []
        for idx in choose_index:
            p = prompts[idx]
            # print(p)
            p = p.split(',')
            p = [item.strip() for item in p]

            #newp = '%s.' % p[0] #The prompt of this image is that
            if len(p) == 1:
                newp = 'The prompt of this image is that %s.' % p[0]
            elif len(p) == 2:
                if 'style' in p[1]:
                    newp = 'The prompt of this image is that %s, and its style is %s.' % (p[0], p[1])
                else:
                    newp = 'The prompt of this image is that %s, and its detail is %s.' % (p[0], p[1])
            else:
                newp = 'The prompt of this image is that %s, ' % p[0]
                news = 'its detail is '

                flag = 0
                for i in range(1, len(p)):
                    if 'style'  in p[i]:
                        flag = 1 # always in the end
                        break
                    #     news += '%s, ' % p[i]
                    # else:
                    #     news += 'and its style is %s.' % p[i]
                if flag:
                    news += ', '.join(p[1:-1])
                    news += ', and its style is %s.' % p[-1]
                else:
                    news += ', '.join(p[1:-1])
                    news += '.'
                newp += news
            # new_p = newp.rstrip(' ')
            # if newp[-1] == ',':
            #     newp = newp[:-1] + '.'
            #print(newp, newp[-1])
            new_p = 'This is a photo aligned with the prompt.'#newp
            # for pp in p:
            #     pp = pp.split(' ')
            #     pp = [item.replace('-', '') for item in pp if item]
            #     pp = ' '.join(pp)
            #     new_p += [pp]
            # new_p = ' '.join(new_p)
            # print(new_p)
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'img', imgnames[idx]), labels[idx], align_labels[idx], new_p))
                na.append(labels[idx])
                nb.append(align_labels[idx])
        self.samples = sample
        self.transform = transform
        # plt.scatter(na, nb)
        # srcc, _ = stats.spearmanr(na, nb)
        # plt.title("%.2f"%srcc)
        # plt.show()
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, align, prompt = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        #print(prompt)
        word_idx = clip.tokenize(prompt)[0]
        #word_idx = word_idx[0].numpy().tolist() # [77]
        # tokenize prompt
        return sample, target/5.0, word_idx, path , align / 5.0 # for 0 - 1

    def __len__(self):
        length = len(self.samples)
        return length
