from multiprocessing.sharedctypes import Value
import torch
import torchvision
from datasets.datasets import * 

class DataLoader(object):
    """Dataset class for IQA databases"""
    def __init__(self, dataset, path, img_indx, patch_num=1, batch_size=1, istrain=True):
        print(dataset)
        self.batch_size = batch_size
        self.istrain = istrain
        if self.istrain:
            patch_size = 224
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=patch_size),
                #torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.RandomHorizontalFlip(),
                #torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                    std=(0.229, 0.224, 0.225))
            ])            
        else:
            transforms = torchvision.transforms.Compose([
            #torchvision.transforms.Resize((224, 224)), # resize may influence image quality prediction
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
            ])

        if dataset == 'IC9600':
            self.data = IC9600(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'IC6720':
            self.data = IC6720(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        else:
            raise ValueError("wrong datasets!!!")

    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=1, shuffle=False)
        return dataloader


if __name__ == '__main__':
    dl = DataLoader()
    