import os
import PIL
from numpy import imag
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2 as cv
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms_tv
import torchvision.transforms.functional as F
from dataset.transform import *

"""Data loader file for imagenet-like shaped datasets"""

def MapClasses(data_dir: str, mapping_file: str) -> dict:
    """
    Maps codes to class names
    
    Args:
        data_dir: path to data directory
        mapping_file: path to file containing dir-to-class mapping
        
    Returns:
        a dictionary with the mappings
    """
    file_ = open(os.path.join(data_dir, mapping_file))
    mappings = file_.readlines()
    
    for i in range(len(mappings)):
        mappings[i] = mappings[i].strip("\n").split("\t")
        
    code_to_class = {}
    
    for record in mappings:
        code_to_class[record[0]] = record[1]
       
    
    return code_to_class    



def GetEntities(data_dir: str) -> dict:
    """
    Get the classes/codes present on the dataset
    
    Args:
        data_dir: path to data directory
    
    Return:
        entites: List of entities included
        ent_to_idx: entities to ids
    """
    dir_path = os.path.join(data_dir, 'train')
    
    entities = sorted(entry.name for entry in os.scandir(dir_path) if entry.is_dir())
    if not entities:
        raise FileNotFoundError((f"Couldn't find any class folder in {data_dir}."))
        
    ent_to_idx = {entity: i for i, entity in enumerate(entities)}
    return entities, ent_to_idx



def SampleList(data_dir:str, entities:list=None, cls_to_idx:dict=None, val_maps:dict=None, mode:str=None) -> dict:
    """
    Create a list of all samples present in the dataset since it is organized hierarchicaly
    Args:
        data_dir: path to data directory
        entites: list of all classes present to the dataset
        cls_to_idx: classes with their corresponding ID
        val_maps: img to class mapping of validation images (needed only when doing validation)
        mode: tells the oprating mode, train or val
    
    Returns:
        samplelist: a dictionary containing all samples and their corresponding classes
    """
    samples = {}

    if mode != 'val':
        for cls in entities:
            cls_dir_path = os.path.join(*[data_dir, mode, cls, "images"])

            for root, dir, files in os.walk(cls_dir_path):
                for file in files:
                    samples[os.path.join(cls_dir_path, file)] = cls_to_idx[cls]

        return samples
    
    else:
        cls_path = os.path.join(*[data_dir, mode, "images"])

        for root, dir, files in os.walk(cls_path):
            for file in files:
                samples[os.path.join(cls_path, file)] = cls_to_idx[val_maps[file]]

        return samples



class TinyImgNet(Dataset):

    """
    Customised dataset generator for tiny image net dataset
    The methods and attributes are made based on the official structure of tiny imagenet dataset
    """

    def __init__(self, data_dir:str, mode: str, transform: str = None, cfg=None) -> None:
        
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.global_mappings = MapClasses(self.data_dir, "words.txt")
        self.entities, self.cls_to_idx = GetEntities(self.data_dir)
        self.cfg = cfg
        
        if self.mode == 'val':
            self.val_maps =  MapClasses(self.data_dir, "val/val_annotations.txt")
            self.samples = SampleList(data_dir=self.data_dir, entities=self.entities, cls_to_idx=self.cls_to_idx, val_maps=self.val_maps, mode=self.mode)
            
        else:
            self.samples = SampleList(data_dir=self.data_dir, entities=self.entities, cls_to_idx=self.cls_to_idx, mode=self.mode)

        
        self.sampleList = list(self.samples.keys())
        print(self.mode)
        if self.transform:
            # self.transformations  = transforms.Compose([
            #                       transforms.Resize((self.cfg.DATA.crop_size, self.cfg.DATA.crop_size)),
            #                     #   transforms.RandomResizedCrop((cfg.img_size,cfg.img_size)), #224 is the size of the imagenet images so this will allow to use transfer learning
            #                       transforms.RandomHorizontalFlip(p=0.5), #Randomly horziontal flip image with probability p, default p = 0.5
            #                       transforms.RandomRotation(45),
            #                       transforms.ToTensor(),#Transforms the image to pytorch tensor (by default the values are in converted in range [0,1])
            #                       transforms.Normalize(
            #                          mean=[0.485, 0.456, 0.406],
            #                          std=[0.229, 0.224, 0.225]
            #                       )
            #                   ])
            self.transformations = prepare_transform(self.cfg, self.mode)

        else:
            self.transformations  = transforms.Compose([
                                  transforms.Resize((self.cfg.DATA.crop_size, self.cfg.DATA.crop_size)),
                                  transforms.ToTensor(),#Transforms the image to pytorch tensor (by default the values are in converted in range [0,1])
                                  transforms.Normalize(
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]
                                  )
                              ])

            
    def __len__(self):
        return len(self.samples)


    
    @staticmethod
    def getEntity(label_id, global_mappings, cls_to_idx):

        """
        Get the entity name from the class id
        """

        class_ = None

        for key, value in cls_to_idx.items():
            if label_id == value:
                class_ = key
        
        entity = global_mappings[class_]

        return entity


    
    def __getitem__(self, idx):

        sample_path = self.sampleList[idx] #img path
        label_id = self.samples[sample_path] #label id

        img = cv.imread(sample_path)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR) # For visualisation purposes
        
        img = F.to_pil_image(img)
        img = self.transformations(img)
            
        label = self.getEntity(label_id, self.global_mappings, self.cls_to_idx)

        return (img, label_id)


             

def make_dataset(data_dir:str, mode:str,  transform:bool=None, cfg=None):
    
    """
    Creates an instance of TinyImgNet class
    
    Parameters
    ==========
    
        data_dir: Path to data directory
        mode: type of data (train/valid)
        transform: bool saying if transformations are goint to be applied to the data or not
    
    Returns
    """
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        raise RuntimeError("No device detected")

    dataset = TinyImgNet(data_dir, mode=mode, transform=transform, cfg=cfg)

    return dataset



def make_data_loader(data_dir:str, mode:str, transform:bool, batch_size:int, shuffle:bool, cfg=None, num_workers:int=12):
    """
    Creates a dataloader for the TinyImgNet class
    
    Parameters
    ==========
    
        data_dir: Path to data directory
        mode: Type of data (train/valid)
        transform: use transformations or not
        cfg: Config data container
        device: Type of device (TPU, GPU or CPU) default is None
        shuffle: Bool for shuffling data or not, default is False
        batch_size: number of batch images
        num_workers: num of virtual cpus to use for loading
        
    """
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        raise RuntimeError("No device detected")

    dataset = make_dataset(data_dir=data_dir, mode=mode, transform=transform, cfg=cfg)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader



def pytorch_dataloader(cfg=None, batch_size:int=None, sampler=None, world_size=None, rank=None):  

    """
    Pytorch way of creating dataset generators for provided datasets
    Args:
        cfg: Config object
        batch_size: batch_size
        sampler: Data sampler (important for distributed training/validation)
        world_size: Number of GPUs
        rank: GPU ID
    Returns:
        train_loader: training set dataloader
        val_loader: val_set dataloader
        sampler: data sampler if defined
    """  
    transformations_train = prepare_transform(cfg, 'train')
    transformations_val = prepare_transform(cfg, 'val')

    if cfg.TRAIN.dataset == "Cars":

        training_data = torchvision.datasets.StanfordCars(
            root=cfg.DATA.path,
            split='train',
            download=False,
            transform=transformations_train,
            # transform= transforms.Compose([
            #                             transforms.Resize((cfg.DATA.crop_size, cfg.DATA.crop_size)),
            #                             # transforms.RandomResizedCrop((cfg.data.img_size, cfg.data.img_size)), #224 is the size of the imagenet images so this will allow to use transfer learning
            #                             # transforms.RandomHorizontalFlip(p=0.5), #Randomly horziontal flip image with probability p, default p = 0.5
            #                             # transforms.RandomRotation(45),
            #                             transforms.ToTensor(),#Transforms the image to pytorch tensor (by default the values are in converted in range [0,1])
            #                             transforms.Normalize(
            #                                         mean=[0.485, 0.456, 0.406],
            #                                         std=[0.229, 0.224, 0.225]
            #                                 )
            #                         ])
        )
        
        test_data = torchvision.datasets.StanfordCars(
            root=cfg.DATA.path,
            split='test',
            download=False,
            transform=transformations_val
            # transform=transforms.Compose([
            #                             transforms.Resize((cfg.DATA.crop_size, cfg.DATA.crop_size)),
            #                             transforms.ToTensor(),#Transforms the image to pytorch tensor (by default the values are in converted in range [0,1])
            #                             transforms.Normalize(
            #                                         mean=[0.485, 0.456, 0.406],
            #                                         std=[0.229, 0.224, 0.225]
            #                                 )
            #                         ])
        )


    elif cfg.TRAIN.dataset == "imagenet":
        training_data = torchvision.datasets.ImageNet(
            root=cfg.DATA.path,
            split='train',
            transform=transformations_train,
            # transform= transforms.Compose([
            #                             transforms.Resize((cfg.DATA.crop_size, cfg.DATA.crop_size)),
            #                             transforms.RandomResizedCrop((cfg.DATA.crop_size, cfg.DATA.crop_size)), #224 is the size of the imagenet images so this will allow to use transfer learning
            #                             transforms.RandomHorizontalFlip(p=0.5), #Randomly horziontal flip image with probability p, default p = 0.5
            #                             transforms.RandomRotation(45),
            #                             transforms.ToTensor(),#Transforms the image to pytorch tensor (by default the values are in converted in range [0,1])
            #                             transforms.Normalize(
            #                                         mean=[0.485, 0.456, 0.406],
            #                                         std=[0.229, 0.224, 0.225]
            #                                 )
            #                         ])
        )
        
        test_data = torchvision.datasets.ImageNet(
            root=cfg.DATA.path,
            split='val',
            transform=transformations_val
            # transform=transforms.Compose([
            #                             transforms.Resize((cfg.DATA.crop_size, cfg.DATA.crop_size)),
            #                             transforms.ToTensor(),#Transforms the image to pytorch tensor (by default the values are in converted in range [0,1])
            #                             transforms.Normalize(
            #                                         mean=[0.485, 0.456, 0.406],
            #                                         std=[0.229, 0.224, 0.225]
            #                                 )
            #                         ])

        )

    elif cfg.TRAIN.dataset == 'cifar10':
        training_data = torchvision.datasets.CIFAR10(
            root = cfg.DATA.path, 
            train = True, 
            transform = transformations_train, 
            target_transform = None, 
            download = True
            )
        
        test_data = torchvision.datasets.CIFAR10(
            root = cfg.DATA.path, 
            train = False, 
            transform = transformations_val, 
            target_transform = None, 
            download = True
            )
        
    elif cfg.TRAIN.dataset == 'cifar100':
        training_data = torchvision.datasets.CIFAR100(
            root = cfg.DATA.path, 
            train = True, 
            transform = transformations_train, 
            target_transform = None, 
            download = True
            )
        
        test_data = torchvision.datasets.CIFAR100(
            root = cfg.DATA.path, 
            train = False, 
            transform = transformations_val, 
            target_transform = None, 
            download = True
            )
        
    elif cfg.TRAIN.dataset == 'pets':
        training_data = torchvision.datasets.OxfordIIITPet(
            root = cfg.DATA.path, 
            transform = transformations_train, 
            target_transform = None, 
            download = True
            )
        
        test_data = torchvision.datasets.OxfordIIITPet(
            root = cfg.DATA.path, 
            split = 'test', 
            transform = transformations_val, 
            target_transform = None, 
            download = True
            )
        
    elif cfg.TRAIN.dataset == 'flowers':
        training_data = torchvision.datasets.Flowers102(
            root = cfg.DATA.path, 
            transform = transformations_train, 
            target_transform = None, 
            download = True
            )
        
        test_data = torchvision.datasets.Flowers102(
            root = cfg.DATA.path, 
            split = 'test', 
            transform = transformations_val, 
            target_transform = None, 
            download = True
            )

    if sampler:

        sampler_tr = DistributedSampler(training_data, num_replicas=world_size, rank=rank)
        sampler_val = DistributedSampler(test_data, num_replicas=world_size, rank=rank)

        train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,
                                                shuffle=False, num_workers=12, sampler=sampler_tr, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                             shuffle=False, num_workers=12, sampler=sampler_val, pin_memory=True)

        return train_loader, val_loader, sampler_tr
    else:
        train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,
                                                shuffle=False, num_workers=12)

        val_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                             shuffle=False, num_workers=12)
       

    
    return train_loader, val_loader

def tv_interp(method):
    if method == "bicubic":
        return transforms.InterpolationMode.BICUBIC
    elif method == "lanczos":
        return transforms.InterpolationMode.LANCZOS
    elif method == "hamming":
        return transforms.InterpolationMode.HAMMING
    else:
        return transforms.InterpolationMode.BILINEAR

def prepare_transform(cfg, mode:str):
    """
    Prepare data transformations (includes augmentations)
    Args:
        cfg: config object
        mode (str): transformations mode (i.e. train or val)

    Returns:
        aug_transform: transform functions with augmentations is specified on the config object 
    """
    # Convert HWC/BGR/int to HWC/RGB/float format for applying transforms
    train_size, test_size = (
        cfg.DATA.crop_size,
        cfg.DATA.crop_size,
    )

    if mode == "train":
        aug_transform = transforms_imagenet_train(
            img_size=(train_size, train_size),
            color_jitter=cfg.AUG.color_jitter,
            auto_augment=cfg.AUG.AA_TYPE,
            interpolation=cfg.AUG.interpolation,
            re_prob=cfg.AUG.re_prob,
            re_mode=cfg.AUG.re_mode,
            re_count=cfg.AUG.re_count,
            mean=cfg.DATA.mean,
            std=cfg.DATA.std,
        )
    else:
        t = []
        if cfg.DATA.VAL_CROP_RATIO == 0.0:
            t.append(
                transforms_tv.Resize((test_size, test_size), interpolation=tv_interp(cfg.AUG.interpolation)),
            )
        else:
            size = int((1.0 / cfg.DATA.VAL_CROP_RATIO) * test_size)
            t.append(
                transforms_tv.Resize(
                    size, interpolation=tv_interp(cfg.AUG.interpolation)
                ),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms_tv.CenterCrop(test_size))
        if cfg.DATA.Affine:
            t.append(transforms_tv.RandomAffine(degrees=0, translate=(cfg.DATA.h_affine, cfg.DATA.v_affine)))
        if cfg.DATA.Pad:
            pad_size = (224 - cfg.DATA.img_size)/2 
            t.append(transforms_tv.Resize(size=(int(cfg.DATA.img_size))))
            t.append(transforms_tv.Pad(padding=int(pad_size)))
        t.append(transforms_tv.ToTensor())
        t.append(transforms_tv.Normalize(cfg.DATA.mean, cfg.DATA.std))
        aug_transform = transforms_tv.Compose(t)
    return aug_transform
        
if __name__ == "__main__":
    pass
