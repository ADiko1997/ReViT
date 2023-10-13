import matplotlib
import numpy as np
from torch.utils.data import Dataset, DataLoader 
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch.nn as nn
from torch.autograd import Variable

def show(imgs):
    
    """Visualizes a grid of images"""

    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

def vizBatch(data_loader, num): 

    "Vizualizes num images from single image dataloader"

    dataiter = iter(data_loader)
    images, labels_, path, labels = dataiter.next()
    images_p = images
    # show images
    img_list = []
    for i in images_p:
        print(i.shape)
        img_list.append(i.permute(2,0,1))
    show(torchvision.utils.make_grid(img_list))


def imshow(imgs, probs, pred_cls, labels):

    """Visualizes images with labels and prediction probabilities"""

    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img), cmap='brg')
        axs[0, i].set_title("{0}, {1:.1f}%\n(label: {2})".format(
            pred_cls[i].split(',')[0],
            probs[i] * 100.0,
            labels[i].split(',')[0]),
            fontdict=({'color':"green"} if pred_cls[i]==labels[i] else {'color':"red"}))
    plt.show()

def images_to_probs(logits):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(logits, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [torch.nn.functional.softmax(el, dim=0)[i].item() for i, el in zip(preds, logits)]


def plot_classes_preds(logits, data_loader, images, labels):
    '''
    Generates matplotlib Figure using a network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(logits)
    # plot the images in the batch, along with predicted and true labels
    img_list = []
    for i in images:
        img_list.append(i.permute(2,0,1))
    pred_cls = []
    for idx in np.arange(4):
        pred_cls.append(data_loader.dataset.getEntity(preds[idx], data_loader.dataset.global_mappings, data_loader.dataset.cls_to_idx))
    imshow(img_list, probs, pred_cls, labels)
#     plt.show()
    return 


class SizeEstimator(object):

    def __init__(self, model, input_size=(1,1,32,32), bits=32):
        '''
        Estimates the size of PyTorch models in memory
        for a given input size
        '''
        self.model = model
        self.input_size = input_size
        self.bits = bits

    def get_parameter_sizes(self):
        '''Get sizes of all parameters in `model`'''
        mods = list(self.model.modules())
        sizes = []
        
        for i in range(1,len(mods)):
            m = mods[i]
            p = list(m.parameters())
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

        self.param_sizes = sizes
        return

    def get_output_sizes(self):
        '''Run sample input through each layer to get output sizes'''
        input_ = Variable(torch.FloatTensor(*self.input_size), volatile=True)
        mods = list(self.model.modules())
        out_sizes = []
        for i in range(1, len(mods)):
            m = mods[i]
            out = m(input_)
            out_sizes.append(np.array(out.size()))
            input_ = out

        self.out_sizes = out_sizes
        return

    def calc_param_bits(self):
        '''Calculate total number of bits to store `model` parameters'''
        total_bits = 0
        for i in range(len(self.param_sizes)):
            s = self.param_sizes[i]
            bits = np.prod(np.array(s))*self.bits
            total_bits += bits
        self.param_bits = total_bits
        return

    def calc_forward_backward_bits(self):
        '''Calculate bits to store forward and backward pass'''
        total_bits = 0
        for i in range(len(self.out_sizes)):
            s = self.out_sizes[i]
            bits = np.prod(np.array(s))*self.bits
            total_bits += bits
        # multiply by 2 for both forward AND backward
        self.forward_backward_bits = (total_bits*2)
        return

    def calc_input_bits(self):
        '''Calculate bits to store input'''
        self.input_bits = np.prod(np.array(self.input_size))*self.bits
        return

    def estimate_size(self):
        '''Estimate model size in memory in megabytes and bits'''
        self.get_parameter_sizes()
        self.get_output_sizes()
        self.calc_param_bits()
        self.calc_forward_backward_bits()
        self.calc_input_bits()
        total = self.param_bits + self.forward_backward_bits + self.input_bits

        total_megabytes = (total/8)/(1024**2)
        return total_megabytes, total


def truncated_normal(t, mean=0.0, std=0.01):
    if isinstance(t, torch.nn.Linear):

        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2*std, t > mean + 2*std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
    
    elif isinstance(t, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(t.weight, gain=1.0)

    return t


def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)