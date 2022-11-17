import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from skimage.util import view_as_windows
from keras.utils import to_categorical
from tqdm import tqdm
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split

working_dir = 'C:/Users/Logan/Desktop/Research/Gleghorn/CV_mainv2'
img_path = working_dir + '/img/'
GT_path = working_dir + '/GT/'

dim = 256

def ax_decorate_box(ax):
    [j.set_linewidth(0) for j in ax.spines.values()]
    ax.tick_params(axis="both", which="both", bottom=False, top=False,
               labelbottom=False, left=False, right=False, labelleft=False)
    return ax

def target_data_process(GTs, num_class):
    return to_categorical(GTs, num_classes=num_class)

'''
class ImageFolder(data.Dataset):
    def __init__(self, root, crop_size=224):
        """Initializes image paths and preprocessing module."""
        self.root = root

        # GT : Ground Truth
        self.GT_paths = root[:-1] + 'GT/'
        self.image_paths = sorted(glob(img_path + '*.png'))
        self.crop_size = crop_size
        self.RotationDegree = [90, 180, 270]
        print("image count in path :{}".format(len(self.image_paths)))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        filename = image_path.split('_')
'''

def crop_augment(img, GT, dim, step, num_class):
#Crops and augments an image and GT the same way
#img, GT - np.array, dim - int, step - int, num_class - int
    imgs = view_as_windows(img, (dim, dim, 3), step=step)
    GTs = view_as_windows(GT, (dim, dim, 1), step=step)
    imgs = imgs.reshape(len(imgs)**2, dim, dim, 3)
    GTs = GTs.reshape(len(GTs)**2, dim, dim, 1)
    print(imgs.shape, GTs.shape)
    if num_class == 2:
        GTs[GTs < 1] = 0
        GTs[GTs > 0] = 1
    if num_class == 3:
        GTs[GTs == 0] = 1
        GTs = GTs - 1
    delete_list = []
    for i in range(len(imgs)):
        per = (np.count_nonzero(np.array(GTs[i])) / (dim * dim)) * 100
        if (per < 0.01):
            delete_list.append(i)
        else:
            continue
    print(len(delete_list))
    imgs = np.delete(imgs, delete_list, 0)
    GTs = np.delete(GTs, delete_list, 0)
    GTs = to_categorical(GTs, num_classes=num_class)
    print(imgs.shape, GTs.shape)
    imgs_90 = np.copy(imgs)
    imgs_vflip = np.copy(imgs)
    imgs_hflip = np.copy(imgs)
    GTs_90 = np.copy(GTs)
    GTs_vflip = np.copy(GTs)
    GTs_hflip = np.copy(GTs)
    for i in range(len(imgs)):
        imgs_90[i] = np.rot90(imgs_90[i])
        imgs_vflip[i] = np.flipud(imgs_vflip[i])
        imgs_hflip[i] = np.fliplr(imgs_hflip[i])
        GTs_90[i] = np.rot90(GTs_90[i])
        GTs_vflip[i] = np.flipud(GTs_vflip[i])
        GTs_hflip[i] = np.fliplr(GTs_hflip[i])
    final_crops = np.concatenate((imgs, imgs_90, imgs_vflip, imgs_hflip), axis=0)
    final_crops_GT = np.concatenate((GTs, GTs_90, GTs_vflip, GTs_hflip), axis=0)
    return final_crops, final_crops_GT


img = Image.open('C:/Users/Logan/Desktop/Research/Gleghorn/CV_mainv2/img/2.png')
img = np.array(img)
GT = Image.open('C:/Users/Logan/Desktop/Research/Gleghorn/CV_mainv2/GT/2.png')
GT = np.array(GT)
a, b = GT.shape
GT = GT.reshape(a, b, 1)
crop_imgs, crop_GTs = crop_augment(img, GT, dim, int(dim/2), 2)
print(crop_imgs.shape, crop_GTs.shape)




rows = 1
columns = 2
for i in range(10):
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    plt.imshow(crop_imgs[i])
    plt.axis('off')
    plt.title('Img')
    fig.add_subplot(rows, columns, 2)
    plt.imshow(crop_GTs[i][:,:,1], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title('GT')
    plt.show()

