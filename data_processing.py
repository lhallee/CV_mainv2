import numpy as np
import os
import torch
import torchvision
import cv2
from natsort import natsorted
from torch.utils import data
from skimage.util import view_as_windows
from tqdm import tqdm
from glob import glob
from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split
'''
def to_categorical(y, num_classes=None, dtype="float32"):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with `categorical_crossentropy`.
    Args:
        y: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
          as `max(y) + 1`.
        dtype: The data type expected by the input. Default: `'float32'`.
    Returns:
        A binary matrix representation of the input. The class axis is placed
        last.
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
'''


#Custom pytorch dataset, simply indexes imgs and gts
class ImageSet(data.Dataset):
    def __init__(self, imgs, GTs):
        self.imgs = imgs
        self.GTs = GTs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img = torch.tensor(self.imgs[index], dtype=torch.float)
        GT = torch.tensor(self.GTs[index], dtype=torch.float)
        return img, GT

class ReconSet(data.Dataset):
    def __init__(self, imgs):
        self.imgs = imgs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img = torch.tensor(self.imgs[index], dtype=torch.float)
        return img


class Imageset_processing:
    def __init__(self, config):
        self.img_path = config.img_path
        self.GT_path = config.GT_path
        self.eval_path = config.eval_img_path
        self.eval_type = config.eval_type
        self.dim = config.image_size
        self.num_class = config.num_class
        self.train_per = config.train_per
        self.batch_size = config.batch_size
        self.num_cpu = os.cpu_count()

    def load_imgs(self):
        img_paths = natsorted(glob(self.img_path + '*.png'))  # natural sort
        GT_paths = natsorted(glob(self.GT_path + '*.png'))
        assert len(img_paths) == len(GT_paths), 'Need GT for every Image.'
        eval_paths = natsorted(glob(self.eval_path + '*.png'))
        return img_paths, GT_paths, eval_paths

    def crop_augment(self, img, GT):
        img = np.array(cv2.imread(img, 1)) / 255.0 #read and scale img
        GT = np.array(cv2.imread(GT, 2), dtype=np.float32)
        a, b = GT.shape
        GT = GT.reshape(a, b, 1) #reshape for view_as_windows
        imgs = view_as_windows(img, (self.dim, self.dim, 3), step=self.dim)
        GTs = view_as_windows(GT, (self.dim, self.dim, 1), step=self.dim)
        a, b, c, d, e, f = imgs.shape
        imgs = imgs.reshape(a * b, self.dim, self.dim, 3) #reshape windowed output into num_images, dim, dim channel
        GTs = GTs.reshape(a * b, self.dim, self.dim, 1)

        if self.num_class == 2: #format GTs (similar to to_categorical)
            GTs[GTs < 1] = 0
            GTs[GTs > 0] = 1
        if self.num_class == 3:
            GTs[GTs == 0] = 1
            GTs = GTs - 1
        imgs_90 = np.copy(imgs) #copy images for various augmentations
        imgs_vflip = np.copy(imgs)
        imgs_hflip = np.copy(imgs)
        #reshape for torch augmentation
        imgs_jitter_1 = torch.tensor(np.transpose(np.copy(imgs), axes=(0, 3, 1, 2)), dtype=torch.float32)
        imgs_jitter_2 = torch.tensor(np.transpose(np.copy(imgs), axes=(0, 3, 1, 2)), dtype=torch.float32)
        imgs_jitter_3 = torch.tensor(np.transpose(np.copy(imgs), axes=(0, 3, 1, 2)), dtype=torch.float32)
        GTs_90 = np.copy(GTs)
        GTs_vflip = np.copy(GTs)
        GTs_hflip = np.copy(GTs)
        GTs_jitter_1 = np.copy(GTs)
        GTs_jitter_2 = np.copy(GTs)
        GTs_jitter_3 = np.copy(GTs)
        for i in range(len(imgs)):
            imgs_90[i] = np.rot90(imgs_90[i])
            imgs_vflip[i] = np.flipud(imgs_vflip[i])
            imgs_hflip[i] = np.fliplr(imgs_hflip[i])
            #perform various jitter augmentations with a new probability each time
            transform_1 = torchvision.transforms.ColorJitter(np.random.uniform(0.0, 0.5),
                                                           np.random.uniform(0.0, 0.5),
                                                           np.random.uniform(0.0, 0.5),
                                                           np.random.uniform(0.0, 0.5))
            transform_2 = torchvision.transforms.ColorJitter(np.random.uniform(0.0, 0.5),
                                                           np.random.uniform(0.0, 0.5),
                                                           np.random.uniform(0.0, 0.5),
                                                           np.random.uniform(0.0, 0.5))
            transform_3 = torchvision.transforms.ColorJitter(np.random.uniform(0.0, 0.5),
                                                           np.random.uniform(0.0, 0.5),
                                                           np.random.uniform(0.0, 0.5),
                                                           np.random.uniform(0.0, 0.5))
            imgs_jitter_1[i] = transform_1(imgs_jitter_1[i])
            imgs_jitter_2[i] = transform_2(imgs_jitter_1[i])
            imgs_jitter_3[i] = transform_3(imgs_jitter_1[i])
            GTs_90[i] = np.rot90(GTs_90[i])
            GTs_vflip[i] = np.flipud(GTs_vflip[i])
            GTs_hflip[i] = np.fliplr(GTs_hflip[i])
        imgs_jitter_1 = np.transpose(np.array(imgs_jitter_1), axes=(0, 2, 3, 1)) #reshape back to normal
        imgs_jitter_2 = np.transpose(np.array(imgs_jitter_2), axes=(0, 2, 3, 1))  # reshape back to normal
        imgs_jitter_3 = np.transpose(np.array(imgs_jitter_3), axes=(0, 2, 3, 1))  # reshape back to normal
        final_crops = np.concatenate((imgs, imgs_90, imgs_vflip, imgs_hflip, imgs_jitter_1, imgs_jitter_2, imgs_jitter_3)) #combine all together
        final_crops_GT = np.concatenate((GTs, GTs_90, GTs_vflip, GTs_hflip, GTs_jitter_1, GTs_jitter_2, GTs_jitter_3))
        return final_crops, final_crops_GT

    def to_dataloader(self):
        img_paths, GT_paths = self.load_imgs()[:2]
        #img_paths = img_paths[:1]
        #GT_paths = GT_paths[:1]
        #Combine results from each image path into one array
        crop_imgs = np.concatenate([self.crop_augment(img_paths[i], GT_paths[i])[0]
                                    for i in tqdm(range(len(img_paths)))], axis=0)
        crop_GTs = np.concatenate([self.crop_augment(img_paths[i], GT_paths[i])[1]
                                   for i in tqdm(range(len(img_paths)))], axis=0)
        #numpy array to torch tensor, move around columns for pytorch convolution
        crop_imgs = np.transpose(crop_imgs, axes=(0, 3, 1, 2))
        crop_GTs = np.transpose(crop_GTs, axes=(0, 3, 1, 2))
        #split into train and mem
        X_train, X_mem, y_train, y_mem = train_test_split(crop_imgs, crop_GTs, train_size=self.train_per)
        #split mem into valid and test
        X_valid, X_test, y_valid, y_test = train_test_split(X_mem, y_mem, test_size=0.33)
        train_data = ImageSet(X_train, y_train) #move to pytorch dataset
        valid_data = ImageSet(X_valid, y_valid)
        test_data = ImageSet(X_test, y_test)
        #init pytorch dataloader
        train_loader = data.DataLoader(train_data, batch_size=self.batch_size,
                                       shuffle=True, drop_last=True, num_workers=self.num_cpu)
        val_loader = data.DataLoader(valid_data, batch_size=self.batch_size,
                                     shuffle=True, drop_last=True, num_workers=self.num_cpu)
        test_loader = data.DataLoader(test_data, batch_size=self.batch_size,
                                      shuffle=True, drop_last=True, num_workers=self.num_cpu)
        return train_loader, val_loader, test_loader

    def crop_recon(self, img):
        img = np.array(cv2.imread(img, 1)) / 255.0 #load and scale img
        imgs = view_as_windows(img, (self.dim, self.dim, 3), step=self.dim)
        a, b, c, d, e, f = imgs.shape
        imgs = imgs.reshape(a*b, self.dim, self.dim, 3)
        imgs = np.transpose(imgs, axes=(0, 3, 1, 2))
        return imgs, a, b

    def eval_dataloader(self):
        eval_paths = self.load_imgs()[2]
        if self.eval_type == 'Windowed':
            #path to crop_recon, concatenate results
            window_imgs = np.concatenate([self.crop_recon(eval_paths[i])[0]
                                          for i in range(len(eval_paths))])
            num_col, num_row = self.crop_recon(eval_paths[0])[1:]
            eval_loader = data.DataLoader(ReconSet(window_imgs), batch_size=self.batch_size,
                                          shuffle=False, drop_last=False, num_workers=self.num_cpu)
            return eval_loader, num_col, num_row

        elif self.eval_type == 'Scaled':
            a, b, c = np.array(cv2.imread(eval_paths[0], 1)).shape
            alpha, beta = int(0.15 * a), int(0.15 * b)
            h = 1024
            w = 1024
            scale_dim = (w, h)
            scaled_imgs = np.concatenate([np.array(cv2.resize(cv2.imread(eval_paths[i], 1),
                                            scale_dim, interpolation=cv2.INTER_NEAREST)).reshape(1, h, w, c) / 255.0
                                            for i in range(len(eval_paths))])
            scaled_imgs = np.transpose(scaled_imgs, axes=(0, 3, 1, 2))
            print(scaled_imgs.shape)
            eval_loader = data.DataLoader(ReconSet(scaled_imgs), batch_size=1, #smaller batch size because bigger than normal runs
                                          shuffle=False, drop_last=False, num_workers=self.num_cpu)
            return eval_loader, None, None

        else:
            print('Wrong eval type, try again.')
            return None


