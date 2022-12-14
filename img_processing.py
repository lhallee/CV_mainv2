import cv2
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from glob import glob
from matplotlib import pyplot as plt
'''
full_LN_path = './aligned_imgs/'
save_path = './squared_aligned_ln_imgs/'
LN_paths = natsorted(glob(full_LN_path + '*.tif'))

max_H = 0
max_W = 0
dim = 0
heights = []
widths = []
for i in tqdm(range(len(LN_paths))):
	img = np.array(cv2.imread(LN_paths[i], 1))
	H, W, C = img.shape
	if H > max_H:
		max_H = H
	if W > max_W:
		max_W = W
	heights.append(H)
	widths.append(W)
if max_H >= max_W:
	dim = max_H
if max_W > max_H:
	dim = max_W

for i in tqdm(range(len(LN_paths))):
	img = cv2.imread(LN_paths[i], 1)
	old_H, old_W = heights[i], widths[i]
	dif_H, dif_W = dim - old_H, dim - old_W
	top = int(dif_H / 2)
	bottom = int(dif_H - top)
	right = int(dif_W / 2)
	left = int(dif_W - right)
	path = save_path + 'squared_' + LN_paths[i].split('\\')[1] + '.png'
	new_img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
	cv2.imwrite(path, new_img)
'''
hev_masks = './GT/'
lob_masks = './lobule masks/'
hev = natsorted(glob(hev_masks + '*.png'))
lob = natsorted(glob(lob_masks + '*.png'))

hev_test = np.array(cv2.imread(hev[0], 2))
hev_test = hev_test.reshape(hev_test.shape[0], hev_test.shape[1], 1)
lob_test = np.array(cv2.imread(lob[0], 2))
lob_test = lob_test.reshape(lob_test.shape[0], lob_test.shape[1], 1)

both = np.concatenate((hev_test, lob_test), axis=2)
rows = 1
    columns = num_class
    SR = np.transpose(np.array(SR), axes=(0, 2, 3, 1))
    GT = np.transpose(np.array(GT), axes=(0, 2, 3, 1))
    feed_img = np.transpose(np.array(feed_img), axes=(0, 2, 3, 1))
    i = np.random.randint(0, len(SR))
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    plt.imshow(SR[i][:,:,0])
    plt.axis('off')
    plt.title('Img')
    if num_class == 1:
        fig.add_subplot(rows, columns, 2)
        plt.imshow(GT[i][:, :, 0], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title('GT')
        plt.show()
    else:
        fig.add_subplot(rows, columns, 2)
        plt.imshow(GT[i][:, :, 0], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title('GT')
        fig.add_subplot(rows, columns, 3)
        plt.imshow(GT[i][:, :, 1], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title('GT')
        plt.show()
