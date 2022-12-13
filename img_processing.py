import cv2
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from glob import glob
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

img = cv2.imread('eval_img/section 5_z1c1+2+3.png', 1)
tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
b, g, r = cv2.split(img)
rgba = [b,g,r, alpha]
dst = cv2.merge(rgba,4)
cv2.imwrite('test.png', dst)