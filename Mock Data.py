import cv2 as cv
import numpy as np


def create_square(dim, x, y, z):
	RED = (0, 0, 255)
	pts = [(x - z, y - z), (x + z, y - z), (x + z, y + z), (x - z, y + z)]
	img = np.zeros((dim, dim, 3), np.uint8)
	img = cv.fillPoly(img, np.array([pts]), RED)
	img = img.reshape(1, dim, dim, 3)
	return img


imgs = np.concatenate([create_square(400, 200, 200, i) for i in range(15)])

test = np.mean(np.zeros((128, 128, 3)), -1)
print(test.shape)
