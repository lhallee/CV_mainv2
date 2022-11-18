import os
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from metrics import *
from models import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from plots import checker, test_saver
import csv


class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.best_unet = None
		self.unet_path = None
		self.optimizer = None
		self.best_epoch = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = torch.nn.BCELoss()

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		self.scheduler = config.scheduler
		# Training settings
		self.num_epochs = config.num_epochs
		self.batch_size = config.batch_size

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type == 'U_Net':
			self.unet = U_Net(img_ch=self.img_ch, output_ch=self.output_ch)
		elif self.model_type == 'R2U_Net':
			self.unet = R2U_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t)
		elif self.model_type == 'AttU_Net':
			self.unet = AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch)
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t)

		self.optimizer = optim.Adam(list(self.unet.parameters()), self.lr, (self.beta1, self.beta2))
		if self.scheduler == 'exp':
			self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99, last_epoch=-1)
		self.unet.to(self.device)
		#self.print_network(self.unet, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def compute_accuracy(self, SR, GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)

		acc = GT_flat.data.cpu() == (SR_flat.data.cpu() > 0.5)
	'''
	def tensor2img(self, x):
		img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
		img = img * 255
		return img
	'''
	def train(self):
		self.unet_path = os.path.join(self.model_path, '%s-%d-%.4f.pkl' % (self.model_type, self.num_epochs, self.lr))
		for epoch in range(self.num_epochs):
			epoch_loss = 0
			acc = 0.  # Accuracy
			SE = 0.  # Sensitivity (Recall)
			SP = 0.  # Specificity
			PC = 0.  # Precision
			F1 = 0.  # F1 Score
			JS = 0.  # Jaccard Similarity
			DC = 0.  # Dice Coefficient
			length = 0
			pbar_train = tqdm(total=len(self.train_loader), desc='Training')
			batch = 0
			for images, GT in self.train_loader:
				images = images.to(self.device)
				GT = GT.to(self.device)
				# SR : Segmentation Result
				SR = self.unet(images)
				SR_probs = torch.sigmoid(SR)
				#SR_flat = SR_probs.view(SR_probs.size(0), -1)
				#GT_flat = GT.view(GT.size(0), -1)

				loss = self.criterion(SR_probs, GT)
				if batch % 50 == 0:
					checker(path=self.result_path, imgs=SR_probs, GTs=GT, epoch=epoch, batch=batch, num_class=self.output_ch + 1)

				epoch_loss += loss.item()

				# Backprop + optimize
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				if self.scheduler is not None:
					self.scheduler.step()

				acc += get_accuracy(SR, GT)
				SE += get_sensitivity(SR, GT)
				SP += get_specificity(SR, GT)
				PC += get_precision(SR, GT)
				F1 += get_F1(SR, GT)
				JS += get_JS(SR, GT)
				DC += get_DC(SR, GT)
				length += images.size(0)
				batch += 1
				pbar_train.update(1)

			acc = acc / length
			SE = SE / length
			SP = SP / length
			PC = PC / length
			F1 = F1 / length
			JS = JS / length
			DC = DC / length

			# Print the log info
			print(
				'Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
					epoch + 1, self.num_epochs,
					epoch_loss,
					acc, SE, SP, PC, F1, JS, DC)
			)
			pbar_train.close()
			self.valid(epoch)
	@torch.no_grad()
	def valid(self, epoch):
		best_unet_score = 0.0
		acc = 0.  # Accuracy
		SE = 0.  # Sensitivity (Recall)
		SP = 0.  # Specificity
		PC = 0.  # Precision
		F1 = 0.  # F1 Score
		JS = 0.  # Jaccard Similarity
		DC = 0.  # Dice Coefficient
		length = 0
		for images, GT in self.valid_loader:
			images = images.to(self.device)
			GT = GT.to(self.device)
			SR = torch.sigmoid(self.unet(images))
			acc += get_accuracy(SR, GT)
			SE += get_sensitivity(SR, GT)
			SP += get_specificity(SR, GT)
			PC += get_precision(SR, GT)
			F1 += get_F1(SR, GT)
			JS += get_JS(SR, GT)
			DC += get_DC(SR, GT)

			length += images.size(0)

		acc = acc / length
		SE = SE / length
		SP = SP / length
		PC = PC / length
		F1 = F1 / length
		JS = JS / length
		DC = DC / length
		unet_score = JS + DC

		print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
		acc, SE, SP, PC, F1, JS, DC))


		'''
				torchvision.utils.save_image(images.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_image.png'%(self.model_type,epoch+1)))
				torchvision.utils.save_image(SR.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
				torchvision.utils.save_image(GT.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
		'''

		# Save Best U-Net model
		if unet_score > best_unet_score:
			best_unet_score = unet_score
			self.best_epoch = epoch
			self.best_unet = self.unet.state_dict()
			print('Best %s model score : %.4f' % (self.model_type, best_unet_score))
			torch.save(self.best_unet, self.unet_path)

	@torch.no_grad()
	def test(self):
		# ===================================== Test ====================================#
		if self.best_unet is not None:
			self.unet = self.best_unet
		else:
			self.build_model()
			self.unet.load_state_dict(torch.load(self.unet_path))

		acc = 0.  # Accuracy
		SE = 0.  # Sensitivity (Recall)
		SP = 0.  # Specificity
		PC = 0.  # Precision
		F1 = 0.  # F1 Score
		JS = 0.  # Jaccard Similarity
		DC = 0.  # Dice Coefficient
		length = 0
		pbar_test = tqdm(total=len(self.test_loader), desc='Testing')
		batch = 0
		for images, GT in self.test_loader:
			batch += 1
			images = images.to(self.device)
			GT = GT.to(self.device)
			SR = torch.sigmoid(self.unet(images))
			test_saver(path=self.result_path, imgs=SR, GTs=GT, batch=batch)
			acc += get_accuracy(SR, GT)
			SE += get_sensitivity(SR, GT)
			SP += get_specificity(SR, GT)
			PC += get_precision(SR, GT)
			F1 += get_F1(SR, GT)
			JS += get_JS(SR, GT)
			DC += get_DC(SR, GT)
			length += images.size(0)
			pbar_test.update(1)

		acc = acc / length
		SE = SE / length
		SP = SP / length
		PC = PC / length
		F1 = F1 / length
		JS = JS / length
		DC = DC / length
		unet_score = JS + DC

		f = open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
		wr = csv.writer(f)
		wr.writerow([self.model_type, acc, SE, SP, PC, F1, JS, DC, self.lr, self.best_epoch, self.num_epochs])
		f.close()
		pbar_test.close()
