import os
import torch
from torch import optim
from tqdm import tqdm
from metrics import _calculate_overlap_metrics
from models import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from custom_model_components import *
from plots import checker, test_saver
import csv


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):
        #Dataloaders
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        #Model
        self.model_type = config.model_type
        self.t = config.t
        self.unet = None
        self.best_unet = None
        self.unet_path = None
        self.optimizer = None
        self.best_epoch = None
        self.criterion = None
        self.loss = config.loss
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.data_type = config.data_type
        self.scheduler = config.scheduler
        #Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        #Training settings
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.best_unet_score = 0
        #Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        #MISC
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unet_path = self.model_path + self.model_type + self.data_type + self.loss + str(self.num_epochs) + str(
            self.lr) + '.pkl' #descriptive name from settings
        self.progress = config.progress

    def build_model(self):
        if self.model_type == 'U_Net':
            self.unet = U_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t)

        if self.loss == 'BCE':
            self.criterion = nn.BCELoss()
        elif self.loss == 'DiceBCE':
            self.criterion = DiceBCELoss()
        elif self.loss == 'IOU':
            self.criterion = IoULoss()
        elif self.loss == 'CE':
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(list(self.unet.parameters()), self.lr, (self.beta1, self.beta2))
        if self.scheduler == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99, last_epoch=-1)
        elif self.scheduler == 'cosine':
            self.scheduler = CosineWarmupScheduler(self.optimizer, warmup=len(self.train_loader) * 4,
                                                   max_iters=len(self.train_loader) * self.num_epochs)
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

    def train(self):
        self.build_model()
        if self.model_path is not None:
            self.unet.load_state_dict(torch.load(self.model_path, map_location=self.device))
        epoch = 0
        #Unet score is the average of our metrics calculated from validation
        while epoch < self.num_epochs and self.best_unet_score < 0.95:
            epoch_loss = 0
            acc = 0.  # Accuracy
            RE = 0.  # Sensitivity (Recall)
            SP = 0.  # Specificity
            PC = 0.  # Precision
            F1 = 0.  # F1 Score
            DC = 0.  # Dice Coefficient
            length = 0
            pbar_train = tqdm(total=len(self.train_loader), desc='Training')
            for images, GT in self.train_loader:
                images = images.to(self.device)
                GT = GT.to(self.device)
                SR = self.unet(images) #SR : Segmentation Result
                loss = self.criterion(SR, GT)
                #if progess = True call checker to image SR and GT
                if self.progress and length == 0:
                    checker(path=self.result_path, feed_img=images.detach().cpu().numpy(),
                            SR=SR.detach().cpu().numpy(), GT=GT.detach().cpu().numpy(),
                            epoch=epoch, batch=length, num_class=self.output_ch + 1)
                epoch_loss += loss.item()
                #Backprop + optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                #calculate metrics
                _acc, _DC, _PC, _RE, _SP, _F1 = _calculate_overlap_metrics(SR.detach().cpu(), GT.detach().cpu())
                acc += _acc.item()
                DC += _DC.item()
                RE += _RE.item()
                SP += _SP.item()
                PC += _PC.item()
                F1 += _F1.item()
                length += 1 #for epoch metrics and number of batches
                pbar_train.update(1) #progress bar update
            acc = acc / length
            RE = RE / length
            SP = SP / length
            PC = PC / length
            F1 = F1 / length
            DC = DC / length
            # Print the log info
            print(
                'Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, RE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, DC: %.4f' % (
                    epoch + 1, self.num_epochs, epoch_loss, acc, RE, SP, PC, F1, DC))
            pbar_train.close() #close progress bar
            self.valid(epoch) #call validation
            epoch += 1

    @torch.no_grad() #don't update weights during validation
    def valid(self, epoch):
        acc = 0.  # Accuracy
        RE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        DC = 0.  # Dice Coefficient
        length = 0
        for images, GT in self.valid_loader:
            images = images.to(self.device)
            GT = GT.to(self.device)
            SR = self.unet(images)
            #Calculate metrics
            _acc, _DC, _PC, _RE, _SP, _F1 = _calculate_overlap_metrics(SR.detach().cpu(), GT.detach().cpu())
            acc += _acc.item()
            DC += _DC.item()
            RE += _RE.item()
            SP += _SP.item()
            PC += _PC.item()
            F1 += _F1.item()
            length += 1
        acc = acc / length
        RE = RE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        DC = DC / length
        #unet_score is an average of metrics
        unet_score = (F1 + DC + PC + SP + RE + acc) / 6
        print('[Validation] Acc: %.4f, RE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, DC: %.4f' % (
            acc, RE, SP, PC, F1, DC))
        #Save Best U-Net model
        if unet_score > self.best_unet_score:
            self.best_unet_score = unet_score
            self.best_epoch = epoch
            self.best_unet = self.unet.state_dict()
            print('Best %s model score : %.4f' % (self.model_type, self.best_unet_score))
            torch.save(self.best_unet, self.unet_path)
            #only saved when validation metrics are improved on average

    @torch.no_grad() #don't update weights while testing
    def test(self):
        del self.unet #remove unet and best_unet, in case testing from training mode
        del self.best_unet
        self.build_model() #rebuild model
        self.unet.load_state_dict(torch.load(self.unet_path)) #load pretrained weights
        acc = 0.  # Accuracy
        RE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        DC = 0.  # Dice Coefficient
        length = 0
        pbar_test = tqdm(total=len(self.test_loader), desc='Testing') #progress bar
        for images, GT in self.test_loader:
            images = images.to(self.device)
            GT = GT.to(self.device)
            SR = self.unet(images)
            #Save test images
            test_saver(path=self.result_path, feed_img=images.detach().cpu().numpy(),
                       SR=SR.detach().cpu().numpy(), GT=GT.detach().cpu().numpy(), batch=length)
            _acc, _DC, _PC, _RE, _SP, _F1 = _calculate_overlap_metrics(SR.detach().cpu(), GT.detach().cpu())
            acc += _acc.item()
            DC += _DC.item()
            RE += _RE.item()
            SP += _SP.item()
            PC += _PC.item()
            F1 += _F1.item()
            length += 1
            pbar_test.update(1)
        acc = acc / length
        RE = RE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        DC = DC / length
        unet_score = (F1 + DC + PC + SP + RE + acc) / 6
        #Store results in result path
        f = open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow([self.model_type, acc, RE, SP, PC, F1, DC, unet_score, self.lr, self.best_epoch, self.num_epochs])
        f.close() #close file
        pbar_test.close() #close progress bar



