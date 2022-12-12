import os
import numpy as np
import torch
from tqdm import tqdm
from models import U_Net, R2U_Net, AttU_Net, R2AttU_Net




class eval_solver:
    def __init__(self, config, eval_loader, num_col=0, num_row=0):
        self.eval_loader = eval_loader
        self.num_col = num_col
        self.num_row = num_row
        self.model_type = config.model_type
        self.t = config.t
        self.unet = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.batch_size = config.batch_size
        self.dim = config.image_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_model(self):
        if self.model_type == 'U_Net':
            self.unet = U_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t)
        self.unet.to(self.device)

    @torch.no_grad()  # don't update weights while evaluating
    def eval(self):
        self.build_model()  # rebuild model
        self.unet.load_state_dict(torch.load(self.model_path))  # load pretrained weights
        loop = tqdm(self.eval_loader, leave=True)
        SRs = np.concatenate([self.unet(batch.to(self.device)).detach().cpu().numpy() for batch in loop])
        if self.eval_type == 'Windowed':
            print(SRs.shape)
            assert len(SRs) % self.num_col * self.num_row, 'Check SRs is sum of images, not sum of batches'
        elif self.eval_type == 'Scaled':
            pass