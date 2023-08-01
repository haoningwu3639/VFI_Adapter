import math
import sys
sys.path.append('.')
import cv2
import torch
import numpy as np
import torch.nn as nn
from model.pipeline import Pipeline
from model.util import ImagePadder
from pytorch_msssim import ssim_matlab
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def SNU_FILM(adap_step):
    width = 720
    PYR_LEVEL = math.ceil(math.log2(width / 448) + 3)
    padder = ImagePadder(dims=(3, 720, 1280), factor=32, mode='sintel')

    model_cfg_dict = dict(load_pretrain = True, model_size = "LARGE", model_file = "../upr-train-log/trained-models/model.pkl")
    adap_model = Pipeline(model_cfg_dict)
    adap_model.eval()

    lap = nn.L1Loss()
    f = open('../../Dataset/SNU-FILM/index_septuplet/test_extreme.txt', 'r')

    adap_psnr_list = []
    adap_ssim_list = []

    for i in f:        
        print(i)
        name = str(i).strip()

        if(len(name) <= 1):
            continue
        frames = i.strip().split(' ')
        
        I1 = cv2.imread(frames[0].replace('data/', '../../Dataset/'))
        I3 = cv2.imread(frames[2].replace('data/', '../../Dataset/'))
        I4 = cv2.imread(frames[3].replace('data/', '../../Dataset/'))
        I5 = cv2.imread(frames[4].replace('data/', '../../Dataset/'))
        I7 = cv2.imread(frames[6].replace('data/', '../../Dataset/'))

        I1 = (torch.tensor(I1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        I3 = (torch.tensor(I3.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        I5 = (torch.tensor(I5.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)    
        I7 = (torch.tensor(I7.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        
        I1 = padder.pad(I1)
        I3 = padder.pad(I3)
        I5 = padder.pad(I5)
        I7 = padder.pad(I7)
        
        adap_model = Pipeline(model_cfg_dict)

        for name, module in adap_model.model.named_modules():
            if "adapter_alpha" in name or "adapter_beta" in name or "adapter_alpha_conv" in name or "adapter_beta_conv" in name:
                for params in module.parameters():
                    params.requires_grad = True
            else:
                for params in module.parameters():
                    params.requires_grad = False
        optimG = AdamW(adap_model.model.parameters(), lr=1e-4, weight_decay=1e-3)
        adap_model.train()
        
        for j in range(adap_step):
            
            with torch.no_grad(): # NOTE: Due to limitation of GPU memomry, we do not calculate gradient here.
                I2_pred = adap_model.inference(I1, I3, time_period=0.5, pyr_level=PYR_LEVEL).clip(0, 1)
                I4_pred = adap_model.inference(I3, I5, time_period=0.5, pyr_level=PYR_LEVEL).clip(0, 1)
                I6_pred = adap_model.inference(I5, I7, time_period=0.5, pyr_level=PYR_LEVEL).clip(0, 1)
            # Cycle-Consistency Adaptation
            I3_pred = adap_model.inference(I2_pred, I4_pred, time_period=0.5, pyr_level=PYR_LEVEL)
            I5_pred = adap_model.inference(I4_pred, I6_pred, time_period=0.5, pyr_level=PYR_LEVEL)
            # Direct Adaptation            
            # I3_pred = adap_model.inference(I1, I5, time_period=0.5, pyr_level=PYR_LEVEL)
            # I5_pred = adap_model.inference(I3, I7, time_period=0.5, pyr_level=PYR_LEVEL)
            
            I3_loss = lap(I3_pred, I3)
            I5_loss = lap(I5_pred, I5)
            loss = (I3_loss + I5_loss) / 2.
            
            optimG.zero_grad()
            loss.backward()
            optimG.step()
            
        adap_model.eval()
        
        with torch.no_grad():
            adap_mid = adap_model.inference(I3, I5, time_period=0.5, pyr_level=PYR_LEVEL)[0].clip(0, 1)
            adap_mid = padder.unpad(adap_mid)

        adap_ssim = ssim_matlab(torch.tensor(I4.transpose(2, 0, 1)).to(device).unsqueeze(0) / 255., torch.round(adap_mid * 255).unsqueeze(0) / 255.).detach().cpu().numpy()
        adap_mid = np.round((adap_mid * 255.).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0) / 255.    
        I4 = I4 / 255.
        adap_psnr = -10 * math.log10(((I4 - adap_mid) * (I4 - adap_mid)).mean())
        
        adap_psnr_list.append(adap_psnr)
        adap_ssim_list.append(adap_ssim)
        print("Avg Adap_PSNR: {} Adap_SSIM: {}".format(np.mean(adap_psnr_list), np.mean(adap_ssim_list)))

if __name__ == "__main__":
    adap_step = 10
    SNU_FILM(adap_step=adap_step)