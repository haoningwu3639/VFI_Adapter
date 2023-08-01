import sys
sys.path.append('.')
import cv2
import math
import torch
import numpy as np
from pytorch_msssim import ssim_matlab
from model.RIFE import Model
from torch.optim import AdamW
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Vimeo(adap_step):
    adap_model = Model()
    adap_model.device()

    lap = nn.L1Loss()
    optimG = AdamW(adap_model.flownet.parameters(), lr=1e-4, weight_decay=1e-3)

    trainable_modules = ("adapter_alpha", "adapter_beta", "adapter_alpha_conv", "adapter_beta_conv")
    for name, module in adap_model.flownet.named_modules():
        if name.endswith(trainable_modules):
            for params in module.parameters():
                params.requires_grad = True
        else:
            for params in module.parameters():
                params.requires_grad = False

    total_params = sum(p.numel() for p in adap_model.flownet.parameters() if p.requires_grad)
    print('the number of network parameters: {}'.format(total_params))

    path = '../../Dataset/Vimeo/vimeo_septuplet/'
    f = open(path + 'sep_testlist.txt', 'r')

    adap_psnr_list = []
    adap_ssim_list = []

    for i in f:
        print(i)
        name = str(i).strip()
        if(len(name) <= 1):
            continue
        I1 = cv2.imread(path + 'sequences/' + name + '/im1.png')
        I3 = cv2.imread(path + 'sequences/' + name + '/im3.png')
        I4 = cv2.imread(path + 'sequences/' + name + '/im4.png')
        I5 = cv2.imread(path + 'sequences/' + name + '/im5.png')
        I7 = cv2.imread(path + 'sequences/' + name + '/im7.png')

        I1 = (torch.tensor(I1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        I3 = (torch.tensor(I3.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        I5 = (torch.tensor(I5.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        I7 = (torch.tensor(I7.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        
        adap_model.load_model('ckpt')
        adap_model.train()
        
        for j in range(adap_step):
            # Cycle Consistency Adaptation
            I2_pred = adap_model.inference(I1, I3)
            I4_pred = adap_model.inference(I3, I5)
            I6_pred = adap_model.inference(I5, I7)
            I3_pred = adap_model.inference(I2_pred, I4_pred)
            I5_pred = adap_model.inference(I4_pred, I6_pred)

            # Direct Adaptation
            # I3_pred = adap_model.inference(I1, I5)
            # I5_pred = adap_model.inference(I3, I7)
            
            I3_loss = lap(I3_pred, I3)
            I5_loss = lap(I5_pred, I5)
            loss = (I3_loss + I5_loss) / 2.
            
            optimG.zero_grad()
            loss.backward()
            optimG.step()

        adap_model.eval()
        
        with torch.no_grad():
            adap_mid = adap_model.inference(I3, I5)[0].clip(0, 1)
            
        adap_ssim = ssim_matlab(torch.tensor(I4.transpose(2, 0, 1)).to(device).unsqueeze(0) / 255., torch.round(adap_mid * 255).unsqueeze(0) / 255.).detach().cpu().numpy()
        adap_mid = np.round((adap_mid * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0) / 255.    
        I4 = I4 / 255.
        adap_psnr = -10 * math.log10(((I4 - adap_mid) * (I4 - adap_mid)).mean())
        
        adap_psnr_list.append(adap_psnr)
        adap_ssim_list.append(adap_ssim)
        print("Avg Adap_PSNR: {} Adap_SSIM: {}".format(np.mean(adap_psnr_list), np.mean(adap_ssim_list)))
    
if __name__ == "__main__":
    adap_step = 10
    Vimeo(adap_step=adap_step)