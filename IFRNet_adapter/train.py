import argparse
import math
import os
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from dataset import VimeoDataset
from model.IFRNet_L import Model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_lr(args, iters):
    ratio = 0.5 * (1.0 + np.cos(iters / (args.epochs * args.iters_per_epoch) * math.pi))
    lr = (args.lr_start - args.lr_end) * ratio + args.lr_end
    return lr

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args, ddp_model):
    local_rank = args.local_rank
    print('Distributed Data Parallel Training IFRNet on Rank {}'.format(local_rank))
    if local_rank == 0:
        os.makedirs(args.log_path, exist_ok=True)

    dataset_train = VimeoDataset('train')
    sampler = DistributedSampler(dataset_train)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
    args.iters_per_epoch = dataloader_train.__len__()
    iters = args.resume_epoch * args.iters_per_epoch
    
    dataset_val = VimeoDataset('val')
    dataloader_val = DataLoader(dataset_val, batch_size=4, num_workers=4, pin_memory=True, shuffle=False, drop_last=True)

    optimizer = optim.AdamW(ddp_model.parameters(), lr=args.lr_start, weight_decay=0)
    best_psnr = 0.0

    for epoch in range(args.resume_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for i, data in enumerate(dataloader_train):
            for l in range(len(data)):
                data[l] = data[l].to(args.device)
            img0, imgt, img1, embt = data
            img0 = img0 / 255.0
            img1 = img1 / 255.0
            imgt = imgt / 255.0

            lr = get_lr(args, iters)
            set_lr(optimizer, lr)
            optimizer.zero_grad()

            imgt_pred, loss_rec = ddp_model(img0, img1, embt, imgt, flow=None)
            loss = loss_rec
            loss.backward()
            optimizer.step()

            if (iters+1) % 100 == 0 and local_rank == 0:
                print('epoch:{}/{} iter:{}/{} lr:{:.5e} loss:{:.5e} psnr:{:.4f}'.format(epoch+1, args.epochs, iters+1, args.epochs * args.iters_per_epoch, lr, loss_rec, best_psnr))
            iters += 1

        if (epoch + 1) % args.eval_interval == 0 and local_rank == 0:
            psnr = evaluate(args, ddp_model, dataloader_val, epoch)
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(ddp_model.module.state_dict(), '{}/IFRNet_{}.pth'.format(args.log_path, 'best'))

        dist.barrier()

def evaluate(args, ddp_model, dataloader_val, epoch):
    psnr_list = []
    for i, data in enumerate(dataloader_val):
        for l in range(len(data)):
            data[l] = data[l].to(args.device)
        img0, imgt, img1, embt = data
        img0 = img0 / 255.0
        img1 = img1 / 255.0
        imgt = imgt / 255.0

        with torch.no_grad():
            imgt_pred, loss_rec = ddp_model(img0, img1, embt, imgt, flow=None)

        for j in range(img0.shape[0]):
            psnr = -10 * math.log10(torch.mean((imgt_pred[j] - imgt[j]) * (imgt_pred[j] - imgt[j])).cpu().data)
            psnr_list.append(psnr)

    print('eval epoch:{}/{} psnr:{:.3f}'.format(epoch+1, args.epochs, np.array(psnr_list).mean()))
    return np.array(psnr_list).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IFRNet')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--eval_interval', default=1, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr_start', default=3e-4, type=float)
    parser.add_argument('--lr_end', default=3e-5, type=float)
    parser.add_argument('--log_path', default='./ckpt', type=str)
    parser.add_argument('--resume_epoch', default=0, type=int)
    parser.add_argument('--resume_path', default='./checkpoints/IFRNet_large/IFRNet_L_Vimeo90K.pth', type=str)
    args = parser.parse_args()

    dist.init_process_group(backend='nccl', world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank)

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    model = Model().to(args.device)
        
    model_dict = model.state_dict()
    ckpt = torch.load(args.resume_path, map_location='cpu')
    model_dict.update(ckpt)
    model.load_state_dict(model_dict)
    
    for name, module in model.named_modules():
        if "adapter_alpha" in name or "adapter_beta" in name or "adapter_alpha_conv" in name or "adapter_beta_conv" in name:
            for params in module.parameters():
                params.requires_grad = True
        else:
            for params in module.parameters():
                params.requires_grad = False
    
    ddp_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    train(args, ddp_model)
    dist.destroy_process_group()