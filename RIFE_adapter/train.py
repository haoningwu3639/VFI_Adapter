import math
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from model.RIFE import Model
from dataset import VimeoDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

device = torch.device("cuda")

def get_learning_rate(step):
    if step < 1000:
        mul = step / 1000.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 1000) / (args.epoch * args.step_per_epoch - 1000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-5) * mul + 3e-5

def train(model, local_rank):
    if local_rank == 0:
        writer = SummaryWriter('train')
        writer_val = SummaryWriter('validate')
    else:
        writer = None
        writer_val = None
    step = 0
    psnr = 0.0
    
    dataset = VimeoDataset('train')
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.train_batch, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()
    dataset_val = VimeoDataset('test')
    val_data = DataLoader(dataset_val, batch_size=args.val_batch, pin_memory=True, num_workers=4)

    for epoch in range(args.epoch):
        sampler.set_epoch(epoch)
        for i, data in enumerate(train_data):
            data = data.to(device, non_blocking=True) / 255.
            imgs = data[:, :6]
            gt = data[:, 6:9]
            learning_rate = get_learning_rate(step)
            pred, info = model.update(imgs, gt, learning_rate, training=True)            
            if step % 500 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/l1', info['loss_l1'], step)
            if step % 1000 == 1 and local_rank == 0:
                gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                
                for i in range(args.val_batch):
                    imgs = np.concatenate((pred[i], gt[i]), 1)[:, :, ::-1]
                    writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                writer.flush()
            if local_rank == 0:
                print('epoch:{} {}/{} learning_rate:{:.4e} loss_l1:{:.4e}, psnr:{:.4}'.format(epoch, i, args.step_per_epoch, learning_rate, info['loss_l1'], psnr))
            step += 1
        
        if epoch % 3 == 0 and local_rank == 0:
            temp = evaluate(model, val_data, step, local_rank, writer_val)
            if temp and temp > psnr:
                psnr = temp
            model.save_model(args.log_path, local_rank)
        
        dist.barrier()

def evaluate(model, val_data, nr_eval, local_rank, writer_val):
    psnr_list = []
    
    for i, data in enumerate(val_data):
        data = data.to(device, non_blocking=True) / 255.
        imgs = data[:, :6]
        gt = data[:, 6:9]
        with torch.no_grad():
            pred, info = model.update(imgs, gt, training=False)
        for j in range(gt.shape[0]):
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
        gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        if i == 0 and local_rank == 0:
            for j in range(args.val_batch):
                imgs = np.concatenate((pred[j], gt[j]), 1)[:, :, ::-1]
                writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
    
    writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
    if local_rank == 0:
        return np.array(psnr_list).mean()
    else: 
        return
    

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--train_batch', default=16, type=int)
    parser.add_argument('--val_batch', default=4, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--log_path', default='./ckpt', type=str)
    
    args = parser.parse_args()
    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args.local_rank)
    model.load_model('train_log', testing=False)
    
    trainable_modules = ("adapter_alpha", "adapter_beta", "adapter_alpha_conv", "adapter_beta_conv")
    for name, module in model.flownet.named_modules():
        if name.endswith(trainable_modules):
            for params in module.parameters():
                params.requires_grad = True
        else:
            for params in module.parameters():
                params.requires_grad = False
                # params.requires_grad = True
    
    train(model, args.local_rank)

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nrpoc_per_node=4 train.py --train_batch 64 --world_size=1