import argparse
import math
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from dataset import VimeoDataset
from model.pipeline import Pipeline
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

def get_learning_rate(total_step, cur_step, init_lr, min_lr=1e-6):
    if cur_step < 1000:
        mul = cur_step / 1000.
        return init_lr * mul
    else:
        mul = np.cos((cur_step - 1000) / (total_step - 1000.) * math.pi) * 0.5 + 0.5
        return  (init_lr - min_lr) * mul + min_lr

def train(ppl, dataset_cfg_dict, optimizer_cfg_dict):
    model_log_dir = os.path.join(TRAIN_LOG_ROOT, "trained-models")
    os.makedirs(model_log_dir, exist_ok=True)
    tf_log_dir = os.path.join(TRAIN_LOG_ROOT, "tensorboard")

    # dataset config
    batch_size = dataset_cfg_dict.get("batch_size", 32)
    dataset_train = VimeoDataset(dataset_name='train')
    dataset_val = VimeoDataset(dataset_name='validation')

    sampler = DistributedSampler(dataset_train)
    train_data = DataLoader(dataset_train, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
    val_data = DataLoader(dataset_val, batch_size=4, num_workers=4, pin_memory=True)

    # optimizer config
    total_step = int(optimizer_cfg_dict["steps"])
    save_interval = int(optimizer_cfg_dict["save_interval"])
    init_lr = optimizer_cfg_dict.get("init_lr", 3e-4)
    min_lr = optimizer_cfg_dict.get("min_lr", 3e-5)

    step = 1
    psnr = 0.0

    if LOCAL_RANK == 0:
        writer = SummaryWriter(tf_log_dir + '/train')

    step_per_epoch = len(train_data)
    epoch_counter = 0
    last_epoch = False

    while step <  total_step + 1:
        if step + step_per_epoch > total_step:
            last_epoch = True

        epoch_counter += 1
        sampler.set_epoch(epoch_counter)

        for data in train_data:
            data_gpu = data.to(DEVICE, dtype=torch.float, non_blocking=True) / 255.
            img0 = data_gpu[:, :3]
            img1 = data_gpu[:, 3:6]
            gt = data_gpu[:, 6:9]
            learning_rate = get_learning_rate(total_step, step, init_lr, min_lr)
            pred, extra_dict = ppl.train_one_iter(img0, img1, gt, learning_rate=learning_rate)

            if step % 1000 == 1 and LOCAL_RANK == 0:
                gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                nr_show = min(4, batch_size)
                for i in range(nr_show):
                    imgs = np.concatenate((gt[i], pred[i]), 1)[:, :, ::-1]
                    writer.add_image(str(i) + '/gt-pred', imgs, step, dataformats='HWC')
                writer.flush()
            if LOCAL_RANK == 0:
                print("train step: {}/{}; learning_rate: {:.4e}; loss: {:.4e}, psnr: {:.4f}".format(
                            step, total_step, learning_rate, extra_dict['loss'], psnr))

            if (LOCAL_RANK == 0) and (step % save_interval == 0):
                psnr_temp = evaluate(ppl, val_data)
                if psnr_temp > psnr:
                    ppl.save_model(model_log_dir, LOCAL_RANK)
                psnr = max(psnr, psnr_temp)
                print("val step: {}; psnr: {:.4f}".format(step, psnr))
                    
            step += 1
            if last_epoch and step == total_step + 1:
                break
        dist.barrier()

def evaluate(ppl, val_data):
    if LOCAL_RANK == 0:
        psnr_list = []
        for i, data in enumerate(val_data):
            data_gpu = data[0] if isinstance(data, list) else data
            data_gpu = data_gpu.to(DEVICE, dtype=torch.float, non_blocking=True) / 255.
            img0 = data_gpu[:, :3]
            img1 = data_gpu[:, 3:6]
            gt = data_gpu[:, 6:9]
            with torch.no_grad():
                pred = ppl.inference(img0, img1)
            for j in range(gt.shape[0]):
                psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
                psnr_list.append(psnr)
        psnr = np.array(psnr_list).mean()
        
    return psnr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UPRNet_Adapter')
    parser.add_argument('--train_log_root', default="./upr-train-log", type=str,
            help='root dir to save all training logs')
    # => args for distributed training
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=1, type=int, help='world size')
    # => args for data loader
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for data loader')
    # => args for model
    parser.add_argument('--pyr_level', type=int, default=3,
            help='the number of pyramid levels of UPR-Net during training')
    parser.add_argument('--load_pretrain', type=bool, default=True, help='whether load pre-trained weight')
    parser.add_argument('--model_file', type=str, default="./checkpoints/upr-llarge.pkl",
            help='weight of UPR-Net')
    # => args for optimizer
    parser.add_argument('--init_lr', type=float, default=3e-4, help='init learning rate')
    parser.add_argument('--min_lr', type=float, default=3e-5, help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='wegith decay')
    parser.add_argument('--steps', type=float, default=50000,
            help='total steps (iteration) for training')
    parser.add_argument('--save_interval', type=float, default=2e3,
            help='iteration interval to save model')

    args = parser.parse_args()

    model_cfg_dict = dict(
            pyr_level = args.pyr_level,
            load_pretrain = args.load_pretrain,
            model_file = args.model_file
            )

    optimizer_cfg_dict = dict(
            init_lr=args.init_lr,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            steps=args.steps,
            save_interval=args.save_interval,
            )

    dataset_cfg_dict = dict(batch_size=args.batch_size)

    # => parse args and init the training environment global variable
    TRAIN_LOG_ROOT = args.train_log_root
    LOCAL_RANK = args.local_rank
    WORLD_SIZE = args.world_size
    DEVICE = torch.device("cuda", LOCAL_RANK)
    
    if args.load_pretrain:
        model_cfg_dict["load_pretrain"] = True

    torch.distributed.init_process_group(backend="nccl", world_size=WORLD_SIZE)
    torch.cuda.set_device(LOCAL_RANK)
    seed = 6666
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    # => init the pipeline and train the pipeline
    ppl = Pipeline(model_cfg_dict, optimizer_cfg_dict, LOCAL_RANK, training=True)
    
    for name, module in ppl.model.named_modules():
        if "adapter_alpha" in name or "adapter_beta" in name or "adapter_alpha_conv" in name or "adapter_beta_conv" in name:
            for params in module.parameters():
                params.requires_grad = True
        else:
            for params in module.parameters():
                params.requires_grad = False
    
    train(ppl, dataset_cfg_dict, optimizer_cfg_dict)
