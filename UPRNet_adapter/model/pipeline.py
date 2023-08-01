import os
import torch
from torch.optim import AdamW
import itertools
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from model.upr_llarge import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Pipeline:
    def __init__(self, model_cfg_dict, optimizer_cfg_dict=None, local_rank=-1, training=False):
        self.model_cfg_dict = model_cfg_dict
        self.optimizer_cfg_dict = optimizer_cfg_dict

        self.init_model()
        self.device()
        self.lap = nn.L1Loss()
        self.training = training

        if training:
            self.optimG = AdamW(itertools.chain(
                filter(lambda p: p.requires_grad, self.model.parameters())),
                lr=optimizer_cfg_dict["init_lr"],
                weight_decay=optimizer_cfg_dict["weight_decay"])

        # `local_rank == -1` is used for testing, which does not need DDP
        if local_rank != -1:
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def device(self):
        self.model.to(DEVICE)

    @staticmethod
    def convert_state_dict(rand_state_dict, pretrained_state_dict):
        param =  {k.replace("module.", "", 1): v for k, v in pretrained_state_dict.items()}
        param = {k: v
                for k, v in param.items()
                if ((k in rand_state_dict) and (rand_state_dict[k].shape == param[k].shape))
                }
        rand_state_dict.update(param)
        return rand_state_dict

    def init_model(self):

        def load_pretrained_state_dict(model, model_file):
            if (model_file == "") or (not os.path.exists(model_file)):
                raise ValueError("Please set the correct path for pretrained model!")

            rand_state_dict = model.state_dict()
            pretrained_state_dict = torch.load(model_file)

            return Pipeline.convert_state_dict(rand_state_dict, pretrained_state_dict)

        # check args
        model_cfg_dict = self.model_cfg_dict
        pyr_level = model_cfg_dict["pyr_level"] if "pyr_level" in model_cfg_dict else 3
        nr_lvl_skipped = model_cfg_dict["nr_lvl_skipped"] if "nr_lvl_skipped" in model_cfg_dict else 0
        load_pretrain = model_cfg_dict["load_pretrain"] if "load_pretrain" in model_cfg_dict else False
        model_file = model_cfg_dict["model_file"] if "model_file" in model_cfg_dict else ""

        self.model = Model(pyr_level, nr_lvl_skipped)
        # load pretrained model weight
        if load_pretrain:
            state_dict = load_pretrained_state_dict(self.model, model_file)
            self.model.load_state_dict(state_dict)
        else:
            print("Train from random initialization.")

    def save_model(self, path, rank):
        if (rank == 0):
            torch.save(self.model.state_dict(), '{}/model.pkl'.format(path))

    def inference(self, img0, img1, time_period=0.5, pyr_level=3, nr_lvl_skipped=0):
        interp_img, _, _ = self.model(img0, img1, time_period=time_period, pyr_level=pyr_level, nr_lvl_skipped=nr_lvl_skipped)
        return interp_img

    def train_one_iter(self, img0, img1, gt, learning_rate=0, time_period=0.5):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        self.train()

        interp_img, bi_flow, info_dict = self.model(img0, img1, time_period)
        loss_G = self.lap(interp_img, gt)

        self.optimG.zero_grad()
        loss_G.backward()
        self.optimG.step()

        extra_dict = {}
        extra_dict["loss"] = loss_G

        return interp_img, extra_dict
