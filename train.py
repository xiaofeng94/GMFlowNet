from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse, configparser
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
# from torch.optim import Adam as AdamW
from torch.optim.adamw import AdamW
from core.onecyclelr import OneCycleLR
from core import create_model

from loss import compute_supervision_coarse, compute_coarse_loss, backwarp

import evaluate
import datasets

from tensorboardX import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self, enabled=False):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremely large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def sequence_loss(train_outputs, image1, image2, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW, use_matching_loss=False):
    """ Loss function defined over sequence of flow predictions """
    flow_preds, softCorrMap = train_outputs

    # original RAFT loss
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exclude invalid pixels and extremely large displacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None].float()  * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    if use_matching_loss:
        # enable global matching loss. Try to use it in late stages of the trianing
        img_2back1 = backwarp(image2, flow_gt)
        occlusionMap = (image1 - img_2back1).mean(1, keepdims=True) #(N, H, W)
        occlusionMap = torch.abs(occlusionMap) > 20
        occlusionMap = occlusionMap.float()

        conf_matrix_gt = compute_supervision_coarse(flow_gt, occlusionMap, 8) # 8 from RAFT downsample

        matchLossCfg = configparser.ConfigParser()
        matchLossCfg.POS_WEIGHT = 1
        matchLossCfg.NEG_WEIGHT = 1
        matchLossCfg.FOCAL_ALPHA = 0.25
        matchLossCfg.FOCAL_GAMMA = 2.0
        matchLossCfg.COARSE_TYPE = 'cross_entropy'
        match_loss = compute_coarse_loss(softCorrMap, conf_matrix_gt, matchLossCfg)

        flow_loss = flow_loss + 0.01*match_loss

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model, last_iters=-1):
    """ Create the optimizer and learning rate scheduler """
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear', last_epoch=last_iters)

    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler, total_steps=0, log_dir=None):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = total_steps
        self.running_loss = {}
        self.writer = None
        self.log_dir = log_dir

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter(logdir=self.log_dir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(logdir=self.log_dir)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(create_model(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=True)

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    if args.restore_ckpt is not None:
        strStep = os.path.split(args.restore_ckpt)[-1].split('_')[0]
        total_steps = int(strStep) if strStep.isdigit() else 0
    else:
        total_steps = 0

    train_loader = datasets.fetch_dataloader(args, TRAIN_DS='C+T+K/S')
    optimizer, scheduler = fetch_optimizer(args, model, total_steps)

    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, total_steps, os.path.join('runs', args.name))

    add_noise = True

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)

            loss, metrics = sequence_loss(flow_predictions, image1, image2, flow, valid, gamma=args.gamma, use_matching_loss=args.use_mix_attn)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)
            
            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))

                logger.write_dict(results)

                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='gmflownet', help="name of your experiment. The saved checkpoint will be named after this in `./checkpoints/.`")
    parser.add_argument('--model', default='gmflownet', help="mdoel class. `<args.model>`_model.py should be in ./core and `<args.model>Model` should be defined in this file")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--use_mix_attn', action='store_true', help='use mixture of POLA and axial attentions')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.set_num_threads(16)

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists('runs'):
        os.mkdir('runs')

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpus))
    args.gpus = [i for i in range(len(args.gpus))]
    train(args)
