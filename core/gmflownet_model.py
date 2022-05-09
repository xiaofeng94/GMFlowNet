import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock
from extractor import BasicEncoder, BasicConvEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
from swin_transformer import POLAUpdate, MixAxialPOLAUpdate

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class GMFlowNetModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if not hasattr(self.args, 'dropout'):
            self.args.dropout = 0

        if not hasattr(self.args, 'alternate_corr'):
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if self.args.use_mix_attn:
            self.fnet = nn.Sequential(
                            BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout),
                            MixAxialPOLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7)
                        )
        else:
            self.fnet = nn.Sequential(
                BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout),
                POLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7, neig_win_num=1)
            )

        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim, input_dim=cdim)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # # Self-attention update
        # fmap1 = self.transEncoder(fmap1)
        # fmap2 = self.transEncoder(fmap2)

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        # Correlation as initialization
        N, fC, fH, fW = fmap1.shape
        corrMap = corr_fn.corrMap

        #_, coords_index = torch.max(corrMap, dim=-1) # no gradient here
        softCorrMap = F.softmax(corrMap, dim=2) * F.softmax(corrMap, dim=1) # (N, fH*fW, fH*fW)

        if flow_init is not None:
            coords1 = coords1 + flow_init
        else:
            # print('matching as init')
            # mutual match selection
            match12, match_idx12 = softCorrMap.max(dim=2) # (N, fH*fW)
            match21, match_idx21 = softCorrMap.max(dim=1)

            for b_idx in range(N):
                match21_b = match21[b_idx,:]
                match_idx12_b = match_idx12[b_idx,:]
                match21[b_idx,:] = match21_b[match_idx12_b]

            matched = (match12 - match21) == 0  # (N, fH*fW)
            coords_index = torch.arange(fH*fW).unsqueeze(0).repeat(N,1).to(softCorrMap.device)
            coords_index[matched] = match_idx12[matched]

            # matched coords
            coords_index = coords_index.reshape(N, fH, fW)
            coords_x = coords_index % fW
            coords_y = coords_index // fW

            coords_xy = torch.stack([coords_x, coords_y], dim=1).float()
            coords1 = coords_xy

        # Iterative update
        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions, softCorrMap
