import time

import kornia.augmentation as K
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from model.layers import point_projection, PadSameConv2d, ConvReLU2, ConvReLU, Upconv, Refine, SSIM, Backprojection
from utils import conditional_flip, filter_state_dict

from utils import parse_config
from .base_models import DepthAugmentation, MaskAugmentation, CostVolumeModule, DepthModule, ResnetEncoder,MonoDepthModule, EfficientNetEncoder
from .ccf_modules import *

class DyMultiDepthModel(nn.Module):
    def __init__(self, inv_depth_min_max=(0.33, 0.0025), cv_depth_steps=32, pretrain_mode=False, pretrain_dropout=0.0, pretrain_dropout_mode=0,
                 augmentation=None, use_mono=True, use_stereo=False, use_ssim=True, sfcv_mult_mask=True,
                 simple_mask=False, mask_use_cv=True, mask_use_feats=True, cv_patch_size=3, depth_large_model=False, no_cv=False,
                 freeze_backbone=True, freeze_module=(), checkpoint_location=None, mask_cp_loc=None, depth_cp_loc=None,
                 fusion_type = 'ccf_fusion', input_size=[256, 512], ccf_mid_dim=32, use_img_in_depthnet=True,
                 backbone_type='resnet18'):
        """
        :param inv_depth_min_max: Min / max (inverse) depth. (Default=(0.33, 0.0025))
        :param cv_depth_steps: Number of depth steps for the cost volume. (Default=32)
        :param pretrain_mode: Which pretrain mode to use:
            0 / False: Run full network.
            1 / True: Only run depth module. In this mode, dropout can be activated to zero out patches from the
            unmasked cost volume. Dropout was not used for the paper.
            2: Only run mask module. In this mode, the network will return the mask as the main result.
            3: Only run depth module, but use the auxiliary masks to mask the cost volume. This mode was not used in
            the paper. (Default=0)
        :param pretrain_dropout: Dropout rate used in pretrain_mode=1. (Default=0)
        :param augmentation: Which augmentation module to use. "mask"=MaskAugmentation, "depth"=DepthAugmentation. The
        exact way to use this is very context dependent. Refer to the training scripts for more details. (Default="none")
        :param use_mono: Use monocular frames during the forward pass. (Default=True)
        :param use_stereo: Use stereo frame during the forward pass. (Default=False)
        :param use_ssim: Use SSIM during cost volume computation. (Default=True)
        :param sfcv_mult_mask: For the single frame cost volumes: If a pixel does not have a valid reprojection at any 
        depth step, all depths get invalidated. (Default=True)
        :param simple_mask: Use the standard cost volume instead of multiple single frame cost volumes in the mask 
        module. (Default=False) 
        :param cv_patch_size: Patchsize, over which the ssim errors get averaged. (Default=3)
        :param freeze_module: Freeze given string list of modules. (Default=())
        :param checkpoint_location: Load given list of checkpoints. (Default=None)
        :param mask_cp_loc: Load list of checkpoints for the mask module. (Default=None)
        :param depth_cp_loc: Load list of checkpoints for the depth module. (Default=None)
        """
        super().__init__()
        self.inv_depth_min_max = inv_depth_min_max
        self.cv_depth_steps = cv_depth_steps
        self.use_mono = use_mono
        self.use_stereo = use_stereo
        self.use_ssim = use_ssim
        self.sfcv_mult_mask = sfcv_mult_mask
        self.pretrain_mode = int(pretrain_mode)
        self.pretrain_dropout = pretrain_dropout
        self.pretrain_dropout_mode = pretrain_dropout_mode
        self.augmentation = augmentation
        self.simple_mask = simple_mask
        self.mask_use_cv = mask_use_cv
        self.mask_use_feats = mask_use_feats
        self.cv_patch_size = cv_patch_size
        self.no_cv = no_cv
        self.depth_large_model = depth_large_model
        self.checkpoint_location = checkpoint_location
        self.mask_cp_loc = mask_cp_loc
        self.depth_cp_loc = depth_cp_loc
        self.freeze_module = freeze_module
        self.freeze_backbone = freeze_backbone

        self.fusion_type = fusion_type
        self.input_size = input_size
        self.ccf_mid_dim = ccf_mid_dim
        self.use_img_in_depthnet = use_img_in_depthnet
        self.backbone_type = backbone_type

        assert self.backbone_type in ["resnet18", "efficientnetb5"]
        
        self.depthmodule_in_chn = self.cv_depth_steps
        if fusion_type == 'ccf_fusion':
            self.extra_input_dim = 0
            self.fusion_module = CrossCueFusion(cv_hypo_num=self.cv_depth_steps, mid_dim=32, input_size=self.input_size)
            self.depthmodule_in_chn = self.ccf_mid_dim * 2
        elif fusion_type == 'mono_guide_multi':
            self.extra_input_dim = 0
            self.fusion_module = MonoGuideMulti(cv_hypo_num=self.cv_depth_steps, mid_dim=32, input_size=self.input_size)
            self.depthmodule_in_chn = self.ccf_mid_dim
        elif fusion_type == 'multi_guide_mono':
            self.extra_input_dim = 0
            self.fusion_module = MultiGuideMono(cv_hypo_num=self.cv_depth_steps, mid_dim=32, input_size=self.input_size)
            self.depthmodule_in_chn = self.ccf_mid_dim

        if self.backbone_type == 'resnet18':
            self._feature_extractor = ResnetEncoder(num_layers=18, pretrained=True)
        elif self.backbone_type == 'efficientnetb5':
            self._feature_extractor = EfficientNetEncoder(pretrained=True)
        
        if self.freeze_backbone:
            for p in self._feature_extractor.parameters(True):
                p.requires_grad_(False)

        self.cv_module = CostVolumeModule(use_mono=use_mono, use_stereo=use_stereo, use_ssim=use_ssim, sfcv_mult_mask=self.sfcv_mult_mask, patch_size=cv_patch_size)

        self.depth_module = DepthModule(self.depthmodule_in_chn, feature_channels=self._feature_extractor.num_ch_enc,
                                            large_model=self.depth_large_model, use_input_img=self.use_img_in_depthnet)
        self.mono_module = MonoDepthModule(extra_input_dim=self.extra_input_dim,
                                                    feature_channels=self._feature_extractor.num_ch_enc, large_model=self.depth_large_model)

        if self.checkpoint_location is not None:
            if not isinstance(checkpoint_location, list):
                checkpoint_location = [checkpoint_location]
            for cp in checkpoint_location:
                checkpoint = torch.load(cp, map_location=torch.device("cpu"))
                checkpoint_state_dict = checkpoint["state_dict"]
                checkpoint_state_dict = filter_state_dict(checkpoint_state_dict, checkpoint["arch"] == "DataParallel")
                self.load_state_dict(checkpoint_state_dict, strict=True)


        for module_name in self.freeze_module:
            module = self.__getattr__(module_name + "_module")
            module.eval()
            for param in module.parameters(True):
                param.requires_grad_(False)

        if self.augmentation == "depth":
            self.augmenter = DepthAugmentation()
        elif self.augmentation == "mask":
            self.augmenter = MaskAugmentation()
        else:
            self.augmenter = None

    def forward(self, data_dict):
        keyframe = data_dict["keyframe"]

        data_dict["inv_depth_min"] = keyframe.new_tensor([self.inv_depth_min_max[0]])
        data_dict["inv_depth_max"] = keyframe.new_tensor([self.inv_depth_min_max[1]])
        data_dict["cv_depth_steps"] = keyframe.new_tensor([self.cv_depth_steps], dtype=torch.int32)

        with torch.no_grad():
            data_dict = self.cv_module(data_dict)

        if self.augmenter is not None and self.training:
            self.augmenter(data_dict)

        # different with MonoRec: the input image should be the reverted 
        data_dict["image_features"] = self._feature_extractor(data_dict["keyframe"] + .5)

        data_dict["cost_volume_init"] = data_dict["cost_volume"]
        
        data_dict = self.mono_module(data_dict)
        data_dict["predicted_inverse_depths_mono"] = [(1-pred) * self.inv_depth_min_max[1] + pred * self.inv_depth_min_max[0]
                                            for pred in data_dict["predicted_inverse_depths_mono"]]
        mono_depth_pred = torch.clamp(1.0 / data_dict["predicted_inverse_depths_mono"][0], min=1e-3, max=80.0).detach()

        b, c, h, w = keyframe.shape


        pseudo_mono_cost = self.pseudocost_from_mono(mono_depth_pred, 
                                    depth_hypothesis = data_dict["cv_bin_steps"].view(1, -1, 1, 1).expand(b, -1, h, w).detach()).detach()


        if self.training:
            if self.pretrain_dropout_mode == 0:
                cv_mask = keyframe.new_ones(b, 1, h // 8, w // 8, requires_grad=False)
                F.dropout(cv_mask, p=1 - self.pretrain_dropout, training=self.training, inplace=True)
                cv_mask = (cv_mask!=0).float()
                cv_mask = F.upsample(cv_mask, (h, w))
            else:
                cv_mask = keyframe.new_ones(b, 1, 1, 1, requires_grad=False)
                F.dropout(cv_mask, p = 1 - self.pretrain_dropout, training=self.training, inplace=True)
                cv_mask = cv_mask.expand(-1, -1, h, w)
        else:
            cv_mask = keyframe.new_zeros(b, 1, h, w, requires_grad=False)
        data_dict["cv_mask"] = cv_mask


        data_dict["cost_volume"] = (1 - data_dict["cv_mask"]) * self.fusion_module(pseudo_mono_cost, data_dict["cost_volume"])

        data_dict = self.depth_module(data_dict)

        data_dict["predicted_inverse_depths"] = [(1-pred) * self.inv_depth_min_max[1] + pred * self.inv_depth_min_max[0]
                                                    for pred in data_dict["predicted_inverse_depths"]]

        if self.augmenter is not None and self.training:
            self.augmenter.revert(data_dict)

        data_dict["result"] = data_dict["predicted_inverse_depths"][0]
        data_dict["result_mono"] = data_dict["predicted_inverse_depths_mono"][0]
        data_dict["mask"] = data_dict["cv_mask"]

        return data_dict


    def pseudocost_from_mono(self, monodepth, depth_hypothesis):
        abs_depth_diff = torch.abs(monodepth - depth_hypothesis)
        # find the closest depth bin that the monodepth correlate with
        min_diff_index = torch.argmin(abs_depth_diff, dim=1, keepdim=True)
        pseudo_cost = depth_hypothesis.new_zeros(depth_hypothesis.shape)
        ones = depth_hypothesis.new_ones(depth_hypothesis.shape)
        
        pseudo_cost.scatter_(dim = 1, index = min_diff_index, src = ones)
        
        return pseudo_cost

    def find_mincost_depth(self, cost_volume, depth_hypos):
        argmax = torch.argmax(cost_volume, dim=1, keepdim=True)
        mincost_depth = torch.gather(input=depth_hypos, dim=1, index=argmax)
        return mincost_depth