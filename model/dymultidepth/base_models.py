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

class DepthAugmentation(nn.Module):
    def single_apply(self, x):
        return conditional_flip(x, self._flip_conditions, inplace=False)

    def single_revert(self, x):
        return conditional_flip(x, self._flip_conditions, inplace=False)

    def forward(self, data_dict):
        n, c, h, w = data_dict["keyframe"].shape
        self._flip_conditions = torch.rand(n) < .5
        if "cost_volume" in data_dict:
            conditional_flip(data_dict["cost_volume"], self._flip_conditions)
            for sfcv in data_dict["single_frame_cvs"]:
                conditional_flip(sfcv, self._flip_conditions)
        data_dict["keyframe"] = conditional_flip(data_dict["keyframe"], self._flip_conditions, inplace=False)
        if "mvobj_mask" in data_dict:
            conditional_flip(data_dict["mvobj_mask"], self._flip_conditions)

    def revert(self, data_dict):
        if "cost_volume" in data_dict:
            conditional_flip(data_dict["cost_volume"], self._flip_conditions)
            for sfcv in data_dict["single_frame_cvs"]:
                conditional_flip(sfcv, self._flip_conditions)
        if "cv_mask" in data_dict:
            data_dict["cv_mask"] = conditional_flip(data_dict["cv_mask"], self._flip_conditions, inplace=False)
        conditional_flip(data_dict["keyframe"], self._flip_conditions)
        if "predicted_inverse_depths" in data_dict:
            data_dict["predicted_inverse_depths"] = [conditional_flip(pid, self._flip_conditions, inplace=False) for pid in data_dict["predicted_inverse_depths"]]
        if "predicted_probabilities" in data_dict:
            for pp in data_dict["predicted_probabilities"]:
                conditional_flip(pp, self._flip_conditions)
        if "mvobj_mask" in data_dict:
            conditional_flip(data_dict["mvobj_mask"], self._flip_conditions)
        if "mono_pred" in data_dict and data_dict["mono_pred"] is not None:
            data_dict["mono_pred"] = [conditional_flip(mp, self._flip_conditions, inplace=False) for mp in data_dict["mono_pred"]]
        if "stereo_pred" in data_dict and data_dict["stereo_pred"] is not None:
            data_dict["stereo_pred"] = [conditional_flip(sp, self._flip_conditions, inplace=False) for sp in data_dict["stereo_pred"]]
        if "mask" in data_dict and data_dict["mask"] is not None:
            data_dict["mask"] = conditional_flip(data_dict["mask"], self._flip_conditions, inplace=False)
        if "result" in data_dict and data_dict["result"] is not None:
            data_dict["result"] = conditional_flip(data_dict["result"], self._flip_conditions, inplace=False)

        if "predicted_inverse_depths_mono" in data_dict:
            data_dict["predicted_inverse_depths_mono"] = [conditional_flip(pid, self._flip_conditions, inplace=False) for pid in data_dict["predicted_inverse_depths_mono"]]
        if "result_mono" in data_dict and data_dict["result_mono"] is not None:
            data_dict["result_mono"] = conditional_flip(data_dict["result_mono"], self._flip_conditions, inplace=False)



class MaskAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = nn.ModuleList([
            K.RandomHorizontalFlip(),
            K.RandomResizedCrop(size=(256, 512), scale=(.8, 1.0), ratio=(1.9, 2.1))
        ])

    def single_apply(self, x):
        for t, p in zip(self.transforms, self.params):
            x = t.apply_transform(x, p)
        return x

    def single_revert(self, x):
        return x

    def forward(self, data_dict):
        with torch.no_grad():
            shape = data_dict["mvobj_mask"].shape
            for t in self.transforms: t.size = (shape[2], shape[3])
            self.params = [t.generate_parameters(shape) for t in self.transforms]
            for t, params in zip(self.transforms, self.params):
                data_dict["keyframe"] = t.apply_transform(data_dict["keyframe"], params)
                data_dict["frames"] = [t.apply_transform(f, params) for f in data_dict["frames"]]
                if "stereo_frame" in data_dict:
                    data_dict["stereo_frame"] = t.apply_transform(data_dict["stereo_frame"], params)
                data_dict["mvobj_mask"] = t.apply_transform(data_dict["mvobj_mask"], params)
                if "cost_volume" in data_dict:
                    data_dict["cost_volume"] = t.apply_transform(data_dict["cost_volume"], params)
                    data_dict["single_frame_cvs"] = [t.apply_transform(sfcv, params) for sfcv in data_dict["single_frame_cvs"]]
            data_dict["mvobj_mask"] = (data_dict["mvobj_mask"] > .5).to(dtype=torch.float32)
            data_dict["target"] = data_dict["mvobj_mask"]

    def revert(self, data_dict):
        return data_dict


class ResnetEncoder(nn.Module):
    def __init__(self, num_layers, pretrained):
        """
        Adapted from monodepth2.resnet_encoder.py
        """
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: torchvision.models.resnet18,
                   34: torchvision.models.resnet34,
                   50: torchvision.models.resnet50,
                   101: torchvision.models.resnet101,
                   152: torchvision.models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features



class EfficientNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetEncoder, self).__init__()
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_b5_ap', pretrained=pretrained)
        # Remove last layer
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        self.original_model = basemodel

        self.num_ch_enc = np.array([24, 40, 64, 176, 2048])

    def forward(self, x):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        for ii in range(3):
            x[:,ii,:,:] = (x[:,ii,:,:] - mean[ii]) / std[ii]
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))

        return (features[4], features[5], features[6], features[8], features[11])


class CostVolumeModule(nn.Module):
    def __init__(self, use_mono=True, use_stereo=False, use_ssim=True, patch_size=3, channel_weights=(5 / 32, 16 / 32, 11 / 32), alpha=10, not_center_cv=False, sfcv_mult_mask=True):
        super().__init__()
        self.use_mono = use_mono
        self.use_stereo = use_stereo
        self.use_ssim = use_ssim
        self.patch_size = patch_size
        self.border_radius = patch_size // 2 + 1
        if not (channel_weights is None):
            self.sad_kernel = (torch.tensor(channel_weights) / (patch_size ** 2)) \
                .view(1, len(channel_weights), 1, 1, 1).repeat(1, 1, 1, patch_size, patch_size)
        else:
            self.sad_kernel = None
        self.alpha = alpha
        self.not_center_cv = not_center_cv
        self.sfcv_mult_mask = sfcv_mult_mask
        self.ssim = SSIM()

    def forward(self, data_dict):
        start_time = time.time()
        keyframe = data_dict["keyframe"]
        keyframe_intrinsics = data_dict["keyframe_intrinsics"]
        keyframe_pose = data_dict["keyframe_pose"]

        frames = []
        intrinsics = []
        poses = []

        if self.use_mono:
            frames += data_dict["frames"]
            intrinsics += data_dict["intrinsics"]
            poses += data_dict["poses"]
        if self.use_stereo:
            frames += [data_dict["stereoframe"]]
            intrinsics += [data_dict["stereoframe_intrinsics"]]
            poses += [data_dict["stereoframe_pose"]]

        batch_size, channels, height, width = keyframe.shape

        extrinsics = [torch.inverse(pose) for pose in poses]

        # If the convolution kernel for the SAD tensor has not been defined in init, do it now.
        if self.sad_kernel is None:
            self.sad_kernel = self.sad_kernel = (
                    torch.tensor(channels * [1 / channels], device=keyframe.device) / (self.patch_size ** 2)) \
                .view(1, channels, 1, 1, 1).repeat(1, 1, 1, self.patch_size, self.patch_size)
        elif self.sad_kernel.device != keyframe.device:
            self.sad_kernel = self.sad_kernel.cuda(keyframe.get_device()) if keyframe.is_cuda else self.sad_kernel

        if "cv_depths" in data_dict:
            depths = data_dict["cv_depths"]
        else:
            depths = (1 / torch.linspace(data_dict["inv_depth_max"][0].item(), data_dict["inv_depth_min"][0].item(), data_dict["cv_depth_steps"][0].item(),
                                    device=keyframe.device))
            data_dict["cv_bin_steps"] = depths
            depths = depths.view(1, -1, 1, 1).expand(batch_size, -1, height, width)

        backproject_depth = Backprojection(1, height, width)
        backproject_depth.to(keyframe.device)

        cost_volumes = []
        single_frame_cvs = [[] for i in range(len(frames))]

        for batch_nr in range(batch_size):
            batch_depths = depths[batch_nr, :, :, :]

            depth_value_count = batch_depths.shape[0]

            inv_k = torch.inverse(keyframe_intrinsics[batch_nr]).unsqueeze(0)
            cam_points = (inv_k[:, :3, :3] @ backproject_depth.coord)
            cam_points = batch_depths.view(depth_value_count, 1, -1) * cam_points
            cam_points = torch.cat([cam_points, backproject_depth.ones.expand(depth_value_count, -1, -1)], 1)

            warped_images = []
            warped_masks = []

            for i, image in enumerate(frames):
                t = extrinsics[i][batch_nr] @ keyframe_pose[batch_nr]
                pix_coords = point_projection(cam_points, depth_value_count, height, width, intrinsics[i][batch_nr].unsqueeze(0), t.unsqueeze(0)).clamp(-2, 2)

                # (D, C, H, W)
                image_to_warp = image[batch_nr, :, :, :].unsqueeze(0).expand(depth_value_count, -1, -1, -1)
                mask_to_warp = self.create_mask(1, height, width, self.border_radius, keyframe.device).expand(
                    depth_value_count, -1, -1, -1)

                warped_image = F.grid_sample(image_to_warp, pix_coords)
                warped_images.append(warped_image)

                warped_mask = F.grid_sample(mask_to_warp, pix_coords)
                warped_mask = mask_to_warp[0] * torch.min(warped_mask != 0, dim=0)[0]
                warped_masks.append(warped_mask)

            # (D, F, C, H, W)
            warped_images = torch.stack(warped_images, dim=1)
            # (F, 1, H, W)
            warped_masks = torch.stack(warped_masks)

            if not self.use_ssim:
                difference = torch.abs(warped_images - keyframe[batch_nr])
            elif self.use_ssim == True:
                b = depth_value_count * len(frames)
                difference = self.ssim(warped_images.view(b, channels, height, width) + .5,
                                       keyframe[batch_nr].unsqueeze(0).expand(b, -1, -1, -1) + .5)
                difference = difference.view(depth_value_count, len(frames), channels, height, width)
            elif self.use_ssim == 2:
                b = depth_value_count * len(frames)
                difference = self.ssim(warped_images.view(b, channels, height, width) + .5,
                                       keyframe[batch_nr].unsqueeze(0).expand(b, -1, -1, -1) + .5)
                difference = difference.view(depth_value_count, len(frames), channels, height, width)
                difference = 0.85 * difference + 0.15 * (torch.abs(warped_images - keyframe[batch_nr]))
            else:
                b = depth_value_count * len(frames)
                difference = F.avg_pool2d(torch.abs(warped_images - keyframe[batch_nr]).view(b, channels, height, width), kernel_size=3, stride=1, padding=1)
                difference = difference.view(depth_value_count, len(frames), channels, height, width)

            # (F, C, D, H, W)
            difference = difference.permute(1, 2, 0, 3, 4)
            sad_volume = F.conv3d(difference, self.sad_kernel, padding=(0, self.patch_size // 2,
                                                                        self.patch_size // 2)).squeeze(1)
            # (F, D, H, W)
            if self.sfcv_mult_mask:
                single_frame_cv = (1 - sad_volume * 2) * warped_masks
            else:
                single_frame_cv = (1 - sad_volume * 2) * (torch.any(warped_images != 0, dim=2) | torch.all(warped_images == keyframe[batch_nr], dim=2)).permute(1, 0, 2, 3)
            for i in range(len(frames)):
                single_frame_cvs[i] += [single_frame_cv[i]]

            sum_item = torch.exp(-self.alpha * torch.pow(sad_volume - torch.min(sad_volume, dim=1, keepdim=True)[0], 2))
            weight = 1 - 1 / (depth_value_count - 1) * (torch.sum(sum_item, dim=1, keepdim=True) - 1)

            weight *= warped_masks

            cost_volume = torch.sum(sad_volume * weight, dim=0)

            weight_sum = torch.sum(weight, dim=0).squeeze()
            weight_zero = weight_sum == 0
            cost_volume[:, torch.logical_not(weight_zero)] /= weight_sum[torch.logical_not(weight_zero)]
            if not self.not_center_cv:
                cost_volume = 1 - 2 * cost_volume
            cost_volume[:, weight_zero] = 0

            cost_volumes.append(cost_volume)

        data_dict["cost_volume"] = torch.stack(cost_volumes)
        data_dict["single_frame_cvs"] = [torch.stack(sf_cv) for sf_cv in single_frame_cvs]

        end_time = time.time()
        time_diff = end_time - start_time
        data_dict["cv_module_time"] = keyframe.new_tensor([time_diff])

        return data_dict

    def create_mask(self, c: int, height: int, width: int, border_radius: int, device=None):
        mask = torch.ones(c, 1, height - 2 * border_radius, width - 2 * border_radius, device=device)
        return F.pad(mask, [border_radius, border_radius, border_radius, border_radius])



class MonoDepthModule(nn.Module):
    def __init__(self, extra_input_dim=0, feature_channels=(64, 64, 128, 256, 512), large_model=False) -> None:
        super().__init__()
        self.extra_input_dim = extra_input_dim
        self.feat_chns = feature_channels
        self._in_channels = self.extra_input_dim + 3
        self._cv_enc_feat_chns = (48, 64, 128, 192, 256) if not large_model else (48, 64, 128, 256, 512)
        self._dec_feat_chns = (256, 128, 64, 48, 32, 24) if not large_model else (512, 256, 128, 64, 32, 24)

        self.enc = nn.ModuleList([
            nn.Sequential(
                ConvReLU2(in_channels=self._in_channels, out_channels=self._cv_enc_feat_chns[0], kernel_size=7),
                ConvReLU2(in_channels=self._cv_enc_feat_chns[0], out_channels=self._cv_enc_feat_chns[0], kernel_size=3)),
            nn.Sequential(
                ConvReLU2(in_channels=self._cv_enc_feat_chns[0], out_channels=self._cv_enc_feat_chns[1], kernel_size=7, stride=2),
                ConvReLU2(in_channels=self._cv_enc_feat_chns[1], out_channels=self._cv_enc_feat_chns[1], kernel_size=3)),
            nn.Sequential(
                ConvReLU2(in_channels=self._cv_enc_feat_chns[1], out_channels=self._cv_enc_feat_chns[2], kernel_size=5, stride=2),
                ConvReLU2(in_channels=self._cv_enc_feat_chns[2], out_channels=self._cv_enc_feat_chns[2], kernel_size=3)),
            nn.Sequential(
                ConvReLU2(in_channels=self._cv_enc_feat_chns[2], out_channels=self._cv_enc_feat_chns[3], kernel_size=5, stride=2),
                ConvReLU2(in_channels=self._cv_enc_feat_chns[3], out_channels=self._cv_enc_feat_chns[3], kernel_size=3)),
            nn.Sequential(
                ConvReLU2(in_channels=self._cv_enc_feat_chns[3], out_channels=self._cv_enc_feat_chns[4], kernel_size=3, stride=2),
                ConvReLU2(in_channels=self._cv_enc_feat_chns[4], out_channels=self._cv_enc_feat_chns[4], kernel_size=3))
        ])

        self.dec = nn.ModuleList([
            # Refine(in_channels=self._cv_enc_feat_chns[4] + self.feat_chns[3], out_channels=self._dec_feat_chns[0]),
            Refine(in_channels=self._cv_enc_feat_chns[4], out_channels=self._dec_feat_chns[0]),
            nn.Sequential(
            Refine(in_channels=self._cv_enc_feat_chns[3] + self.feat_chns[2] + self._dec_feat_chns[0], out_channels=self._dec_feat_chns[1]),
            ConvReLU2(in_channels=self._dec_feat_chns[1], out_channels=self._dec_feat_chns[1], kernel_size=3)),
            nn.Sequential(
            Refine(in_channels=self._cv_enc_feat_chns[2] + self.feat_chns[1] + self._dec_feat_chns[1], out_channels=self._dec_feat_chns[2]),
            ConvReLU2(in_channels=self._dec_feat_chns[2], out_channels=self._dec_feat_chns[2], kernel_size=3)),
            Refine(in_channels=self._cv_enc_feat_chns[1] + self.feat_chns[0] + self._dec_feat_chns[2], out_channels=self._dec_feat_chns[3]),
            nn.Sequential(
            ConvReLU2(in_channels=self._cv_enc_feat_chns[0] + self._dec_feat_chns[3], out_channels=self._dec_feat_chns[4], kernel_size=3),
            PadSameConv2d(kernel_size=3),
            nn.Conv2d(in_channels=self._dec_feat_chns[4], out_channels=self._dec_feat_chns[5], kernel_size=3),
            nn.LeakyReLU(negative_slope=0.1)
            )
        ])

        self.predictors = nn.ModuleList([nn.Sequential(
            PadSameConv2d(kernel_size=3),
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3))
            for channels in self._dec_feat_chns[:3] + self._dec_feat_chns[-1:]])

    def forward(self, data_dict, extra_input=None):
        keyframe = data_dict["keyframe"]
        image_features = data_dict["image_features"]

        x = torch.cat([keyframe, extra_input], dim=1) if extra_input!=None else keyframe

        cv_feats = []
        for layer in self.enc:
            x = layer(x)
            cv_feats.append(x)

        predictions = []
        for i, layer in enumerate(self.dec):
            if i == 0:
                x = torch.cat([cv_feats[-1]], dim=1)
            elif i == len(self.dec) - 1:
                x = torch.cat([cv_feats[0], x], dim=1)
            else:
                x = torch.cat([cv_feats[-(i + 1)], image_features[-(i + 2)], x], dim=1)
            x = layer(x)
            if i != len(self.dec) - 2:
                predictions = [(self.predict_depth(x, scale=i - (1 if i == len(self.dec) - 1 else 0)))] + predictions

        data_dict["predicted_inverse_depths_mono"] = predictions

        return data_dict

    def predict_depth(self, x, scale):
        x = self.predictors[scale](x)
        x = torch.abs(F.tanh(x))
        return x



class DepthModule(nn.Module):
    def __init__(self, depth_steps=32, feature_channels=(64, 64, 128, 256, 512), large_model=False, use_input_img=True) -> None:
        super().__init__()
        self.depth_steps = depth_steps
        self.feat_chns = feature_channels
        self.use_input_img = use_input_img

        if use_input_img:
            self._in_channels = self.depth_steps + 3
        else:
            self._in_channels = self.depth_steps

        self._cv_enc_feat_chns = (48, 64, 128, 192, 256) if not large_model else (48, 64, 128, 256, 512)
        self._dec_feat_chns = (256, 128, 64, 48, 32, 24) if not large_model else (512, 256, 128, 64, 32, 24)

        self.enc = nn.ModuleList([
            nn.Sequential(
                ConvReLU2(in_channels=self._in_channels, out_channels=self._cv_enc_feat_chns[0], kernel_size=7),
                ConvReLU2(in_channels=self._cv_enc_feat_chns[0], out_channels=self._cv_enc_feat_chns[0], kernel_size=3)),
            nn.Sequential(
                ConvReLU2(in_channels=self._cv_enc_feat_chns[0], out_channels=self._cv_enc_feat_chns[1], kernel_size=7, stride=2),
                ConvReLU2(in_channels=self._cv_enc_feat_chns[1], out_channels=self._cv_enc_feat_chns[1], kernel_size=3)),
            nn.Sequential(
                ConvReLU2(in_channels=self._cv_enc_feat_chns[1], out_channels=self._cv_enc_feat_chns[2], kernel_size=5, stride=2),
                ConvReLU2(in_channels=self._cv_enc_feat_chns[2], out_channels=self._cv_enc_feat_chns[2], kernel_size=3)),
            nn.Sequential(
                ConvReLU2(in_channels=self._cv_enc_feat_chns[2], out_channels=self._cv_enc_feat_chns[3], kernel_size=5, stride=2),
                ConvReLU2(in_channels=self._cv_enc_feat_chns[3], out_channels=self._cv_enc_feat_chns[3], kernel_size=3)),
            nn.Sequential(
                ConvReLU2(in_channels=self._cv_enc_feat_chns[3], out_channels=self._cv_enc_feat_chns[4], kernel_size=3, stride=2),
                ConvReLU2(in_channels=self._cv_enc_feat_chns[4], out_channels=self._cv_enc_feat_chns[4], kernel_size=3))
        ])

        self.dec = nn.ModuleList([
            # Refine(in_channels=self._cv_enc_feat_chns[4] + self.feat_chns[3], out_channels=self._dec_feat_chns[0]),
            Refine(in_channels=self._cv_enc_feat_chns[4], out_channels=self._dec_feat_chns[0]),
            nn.Sequential(
            Refine(in_channels=self._cv_enc_feat_chns[3] + self.feat_chns[2] + self._dec_feat_chns[0], out_channels=self._dec_feat_chns[1]),
            ConvReLU2(in_channels=self._dec_feat_chns[1], out_channels=self._dec_feat_chns[1], kernel_size=3)),
            nn.Sequential(
            Refine(in_channels=self._cv_enc_feat_chns[2] + self.feat_chns[1] + self._dec_feat_chns[1], out_channels=self._dec_feat_chns[2]),
            ConvReLU2(in_channels=self._dec_feat_chns[2], out_channels=self._dec_feat_chns[2], kernel_size=3)),
            Refine(in_channels=self._cv_enc_feat_chns[1] + self.feat_chns[0] + self._dec_feat_chns[2], out_channels=self._dec_feat_chns[3]),
            nn.Sequential(
            ConvReLU2(in_channels=self._cv_enc_feat_chns[0] + self._dec_feat_chns[3], out_channels=self._dec_feat_chns[4], kernel_size=3),
            PadSameConv2d(kernel_size=3),
            nn.Conv2d(in_channels=self._dec_feat_chns[4], out_channels=self._dec_feat_chns[5], kernel_size=3),
            nn.LeakyReLU(negative_slope=0.1)
            )
        ])

        self.predictors = nn.ModuleList([nn.Sequential(
            PadSameConv2d(kernel_size=3),
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3))
            for channels in self._dec_feat_chns[:3] + self._dec_feat_chns[-1:]])

    def forward(self, data_dict):
        cost_volume = data_dict["cost_volume"]
        image_features = data_dict["image_features"]
        
        if self.use_input_img:
            keyframe = data_dict["keyframe"]
            x = torch.cat([cost_volume, keyframe], dim=1)
        else:
            x = cost_volume

        cv_feats = []
        for layer in self.enc:
            x = layer(x)
            cv_feats.append(x)

        predictions = []
        for i, layer in enumerate(self.dec):
            if i == 0:
                # x = torch.cat([cv_feats[-1], image_features[-2]], dim=1)
                x = torch.cat([cv_feats[-1]], dim=1)
            elif i == len(self.dec) - 1:
                x = torch.cat([cv_feats[0], x], dim=1)
            else:
                x = torch.cat([cv_feats[-(i + 1)], image_features[-(i + 2)], x], dim=1)
            x = layer(x)
            if i != len(self.dec) - 2:
                predictions = [(self.predict_depth(x, scale=i - (1 if i == len(self.dec) - 1 else 0)))] + predictions

        data_dict["predicted_inverse_depths"] = predictions

        return data_dict

    def predict_depth(self, x, scale):
        x = self.predictors[scale](x)
        x = torch.abs(F.tanh(x))
        return x

