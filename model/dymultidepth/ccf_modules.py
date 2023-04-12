import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn


class CrossCueFusion(nn.Module):
    def __init__(self, cv_hypo_num=32, mid_dim=32, input_size=(256,512)):
        super().__init__()
        self.cv_hypo_num = cv_hypo_num
        self.mid_dim = mid_dim
        self.residual_connection =True
        self.is_reduce = True if input_size[1]>650 else False

        if not self.is_reduce:
            self.mono_expand = nn.Sequential(
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True)
            )
            self.multi_expand = nn.Sequential(
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.mono_expand = nn.Sequential(
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True)
            )

            self.multi_expand = nn.Sequential(
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True)
            )
        self.kq_dim = self.mid_dim //4 if self.mid_dim>128 else self.mid_dim

        self.lin_mono_k = nn.Conv2d(self.mid_dim, self.kq_dim, kernel_size=1)
        self.lin_mono_q = nn.Conv2d(self.mid_dim, self.kq_dim, kernel_size=1)
        self.lin_mono_v = nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=1)

        self.lin_multi_k = nn.Conv2d(self.mid_dim, self.kq_dim, kernel_size=1)
        self.lin_multi_q = nn.Conv2d(self.mid_dim, self.kq_dim, kernel_size=1)
        self.lin_multi_v = nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

        if self.residual_connection:
            self.mono_reg = nn.Sequential(
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True)
            )
            self.multi_reg = nn.Sequential(
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=1, padding=0),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True)
            )
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, mono_pseudo_cost, cost_volume):
        init_b, init_c, init_h, init_w = cost_volume.shape
        mono_feat = self.mono_expand(mono_pseudo_cost)
        multi_feat = self.multi_expand(cost_volume)
        b,c,h,w = multi_feat.shape

        # cross-cue attention
        mono_q = self.lin_mono_q(mono_feat).view(b,-1,h*w).permute(0,2,1)
        mono_k = self.lin_mono_k(mono_feat).view(b,-1,h*w)
        mono_score = torch.bmm(mono_q, mono_k)
        mono_atten = self.softmax(mono_score)

        multi_q = self.lin_multi_q(multi_feat).view(b,-1,h*w).permute(0,2,1)
        multi_k = self.lin_multi_k(multi_feat).view(b,-1,h*w)
        multi_score = torch.bmm(multi_q, multi_k)
        multi_atten = self.softmax(multi_score)

        mono_v = self.lin_mono_v(mono_feat).view(b,-1,h*w)
        mono_out = torch.bmm(mono_v, multi_atten.permute(0,2,1))
        mono_out = mono_out.view(b,self.mid_dim, h,w)

        multi_v = self.lin_multi_v(multi_feat).view(b,-1,h*w)
        multi_out = torch.bmm(multi_v, mono_atten.permute(0,2,1))
        multi_out = multi_out.view(b,self.mid_dim, h,w)


        # concatenate and upsample
        fused = torch.cat((multi_out,mono_out), dim=1)
        fused = torch.nn.functional.interpolate(fused, size=(init_h,init_w))

        if self.residual_connection:
            mono_residual = self.mono_reg(mono_pseudo_cost)
            multi_residual = self.multi_reg(cost_volume)
            fused_cat = torch.cat((mono_residual,multi_residual), dim=1)
            fused = fused_cat + self.gamma * fused
        
        return fused



class MultiGuideMono(nn.Module):
    def __init__(self, cv_hypo_num=32, mid_dim=32, input_size=(256,512)):
        super().__init__()
        self.cv_hypo_num = cv_hypo_num
        self.mid_dim = mid_dim
        self.residual_connection =True
        self.is_reduce = True if input_size[1]>650 else False

        if not self.is_reduce:
            self.mono_expand = nn.Sequential(
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True)
            )
            self.multi_expand = nn.Sequential(
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.mono_expand = nn.Sequential(
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True)
            )

            self.multi_expand = nn.Sequential(
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True)
            )
        self.kq_dim = self.mid_dim //4 if self.mid_dim>128 else self.mid_dim

        self.lin_mono_v = nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=1)

        self.lin_multi_k = nn.Conv2d(self.mid_dim, self.kq_dim, kernel_size=1)
        self.lin_multi_q = nn.Conv2d(self.mid_dim, self.kq_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

        if self.residual_connection:
            self.mono_reg = nn.Sequential(
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True)
            )
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, mono_pseudo_cost, cost_volume):
        init_b, init_c, init_h, init_w = cost_volume.shape
        mono_feat = self.mono_expand(mono_pseudo_cost)
        multi_feat = self.multi_expand(cost_volume)
        b,c,h,w = multi_feat.shape

        # multi attention

        multi_q = self.lin_multi_q(multi_feat).view(b,-1,h*w).permute(0,2,1)
        multi_k = self.lin_multi_k(multi_feat).view(b,-1,h*w)
        multi_score = torch.bmm(multi_q, multi_k)
        multi_atten = self.softmax(multi_score)

        mono_v = self.lin_mono_v(mono_feat).view(b,-1,h*w)
        mono_out = torch.bmm(mono_v, multi_atten.permute(0,2,1))
        mono_out = mono_out.view(b,self.mid_dim, h,w)

        # upsample
        fused = torch.nn.functional.interpolate(mono_out, size=(init_h,init_w))

        if self.residual_connection:
            mono_residual = self.mono_reg(mono_pseudo_cost)
            fused = mono_residual + self.gamma * fused
        
        return fused


class MonoGuideMulti(nn.Module):
    def __init__(self, cv_hypo_num=32, mid_dim=32, input_size=(256,512)):
        super().__init__()
        self.cv_hypo_num = cv_hypo_num
        self.mid_dim = mid_dim
        self.residual_connection =True
        self.is_reduce = True if input_size[1]>650 else False

        if not self.is_reduce:
            self.mono_expand = nn.Sequential(
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True)
            )
            self.multi_expand = nn.Sequential(
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.mono_expand = nn.Sequential(
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True)
            )

            self.multi_expand = nn.Sequential(
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True)
            )
        self.kq_dim = self.mid_dim //4 if self.mid_dim>128 else self.mid_dim

        self.lin_mono_k = nn.Conv2d(self.mid_dim, self.kq_dim, kernel_size=1)
        self.lin_mono_q = nn.Conv2d(self.mid_dim, self.kq_dim, kernel_size=1)

        self.lin_multi_v = nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

        if self.residual_connection:
            self.multi_reg = nn.Sequential(
                nn.Conv2d(self.cv_hypo_num, self.mid_dim, kernel_size=1, padding=0),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True)
            )
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, mono_pseudo_cost, cost_volume):
        init_b, init_c, init_h, init_w = cost_volume.shape
        mono_feat = self.mono_expand(mono_pseudo_cost)
        multi_feat = self.multi_expand(cost_volume)
        b,c,h,w = multi_feat.shape

        # mono attention
        mono_q = self.lin_mono_q(mono_feat).view(b,-1,h*w).permute(0,2,1)
        mono_k = self.lin_mono_k(mono_feat).view(b,-1,h*w)
        mono_score = torch.bmm(mono_q, mono_k)
        mono_atten = self.softmax(mono_score)

        multi_v = self.lin_multi_v(multi_feat).view(b,-1,h*w)
        multi_out = torch.bmm(multi_v, mono_atten.permute(0,2,1))
        multi_out = multi_out.view(b,self.mid_dim, h,w)


        # upsample
        fused = torch.nn.functional.interpolate(multi_out, size=(init_h,init_w))

        if self.residual_connection:
            multi_residual = self.multi_reg(cost_volume)
            fused = multi_residual + self.gamma * fused
        
        return fused
