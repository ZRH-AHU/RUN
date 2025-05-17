from lib.Modules import Selectivemamba, GPM, REM11, BasicConv2d, SpatialAttentionExtractor, GCM_pvt, GCM3, REU7, VSSBlock
import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.get_m_hat import SparsePipeline

from lib.get_b_hat import MST
from torchvision import models


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class ContextualFeatureAggregation(nn.Module):
    def __init__(self, in_channels):
        """
        上下文特征聚合模块 (CFA)
        :param in_channels: 输入通道数
        """
        super(ContextualFeatureAggregation, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()
        self.conv3 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv_final = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, fc_k_plus1, fa_k):
        """
        :param fc_k_plus1: 上一层的特征 (上采样后的)
        :param fa_k: 当前层的聚合特征
        :return: 聚合后的上下文特征
        """
        # 上采样 fc_k+1
        fc_k_plus1_up = self.upsample(fc_k_plus1)

        # 拼接上采样特征和当前层特征
        combined_features = torch.cat([fc_k_plus1_up, fa_k], dim=1)

        # 3x3 卷积
        combined_features = self.conv3(combined_features)

        # 通道注意力
        ca_out = self.channel_attention(combined_features)
        combined_features = combined_features * ca_out

        # 空间注意力
        sa_out = self.spatial_attention(combined_features)
        fc_k = combined_features * sa_out

        combined = torch.cat([fa_k, fc_k], dim=1)
        return self.conv_final(combined)



class IntraLayerFeatureAggregation(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        IFA模块: 融合多尺度特征，捕获尺度不变信息。
        :param in_channels: 输入特征的通道数
        :param out_channels: 输出特征的通道数
        """
        super(IntraLayerFeatureAggregation, self).__init__()

        # 1x1 卷积用于通道压缩
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 3x3 和 5x5 卷积，用于不同尺度的特征提取
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)

        self.conv3_2 = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels*2, in_channels, kernel_size=5, padding=2)

        # 并行卷积后的 CRB 块（3x3卷积 + ReLU + BN）
        self.crb = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # 1. 通道压缩
        x_compressed = self.conv1(x)  # [B, out_channels, H, W]

        # 2. 多尺度特征提取
        f3 = self.conv3(x_compressed)  # 3x3卷积
        f5 = self.conv5(x_compressed)  # 5x5卷积

        # 3. 特征聚合和多尺度信息融合
        f_combined = torch.cat([f3, f5], dim=1)  # 拼接 [B, 2*out_channels, H, W]
        f_combined_3x3 = self.conv3_2(f_combined)
        f_combined_5x5 = self.conv5_2(f_combined)

        # 输出特征相乘，捕获尺度不变信息
        f_35 = f_combined_3x3 * f_combined_5x5

        # 4. 整合三种特征 (f3, f5, f_35)，并经过 CRB 块处理
        f_aggregated = self.crb(torch.cat([f3, f5, f_35], dim=1))  # 拼接后输入CRB

        # 5. 和通道压缩的特征相加
        output = x_compressed + f_aggregated
        return output



class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)
        z = self.conv_du(y)
        return z * y + x


class UNetConvBlock_fre(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock_fre, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out



class InvBlock(nn.Module):
    def __init__(self, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = UNetConvBlock_fre(self.split_len2, self.split_len1)
        self.G = UNetConvBlock_fre(self.split_len1, self.split_len2)
        self.H = UNetConvBlock_fre(self.split_len1, self.split_len2)

        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x):
        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out

class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = InvBlock(nc,nc//2)

    def forward(self, x):
        return x+self.block(x)


class FreBlockSpa(nn.Module):
    def __init__(self, nc):
        super(FreBlockSpa, self).__init__()
        self.processreal = nn.Sequential(
            nn.Conv2d(nc,nc,kernel_size=3,padding=1,stride=1,groups=nc),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc,nc,kernel_size=3,padding=1,stride=1,groups=nc))
        self.processimag = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1, groups=nc),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1, groups=nc))

    def forward(self,x):
        real = self.processreal(x.real)
        imag = self.processimag(x.imag)
        x_out = torch.complex(real, imag)

        return x_out


class FreBlockCha(nn.Module):
    def __init__(self, nc):
        super(FreBlockCha, self).__init__()
        self.processreal = nn.Sequential(
            nn.Conv2d(nc,nc,kernel_size=1,padding=0,stride=1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc,nc,kernel_size=1,padding=0,stride=1))
        self.processimag = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1))

    def forward(self,x):
        real = self.processreal(x.real)
        imag = self.processimag(x.imag)
        x_out = torch.complex(real, imag)

        return x_out


class SpatialFuse(nn.Module):
    def __init__(self, in_nc):
        super(SpatialFuse,self).__init__()
        # self.fpre = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlockSpa(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc,in_nc,3,1,1)
        self.cat = nn.Conv2d(2*in_nc,in_nc,3,1,1)


    def forward(self, x):
        xori = x
        _, _, H, W = x.shape
        # x_freq = torch.fft.rfft2(x, norm='backward')
        x = self.spatial_process(x)
        # x_freq = self.frequency_process(x)
        # x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        x_freq_spatial = self.frequency_spatial(x)
        xcat = torch.cat([x,x_freq_spatial],1)
        x_out = self.cat(xcat)

        return x_out+xori


class ChannelFuse(nn.Module):
    def __init__(self, in_nc):
        super(ChannelFuse,self).__init__()
        # self.fpre = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlockCha(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc,in_nc,1,1,0)
        self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,0)


    def forward(self, x):
        xori = x
        _, _, H, W = x.shape
        # x_freq = torch.fft.rfft2(x, norm='backward')
        x = self.spatial_process(x)
        # x_freq = self.frequency_process(x)
        # x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        x_freq_spatial = self.frequency_spatial( x)
        xcat = torch.cat([x,x_freq_spatial],1)
        x_out = self.cat(xcat)

        return x_out+xori


class ProcessBlock(nn.Module):
    def __init__(self, nc):
        super(ProcessBlock, self).__init__()
        self.spa = SpatialFuse(nc)
        self.cha = ChannelFuse(nc)

    def forward(self,x):
        x = self.spa(x)
        x = self.cha(x)

        return x


class ProcessNet(nn.Module):
    def __init__(self, nc):
        super(ProcessNet,self).__init__()
        self.conv0 = nn.Conv2d(nc, nc, 3, 1, 1)
        self.conv1 = ProcessBlock(nc)
        self.downsample1 = nn.Conv2d(nc, nc * 2, stride=2, kernel_size=2, padding=0)
        self.conv2 = ProcessBlock(nc * 2)
        self.downsample2 = nn.Conv2d(nc * 2, nc * 3, stride=2, kernel_size=2, padding=0)
        self.conv3 = ProcessBlock(nc * 3)
        self.up1 = nn.ConvTranspose2d(nc * 5, nc * 2, 1, 1)
        self.conv4 = ProcessBlock(nc * 2)
        self.up2 = nn.ConvTranspose2d(nc * 3, nc * 1, 1, 1)
        self.conv5 = ProcessBlock(nc)
        self.convout = nn.Conv2d(nc, nc, 3, 1, 1)

    def forward(self, x):
        x = self.conv0(x)
        x01 = self.conv1(x)
        x1 = self.downsample1(x01)
        x12 = self.conv2(x1)
        x2 = self.downsample2(x12)
        x3 = self.conv3(x2)
        x34 = self.up1(torch.cat([F.interpolate(x3, size=(x12.size()[2], x12.size()[3]), mode='bilinear'), x12], 1))
        x4 = self.conv4(x34)
        x4 = self.up2(torch.cat([F.interpolate(x4, size=(x01.size()[2], x01.size()[3]), mode='bilinear'), x01], 1))
        x5 = self.conv5(x4)
        xout = self.convout(x5)

        return xout



class InteractNet(nn.Module):
    def __init__(self, inchannel, nc, outchannel):
        super(InteractNet,self).__init__()
        self.extract = nn.Conv2d(inchannel, nc,1,1,0)
        self.process = ProcessNet(nc)
        self.recons = nn.Conv2d(nc, outchannel, 1, 1, 0)

    def forward(self, x):
        x_f = self.extract(x)
        x_f = self.process(x_f)+x_f
        y = self.recons(x_f)

        return y

class ChannelAttentionEnhancement(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionEnhancement, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


def compute_B_k_hat( B_k_1, image, m_k, Lambda):

    ones_matrix = torch.ones_like(image)
    L = (1 + Lambda) * ones_matrix + 1e-6
    term2 = (Lambda * B_k_1 + image - m_k * image)
    B_k_hat = term2 / L
    return B_k_hat

class Network_res2net50(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=96):
        super(Network_res2net50, self).__init__()

        self.GCM3 = GCM3(256, channels)
        self.GPM = GPM(in_channels=2048)
        self.REM11 = REM11(channels, channels)

        self.adjustIllu_model = InteractNet(inchannel=7, nc=8, outchannel=3)

        self.Lambda = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)
        self.pipeline = SparsePipeline()

        self.IAF = IntraLayerFeatureAggregation(channels, channels)
        self.CAF = ContextualFeatureAggregation(channels)
        self.conv3_fea = nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0)

        self.pipeline = SparsePipeline()

        self.conv_final = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.conv_b = nn.Conv2d(7, 3, kernel_size=3, padding=1)

        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.conv_final2 = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, x):
        image = x

        # Feature Extraction
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11
        LL, LH, HL, HH, f1, f2, f3, f4 = self.GCM3(x1, x2, x3, x4)

        f1_a, f2_a, f3_a, f4_a = self.IAF(f1), self.IAF(f2), self.IAF(f3), self.IAF(f4)
        f3 = self.CAF(f4_a, f3_a)
        x = torch.cat([f3, f3_a], dim=1)
        f3 = self.conv3_fea(x)
        f2 = self.CAF(f3, f2_a)
        x = torch.cat([f2, f2_a], dim=1)
        f2 = self.conv3_fea(x)

        f1 = self.CAF(f2, f1_a)
        x = torch.cat([f1, f1_a], dim=1)
        f1 = self.conv3_fea(x)

       
        prior_cam = self.GPM(x4)
        pred_0 = F.interpolate(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        pred_m5_hat = pred_0
        I = torch.ones_like(pred_0)
        pred_b0 = (I - pred_0) * image

        pred_m4_hat = self.pipeline(image, pred_b0, pred_0, pred_m5_hat, pred_0)

        selective_mamba = Selectivemamba().cuda()
        prior_cam4 = selective_mamba(f4, pred_m4_hat, pred_b0, image)
        pred_m4 = F.interpolate(prior_cam4, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)

        f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = self.REM11([f1, f2, f3, f4], prior_cam, image)

        x = torch.cat([pred_m4 * 0.1, f4], dim=1)
        f4 = self.conv_final2(x)


        x = torch.cat([pred_m4_hat * 0.1, f4], dim=1)

        f4 = self.conv_final(x)

        pred_b4_hat = compute_B_k_hat(pred_b0, image, f4, Lambda=self.Lambda)
        x = torch.cat([f4, pred_b4_hat, image], dim=1)
        pred_b4 = self.adjustIllu_model(x)
        pred_b3_hat = compute_B_k_hat(pred_b4, image, f3, Lambda=self.Lambda)
        x = torch.cat([f3, pred_b3_hat, image], dim=1)
        pred_b3 = self.adjustIllu_model(x)
        pred_b2_hat = compute_B_k_hat(pred_b3, image, f2, Lambda=self.Lambda)
        x = torch.cat([f2, pred_b2_hat, image], dim=1)
        pred_b2 = self.adjustIllu_model(x)
        pred_b1_hat = compute_B_k_hat(pred_b2, image, f1, Lambda=self.Lambda)
        x = torch.cat([f1, pred_b1_hat, image], dim=1)
        pred_b1 = self.adjustIllu_model(x)

        return pred_0, f4, f3, f2, f1, bound_f3, bound_f2, bound_f1, pred_b0, pred_b4, pred_b3, pred_b2, pred_b1


class Network_pvt(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=128):
        super(Network_pvt, self).__init__()
        # self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.GCM3 = GCM_pvt(64, channels)
        self.GPM = GPM(in_channels=512)
        self.REM11 = REM11(channels, channels)
        self.cam = ChannelAttentionEnhancement(channels)
        self.sam = SpatialAttentionExtractor()

        self.adjustIllu_model = InteractNet(inchannel=7, nc=8, outchannel=3)

        self.Lambda = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)
        self.pipeline = SparsePipeline()

        self.IAF = IntraLayerFeatureAggregation(channels, channels)
        self.CAF = ContextualFeatureAggregation(channels)
        self.conv3_fea = nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0)

        self.pipeline = SparsePipeline()
        self.encoder = pvt_v2_b4()
        pretrained_dict = torch.load('/home/chunming/FEDER/pvt_v2_b4.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(pretrained_dict)
        print('Pretrained encoder loaded.')
        self.conv_final = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.conv_b = nn.Conv2d(7, 3, kernel_size=3, padding=1)
        self.conv_final2 = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, x):
        image = x
        # Feature Extraction
        features = self.encoder(x)
        x1 = features[3]
        x2 = features[2]
        x3 = features[1]
        x4 = features[0]
        LL, LH, HL, HH, f1, f2, f3, f4 = self.GCM3(x1, x2, x3, x4)

        f1_a, f2_a, f3_a, f4_a = self.IAF(f1), self.IAF(f2), self.IAF(f3), self.IAF(f4)
        f3 = self.CAF(f4_a, f3_a)
        x = torch.cat([f3, f3_a], dim=1)
        f3 = self.conv3_fea(x)

        f2 = self.CAF(f3, f2_a)
        x = torch.cat([f2, f2_a], dim=1)
        f2 = self.conv3_fea(x)

        f1 = self.CAF(f2, f1_a)
        x = torch.cat([f1, f1_a], dim=1)
        f1 = self.conv3_fea(x)


        prior_cam = self.GPM(x4)
        pred_0 = F.interpolate(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        pred_m5_hat = pred_0
        I = torch.ones_like(pred_0)
        pred_b0 = (I - pred_0) * image

        pred_m4_hat = self.pipeline(image, pred_b0, pred_0, pred_m5_hat, pred_0)


        selective_mamba = Selectivemamba().cuda()
        prior_cam4 = selective_mamba(f4, pred_m4_hat, pred_b0, image)
        pred_m4 = F.interpolate(prior_cam4, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)

        f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = self.REM11([f1, f2, f3, f4], prior_cam, image)

        x = torch.cat([pred_m4*0.1, f4], dim=1)
        f4 = self.conv_final2(x)


        x = torch.cat([pred_m4_hat*0.1, f4], dim=1)

        f4 = self.conv_final(x)

        pred_b4_hat = compute_B_k_hat(pred_b0, image, f4, Lambda=self.Lambda)
        x = torch.cat([f4, pred_b4_hat, image], dim=1)
        pred_b4 = self.adjustIllu_model(x)
        pred_b3_hat = compute_B_k_hat(pred_b4, image, f3, Lambda=self.Lambda)
        x = torch.cat([f3, pred_b3_hat, image], dim=1)
        pred_b3 = self.adjustIllu_model(x)
        pred_b2_hat = compute_B_k_hat(pred_b3, image, f2, Lambda=self.Lambda)
        x = torch.cat([f2, pred_b2_hat, image], dim=1)
        pred_b2 = self.adjustIllu_model(x)
        pred_b1_hat = compute_B_k_hat(pred_b2, image, f1, Lambda=self.Lambda)
        x = torch.cat([f1, pred_b1_hat, image], dim=1)
        pred_b1 = self.adjustIllu_model(x)
        # return pred_0, f4, f3, f2, f1

        return pred_0, f4, f3, f2, f1, bound_f3, bound_f2, bound_f1, pred_b0, pred_b4, pred_b3, pred_b2, pred_b1

class Network(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=128):
        super(Network, self).__init__()
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.GCM3 = GCM3(256, channels)
        self.GPM = GPM(in_channels=2048)
        self.REM11 = REM11(channels, channels)
        self.cam = ChannelAttentionEnhancement(channels)
        self.sam = SpatialAttentionExtractor()

        self.adjustIllu_model = InteractNet(inchannel=7, nc=8, outchannel=3)

        self.Lambda = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)
        self.pipeline = SparsePipeline()

        self.IAF = IntraLayerFeatureAggregation(channels, channels)
        self.CAF = ContextualFeatureAggregation(channels)
        self.conv3_fea = nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0)

        self.pipeline = SparsePipeline()

        self.conv_final = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.conv_b = nn.Conv2d(7, 3, kernel_size=3, padding=1)
        self.conv_final2 = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, x):
        image = x
        # Feature Extraction
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats
        LL, LH, HL, HH, f1, f2, f3, f4 = self.GCM3(x1, x2, x3, x4)

        f1_a, f2_a, f3_a, f4_a = self.IAF(f1), self.IAF(f2), self.IAF(f3), self.IAF(f4)
        f3 = self.CAF(f4_a, f3_a)
        x = torch.cat([f3, f3_a], dim=1)
        f3 = self.conv3_fea(x)

        f2 = self.CAF(f3, f2_a)
        x = torch.cat([f2, f2_a], dim=1)
        f2 = self.conv3_fea(x)

        f1 = self.CAF(f2, f1_a)
        x = torch.cat([f1, f1_a], dim=1)
        f1 = self.conv3_fea(x)


        prior_cam = self.GPM(x4)
        pred_0 = F.interpolate(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        pred_m5_hat = pred_0
        I = torch.ones_like(pred_0)
        pred_b0 = (I - pred_0) * image

        pred_m4_hat = self.pipeline(image, pred_b0, pred_0, pred_m5_hat, pred_0)


        selective_mamba = Selectivemamba().cuda()
        prior_cam4 = selective_mamba(f4, pred_m4_hat, pred_b0, image)
        pred_m4 = F.interpolate(prior_cam4, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)

        f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = self.REM11([f1, f2, f3, f4], prior_cam, image)

        x = torch.cat([pred_m4*0.1, f4], dim=1)
        f4 = self.conv_final2(x)


        x = torch.cat([pred_m4_hat*0.1, f4], dim=1)

        f4 = self.conv_final(x)

        pred_b4_hat = compute_B_k_hat(pred_b0, image, f4, Lambda=self.Lambda)
        x = torch.cat([f4, pred_b4_hat, image], dim=1)
        pred_b4 = self.adjustIllu_model(x)
        pred_b3_hat = compute_B_k_hat(pred_b4, image, f3, Lambda=self.Lambda)
        x = torch.cat([f3, pred_b3_hat, image], dim=1)
        pred_b3 = self.adjustIllu_model(x)
        pred_b2_hat = compute_B_k_hat(pred_b3, image, f2, Lambda=self.Lambda)
        x = torch.cat([f2, pred_b2_hat, image], dim=1)
        pred_b2 = self.adjustIllu_model(x)
        pred_b1_hat = compute_B_k_hat(pred_b2, image, f1, Lambda=self.Lambda)
        x = torch.cat([f1, pred_b1_hat, image], dim=1)
        pred_b1 = self.adjustIllu_model(x)

        return pred_0, f4, f3, f2, f1, bound_f3, bound_f2, bound_f1, pred_b0, pred_b4, pred_b3, pred_b2, pred_b1
    
    
if __name__ == '__main__':
    image = torch.rand(2, 3, 384, 384).cuda()
    model = Network(64).cuda()

    pred_0, f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = model(image)
    print(pred_0.shape)
    print(f4.shape)
    print(f3.shape)
    print(f2.shape)
    print(f1.shape)
    print(bound_f4.shape)
    print(bound_f3.shape)
    print(bound_f2.shape)
    print(bound_f1.shape)

