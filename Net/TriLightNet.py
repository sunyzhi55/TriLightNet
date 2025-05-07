import torch
from torch import nn
from Net.basic import *
from Net.defineViT import ViT
from Net.metaformer3D import poolformerv2_3d_s12
import torch.nn.functional as F
from efficientnet_pytorch_3d import EfficientNet3D
import itertools
class Conv1d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', nn.Conv1d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm1d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

class CascadedGroupAttention1D(nn.Module):
    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=128, kernels=None):
        super().__init__()
        if kernels is None:
            kernels = [5] * num_heads
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio
        self.scale = key_dim ** -0.5

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(Conv1d_BN(dim // num_heads, self.key_dim * 2 + self.d, resolution=resolution))
            dws.append(Conv1d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i] // 2, groups=self.key_dim,
                                 resolution=resolution))
        self.qkvs = nn.ModuleList(qkvs)
        self.dws = nn.ModuleList(dws)
        self.proj = nn.Sequential(nn.ReLU(), Conv1d_BN(self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        attention_biases = torch.zeros(num_heads, resolution)
        self.attention_biases = nn.Parameter(attention_biases)
        self.register_buffer('attention_bias_idxs', torch.arange(resolution).unsqueeze(0).expand(resolution, -1))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B, L, C)
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # (B, C, L)
        feats_in = x.chunk(self.num_heads, dim=1)
        feats_out = []
        feat = feats_in[0]
        ab = self.attention_biases[:, self.attention_bias_idxs]

        for i, qkv in enumerate(self.qkvs):
            if i > 0:
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.split([self.key_dim, self.key_dim, self.d], dim=1)
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
            attn = (q.transpose(1, 2) @ k) * self.scale + (ab[i] if self.training else self.ab[i])
            attn = attn.softmax(dim=-1)
            feat = (v @ attn.transpose(1, 2)).view(B, self.d, L)
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, dim=1))
        x = x.permute(0, 2, 1)
        # print("Output:", x.shape)
        return x


class CascadedGroupCrossAttention1D(nn.Module):
    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=128, kernels=None):
        super().__init__()
        if kernels is None:
            kernels = [5] * num_heads
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio
        self.scale = key_dim ** -0.5
        qkvs_x = []
        dws_x = []
        qkvs_y = []
        dws_y = []
        for i in range(num_heads):
            qkvs_x.append(Conv1d_BN(dim // num_heads, self.key_dim * 2 + self.d, resolution=resolution))
            dws_x.append(Conv1d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i] // 2, groups=self.key_dim,
                                 resolution=resolution))
        self.qkvs_x = nn.ModuleList(qkvs_x)
        self.dws_x = nn.ModuleList(dws_x)
        self.proj_x = nn.Sequential(nn.ReLU(), Conv1d_BN(self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        for i in range(num_heads):
            qkvs_y.append(Conv1d_BN(dim // num_heads, self.key_dim * 2 + self.d, resolution=resolution))
            dws_y.append(Conv1d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i] // 2, groups=self.key_dim,
                                 resolution=resolution))
        self.qkvs_y = nn.ModuleList(qkvs_y)
        self.dws_y = nn.ModuleList(dws_y)
        self.proj_y = nn.Sequential(nn.ReLU(), Conv1d_BN(self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        attention_biases = torch.zeros(num_heads, resolution)
        self.attention_biases = nn.Parameter(attention_biases)
        self.register_buffer('attention_bias_idxs', torch.arange(resolution).unsqueeze(0).expand(resolution, -1))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def compute_attention(self, q, k, v, idx, i):
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        attn = (q.transpose(1, 2) @ k) * self.scale
        ab = self.attention_biases[:, idx] if self.training else self.ab
        attn += ab[i]
        attn = attn.softmax(dim=-1)
        return v @ attn.transpose(1, 2)
    def forward(self, x, y):  # x (B, L, C)
        assert x.shape == y.shape, "x, y shapes must be the same"
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # (B, C, L)
        y = y.permute(0, 2, 1)
        feats_in_x = x.chunk(self.num_heads, dim=1)
        feats_out_x = []
        feat_x = feats_in_x[0]
        feats_in_y = y.chunk(self.num_heads, dim=1)
        feats_out_y = []
        feat_y = feats_in_y[0]
        for i in range(self.num_heads):
            if i > 0:
                feat_x = feat_x + feats_in_x[i]
                feat_y = feat_y + feats_in_y[i]
            feat_x = self.qkvs_x[i](feat_x)
            feat_y = self.qkvs_y[i](feat_y)
            q_x, k_x, v_x = feat_x.split([self.key_dim, self.key_dim, self.d], dim=1)
            q_y, k_y, v_y = feat_y.split([self.key_dim, self.key_dim, self.d], dim=1)
            q_x = self.dws_x[i](q_x)
            q_y = self.dws_y[i](q_y)
            feat_x = self.compute_attention(q_y, k_x, v_x, self.attention_bias_idxs, i).view(B, self.d, L)
            feat_y = self.compute_attention(q_x, k_y, v_y, self.attention_bias_idxs, i).view(B, self.d, L)
            feats_out_x.append(feat_x)
            feats_out_y.append(feat_y)
        result_x = self.proj_x(torch.cat(feats_out_x, dim=1))
        result_y = self.proj_y(torch.cat(feats_out_y, dim=1))
        result_x = result_x.permute(0, 2, 1)
        result_y = result_y.permute(0, 2, 1)
        return result_x, result_y
class MlpKan(torch.nn.Module):
    def __init__(self, init_features=64, classes=2):
        """
        1D-DenseNet Module, use to conv global feature and generate final target
        """
        super(MlpKan, self).__init__()
        self.feature_channel_num = init_features

        self.classifer = torch.nn.Sequential(
            # KAN([self.feature_channel_num, self.feature_channel_num // 2]),
            torch.nn.Linear(self.feature_channel_num, self.feature_channel_num // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            # KAN([self.feature_channel_num // 2, classes]),
            torch.nn.Linear(self.feature_channel_num // 2, classes),
            # torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, self.feature_channel_num)
        x = self.classifer(x)
        return x

class CrossModalCBAMOld(nn.Module):
    def __init__(self, img_channels, table_dim, reduction=16):
        super().__init__()
        self.table_proj = nn.Sequential(
            nn.Linear(table_dim, img_channels // reduction),
            nn.ReLU(),
            nn.Linear(img_channels // reduction, img_channels)
        )
        self.sigmoid = nn.Sigmoid()

        # 可选空间注意力（可加可不加）
        self.spatial = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, img_feat, table_feat):
        # 通道注意力（cross-modal）
        B, C, D, H, W = img_feat.shape
        table_channel_attn = self.table_proj(table_feat).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        channel_attn_map = self.sigmoid(table_channel_attn)  # [B, C, 1, 1, 1]
        img_feat = img_feat * channel_attn_map

        # 空间注意力（optional）
        max_pool = torch.max(img_feat, dim=1, keepdim=True)[0]
        mean_pool = torch.mean(img_feat, dim=1, keepdim=True)
        spatial_attn_map = self.spatial(torch.cat([max_pool, mean_pool], dim=1))
        img_feat = img_feat * spatial_attn_map

        return img_feat

class CrossModalCBAM(nn.Module):
    def __init__(self, img_channels, table_dim, reduction=16):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool3d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.image_proj = nn.Sequential(
            nn.Linear(img_channels, img_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(img_channels // reduction, img_channels, bias=False)
        )
        self.table_proj = nn.Sequential(
            nn.Linear(table_dim, img_channels // reduction),
            nn.ReLU(),
            nn.Linear(img_channels // reduction, img_channels)
        )
        self.sigmoid = nn.Sigmoid()

        # 可选空间注意力（可加可不加）
        self.spatial = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, img_feat, table_feat):
        B, C, D, H, W = img_feat.shape
        # 交叉通道注意力机制
        max_out = self.max_pool(img_feat)
        # print the shape
        # print("max_out", max_out.shape) # max_out torch.Size([8, 512, 1, 1])
        max_out = self.image_proj(max_out.view(max_out.size(0), -1))
        avg_out = self.avg_pool(img_feat)
        # print("avg_out", avg_out.shape) # avg_out torch.Size([8, 512, 1, 1])
        avg_out = self.image_proj(avg_out.view(avg_out.size(0), -1))
        table_channel_attn = self.table_proj(table_feat)
        channel_attn_map = self.sigmoid(table_channel_attn + max_out + avg_out)
        channel_attn_map = channel_attn_map.view(B, C, 1, 1, 1)  # [B, C, 1, 1, 1]
        img_feat = img_feat * channel_attn_map

        # 空间注意力（optional）
        max_pool = torch.max(img_feat, dim=1, keepdim=True)[0]
        mean_pool = torch.mean(img_feat, dim=1, keepdim=True)
        spatial_attn_map = self.spatial(torch.cat([max_pool, mean_pool], dim=1))
        img_feat = img_feat * spatial_attn_map

        return img_feat



class Expert(nn.Module):
    """单个专家模块，保持与原MlpKan相似的容量"""

    def __init__(self, in_features, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class MoE_Classifier(nn.Module):
    """解耦辅助损失的MoE分类头"""

    def __init__(self, in_features=512, num_experts=4, hidden_dim=256, num_classes=2, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = 0.01  # 辅助损失系数（可在训练时调整）
        self.num_classes = num_classes

        # 专家池
        self.experts = nn.ModuleList([
            Expert(in_features, hidden_dim, num_classes)
            for _ in range(num_experts)
        ])

        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )

    def forward(self, x):
        """纯前向计算，不包含损失计算"""
        # 输入形状: [B, 512]

        # 门控计算
        gate_logits = self.gate(x)
        gate_scores = F.softmax(gate_logits, dim=-1)

        # Top-k专家选择
        top_k_values, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_values = top_k_values / top_k_values.sum(dim=-1, keepdim=True)  # 重标准化

        # 稀疏计算
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [B, num_experts, num_classes]

        # 通过scatter_add实现高效加权求和
        batch_range = torch.arange(x.size(0), device=x.device)
        output = torch.zeros_like(expert_outputs[:, 0])  # [B, num_classes]
        for i in range(self.top_k):
            output.scatter_add_(
                dim=0,
                index=top_k_indices[:, i].unsqueeze(-1).expand(-1, self.num_classes),
                src=expert_outputs[batch_range, top_k_indices[:, i]] * top_k_values[:, i].unsqueeze(-1)
            )

        # 返回输出和门控信息（用于外部计算损失）
        return {
            "output": output,  # 保持[B, num_classes]形状
            "gate_scores": gate_scores,
            "expert_indices": top_k_indices
        }

    # def compute_aux_loss(self, gate_scores, expert_indices):
    #     """显式计算负载均衡辅助损失"""
    #     # 计算每个专家的选择频率
    #     expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()  # [B, top_k, num_experts]
    #     selection_frequency = expert_mask.mean(dim=1)  # [B, num_experts]
    #
    #     # 计算门控分布的均值
    #     gate_mean = gate_scores.mean(dim=0)  # [num_experts]
    #
    #     # 负载损失 = 专家选择频率的方差
    #     load_loss = torch.var(selection_frequency, dim=0).mean() * self.num_experts
    #     return load_loss * self.aux_loss_coef
    def compute_aux_loss(self, gate_scores, expert_indices):
        """改进版辅助损失，促进选择均衡"""
        # expert_indices: [B, top_k]
        # gate_scores: [B, num_experts]

        B = expert_indices.size(0)

        # 计算每个专家被选中的频率
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()  # [B, top_k, num_experts]
        selection_frequency = expert_mask.sum(dim=1).sum(dim=0)  # [num_experts]

        selection_frequency = selection_frequency / (B * self.top_k)  # 归一化，变成频率

        # 理想频率是每个专家被均匀选中
        ideal_frequency = torch.full_like(selection_frequency, 1.0 / self.num_experts)

        # 用MSE度量偏离
        balance_loss = F.mse_loss(selection_frequency, ideal_frequency)

        # 也可以加上原先的 gate 分布负载均衡（可选）
        gate_mean = gate_scores.mean(dim=0)
        load_loss = torch.var(gate_mean, dim=0).mean() * self.num_experts

        return (balance_loss + load_loss) * self.aux_loss_coef



class TriLightNet(nn.Module):
    def __init__(self, num_classes=2):
        super(TriLightNet, self).__init__()
        self.name = 'TriLightNet'
        self.MriExtraction = get_no_pretrained_vision_encoder()
        self.PetExtraction = get_no_pretrained_vision_encoder()
        self.Table = TransformerEncoder(output_dim=128)
        self.mri_cli_fusion = CrossModalCBAM(img_channels=256, table_dim=128)
        self.pet_cli_fusion = CrossModalCBAM(img_channels=256, table_dim=128)
        self.fusion = CascadedGroupCrossAttention1D(dim=256, key_dim=16, num_heads=4, resolution=1, attn_ratio=4)

        # self.mamba1 = SelfMamba(256, 256, hidden_dropout_prob=0.2, d_state=64)
        # self.mamba2 = SelfMamba(256, 256, hidden_dropout_prob=0.2, d_state=64)
        # self.mamba3 = SelfMamba(256, 256, hidden_dropout_prob=0.2, d_state=64)

        # self.SA1 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        # self.SA2 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        # self.SA3 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)

        # self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=16, in_channels=1, classes=num_classes)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classify_head = MlpKan(init_features=256 * 2, classes=num_classes)
        # self.classify_head = MoE_Classifier(
        #     in_features=512,
        #     num_experts=4,
        #     hidden_dim=256,
        #     num_classes=num_classes,
        #     top_k=2
        # )

    def forward(self, mri, pet, cli):
        """
        Mri: [8, 1, 96, 128, 96]
        Pet: [8, 1, 96, 128, 96]
        Clinical: [8, 9]
        """
        # print("mri", mri.shape)
        # print("pet", pet.shape)
        # print("cli", cli.shape)
        mri_feature = self.MriExtraction(mri)  # torch.Size([8, 256, 3, 4, 3])
        # print(f'mri feature shape: {mri_feature.shape}')
        pet_feature = self.PetExtraction(pet)  # torch.Size([8, 256, 3, 4, 3])
        # print(f'pet feature.shape:{pet_feature.shape}')
        cli_feature = self.Table(cli)  # [8, 256]

        # mri_feature = self.SA1(mri_feature)
        # pet_feature = self.SA2(pet_feature)
        # cli_feature = self.SA3(cli_feature)
        mri_cli_feature = self.mri_cli_fusion(mri_feature, cli_feature) # torch.Size([8, 256, 3, 4, 3])
        pet_cli_feature = self.pet_cli_fusion(pet_feature, cli_feature) # torch.Size([8, 256, 3, 4, 3])
        mri_cli_feature = self.avgpool(mri_cli_feature).view(mri_cli_feature.shape[0], -1)
        pet_cli_feature = self.avgpool(pet_cli_feature).view(pet_cli_feature.shape[0], -1)
        mri_cli_feature = mri_cli_feature.unsqueeze(1)# torch.Size([8, 1, 256])
        pet_cli_feature = pet_cli_feature.unsqueeze(1)# torch.Size([8, 1, 256])

        # torch.Size([8, 1, 256])

        result_mri_cli, result_pet_cli = self.fusion(mri_cli_feature, pet_cli_feature)
        result = torch.cat((result_mri_cli, result_pet_cli), dim=-1)

        # 获取MoE输出（不再包含损失计算）
        # moe_output = self.classify_head(result)
        # return moe_output  # 只返回预测结果
        output = self.classify_head(result)
        return output

class TriLightNetWithoutCBAMWithCasCade(nn.Module):
    def __init__(self, num_classes=2):
        super(TriLightNetWithoutCBAMWithCasCade, self).__init__()
        self.name = 'TriLightNetWithoutCBAMWithCasCade'
        self.MriExtraction = get_no_pretrained_vision_encoder()
        self.PetExtraction = get_no_pretrained_vision_encoder()
        self.Table = TransformerEncoder(output_dim=256)
        # self.mri_cli_fusion = CrossModalCBAM(img_channels=256, table_dim=128)
        # self.pet_cli_fusion = CrossModalCBAM(img_channels=256, table_dim=128)
        self.fusion = CascadedGroupCrossAttention1D(dim=256, key_dim=16, num_heads=4, resolution=1, attn_ratio=4)

        # self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=16, in_channels=1, classes=num_classes)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classify_head = MlpKan(init_features=256 * 2, classes=num_classes)
        # self.classify_head = MoE_Classifier(
        #     in_features=512,
        #     num_experts=4,
        #     hidden_dim=256,
        #     num_classes=num_classes,
        #     top_k=2
        # )

    def forward(self, mri, pet, cli):
        """
        Mri: [8, 1, 96, 128, 96]
        Pet: [8, 1, 96, 128, 96]
        Clinical: [8, 9]
        """
        mri_feature = self.MriExtraction(mri)  # torch.Size([8, 256, 3, 4, 3])
        # print(f'mri feature shape: {mri_feature.shape}')
        mri_feature = self.avgpool(mri_feature)
        mri_feature = mri_feature.view(mri_feature.size(0), -1)
        pet_feature = self.PetExtraction(pet)  # torch.Size([8, 256, 3, 4, 3])
        pet_feature = self.avgpool(pet_feature)
        pet_feature = pet_feature.view(pet_feature.size(0), -1)
        # print(f'pet feature.shape:{pet_feature.shape}')
        cli_feature = self.Table(cli)  # [8, 256]
        mri_cli_feature = mri_feature + cli_feature # torch.Size([8, 256])
        pet_cli_feature = pet_feature + cli_feature # torch.Size([8, 256])
        mri_cli_feature = mri_cli_feature.unsqueeze(1)# torch.Size([8, 1, 256])
        pet_cli_feature = pet_cli_feature.unsqueeze(1)# torch.Size([8, 1, 256])
        result_mri_cli, result_pet_cli = self.fusion(mri_cli_feature, pet_cli_feature)
        result = torch.cat((result_mri_cli, result_pet_cli), dim=-1).squeeze(1)
        # 获取MoE输出（不再包含损失计算）
        # moe_output = self.classify_head(result)
        # return moe_output  # 只返回预测结果
        output = self.classify_head(result)
        return output


class TriLightNetWithCBAMWithOutCasCade(nn.Module):
    def __init__(self, num_classes=2):
        super(TriLightNetWithCBAMWithOutCasCade, self).__init__()
        self.name = 'TriLightNet'
        self.MriExtraction = get_no_pretrained_vision_encoder()
        self.PetExtraction = get_no_pretrained_vision_encoder()
        self.Table = TransformerEncoder(output_dim=128)
        self.mri_cli_fusion = CrossModalCBAM(img_channels=256, table_dim=128)
        self.pet_cli_fusion = CrossModalCBAM(img_channels=256, table_dim=128)
        # self.fusion = CascadedGroupCrossAttention1D(dim=256, key_dim=16, num_heads=4, resolution=1, attn_ratio=4)


        # self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=16, in_channels=1, classes=num_classes)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classify_head = MlpKan(init_features=256 * 2, classes=num_classes)
        # self.classify_head = MoE_Classifier(
        #     in_features=512,
        #     num_experts=4,
        #     hidden_dim=256,
        #     num_classes=num_classes,
        #     top_k=2
        # )

    def forward(self, mri, pet, cli):
        """
        Mri: [8, 1, 96, 128, 96]
        Pet: [8, 1, 96, 128, 96]
        Clinical: [8, 9]
        """
        mri_feature = self.MriExtraction(mri)  # torch.Size([8, 256, 3, 4, 3])
        # print(f'mri feature shape: {mri_feature.shape}')
        pet_feature = self.PetExtraction(pet)  # torch.Size([8, 256, 3, 4, 3])
        # print(f'pet feature.shape:{pet_feature.shape}')
        cli_feature = self.Table(cli)  # [8, 256]

        # mri_feature = self.SA1(mri_feature)
        # pet_feature = self.SA2(pet_feature)
        # cli_feature = self.SA3(cli_feature)
        mri_cli_feature = self.mri_cli_fusion(mri_feature, cli_feature) # torch.Size([8, 256, 3, 4, 3])
        pet_cli_feature = self.pet_cli_fusion(pet_feature, cli_feature) # torch.Size([8, 256, 3, 4, 3])
        mri_cli_feature = self.avgpool(mri_cli_feature).view(mri_cli_feature.shape[0], -1)
        pet_cli_feature = self.avgpool(pet_cli_feature).view(pet_cli_feature.shape[0], -1)
        # mri_cli_feature = mri_cli_feature.unsqueeze(1)# torch.Size([8, 1, 256])
        # pet_cli_feature = pet_cli_feature.unsqueeze(1)# torch.Size([8, 1, 256])

        # torch.Size([8, 1, 256])

        result = torch.cat((mri_cli_feature, pet_cli_feature), dim=-1)

        # 获取MoE输出（不再包含损失计算）
        # moe_output = self.classify_head(result)
        # return moe_output  # 只返回预测结果
        output = self.classify_head(result)
        return output

class TriLightNetWithOutCBAMWithoutCascade(nn.Module):
    def __init__(self, num_classes=2):
        super(TriLightNetWithOutCBAMWithoutCascade, self).__init__()
        self.name = 'TriLightNet'
        self.MriExtraction = get_no_pretrained_vision_encoder()
        self.PetExtraction = get_no_pretrained_vision_encoder()
        self.Table = TransformerEncoder(output_dim=256)
        # self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=16, in_channels=1, classes=num_classes)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classify_head = MlpKan(init_features=256 * 3, classes=num_classes)
        # self.classify_head = MoE_Classifier(
        #     in_features=512,
        #     num_experts=4,
        #     hidden_dim=256,
        #     num_classes=num_classes,
        #     top_k=2
        # )

    def forward(self, mri, pet, cli):
        """
        Mri: [8, 1, 96, 128, 96]
        Pet: [8, 1, 96, 128, 96]
        Clinical: [8, 9]
        """
        mri_feature = self.MriExtraction(mri)  # torch.Size([8, 256, 3, 4, 3])
        # print(f'mri feature shape: {mri_feature.shape}')
        mri_feature = self.avgpool(mri_feature)
        mri_feature = mri_feature.view(mri_feature.size(0), -1)
        pet_feature = self.PetExtraction(pet)  # torch.Size([8, 256, 3, 4, 3])
        pet_feature = self.avgpool(pet_feature)
        pet_feature = pet_feature.view(pet_feature.size(0), -1)
        # print(f'pet feature.shape:{pet_feature.shape}')
        cli_feature = self.Table(cli)  # [8, 256]
        result = torch.cat([mri_feature, pet_feature, cli_feature], dim=-1)
        output = self.classify_head(result)
        return output


if __name__ == '__main__':
    # model = TriLightNet()
    # model = TriLightNetWithoutCBAMWithCasCade()
    # model = TriLightNetWithCBAMWithOutCasCade()
    model = TriLightNetWithOutCBAMWithoutCascade()
    x = torch.randn(8, 1, 96, 128, 96)
    y = torch.randn(8, 1, 96, 128, 96)
    z = torch.randn(8, 9)
    output = model(x, y, z)
    # print(output.shape)
    from thop import profile, clever_format
    flops, params = profile(model, inputs=(x, y, z))
    flops, params = clever_format([flops, params], "%.3f")
    print(f'flops:{flops}, params:{params}')
    # torch.save(model.state_dict(), './TriLightNet.pth')

    # moe_output = model(x, y, z)
    # aux_loss = model.classify_head.compute_aux_loss(
    #     moe_output["gate_scores"],
    #     moe_output["expert_indices"]
    # )
    # print(f'aux_loss:{aux_loss}') # aux_loss:0.0006294594495557249

    # model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), n_classes=128)
    # from thop import profile, clever_format
    # flops, params = profile(model, inputs=(x,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(f'flops:{flops}, params:{params}')
    # torch.save(model.state_dict(), 'TriLightNet.pth')
"""
TriLightNet flops:84.139G, params:17.852M
TriLightNetWithoutCBAMWithCasCade flops:84.135G, params:17.391M
TriLightNetWithCBAMWithOutCasCade flops:84.134G, params:17.221M
TriLightNetWithOutCBAMWithoutCascade flops:84.135G, params:17.371M
"""


