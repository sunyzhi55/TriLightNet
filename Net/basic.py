import torch
import torch.nn as nn
from timm.layers import DropPath
from Net.kan import *
# from mamba_ssm import Mamba

# 定义交叉洗牌函数，交叉/穿插拼接
def shuffle_interleave(tensor1, tensor2):
    batch_size, length, dim= tensor1.shape

    # 确保两个张量的形状相同
    assert tensor1.shape == tensor2.shape, "两个张量的形状必须相同"

    # 将两个张量在通道维度进行交叉洗牌

    # 交替拼接两个张量的通道
    interleaved = torch.empty(batch_size, length * 2, dim, device=tensor1.device)  # (B, 1024, D*H*W)
    interleaved[:, 0::2, :] = tensor1  # 从偶数位置开始填充tensor1
    interleaved[:, 1::2, :] = tensor2  # 从奇数位置开始填充tensor2
    return interleaved
def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)
def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)
def get_inplanes():
    return [32, 64, 128, 256]
    # return [64, 128, 256, 512]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
def get_pretrained_vision_encoder(pretrained_path, **kwargs):
    pretrained_path = r"/data3/wangchangmiao/shenxy/Code/MedicalNet/pretrain/resnet_18.pth"\
        if pretrained_path is None else pretrained_path
    # model = ResNetDualInput(Bottleneck, [3, 4, 6, 3], get_inplanes())
    # model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    # /mntcephfs/lab_data/wangcm/hxy/Pre-model/r3d50_K_200ep.pth
    # /home/shenxiangyuhd/PretrainedResnet/r3d50_K_200ep.pth
    # /home/wangchangmiao/syz/PretrainedResnet/r3d50_K_200ep.pth
    # state_dict = torch.load(r"/home/shenxiangyuhd/PretrainedResnet/r3d50_K_200ep.pth")['state_dict']
    # keys = list(state_dict.keys())
    # state_dict.pop(keys[0])
    # state_dict.pop(keys[-1])
    # state_dict.pop(keys[-2])
    # model.load_state_dict(state_dict, strict=False)

    state_dict = torch.load(pretrained_path)['state_dict']
    # keys = list(state_dict.keys())
    # print("keys", keys)
    checkpoint = dict()
    for k in state_dict.keys():
        checkpoint[".".join(k.split(".")[1:])] = state_dict[k]
    model.load_state_dict(checkpoint, strict=False)

    # for name, param in model.named_parameters():
    #     if name in state_dict.keys():
    #         param.requires_grad = False
    return model

def get_pretrained_multilayer_vision_encoder(**kwargs):
    model = MultiLayerResNetEncoder(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    # /mntcephfs/lab_data/wangcm/hxy/Pre-model/r3d50_K_200ep.pth
    # /home/shenxiangyuhd/PretrainedResnet/r3d50_K_200ep.pth
    # /home/wangchangmiao/syz/PretrainedResnet/r3d50_K_200ep.pth
    state_dict = torch.load(r"/home/shenxiangyuhd/PretrainedResnet/r3d50_K_200ep.pth")['state_dict']
    keys = list(state_dict.keys())
    state_dict.pop(keys[0])
    state_dict.pop(keys[-1])
    state_dict.pop(keys[-2])
    model.load_state_dict(state_dict, strict=False)
    # for name, param in model.named_parameters():
    #     if name in state_dict.keys():
    #         param.requires_grad = False
    return model

# 提取特征的Resnet不是预训练的
def get_no_pretrained_vision_encoder(**kwargs):
    # model = ResNetDualInput(Bottleneck, [3, 4, 6, 3], get_inplanes())
    # model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes())
    # resnet 18
    model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    return model

def get_no_pretrained_multilayer_vision_encoder(**kwargs):
    # model = ResNetDualInput(Bottleneck, [3, 4, 6, 3], get_inplanes())
    # model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes())
    # resnet 18
    model = MultiLayerResNetEncoder(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    return model

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class PoolingTokenMixer(nn.Module):
    def __init__(self, dim, pool_size=3):
        """
        用池化实现Token Mixing
        Args:
            dim (int): 输入和输出特征维度
            pool_size (int): 池化窗口大小
        """
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=1, padding=pool_size // 2)
        self.linear = nn.Linear(dim, dim)  # 保证输出形状与输入一致

    def forward(self, x):
        # x: [seq_len, batch_size, feature_dim]
        x = x.permute(1, 2, 0)  # 转为 [batch_size, feature_dim, seq_len] 适配1D池化
        x = self.pool(x)  # 池化操作
        x = x.permute(2, 0, 1)  # 转回 [seq_len, batch_size, feature_dim]
        x = self.linear(x)  # 线性变换
        return x

class TransformerStylePoolFormerBlock(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mixer = PoolingTokenMixer(dim, pool_size)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # x: [seq_len, batch_size, dim]
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerStylePoolFormer(nn.Module):
    def __init__(self, num_layers, dim, pool_size=3, mlp_ratio=4.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerStylePoolFormerBlock(dim, pool_size, mlp_ratio, drop, drop_path)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: [seq_len, batch_size, feature_dim]
        for block in self.blocks:
            x = block(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=9, embed_dim=128, num_heads=4, ff_hidden_dim=512, num_layers=3, output_dim=400):
        super(TransformerEncoder, self).__init__()

        # Step 1: Input Embedding (Projection)
        # self.embedding = nn.Linear(input_dim, embed_dim)  # 把输入投影到更高维度（可调）
        self.fc1 = KAN([9, embed_dim])
        # =======================================================================
        # Step 2: Transformer Encoder Layer
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=embed_dim,
        #     nhead=num_heads,
        #     dim_feedforward=ff_hidden_dim
        # )
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # =======================================================================

        # pool_transformer
        self.transformer_encoder = TransformerStylePoolFormer(num_layers=num_layers, dim=embed_dim)


        # Step 3: Output Projection to required output_dim
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # x shape: [batchsize, 9]

        # Project input to higher dimension (embedding)
        x = self.fc1(x)  # shape: [batchsize, 9] -> [batchsize, embed_dim]

        # Add an extra dimension for the sequence length (needed for Transformer)
        x = x.unsqueeze(1)  # shape: [batchsize, 1, embed_dim]

        # Transformer Encoder expects input shape [sequence length, batchsize, embedding_dim]
        x = x.permute(1, 0, 2)  # shape: [1, batchsize, embed_dim]

        # Apply Transformer Encoder
        x = self.transformer_encoder(x)  # shape: [1, batchsize, embed_dim]

        # Remove the sequence dimension (transpose back)
        x = x.squeeze(0)  # shape: [batchsize, embed_dim]

        # Project to the required output dimension
        output = self.fc_out(x)  # shape: [batchsize, embed_dim] -> [batchsize, 400]

        return output

class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        """
        SelfAttention Module
        """
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class SelfMamba(nn.Module):
    def __init__(self,  input_size, hidden_size, hidden_dropout_prob, d_state=64):
        """
        SelfAttention Module
        """
        super(SelfMamba, self).__init__()
        self.mamba = Mamba(d_model=input_size, d_state=d_state, d_conv=4, expand=2,)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_tensor):
        context_layer = self.mamba(input_tensor)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class DenseLayer(torch.nn.Module):
    def __init__(self, in_channels, middle_channels=128, out_channels=32):
        super(DenseLayer, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(in_channels, middle_channels, 1),
            torch.nn.BatchNorm1d(middle_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(middle_channels, out_channels, 3, padding=1)
        )

    def forward(self, x):
        return torch.cat([x, self.layer(x)], dim=1)

class DenseBlock(torch.nn.Sequential):
    def __init__(self, layer_num, growth_rate, in_channels, middele_channels=128):
        super(DenseBlock, self).__init__()
        for i in range(layer_num):
            layer = DenseLayer(in_channels + i * growth_rate, middele_channels, growth_rate)
            self.add_module('denselayer%d' % (i), layer)

class Transition(torch.nn.Sequential):
    def __init__(self, channels):
        super(Transition, self).__init__()
        self.add_module('norm', torch.nn.BatchNorm1d(channels))
        self.add_module('relu', torch.nn.ReLU(inplace=True))
        self.add_module('conv', torch.nn.Conv1d(channels, channels // 2, 3, padding=1))
        self.add_module('Avgpool', torch.nn.AvgPool1d(2))

class DenseNet(torch.nn.Module):
    def __init__(self, layer_num=(6, 12, 24, 16), growth_rate=32, init_features=64, in_channels=1, middele_channels=128,
                 classes=2):
        """
        1D-DenseNet Module, use to conv global feature and generate final target
        """
        super(DenseNet, self).__init__()
        self.feature_channel_num = init_features
        self.conv = torch.nn.Conv1d(in_channels, self.feature_channel_num, 7, 2, 3)
        self.norm = torch.nn.BatchNorm1d(self.feature_channel_num)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(3, 2, 1)

        self.DenseBlock1 = DenseBlock(layer_num[0], growth_rate, self.feature_channel_num, middele_channels)
        self.feature_channel_num = self.feature_channel_num + layer_num[0] * growth_rate
        self.Transition1 = Transition(self.feature_channel_num)

        self.DenseBlock2 = DenseBlock(layer_num[1], growth_rate, self.feature_channel_num // 2, middele_channels)
        self.feature_channel_num = self.feature_channel_num // 2 + layer_num[1] * growth_rate
        self.Transition2 = Transition(self.feature_channel_num)

        self.DenseBlock3 = DenseBlock(layer_num[2], growth_rate, self.feature_channel_num // 2, middele_channels)
        self.feature_channel_num = self.feature_channel_num // 2 + layer_num[2] * growth_rate
        self.Transition3 = Transition(self.feature_channel_num)

        self.DenseBlock4 = DenseBlock(layer_num[3], growth_rate, self.feature_channel_num // 2, middele_channels)
        self.feature_channel_num = self.feature_channel_num // 2 + layer_num[3] * growth_rate

        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)

        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(self.feature_channel_num, self.feature_channel_num // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.feature_channel_num // 2, classes),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        # print("x1 conv", x.shape)
        x = self.norm(x)
        # print("x2 norm", x.shape)
        x = self.relu(x)
        # print("x3 relu", x.shape)
        x = self.maxpool(x)
        # print("x4 maxpool", x.shape)
        x = self.DenseBlock1(x)
        # print("x5 DenseBlock1", x.shape)
        x = self.Transition1(x)
        # print("x6 Transition1", x.shape)

        x = self.DenseBlock2(x)
        # print("x7 DenseBlock2", x.shape)
        x = self.Transition2(x)
        # print("x8 Transition2", x.shape)
        x = self.DenseBlock3(x)
        # print("x9 DenseBlock3", x.shape)
        x = self.Transition3(x)
        # print("x10 Transition3", x.shape)

        x = self.DenseBlock4(x)
        # print("x11 DenseBlock4", x.shape)
        x = self.avgpool(x)
        # print("x12 avgpool", x.shape)
        x = x.view(-1, self.feature_channel_num)
        # print("x13 view", x.shape)
        x = self.classifer(x)
        # print("x14 classifer", x.shape)
        """
        x1 conv torch.Size([8, 64, 384])
        x2 norm torch.Size([8, 64, 384])
        x3 relu torch.Size([8, 64, 384])
        x4 maxpool torch.Size([8, 64, 192])
        x5 DenseBlock1 torch.Size([8, 160, 192])
        x6 Transition1 torch.Size([8, 80, 96])
        x7 DenseBlock2 torch.Size([8, 272, 96])
        x8 Transition2 torch.Size([8, 136, 48])
        x9 DenseBlock3 torch.Size([8, 520, 48])
        x10 Transition3 torch.Size([8, 260, 24])
        x11 DenseBlock4 torch.Size([8, 516, 24])
        x12 avgpool torch.Size([8, 516, 1])
        x13 view torch.Size([8, 516])
        x14 classifer torch.Size([8, 2])
        """
        return x

class MlpKan(torch.nn.Module):
    def __init__(self, init_features=64, classes=2):
        """
        1D-DenseNet Module, use to conv global feature and generate final target
        """
        super(MlpKan, self).__init__()
        self.feature_channel_num = init_features

        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(self.feature_channel_num, self.feature_channel_num // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.feature_channel_num // 2, classes),
            # torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, self.feature_channel_num)
        x = self.classifer(x)
        return x

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=2,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                from functools import partial
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)

        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

class MultiLayerResNetEncoder(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                from functools import partial
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # print("conv1", x.shape)
        x = self.bn1(x)
        # print("bn1", x.shape)
        x = self.relu(x)
        # print("relu", x.shape)
        if not self.no_max_pool:
            x = self.maxpool(x)

        layer1_x = self.layer1(x)  # [8, 64, 48, 32, 24]
        # print("layer1", layer1_x.shape)
        layer2_x = self.layer2(layer1_x)  # [8, 128, 24, 16, 12]
        # print("layer2", layer2_x.shape)
        layer3_x = self.layer3(layer2_x)   # [8, 256, 12, 8, 6]
        # print("layer3", layer3_x.shape)
        layer4_x = self.layer4(layer3_x)   # [8, 512, 6, 4, 3]
        # print("layer4", layer4_x.shape)

        x = self.avgpool(layer4_x)
        # print("avgpool", x.shape)

        x = x.view(x.size(0), -1)
        # print("view", x.shape)
        x = self.fc(x)

        return layer1_x, layer2_x, layer3_x, layer4_x, x



# if __name__ == '__main__':
#
#     B, L, C = 2, 128, 192 # batch size, sequence length, embedding dimension
#     x = torch.randn(B, L, C)
#     model = CascadedGroupAttention1D(dim=C, key_dim=16, num_heads=4, resolution=L, attn_ratio=3)
#     print("params", sum([p.numel() for p in model.parameters()]))
#
#     out = model(x)




if __name__ == '__main__':
    MriExtraction = get_no_pretrained_vision_encoder() # input [B,C,128,128,128] OUT[8, 400]
    PetExtraction = get_no_pretrained_vision_encoder() # input [B,C,128,128,128] OUT[8, 400]
    mri_demo = torch.randn(8, 1, 96, 128, 96)
    pet_demo = torch.randn(8, 1, 96, 128, 96)
    mri_feature = MriExtraction(mri_demo)
    pet_feature = PetExtraction(pet_demo)
    # mri_extraction = model(x)
    print("mri_feature", mri_feature.shape)
    print("pet_feature", pet_feature.shape)
    """
    mri_feature torch.Size([8, 256, 3, 4, 3])
    pet_feature torch.Size([8, 256, 3, 4, 3])
    """

