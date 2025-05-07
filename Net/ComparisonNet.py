import torch
from torch import nn
from Net.basic import *
from Net.defineViT import ViT
from Net.metaformer3D import poolformerv2_3d_s12
import torch.nn.functional as F
from efficientnet_pytorch_3d import EfficientNet3D

class MLPTableEncoder(nn.Module):
    def __init__(self, input_dim=9, output_dim=256):
        super(MLPTableEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),  # 激活函数
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(SubNet, self).__init__()
        encoder1 = nn.Sequential(nn.Linear(in_size, hidden_size),nn.Tanh())
        encoder2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.encoder = nn.Sequential(encoder1, encoder2)
    def forward(self, x):
        y = self.encoder(x)
        return y

class HFBSurv(nn.Module):
    def __init__(self, input_dims=(256, 256, 256), hidden_dims=(50, 50, 50, 256), output_dims=(20, 20, 2),
                 dropouts=(0.1, 0.1, 0.1, 0.3), rank=20, fac_drop=0.1):
        super(HFBSurv, self).__init__()
        # self.Radio_encoder = Radiomic_encoder(num_features=1781)
        # self.Radio_encoder.projection_head = nn.Identity()

        # self.Resnet = get_pretrained_vision_encoder() # input [B,C,128,128,128] OUT[8.400]
        self.Resnet = get_no_pretrained_vision_encoder() # input [B,C,128,128,128] OUT[8.400]
        self.Table = TransformerEncoder(output_dim=256)

        # self.bert = AutoModel.from_pretrained("./models/Bio_ClinicalBERT")

        # self.fc_Radio = nn.Linear(512, 256)
        self.fc_vis = nn.Linear(400, 256)
        self.fc_text = nn.Linear(768, 256)

        self.gene_in = input_dims[0]
        self.path_in = input_dims[1]
        self.cona_in = input_dims[2]

        self.gene_hidden = hidden_dims[0]
        self.path_hidden = hidden_dims[1]
        self.cona_hidden = hidden_dims[2]
        self.cox_hidden = hidden_dims[3]

        self.output_intra = output_dims[0]
        self.output_inter = output_dims[1]
        self.label_dim = output_dims[2]
        self.rank = rank
        self.factor_drop = fac_drop

        self.gene_prob = dropouts[0]
        self.path_prob = dropouts[1]
        self.cona_prob = dropouts[2]
        self.cox_prob = dropouts[3]

        self.joint_output_intra = self.rank * self.output_intra
        self.joint_output_inter = self.rank * self.output_inter
        self.in_size = self.gene_hidden + self.output_intra + self.output_inter
        self.hid_size = self.gene_hidden

        self.norm = nn.BatchNorm1d(self.in_size)
        self.factor_drop = nn.Dropout(self.factor_drop)
        self.attention = nn.Sequential(nn.Linear((self.hid_size + self.output_intra), 1), nn.Sigmoid())

        self.encoder_gene = SubNet(self.gene_in, self.gene_hidden)
        self.encoder_path = SubNet(self.path_in, self.path_hidden)
        self.encoder_cona = SubNet(self.cona_in, self.cona_hidden)

        self.Linear_gene = nn.Linear(self.gene_hidden, self.joint_output_intra)
        self.Linear_path = nn.Linear(self.path_hidden, self.joint_output_intra)
        self.Linear_cona = nn.Linear(self.cona_hidden, self.joint_output_intra)

        self.Linear_gene_a = nn.Linear(self.gene_hidden + self.output_intra, self.joint_output_inter)
        self.Linear_path_a = nn.Linear(self.path_hidden + self.output_intra, self.joint_output_inter)
        self.Linear_cona_a = nn.Linear(self.cona_hidden + self.output_intra, self.joint_output_inter)

        #########################the layers of survival prediction#####################################
        encoder1 = nn.Sequential(nn.Linear(self.in_size, self.cox_hidden), nn.Tanh(), nn.Dropout(p=self.cox_prob))
        encoder2 = nn.Sequential(nn.Linear(self.cox_hidden, 64), nn.Tanh(), nn.Dropout(p=self.cox_prob))
        self.encoder = nn.Sequential(encoder1, encoder2)
        self.classifier = nn.Sequential(nn.Linear(64, self.label_dim), nn.Sigmoid())

        # self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        # self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def mfb(self, x1, x2, output_dim):
        self.output_dim = output_dim
        fusion = torch.mul(x1, x2)
        fusion = self.factor_drop(fusion)
        fusion = fusion.view(-1, 1, self.output_dim, self.rank)
        fusion = torch.squeeze(torch.sum(fusion, 3))
        fusion = torch.sqrt(F.relu(fusion)) - torch.sqrt(F.relu(-fusion))
        fusion = F.normalize(fusion)
        return fusion

    def forward(self, mri, pet, cli):
        # radio_feature = self.Radio_encoder(radio)[0]
        mri_feature = self.Resnet(mri)
        pet_feature = self.Resnet(pet)
        cli_feature = self.Table(cli)
        # cli_feature = self.bert(input_ids=input_ids, attention_mask=attention_mask,
        #                         token_type_ids=token_type_ids).pooler_output

        mri_feature = self.fc_vis(mri_feature)
        pet_feature = self.fc_vis(pet_feature)
        # cli_feature = self.fc_text(cli_feature)

        x1 = mri_feature
        x2 = pet_feature
        x3 = cli_feature

        gene_feature = self.encoder_gene(x1.squeeze(1))
        path_feature = self.encoder_path(x2.squeeze(1))
        cona_feature = self.encoder_cona(x3.squeeze(1))

        gene_h = self.Linear_gene(gene_feature)
        path_h = self.Linear_path(path_feature)
        cona_h = self.Linear_cona(cona_feature)

        ######################### modelity-specific###############################
        # intra_interaction#
        intra_gene = self.mfb(gene_h, gene_h, self.output_intra)
        intra_path = self.mfb(path_h, path_h, self.output_intra)
        intra_cona = self.mfb(cona_h, cona_h, self.output_intra)

        gene_x = torch.cat((gene_feature, intra_gene), 1)
        path_x = torch.cat((path_feature, intra_path), 1)
        cona_x = torch.cat((cona_feature, intra_cona), 1)

        sg = self.attention(gene_x)
        sp = self.attention(path_x)
        sc = self.attention(cona_x)

        sg_a = (sg.expand(gene_feature.size(0), (self.gene_hidden + self.output_intra)))
        sp_a = (sp.expand(path_feature.size(0), (self.path_hidden + self.output_intra)))
        sc_a = (sc.expand(cona_feature.size(0), (self.cona_hidden + self.output_intra)))

        gene_x_a = sg_a * gene_x
        path_x_a = sp_a * path_x
        cona_x_a = sc_a * gene_x

        unimodal = gene_x_a + path_x_a + cona_x_a

        ######################### cross-modelity######################################
        g = F.softmax(gene_x_a, 1)
        p = F.softmax(path_x_a, 1)
        c = F.softmax(cona_x_a, 1)

        sg = sg.squeeze()
        sp = sp.squeeze()
        sc = sc.squeeze()

        sgp = (1 / (torch.matmul(g.unsqueeze(1), p.unsqueeze(2)).squeeze() + 0.5) * (sg + sp))
        sgc = (1 / (torch.matmul(g.unsqueeze(1), c.unsqueeze(2)).squeeze() + 0.5) * (sg + sc))
        spc = (1 / (torch.matmul(p.unsqueeze(1), c.unsqueeze(2)).squeeze() + 0.5) * (sp + sc))
        normalize = torch.cat((sgp.unsqueeze(1), sgc.unsqueeze(1), spc.unsqueeze(1)), 1)
        normalize = F.softmax(normalize, 1)
        sgp_a = normalize[:, 0].unsqueeze(1).expand(gene_feature.size(0), self.output_inter)
        sgc_a = normalize[:, 1].unsqueeze(1).expand(path_feature.size(0), self.output_inter)
        spc_a = normalize[:, 2].unsqueeze(1).expand(cona_feature.size(0), self.output_inter)

        # inter_interaction#
        gene_l = self.Linear_gene_a(gene_x_a)
        path_l = self.Linear_gene_a(path_x_a)
        cona_l = self.Linear_gene_a(cona_x_a)

        inter_gene_path = self.mfb(gene_l, path_l, self.output_inter)
        inter_gene_cona = self.mfb(gene_l, cona_l, self.output_inter)
        inter_path_cona = self.mfb(path_l, cona_l, self.output_inter)

        bimodal = sgp_a * inter_gene_path + sgc_a * inter_gene_cona + spc_a * inter_path_cona
        ############################################### fusion layer ###################################################

        fusion = torch.cat((unimodal, bimodal), 1)
        fusion = self.norm(fusion)
        code = self.encoder(fusion)
        out = self.classifier(code)
        # out = out * self.output_range + self.output_shift
        return out

# 三模态融合模块
class TriModalCrossAttention(nn.Module):
    def __init__(self, input_dim):
        super(TriModalCrossAttention, self).__init__()
        self.W_q1 = nn.Linear(input_dim, input_dim)
        self.W_k1 = nn.Linear(input_dim, input_dim)
        self.W_v1 = nn.Linear(input_dim, input_dim)

        self.W_q2 = nn.Linear(input_dim, input_dim)
        self.W_k2 = nn.Linear(input_dim, input_dim)
        self.W_v2 = nn.Linear(input_dim, input_dim)

        self.W_q3 = nn.Linear(input_dim, input_dim)
        self.W_k3 = nn.Linear(input_dim, input_dim)
        self.W_v3 = nn.Linear(input_dim, input_dim)

        self.W_o1 = nn.Linear(input_dim * 2, input_dim)
        self.W_o2 = nn.Linear(input_dim * 2, input_dim)
        self.W_o3 = nn.Linear(input_dim * 2, input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x1, x2, x3):
        # x1, x2, x3: [B, N, input_dim]
        batch_size, seq_len, _ = x1.size()

        # Linear transformations for each modality
        queries1 = self.W_q1(x1)
        keys2 = self.W_k2(x2)
        values2 = self.W_v2(x2)

        queries2 = self.W_q2(x2)
        keys3 = self.W_k3(x3)
        values3 = self.W_v3(x3)

        queries3 = self.W_q3(x3)
        keys1 = self.W_k1(x1)
        values1 = self.W_v1(x1)

        # Scaled dot-product attention
        attention_scores1 = torch.matmul(queries1, keys2.transpose(-2, -1)) / (x1.size(-1) ** 0.5)  # [B, N, N]
        attention_weights1 = F.softmax(attention_scores1, dim=-1)
        context1 = torch.matmul(self.dropout(attention_weights1), values2)  # [B, N, input_dim]

        attention_scores2 = torch.matmul(queries2, keys3.transpose(-2, -1)) / (x2.size(-1) ** 0.5)  # [B, N, N]
        attention_weights2 = F.softmax(attention_scores2, dim=-1)
        context2 = torch.matmul(self.dropout(attention_weights2), values3)  # [B, N, input_dim]

        attention_scores3 = torch.matmul(queries3, keys1.transpose(-2, -1)) / (x3.size(-1) ** 0.5)  # [B, N, N]
        attention_weights3 = F.softmax(attention_scores3, dim=-1)
        context3 = torch.matmul(self.dropout(attention_weights3), values1)  # [B, N, input_dim]

        # Concatenate context with input for each modality
        combined1 = torch.cat((x1, context1), dim=-1)  # [B, N, input_dim * 2]
        combined2 = torch.cat((x2, context2), dim=-1)  # [B, N, input_dim * 2]
        combined3 = torch.cat((x3, context3), dim=-1)  # [B, N, input_dim * 2]

        # Linear transformations and output for each modality
        output1 = self.W_o1(combined1)
        output2 = self.W_o2(combined2)
        output3 = self.W_o3(combined3)

        global_feature = torch.cat((output1, output2, output3), dim=1) # [B, N * 3, input_dim ]
        return output1, output2, output3, global_feature

class TriModalCrossAttention_ver2(nn.Module):
    def __init__(self, input_dim):
        super(TriModalCrossAttention_ver2, self).__init__()

        # 定义 Query, Key, Value 的线性变换
        self.W_q = nn.Linear(input_dim * 3, input_dim)  # 用于拼接后的查询
        self.W_k1 = nn.Linear(input_dim, input_dim)
        self.W_v1 = nn.Linear(input_dim, input_dim)

        self.W_k2 = nn.Linear(input_dim, input_dim)
        self.W_v2 = nn.Linear(input_dim, input_dim)

        self.W_k3 = nn.Linear(input_dim, input_dim)
        self.W_v3 = nn.Linear(input_dim, input_dim)

        # 输出投影
        self.W_o1 = nn.Linear(input_dim * 2, input_dim)
        self.W_o2 = nn.Linear(input_dim * 2, input_dim)
        self.W_o3 = nn.Linear(input_dim * 2, input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x1, x2, x3):
        # x1, x2, x3: [B, N, input_dim] [8, 256, 1]
        batch_size, seq_len, input_dim = x1.size()

        # 将三个模态在最后一个维度拼接得到 [B, N, input_dim * 3]
        combined_input = torch.cat([x1, x2, x3], dim=-1)
        queries = self.W_q(combined_input)  # [B, N, input_dim * 3]

        # 对每个模态的键和值进行投影
        keys1 = self.W_k1(x1)  # [B, N, input_dim]
        values1 = self.W_v1(x1)

        keys2 = self.W_k2(x2)  # [B, N, input_dim]
        values2 = self.W_v2(x2)

        keys3 = self.W_k3(x3)  # [B, N, input_dim]
        values3 = self.W_v3(x3)

        # 分别计算每个模态的交叉注意力输出
        attention_scores1 = torch.matmul(queries, keys1.transpose(-2, -1)) / (input_dim ** 0.5)  # [B, N, N]
        attention_weights1 = F.softmax(attention_scores1, dim=-1)
        context1 = torch.matmul(self.dropout(attention_weights1), values1)  # [B, N, input_dim]

        attention_scores2 = torch.matmul(queries, keys2.transpose(-2, -1)) / (input_dim ** 0.5)  # [B, N, N]
        attention_weights2 = F.softmax(attention_scores2, dim=-1)
        context2 = torch.matmul(self.dropout(attention_weights2), values2)  # [B, N, input_dim]

        attention_scores3 = torch.matmul(queries, keys3.transpose(-2, -1)) / (input_dim ** 0.5)  # [B, N, N]
        attention_weights3 = F.softmax(attention_scores3, dim=-1)
        context3 = torch.matmul(self.dropout(attention_weights3), values3)  # [B, N, input_dim]

        # 将原输入与注意力上下文拼接
        combined1 = torch.cat((x1, context1), dim=-1)  # [B, N, input_dim * 2]
        combined2 = torch.cat((x2, context2), dim=-1)  # [B, N, input_dim * 2]
        combined3 = torch.cat((x3, context3), dim=-1)  # [B, N, input_dim * 2]

        # 线性变换生成最终输出
        output1 = self.W_o1(combined1)  # [B, N, input_dim]
        output2 = self.W_o2(combined2)  # [B, N, input_dim]
        output3 = self.W_o3(combined3)  # [B, N, input_dim]

        # 将三个模态的输出在第二维度拼接 [B, N * 3, input_dim]
        vision_feature = shuffle_interleave(output1, output2)
        # global_feature = torch.cat((output1, output2, output3), dim=1)
        global_feature = torch.cat((vision_feature,output3), dim=1)

        return output1, output2, output3, global_feature

class Triple_model_Fusion(nn.Module):
    def __init__(self, num_classes=2):
        super(Triple_model_Fusion, self).__init__()
        self.name = 'Triple_model_CrossAttentionFusion_self_KAN'
        self.Resnet = get_no_pretrained_vision_encoder() # input [B,C,128,128,128] OUT[8.400]
        self.Table = TransformerEncoder(output_dim=256)
        self.fc_vis = nn.Linear(400, 256)
        self.fusion = TriModalCrossAttention_ver2(input_dim=1)
        self.SA1 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        self.SA2 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        self.SA3 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=16, in_channels=1, classes=num_classes)

    def forward(self, mri, pet, cli):
        mri_feature = self.Resnet(mri)
        # print(f'mri feature shape: {mri_feature.shape}')
        pet_feature = self.Resnet(pet)
        # print(f'pet feature.shape:{pet_feature.shape}')
        cli_feature = self.Table(cli)

        mri_feature = self.fc_vis(mri_feature)
        pet_feature = self.fc_vis(pet_feature)
        # cli_feature = self.fc_cli(cli_feature)

        mri_feature = torch.unsqueeze(mri_feature, dim=1)
        pet_feature = torch.unsqueeze(pet_feature, dim=1)
        cli_feature = torch.unsqueeze(cli_feature, dim=1)

        mri_feature = self.SA2(mri_feature)
        pet_feature = self.SA3(pet_feature)
        cli_feature = self.SA1(cli_feature)
        # mri_feature.shape torch.Size([8, 256, 1])
        mri_feature_tr = mri_feature.permute(0, 2, 1)
        pet_feature_tr = pet_feature.permute(0, 2, 1)
        cli_feature_tr = cli_feature.permute(0, 2, 1)
        # print(mri_feature_tr.shape)
        # _, _, _, global_feature  = self.fusion(mri_feature_tr, pet_feature_tr, cli_feature_tr)
        # print(global_feature.shape)
        global_feature = torch.cat((mri_feature, pet_feature, cli_feature), dim=-1)
        # global_feature = global_feature.permute(0, 2, 1)
        output = self.classify_head(global_feature)
        return mri_feature, pet_feature, cli_feature, output

class Triple_model_CoAttention_Fusion(nn.Module):
    def __init__(self, num_classes=2):
        super(Triple_model_CoAttention_Fusion, self).__init__()
        self.name = 'Triple_model_CrossAttentionFusion_self_KAN'
        self.MriExtraction = get_no_pretrained_vision_encoder() # input [B,C,96,128,96] OUT[8, 400]
        self.PetExtraction = get_no_pretrained_vision_encoder() # input [B,C,96,128,96] OUT[8, 400]
        self.Table = TransformerEncoder(output_dim=256)
        self.fc_vis = nn.Linear(400, 256)
        self.fusion = TriModalCrossAttention_ver2(input_dim=1)

        # self.mamba1 = SelfMamba(256, 256, hidden_dropout_prob=0.2, d_state=64)
        # self.mamba2 = SelfMamba(256, 256, hidden_dropout_prob=0.2, d_state=64)
        # self.mamba3 = SelfMamba(256, 256, hidden_dropout_prob=0.2, d_state=64)

        self.SA1 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        self.SA2 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        self.SA3 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)

        # self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=16, in_channels=1, classes=num_classes)
        self.classify_head = MlpKan(init_features=768, classes=num_classes)

    def forward(self, mri, pet, cli):
        """
        Mri: [8, 1, 96, 128, 96]
        Pet: [8, 1, 96, 128, 96]
        Clinicla: [8, 9]
        """
        mri_feature = self.MriExtraction(mri)  # [8, 400]
        # print(f'mri feature shape: {mri_feature.shape}')
        pet_feature = self.PetExtraction(pet)  # [8, 400]
        # print(f'pet feature.shape:{pet_feature.shape}')
        cli_feature = self.Table(cli)  # [8, 256]

        mri_feature = self.fc_vis(mri_feature)
        pet_feature = self.fc_vis(pet_feature)
        # cli_feature = self.fc_cli(cli_feature)

        mri_feature = torch.unsqueeze(mri_feature, dim=1)
        pet_feature = torch.unsqueeze(pet_feature, dim=1)
        cli_feature = torch.unsqueeze(cli_feature, dim=1)

        mri_feature = self.SA1(mri_feature)
        pet_feature = self.SA2(pet_feature)
        cli_feature = self.SA3(cli_feature)
        # mri_feature.shape torch.Size([8, 1, 256])
        # pet_feature.shape torch.Size([8, 1, 256])
        # cli_feature.shape torch.Size([8, 1, 256])
        mri_feature_tr = mri_feature.permute(0, 2, 1)
        pet_feature_tr = pet_feature.permute(0, 2, 1)
        cli_feature_tr = cli_feature.permute(0, 2, 1)
        # mri_feature_tr.shape torch.Size([8, 256, 1])
        # pet_feature_tr.shape torch.Size([8, 256, 1])
        # cli_feature_tr.shape torch.Size([8, 256, 1])

        _, _, _, global_feature  = self.fusion(mri_feature_tr, pet_feature_tr, cli_feature_tr)
        # print("global_feature", global_feature.shape) # global_feature2 torch.Size([8, 768, 1])
        global_feature = global_feature.permute(0, 2, 1)
        output = self.classify_head(global_feature)
        return mri_feature, pet_feature, cli_feature, output

# MutanLayer In Interactive Multimodal Fusion Model
class MutanLayer(nn.Module):
    def __init__(self, dim, multi):
        super(MutanLayer, self).__init__()

        self.dim = dim
        self.multi = multi

        modal1 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal1.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal1_layers = nn.ModuleList(modal1)

        modal2 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal2.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal2_layers = nn.ModuleList(modal2)

        modal3 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal3.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal3_layers = nn.ModuleList(modal3)

    def forward(self, modal1_emb, modal2_emb, modal3_emb):
        bs = modal1_emb.size(0)
        x_mm = []
        for i in range(self.multi):
            x_modal1 = self.modal1_layers[i](modal1_emb)
            x_modal2 = self.modal2_layers[i](modal2_emb)
            x_modal3 = self.modal3_layers[i](modal3_emb)
            x_mm.append(torch.mul(torch.mul(x_modal1, x_modal2), x_modal3))
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(bs, self.dim)
        x_mm = torch.relu(x_mm)
        return x_mm

# IMF Net
class Interactive_Multimodal_Fusion_Model(nn.Module):
    def __init__(self, num_classes=2):
        super(Interactive_Multimodal_Fusion_Model, self).__init__()
        self.name = 'Interactive_Multimodal_Fusion_Model'
        self.MriExtraction = get_no_pretrained_vision_encoder()  # input [B,C,96,128,96] OUT[8, 400]
        # self.MriExtraction = Simple3DResNet(num_classes=400)  # input [B,C,96,128,96] OUT[8, 400]
        self.PetExtraction = get_no_pretrained_vision_encoder()  # input [B,C,96,128,96] OUT[8, 400]
        # self.PetExtraction = Simple3DResNet(num_classes=400)  # input [B,C,96,128,96] OUT[8, 400]
        self.Table = TransformerEncoder(output_dim=256)
        self.fc_mri = nn.Linear(400, 256)
        self.fc_pet = nn.Linear(400, 256)
        self.fusion = MutanLayer(dim=256, multi=2)

        # self.SA1 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        # self.SA2 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        # self.SA3 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)

        # self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=16, in_channels=1, classes=2)
        self.classify_head = MlpKan(init_features=256, classes=num_classes)
    def forward(self, mri, pet, cli):
        """
        Mri: [8, 1, 96, 128, 96]
        Pet: [8, 1, 96, 128, 96]
        Clinicla: [8, 9]
        """
        mri_feature = self.MriExtraction(mri)  # [8, 400]
        # print(f'mri feature shape: {mri_feature.shape}')
        pet_feature = self.PetExtraction(pet)  # [8, 400]
        # print(f'pet feature.shape:{pet_feature.shape}')
        cli_feature = self.Table(cli)  # [8, 256]

        mri_feature = self.fc_mri(mri_feature)
        pet_feature = self.fc_pet(pet_feature)
        # cli_feature = self.fc_cli(cli_feature)

        # mri_feature.shape torch.Size([8, 256])
        # pet_feature.shape torch.Size([8, 256])
        # cli_feature.shape torch.Size([8, 256])

        # mri_feature = torch.unsqueeze(mri_feature, dim=1)
        # pet_feature = torch.unsqueeze(pet_feature, dim=1)
        # cli_feature = torch.unsqueeze(cli_feature, dim=1)
        #
        # mri_feature = self.SA1(mri_feature)
        # pet_feature = self.SA2(pet_feature)
        # cli_feature = self.SA3(cli_feature)

        # mri_feature.shape torch.Size([8, 1, 256])
        # pet_feature.shape torch.Size([8, 1, 256])
        # cli_feature.shape torch.Size([8, 1, 256])

        # mri_feature_tr = mri_feature.permute(0, 2, 1)
        # pet_feature_tr = pet_feature.permute(0, 2, 1)
        # cli_feature_tr = cli_feature.permute(0, 2, 1)
        # mri_feature_tr.shape torch.Size([8, 256, 1])
        # pet_feature_tr.shape torch.Size([8, 256, 1])
        # cli_feature_tr.shape torch.Size([8, 256, 1])

        global_feature = self.fusion(mri_feature, pet_feature, cli_feature)
        # print("global_feature", global_feature.shape) # global_feature2 torch.Size([8, 768, 1])

        # print("global_feature", global_feature.shape) # global_feature2 torch.Size([8, 256])

        mri_output = torch.sigmoid(self.classify_head(mri_feature))
        pet_output = torch.sigmoid(self.classify_head(pet_feature))
        cli_output = torch.sigmoid(self.classify_head(cli_feature))
        global_output = torch.sigmoid(self.classify_head(global_feature))

        return [mri_output, pet_output, cli_output, global_output]

# Resnet mri pet
class ResnetMriPet(nn.Module):
    def __init__(self, num_classes=2, pretrained_path=None):
        super(ResnetMriPet, self).__init__()
        self.name = 'Resnet_mri_pet'
        self.MriExtraction = get_pretrained_vision_encoder(pretrained_path=pretrained_path)  # input [B,C,96,128,96] OUT[8, 400]
        self.PetExtraction = get_pretrained_vision_encoder(pretrained_path=pretrained_path)  # input [B,C,96,128,96] OUT[8, 400]
        self.fc = nn.Sequential(
            nn.Linear(800, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, mri_pet):
        """
        Mri: [8, 1, 96, 128, 96]
        Pet: [8, 1, 96, 128, 96]
        """
        # 已知输入 mri_pet 为[8, 2, 96, 128, 96]，现在通过一下代码获得mri pet 为 torch.Size([8, 96, 128, 96])，但是应该为torch.Size([8, 1, 96, 128, 96])，请解决
        mri = mri_pet[:, 0:1, :, :, :]
        # print(f'mri shape: {mri.shape}')
        pet = mri_pet[:, 1:2, :, :, :]
        # print(f'pet shape: {pet.shape}')
        mri_feature = self.MriExtraction(mri)  # [8, 400]
        # print(f'mri feature shape: {mri_feature.shape}')
        pet_feature = self.PetExtraction(pet)  # [8, 400]
        # print(f'pet feature.shape:{pet_feature.shape}')
        result = self.fc(torch.cat((mri_feature, pet_feature), dim=1))
        return result

class ViTMriPet(nn.Module):
    def __init__(self, num_classes=2, pretrained_path=None):
        super(ViTMriPet, self).__init__()
        self.name = 'ViT_mri_pet'
        self.MriExtraction = ViT(image_size=[96, 128, 96], patch_size=16, num_classes=2, dim=128, depth=2,
                                 heads=16, mlp_dim=512, channels=1, dropout=0.1, emb_dropout=0.1)
        self.PetExtraction = ViT(image_size=[96, 128, 96], patch_size=16, num_classes=2, dim=128, depth=2,
                                 heads=16, mlp_dim=512, channels=1, dropout=0.1, emb_dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(2 * 2, num_classes),
            # nn.ReLU(),
            # nn.Linear(256, 64),
            # nn.ReLU(),
            # nn.Linear(64, num_classes)
        )
    def forward(self, mri_pet):
        """
        Mri: [8, 1, 96, 128, 96]
        Pet: [8, 1, 96, 128, 96]
        """
        # 已知输入 mri_pet 为[8, 2, 96, 128, 96]，现在通过一下代码获得mri pet 为 torch.Size([8, 96, 128, 96])，但是应该为torch.Size([8, 1, 96, 128, 96])，请解决
        mri = mri_pet[:, 0:1, :, :, :]
        # print(f'mri shape: {mri.shape}')
        pet = mri_pet[:, 1:2, :, :, :]
        # print(f'pet shape: {pet.shape}')
        mri_feature = self.MriExtraction(mri)  # [8, 400]
        # print(f'mri feature shape: {mri_feature.shape}')
        pet_feature = self.PetExtraction(pet)  # [8, 400]
        # print(f'pet feature.shape:{pet_feature.shape}')
        result = self.fc(torch.cat((mri_feature, pet_feature), dim=1))
        return result

class EfficientNetMriPet(nn.Module):
    def __init__(self, num_classes=2, pretrained_path=None):
        super(EfficientNetMriPet, self).__init__()
        self.name = 'EfficientNet_mri_pet'
        self.MriExtraction = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 2},
                                                      in_channels=1)
        self.PetExtraction = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 2},
                                                      in_channels=1)
        pretrained_path = r'/data3/wangchangmiao/shenxy/PretrainedModel/kaggle_efficientNet3D/T1w-e6-loss0.691-auc0.581.pth'
        # pretrained_path = None

        if pretrained_path:
            state_dict = torch.load(pretrained_path, weights_only=True)['model_state_dict']
            # print(state_dict)
            keys = list(state_dict.keys())
            state_dict.pop(keys[-1])
            state_dict.pop(keys[-2])
            # model.load_state_dict(state_dict, strict=False)
            self.MriExtraction.load_state_dict(state_dict, strict=False)
            self.PetExtraction.load_state_dict(state_dict, strict=False)

        self.fc = nn.Sequential(
            nn.Linear(2 * 2, num_classes),
            # nn.ReLU(),
            # nn.Linear(256, 64),
            # nn.ReLU(),
            # nn.Linear(64, num_classes)
        )
    def forward(self, mri_pet):
        """
        Mri: [8, 1, 96, 128, 96]
        Pet: [8, 1, 96, 128, 96]
        """
        # 已知输入 mri_pet 为[8, 2, 96, 128, 96]，现在通过一下代码获得mri pet 为 torch.Size([8, 96, 128, 96])，但是应该为torch.Size([8, 1, 96, 128, 96])，请解决
        mri = mri_pet[:, 0:1, :, :, :]
        # print(f'mri shape: {mri.shape}')
        pet = mri_pet[:, 1:2, :, :, :]
        # print(f'pet shape: {pet.shape}')
        mri_feature = self.MriExtraction(mri)  # [8, 400]
        # print(f'mri feature shape: {mri_feature.shape}')
        pet_feature = self.PetExtraction(pet)  # [8, 400]
        # print(f'pet feature.shape:{pet_feature.shape}')
        result = self.fc(torch.cat((mri_feature, pet_feature), dim=1))
        return result

class MetaFormerMriPet(nn.Module):
    def __init__(self, num_classes=2, pretrained_path=None):
        super(MetaFormerMriPet, self).__init__()
        self.name = 'MetaFormer_mri_pet'
        self.MriExtraction = poolformerv2_3d_s12(num_classes=500)
        self.PetExtraction = poolformerv2_3d_s12(num_classes=500)
        self.fc = nn.Sequential(
            nn.Linear(2 * 500, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, mri_pet):
        """
        Mri: [8, 1, 96, 128, 96]
        Pet: [8, 1, 96, 128, 96]
        """
        # 已知输入 mri_pet 为[8, 2, 96, 128, 96]，现在通过一下代码获得mri pet 为 torch.Size([8, 96, 128, 96])，但是应该为torch.Size([8, 1, 96, 128, 96])，请解决
        mri = mri_pet[:, 0:1, :, :, :]
        # print(f'mri shape: {mri.shape}')
        pet = mri_pet[:, 1:2, :, :, :]
        # print(f'pet shape: {pet.shape}')
        mri_feature = self.MriExtraction(mri)  # [8, 400]
        # print(f'mri feature shape: {mri_feature.shape}')
        pet_feature = self.PetExtraction(pet)  # [8, 400]
        # print(f'pet feature.shape:{pet_feature.shape}')
        result = self.fc(torch.cat((mri_feature, pet_feature), dim=1))
        return result
