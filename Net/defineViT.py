from typing import Union
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
import torch
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        mlp_dim = 2048
        for _ in range(depth):
            #print (dim, mlp_dim)
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size:Union[tuple, list], patch_size, num_classes, dim, depth, heads,
                 mlp_dim, channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width, image_depth = image_size
        assert image_height % patch_size == 0, 'height dimensions must be divisible by the patch size'
        assert image_width % patch_size == 0, 'width dimensions must be divisible by the patch size'
        assert image_depth % patch_size == 0, 'depth must be divisible by the patch size'
        num_patches = (image_height // patch_size) * (image_width // patch_size) * (image_depth // patch_size)
        patch_dim = channels * patch_size ** 3
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        #print (mlp_dim)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        #print (dim)
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
            nn.Dropout(dropout)
        )

    def forward(self, img, mask = None):
        p = self.patch_size
        # print (img.shape) # torch.Size([2, 1, 128, 128, 128])
        x = rearrange(img, 'b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1 = p, p2 = p, p3 = p)
        # print ("x_rearrange", x.shape)# x_rearrange torch.Size([2, 512, 4096])
        x = self.patch_to_embedding(x)
        # print ("x_patch_to_embedding", x.shape)# x_patch_to_embedding torch.Size([2, 512, 512])
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        # print ("cls_tokens", cls_tokens.shape)# cls_tokens torch.Size([2, 1, 512])
        x = torch.cat((cls_tokens, x), dim=1)
        # print (x.shape)# torch.Size([2, 513, 512])
        # print (self.pos_embedding.shape)# torch.Size([1, 129, 512])
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)
# test the model
if __name__ == '__main__':
    # model = Model(
    #     image_size = 32,
    #     patch_size = 4,
    #     num_classes = 2,
    #     dim = 512,
    #     depth = 6,
    #     heads = 8,
    #     mlp_dim = 512,
    #     dropout = 0.1,
    #     emb_dropout = 0.1
    # )
    # model = Model(
    #     image_size = 128,
    #     patch_size = 32,
    #     num_classes = 2,
    #     dim = 1024,
    #     depth = 2,
    #     heads = 16,
    #     mlp_dim = 2048,
    #     channels = 1,
    #     dropout = 0.1,
    #     emb_dropout = 0.1
    # )
    # 创建模型
    model = ViT(
        image_size=[96, 128, 96],
        patch_size=16,
        num_classes=2,
        dim=128,
        depth=2,
        heads=16,
        mlp_dim=512,
        channels=1,
        dropout=0.1,
        emb_dropout=0.1
    )
    print (model)
    print("model params", sum(p.numel() for p in model.parameters()))
    x = torch.randn(2, 1, 96, 128, 96)
    y = model(x)
    print ('y', y.shape)