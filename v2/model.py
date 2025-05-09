import torch
import torch.nn as nn
try:
    from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
except ImportError:
    from spikingjelly.activation_based.neuron import ParametricLIFNode as MultiStepParametricLIFNode, LIFNode as MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from functools import partial
from timm.models import create_model
# 从 SpikingResformer 中引入稀疏矩阵乘、分支前馈和 SSDP 模块
from spikingresformer import SpikingMatmul, GWFFN, SSDPModule

__all__ = ['QKFormer', 'QKResformer']

class Token_QK_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Q、K 一维卷积 + BN + LIF
        self.q_conv = nn.Conv1d(dim, dim, 1, bias=False)
        self.q_bn   = nn.BatchNorm1d(dim)
        self.q_lif  = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.k_conv = nn.Conv1d(dim, dim, 1, bias=False)
        self.k_bn   = nn.BatchNorm1d(dim)
        self.k_lif  = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        # 门控 LIF
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        # 输出投影
        self.proj_conv = nn.Conv1d(dim, dim, 1)
        self.proj_bn   = nn.BatchNorm1d(dim)
        self.proj_lif  = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        # —— 新增：动态脉冲率归一 buffer 和动量
        self.register_buffer('fr_q',    torch.zeros(1,1,self.num_heads,1,1))
        self.register_buffer('fr_attn', torch.zeros(1,1,self.num_heads,1,1))
        self.momentum = 0.999

    def forward(self, x):
        # x: [T, B, C, H, W]
        T, B, C, H, W = x.shape

        # Tokenize: flatten H×W → N
        x = x.flatten(3)              # [T, B, C, N]
        N = x.size(-1)
        x_flat = x.flatten(0,1)       # [T*B, C, N]

        # Q path
        q = self.q_conv(x_flat)       # [T*B, C, N]
        q = self.q_bn(q).view(T, B, C, N)
        q = self.q_lif(q)             # [T, B, C, N]
        q = q.view(T, B, self.num_heads, self.head_dim, N)  # [T,B,heads,hd,N]

        # K path
        k = self.k_conv(x_flat)
        k = self.k_bn(k).view(T, B, C, N)
        k = self.k_lif(k)
        k = k.view(T, B, self.num_heads, self.head_dim, N)  # [T,B,heads,hd,N]

        # —— 多头保留：用 mean 而非 sum 做门控
        q_gate = q.mean(dim=3, keepdim=True)  # [T,B,heads,1,N]

        # —— 1) 更新 q 脉冲率，动态归一
        fr_q_batch = q_gate.detach().mean((0,1,3,4), keepdim=True)  # [1,1,heads,1,1]
        self.fr_q = self.fr_q * self.momentum + fr_q_batch * (1-self.momentum)
        scale = 1.0 / torch.sqrt(self.fr_q * self.head_dim + 1e-6)  # 防止除零

        # —— 2) 门控 + LIF
        attn = self.attn_lif(q_gate * scale)  # [T,B,heads,1,N]

        # —— 更新 attn 脉冲率
        fr_attn_batch = attn.detach().mean((0,1,3,4), keepdim=True)
        self.fr_attn = self.fr_attn * self.momentum + fr_attn_batch * (1-self.momentum)

        # --- 前面计算 q, k, attn 后得到 ---
        # q  -> [T,B,heads,head_dim,N]
        # k  -> [T,B,heads,head_dim,N]
        # attn-> [T,B,heads,1,N]
        # ------------------------------------------------
        # 1) 计算加权输出
        x_out = attn * k  # [T, B, heads, head_dim, N]

        # 2) 合并 heads 和 head_dim 维度 => 通道 C
        #    flatten dims 2 (heads) 和 3 (head_dim)
        x_out = x_out.flatten(2, 3)  # [T, B, C, N]

        # 3) 合并 时间 T 和 批次 B 维度 => batch
        x_out = x_out.flatten(0, 1)  # [T*B, C, N]

        # 4) 时序卷积投影
        x_out = self.proj_conv(x_out)  # [T*B, C, N]
        x_out = self.proj_bn(x_out)  # [T*B, C, N]
        x_out = self.proj_lif(x_out)  # [T*B, C, N]

        # 5) 恢复到 [T, B, C, N]
        x_out = x_out.view(T, B, C, -1)  # N = H*W

        # 6) 最终 reshape 回空间维度
        x_out = x_out.view(T, B, C, H, W)  # [T, B, C, H, W]

        return x_out


class Spiking_Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.qkv_mp = nn.MaxPool1d(4)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2,-1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0,1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T,B,C,W,H)

        return x

# 用 GWFFN 替换原 MLP
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        hidden = hidden_features or in_features * 4
        ratio = hidden // in_features
        # GWFFN 内部含多分支卷积 + 残差
        self.ffn = GWFFN(in_features, num_conv=2, ratio=ratio, activation=MultiStepLIFNode)

    def forward(self, x):
        # x: [T,B,C,H,W]
        return self.ffn(x)


class TokenSpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.tssa = Token_QK_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        # 残差连接
        x = x + self.tssa(x)
        x = x + self.mlp(x)
        return x


class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.ssa = Spiking_Self_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.ssa(x)
        x = x + self.mlp(x)
        return x



class PatchEmbedInit(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj1_conv = nn.Conv2d(embed_dims // 2, embed_dims // 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims // 1)
        self.proj1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims //1, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')



    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W)
        x = self.proj_lif(x).flatten(0, 1)

        x_feat = x
        x = self.proj1_conv(x)
        x = self.proj1_bn(x).reshape(T, B, -1, H, W)
        x = self.proj1_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H, W).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat # shortcut

        return x


class PatchEmbeddingStage(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj3_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.proj3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj4_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, C, H, W = x.shape
        # Downsampling + Res

        x = x.flatten(0, 1).contiguous()
        x_feat = x

        x = self.proj3_conv(x)
        x = self.proj3_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj3_lif(x).flatten(0, 1).contiguous()

        x = self.proj4_conv(x)
        x = self.proj4_bn(x)
        x = self.proj4_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H//2, W//2).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat # shortcut

        return x


class spiking_transformer(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=3,
        num_classes=100,
        embed_dims=(64, 128, 256),
        num_heads=(8, 8, 8),
        mlp_ratios=(4, 4, 4),
        depths=(6, 8, 6),
        sr_ratios=(8, 4, 2),
        T=4,
        device='cuda',
        **kwargs
    ):
        super().__init__()
        # 兼容 depths、num_heads、mlp_ratios、sr_ratios 为 int 或列表
        if isinstance(depths, int):       depth_list = [depths]*3
        else:                              depth_list = list(depths)
        if isinstance(num_heads, int):     heads_list = [num_heads]*3
        else:                              heads_list = list(num_heads)
        if isinstance(mlp_ratios, (int,float)): mlp_list = [mlp_ratios]*3
        else:                              mlp_list = list(mlp_ratios)

        # 兼容 embed_dims：最后一个维度为 embed_dim
        if isinstance(embed_dims, int):
            e1 = embed_dims//4; e2 = embed_dims//2; e3 = embed_dims
            embed_dim = embed_dims
        else:
            e1, e2, e3 = embed_dims
            embed_dim = e3

        self.num_classes = num_classes
        self.depths = depth_list
        self.T = T

        # Stage 1
        self.patch_embed1 = PatchEmbedInit(img_size_h, img_size_w, patch_size, in_channels, e1)
        self.stage1 = nn.ModuleList([
            TokenSpikingTransformer(dim=e1, num_heads=heads_list[0], mlp_ratio=mlp_list[0])
            for _ in range(depth_list[0])
        ])
        # Stage 2
        self.patch_embed2 = PatchEmbeddingStage(img_size_h, img_size_w, patch_size, in_channels, e2)
        self.stage2 = nn.ModuleList([
            TokenSpikingTransformer(dim=e2, num_heads=heads_list[1], mlp_ratio=mlp_list[1])
            for _ in range(depth_list[1])
        ])
        # Stage 3
        self.patch_embed3 = PatchEmbeddingStage(img_size_h, img_size_w, patch_size, in_channels, e3)
        self.stage3 = nn.ModuleList([
            SpikingTransformer(dim=e3, num_heads=heads_list[2], mlp_ratio=mlp_list[2])
            for _ in range(depth_list[2])
        ])

        # Classification head + SSDP
        self.head = nn.Linear(embed_dim, num_classes) if num_classes>0 else nn.Identity()
        self.ssdp = SSDPModule(input_dim=embed_dim, output_dim=num_classes, device=device)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # x: [T, B, C_in, H, W]
        x = self.patch_embed1(x)
        for blk in self.stage1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.stage2:
            x = blk(x)
        x = self.patch_embed3(x)
        for blk in self.stage3:
            x = blk(x)
        # flatten 空间后平均
        return x.flatten(3).mean(3)  # [T, B, C_embed]

    def forward(self, x):
        # x: [B, C_in, H, W] → [T, B, C_in, H, W]
        if x.dim()==4:
            x = x.unsqueeze(0).repeat(self.T,1,1,1,1)

        feats = self.forward_features(x)      # [T, B, C_embed]
        pre_spikes = (feats>0).float()        # [T, B, C_embed]

        # 每步分类
        outputs = [ self.head(feats[t]) for t in range(self.T) ]  # list of [B, num_classes]
        outputs = torch.stack(outputs, dim=0)                     # [T, B, num_classes]
        post_spikes = (outputs>0).float()                         # [T, B, num_classes]

        if self.training:
            # 1) 首发时刻
            t_pre  = pre_spikes.float().cumsum(0).argmax(0).float()   # [B, C_embed]
            t_post = post_spikes.float().cumsum(0).argmax(0).float()  # [B, num_classes]
            delta_t = (t_post[:,:,None] - t_pre[:,None,:])           # [B, num_classes, C_embed]

            # 2) 以 batch×channel 形式传入 SSDP
            #    这里用 spike 总数/频率都行，sum 更直观
            pre_spike  = pre_spikes.sum(dim=0)      # [B, C_embed]
            post_spike = post_spikes.sum(dim=0)     # [B, num_classes]

            # 3) 计算并应用 Δw
            delta_w = self.ssdp(pre_spike, post_spike, delta_t)  # [num_classes, C_embed]
            with torch.no_grad():
                self.head.weight += delta_w

        # 平均输出
        return outputs.mean(0)  # [B, num_classes]




@register_model
def QKFormer(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('cache_dir', None)
    model = spiking_transformer(
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def QKResformer(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('cache_dir', None)
    """
    基于 QKFormer + SpikingResformer DSSA 优点的混合模型
    """
    model = spiking_transformer(**kwargs)
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    input = torch.randn(2, 3, 32, 32).cuda()
    model = create_model(
        'QKFormer',
        pretrained=False,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size_h=32, img_size_w=32,
        patch_size=4, embed_dims=384, num_heads=8, mlp_ratios=4,
        in_channels=3, num_classes=100, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
        T=4,
    ).cuda()

    from torchinfo import summary
    summary(model, input_size=(2, 3, 32, 32))