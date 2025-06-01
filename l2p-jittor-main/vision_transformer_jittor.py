import jittor as jt
from jittor import nn
import math
from collections import OrderedDict
from prompt_jittor import Prompt_jittor
from functools import partial

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    Jittor implementation
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def execute(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Jittor implementation modified from timm
    """
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        if self.drop_prob == 0. or not self.is_training():
            return x
        keep_prob = 1 - self.drop_prob
        mask = jt.random(x.shape[:1] + (1,)*(x.ndim-1), dtype=x.dtype)
        mask = (mask < keep_prob).stop_grad()
        return x * mask / keep_prob

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def execute(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Jittor doesn't support unbind

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = nn.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = jt.array(init_values * jt.ones(dim))

    def execute(self, x):
        return x.multiply(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def execute(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

# 注意：Mlp类和PatchEmbed类需要定义
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    Jittor implementation
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class VisionTransformer_jittor(nn.Module):
    """ Vision Transformer
    A Jittor impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
            class_token=True, no_embed_class=False, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block,
            prompt_length=None, embedding_key='cls', prompt_init='uniform', prompt_pool=False, prompt_key=False, pool_size=None,
            top_k=None, batchwise_prompt=False, prompt_key_init='uniform', head_type='token', use_prompt_mask=False,):
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.img_size = img_size
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.class_token = class_token
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = jt.zeros(1, 1, embed_dim) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        if prompt_length is not None and pool_size is not None and prompt_pool:
            embed_len += prompt_length * top_k
        self.pos_embed = jt.randn(1, embed_len, embed_dim) * .02
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.prompt_pool = prompt_pool
        self.head_type = head_type
        self.use_prompt_mask = use_prompt_mask
        
        if prompt_length is not None and pool_size is not None and prompt_pool: 
            self.prompt = Prompt_jittor(length=prompt_length, embed_dim=embed_dim, embedding_key=embedding_key, 
                               prompt_init=prompt_init, prompt_pool=prompt_pool, prompt_key=prompt_key, 
                               pool_size=pool_size, top_k=top_k, batchwise_prompt=batchwise_prompt,
                               prompt_key_init=prompt_key_init)
        # Stochastic depth decay rule
        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if weight_init != 'skip':
            self.init_weights(weight_init)
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        
        # 使用Jittor版的trunc_normal_
        self.pos_embed = trunc_normal_(self.pos_embed, std=.02)
        
        if self.cls_token is not None:
            # Jittor版的正态分布初始化
            jt.init.gauss_(self.cls_token, 0, 1e-6)
        
        # 自定义的named_apply替代
        self._named_apply(self._get_init_weights_vit(mode, head_bias))
        
    def _named_apply(self, fn):
        for name, module in self.named_modules():
            fn(module)
            
    def _get_init_weights_vit(self, mode, head_bias):
        def init_fn(m):
            if isinstance(m, nn.Linear):
                if mode.startswith('jax'):
                    lecun_normal_(m.weight)
                else:
                    trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    if 'nlhb' in mode:
                        m.bias.set_data(head_bias * jt.ones_like(m.bias))
                    else:
                        jt.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                if mode.startswith('jax'):
                    lecun_normal_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
        return init_fn
    
    @jt.no_grad()
    def load_pretrained(self, checkpoint_path, prefix=''):     
        # 暂留
        
        # 这里需要实现自定义的权重加载逻辑
        pass
    
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}
    
    def get_classifier(self):
        return self.head
    
    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
    def execute_features(self, x, task_id=-1, cls_features=None, train=False):
        x = self.patch_embed(x)
        if hasattr(self, 'prompt'):
            if self.use_prompt_mask and train:
                start = task_id * self.prompt.top_k
                end = (task_id + 1) * self.prompt.top_k
                single_prompt_mask = jt.arange(start, end).to(x.device)
                prompt_mask = single_prompt_mask.unsqueeze(0).expand(x.shape[0], -1)
                if end > self.prompt.pool_size:
                    prompt_mask = None
            else:
                prompt_mask = None
            res = self.prompt(x, prompt_mask=prompt_mask, cls_features=cls_features)
            self.total_prompt_len = res['total_prompt_len']
            x = res['prompted_embedding']
        else:
            res = dict()
        if self.cls_token is not None:
            x = jt.concat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        res['x'] = x
        return res
    def execute_head(self, res, pre_logits=False):
        x = res['x']
        if self.class_token and self.head_type == 'token':
            x = x[:, 0]
        elif self.head_type == 'gap' and self.global_pool == 'avg':
            x = x.mean(dim=1)
        elif self.head_type == 'prompt' and self.prompt_pool:
            x = x[:, 1:(1 + self.total_prompt_len)] if self.class_token else x[:, 0:self.total_prompt_len]
            x = x.mean(dim=1)
        elif self.head_type == 'token+prompt' and self.prompt_pool and self.class_token:
            x = x[:, 0:self.total_prompt_len + 1]
            x = x.mean(dim=1)
        else:
            raise ValueError(f'Invalid classifier={self.classifier}')
        
        res['pre_logits'] = x
        x = self.fc_norm(x)
        res['logits'] = self.head(x)
        return res
    def execute(self, x, task_id=-1, cls_features=None, train=False):
        res = self.execute_features(x, task_id=task_id, cls_features=cls_features, train=train)
        res = self.execute_head(res)
        return res
# 辅助函数
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Jittor版本的截断正态分布初始化"""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with jt.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(l, u)
        tensor = tensor.erfinv()
        tensor = tensor.multiply(std * math.sqrt(2.))
        tensor = tensor.add(mean)
        tensor = tensor.clamp(min=a, max=b)
        return tensor
def lecun_normal_(tensor):
    """Jittor版LeCun正态初始化"""
    fan_in = tensor.shape[1] * tensor[0][0].numel() if tensor.ndim > 2 else tensor.shape[1]
    std = math.sqrt(1. / fan_in)
    return trunc_normal_(tensor, std=std)