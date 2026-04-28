import torch.nn as nn
from einops.layers.torch import Rearrange
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange
from models.embed import DataEmbedding, TokenEmbedding
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
import torch.nn.functional as F
from functools import partial


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, inner_dim, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

class MixerBlock(nn.Module):
    def __init__(self, dim, patch_num, token_mixing_dim, channel_mixing_dim, dropout, chan_first, chan_last):
        super().__init__()
        self.mixer_i = nn.Sequential(
        PreNormResidual(dim, FeedForward(patch_num, token_mixing_dim, dropout, chan_first)),
        PreNormResidual(dim, FeedForward(dim, channel_mixing_dim, dropout, chan_last)))
    def forward(self,x):
        # token-mixing [B, D, #tokens]

        return self.mixer_i(x)

class GPT4TS(nn.Module):

    def __init__(self, configs):
        super(GPT4TS, self).__init__()

        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride +1
        if (configs.seq_len - self.patch_size) % self.stride != 0:
            self.patch_num += 1
            self.padding_patch_layer = nn.ReplicationPad1d((0, configs.stride))
        else:
            self.padding_patch_layer = nn.ReplicationPad1d((0, 0))
        
        
        dim = configs.hidden_dim
        self.embedding = TokenEmbedding(c_in=self.patch_size * configs.feature_num, d_model=dim)

        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        token_mixing_dim = self.patch_num * configs.token_mixing_factor
        channel_mixing_dim = dim * configs.channel_mixing_factor

        self.mlp_blocks = nn.ModuleList([
            MixerBlock(dim, self.patch_num, token_mixing_dim, channel_mixing_dim, 
            configs.dropout, chan_first, chan_last) for _ in
            range(configs.block_num)
        ])
        self.ln_proj1 = nn.LayerNorm(configs.d_model)
        self.map_ = nn.Linear(dim, configs.d_model)

        # loads a pretrained GPT-2 base model
        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True,
                                              output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        print("gpt2 = {}".format(self.gpt2))
        for param in self.gpt2.parameters():
            param.requires_grad = False

        # self.ln_proj2 = nn.LayerNorm(configs.d_model)
        self.act = F.gelu
        self.ln_proj3 = nn.LayerNorm(configs.d_model * self.patch_num)
        self.dropout = nn.Dropout(0.5)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pre_len)

    def forward(self, x_enc):
        B, L, M = x_enc.shape

        input_x = rearrange(x_enc, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')
        input_x = self.embedding(input_x)
        for block in self.mlp_blocks:
            input_x = block(input_x)
        map_x = self.map_(input_x)
        outputs = self.gpt2(inputs_embeds=self.ln_proj1(map_x)).last_hidden_state + map_x
        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.dropout(self.ln_proj3(outputs))
        outputs = self.out_layer(outputs)

        return outputs.squeeze()
