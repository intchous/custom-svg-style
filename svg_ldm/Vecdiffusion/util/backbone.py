import copy
import math
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from svg_ldm.Vecdiffusion.util.basic_transformerblock import BasicTransformerBlock
from diffusers.models.normalization import AdaLayerNormContinuous


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _gelu2(x):
    return x * F.sigmoid(1.702 * x)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "gelu2":
        return _gelu2
    else:
        raise RuntimeError(
            "activation should be relu/gelu/gelu2, not {}".format(activation)
        )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps: int, dim: int, rescale_steps: int = 4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x: Tensor):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AdaLayerNorm_diffusers(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`, *optional*): The size of the embeddings dictionary.
        output_dim (`int`, *optional*):
        norm_elementwise_affine (`bool`, defaults to `False):
        norm_eps (`bool`, defaults to `False`):
        chunk_dim (`int`, defaults to `0`):
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        output_dim: Optional[int] = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 1,
    ):
        super().__init__()

        self.chunk_dim = chunk_dim
        output_dim = output_dim or embedding_dim * 2

        if num_embeddings is not None:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(
            output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.emb is not None:
            temb = self.emb(timestep)

        temb = self.linear(self.silu(temb))

        if self.chunk_dim == 1:
            # This is a bit weird why we have the order of "shift, scale" here and "scale, shift" in the
            # other if-branch. This branch is specific to CogVideoX for now.
            shift, scale = temb.chunk(2, dim=1)
            shift = shift[:, None, :]
            scale = scale[:, None, :]

        else:
            scale, shift = temb.chunk(2, dim=0)

        x = self.norm(x) * (1 + scale) + shift

        return x


class _AdaNorm(nn.Module):
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(max_timestep, n_embd)
        elif "mlp" in emb_type:
            self.emb = nn.Sequential(
                Rearrange("b -> b 1"),
                nn.Linear(1, n_embd // 2),
                nn.ReLU(),
                nn.Linear(n_embd // 2, n_embd),
            )
        else:
            self.emb = nn.Embedding(max_timestep, n_embd)

        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)


class AdaLayerNorm(_AdaNorm):
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__(n_embd, max_timestep, emb_type)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x: Tensor, timestep: int):

        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)

        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift

        return x


class AdaInsNorm(_AdaNorm):
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__(n_embd, max_timestep, emb_type)
        self.instancenorm = nn.InstanceNorm1d(n_embd)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = (self.instancenorm(x.transpose(-1, -2)
                               ).transpose(-1, -2) * (1 + scale) + shift)
        return x


class TransformerEncoder(nn.Module):
    """
    Close to torch.nn.TransformerEncoder, but with timestep support for diffusion
    """

    __constants__ = ["norm"]

    def __init__(self, num_layers=4, max_len=25, dim_seq=10, dim_transformer=512, nhead=8, dim_feedforward=2048, diffusion_step=100, encoder_hid_dim=None, cross_attention_dim=None, use_norm_out=False, device='cuda'):
        super(TransformerEncoder, self).__init__()

        self.pos_encoder = SinusoidalPosEmb(
            num_steps=max_len, dim=dim_transformer).to(device)
        pos_i = torch.tensor([i for i in range(max_len)]).to(device)
        self.pos_embed = self.pos_encoder(pos_i)

        self.layer_in = nn.Linear(
            in_features=dim_seq, out_features=dim_transformer).to(device)

        if ((encoder_hid_dim is not None) and (cross_attention_dim is not None)):
            self.encoder_hid_proj = nn.Linear(
                encoder_hid_dim, cross_attention_dim).to(device)
        else:
            self.encoder_hid_proj = None

        timestep_input_dim = int(dim_transformer/4.0)
        time_embed_dim = dim_transformer

        self.time_proj = Timesteps(
            timestep_input_dim, True, downscale_freq_shift=0)

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim, time_embed_dim)

        attention_head_dim = dim_transformer // nhead
        encoder_layer = BasicTransformerBlock(
            dim=dim_transformer,
            num_attention_heads=nhead,
            attention_head_dim=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            # num_embeds_ada_norm=diffusion_step,
            num_embeds_ada_norm=None,
            # 'layer_norm', 'ada_norm', 'ada_norm_zero'
            norm_type="ada_norm",
            ff_inner_dim=dim_feedforward,
        )

        self.layers = _get_clones(encoder_layer, num_layers).to(device)
        self.num_layers = num_layers

        self.use_norm_out = use_norm_out
        if (use_norm_out):
            self.norm_out = AdaLayerNormContinuous(
                dim_transformer, time_embed_dim, elementwise_affine=False, eps=1e-6)

        self.layer_out = nn.Linear(
            in_features=dim_transformer, out_features=dim_seq).to(device)

    def forward(
        self,
        # src: Tensor,
        # timestep: Tensor = None,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Tensor:

        output = hidden_states

        output = self.layer_in(output)
        # output = F.softplus(output)
        output = output + self.pos_embed

        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)

        if (self.encoder_hid_proj is not None):
            encoder_hidden_states = self.encoder_hid_proj(
                encoder_hidden_states)

        for i, mod in enumerate(self.layers):

            output = mod(
                hidden_states=output,
                encoder_hidden_states=encoder_hidden_states,
                # timestep=timestep,
                temb=t_emb,
            )

            # if i < self.num_layers - 1:
            #     output = F.softplus(output)

        # final layer
        if (self.use_norm_out):
            output = self.norm_out(output, t_emb)

        output = self.layer_out(output)

        return output
