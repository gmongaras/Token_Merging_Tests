from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    FluxAttnProcessor2_0,
    FluxSingleAttnProcessor2_0,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from typing import List

from diffusers import FluxPipeline


from Merge import BipartiteSoftMatching, calculate_indices












# YiYi to-do: refactor rope related functions/classes
def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)

    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()


# YiYi to-do: refactor rope related functions/classes
class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)
    


pos_embed = EmbedND(dim=128*24, theta=10000, axes_dim=[16, 56, 56])


































class FluxTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)

        self.norm1_context = AdaLayerNormZero(dim)

        if hasattr(F, "scaled_dot_product_attention"):
            processor = FluxAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

        self.merge_attn = True
        self.merge_ff = True

        self.merge_per = 0.5

        self.merge_method = "Random"

        self.reencode_pos_enc = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):







        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}












        # Merging
        if self.merge_attn:
            text_size = norm_encoder_hidden_states.shape[1]
            # Calculate indices
            if not hasattr(self, "hidden_states_merge"):
                r = int(norm_hidden_states.shape[1]*self.merge_per)
                generator = torch.Generator()#.manual_seed(42)
                self.hidden_states_merge = BipartiteSoftMatching(norm_hidden_states, r, generator=generator, method=self.merge_method)

            # Merge the hidden states
            norm_hidden_states = self.hidden_states_merge.merge(norm_hidden_states, mode="mean")

            # Slice the rotary embeddings
            # The first part of the rotary embeddings is the text part and the last part is the second part
            # image_rotary_emb = torch.cat((image_rotary_emb[:, :, self.hidden_states_merge.unm_idx[0, :, 0]], ((image_rotary_emb[:, :, self.hidden_states_merge.dst_idx[0, :, 0]] + image_rotary_emb[:, :, self.hidden_states_merge.src_idx[0, :, 0]])/2), image_rotary_emb[:, :, -norm_encoder_hidden_states.shape[1]:]), dim=2)
            # image_rotary_emb = torch.cat((image_rotary_emb[:, :, :text_size], image_rotary_emb[:, :, text_size+self.hidden_states_merge.unm_idx[0, :, 0]], ((image_rotary_emb[:, :, text_size+self.hidden_states_merge.dst_idx[0, :, 0]] + image_rotary_emb[:, :, text_size+self.hidden_states_merge.src_idx[0, :, 0]])/2)), dim=2)




            # Reencode the positional embeddings
            if hasattr(self, "image_rotary_emb"):
                image_rotary_emb = self.image_rotary_emb
            else:
                if self.reencode_pos_enc:
                    latent_image_ids = FluxPipeline._prepare_latent_image_ids(1, 1024//8, 1024//8, "cuda", torch.bfloat16)
                    text_ids = torch.zeros(1, 512, 3).to(device=torch.device("cuda"), dtype=torch.bfloat16)
                    text_ids = text_ids.repeat(1, 1, 1)
                    # image_rotary_emb = pos_embed(torch.cat([text_ids, latent_image_ids], dim=1))
                    image_rotary_emb = pos_embed(torch.cat([
                        text_ids, 
                        latent_image_ids.gather(1, self.hidden_states_merge.unm_idx.repeat(1, 1, 3)),
                        (
                            (latent_image_ids.gather(1, self.hidden_states_merge.src_idx.repeat(1, 1, 3)) + latent_image_ids.gather(1, self.hidden_states_merge.dst_idx.repeat(1, 1, 3))) / 2
                        )
                        ], dim=1
                    ))
                    self.image_rotary_emb = image_rotary_emb
                else:
                    # Slice the rotary embeddings
                    # The first part of the rotary embeddings is the text part and the last part is the second part
                    # image_rotary_emb = torch.cat((image_rotary_emb[:, :, self.hidden_states_merge.unm_idx[0, :, 0]], ((image_rotary_emb[:, :, self.hidden_states_merge.dst_idx[0, :, 0]] + image_rotary_emb[:, :, self.hidden_states_merge.src_idx[0, :, 0]])/2), image_rotary_emb[:, :, -norm_encoder_hidden_states.shape[1]:]), dim=2)
                    if self.merge_method == "Random":
                        # if not hasattr(self, "image_rotary_emb"):
                        self.image_rotary_emb = torch.cat(
                            (
                                image_rotary_emb[:, :, :text_size], 
                                image_rotary_emb[:, :, text_size+self.hidden_states_merge.unm_idx[0, :, 0]], 
                                ((image_rotary_emb[:, :, text_size+self.hidden_states_merge.dst_idx[0, :, 0]] + image_rotary_emb[:, :, text_size+self.hidden_states_merge.src_idx[0, :, 0]])/2)
                            ), dim=2)
                        image_rotary_emb = self.image_rotary_emb
                            
                    elif self.merge_method == "BipartiteSoftMatching":
                        image_rotary_emb_txt = image_rotary_emb[:, :, :text_size]
                        image_rotary_emb = image_rotary_emb[:, :, text_size:]

                        def split(x):
                            B, N = x.shape[0], x.shape[2]
                            C1, C2, C3 = x.shape[3], x.shape[4], x.shape[5]
                            src = x[:, :, self.hidden_states_merge.a_idx.squeeze()]
                            dst = x[:, :, self.hidden_states_merge.b_idx.squeeze()]
                            return src, dst
                        src, dst = split(image_rotary_emb)
                        
                        unm = src[:, :, self.hidden_states_merge.unm_idx.squeeze()]
                        src = src[:, :, self.hidden_states_merge.src_idx.squeeze()]
                        C1, C2, C3 = src.shape[3], src.shape[4], src.shape[5]
                        dst = dst.scatter_reduce(2, self.hidden_states_merge.dst_idx[:, None, :, :, None, None].expand(1, 1, self.hidden_states_merge.r, C1, C2, C3), src, reduce="mean")

                        image_rotary_emb = torch.cat(
                            (
                                image_rotary_emb_txt, 
                                unm,
                                dst,
                            ), dim=2)
                        
                        self.image_rotary_emb = image_rotary_emb
                    else:
                        raise ValueError("Invalid merge method")




        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # Unmerge
        if self.merge_attn:
            attn_output = self.hidden_states_merge.unmerge(attn_output)

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output














        if self.merge_ff:
            # Calculate indices
            if not hasattr(self, "hidden_states_merge_ff"):
                r = int(hidden_states.shape[1]*self.merge_per)
                generator = torch.Generator()
                self.hidden_states_merge_ff = BipartiteSoftMatching(hidden_states, r, generator=generator, method=self.merge_method)

            # Merge the hidden states
            hidden_states = self.hidden_states_merge_ff.merge(hidden_states, mode="mean")

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Unmerge
        if self.merge_ff:
            hidden_states = self.hidden_states_merge_ff.unmerge(hidden_states)

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states
    









































class FluxSingleTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        processor = FluxSingleAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

        self.merge_attn = True
        self.merge_ff = True

        self.merge_per = 0.5

        self.merge_method = "Random"

        self.reencode_pos_enc = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}


















        # Merging
        text_size = 512
        if self.merge_attn:
            # Calculate indices
            if not hasattr(self, "hidden_states_merge"):
                generator = torch.Generator()#.manual_seed(42)
                r = int((norm_hidden_states.shape[1]-text_size)*self.merge_per)
                self.hidden_states_merge = BipartiteSoftMatching(norm_hidden_states[:, text_size:], r, generator=generator, method=self.merge_method)

            # Merge the hidden states
            norm_hidden_states = torch.cat(
                (
                    norm_hidden_states[:, :text_size],
                    self.hidden_states_merge.merge(norm_hidden_states[:, text_size:], mode="mean")
                ), dim=1
            )


            # Reencode the positional embeddings
            if hasattr(self, "image_rotary_emb"):
                image_rotary_emb = self.image_rotary_emb
            else:
                if self.reencode_pos_enc:
                    latent_image_ids = FluxPipeline._prepare_latent_image_ids(1, 1024//8, 1024//8, "cuda", torch.bfloat16)
                    text_ids = torch.zeros(1, 512, 3).to(device=torch.device("cuda"), dtype=torch.bfloat16)
                    text_ids = text_ids.repeat(1, 1, 1)
                    image_rotary_emb = pos_embed(torch.cat([
                        text_ids, 
                        latent_image_ids.gather(1, self.hidden_states_merge.unm_idx.repeat(1, 1, 3)),
                        (
                            (latent_image_ids.gather(1, self.hidden_states_merge.src_idx.repeat(1, 1, 3)) + latent_image_ids.gather(1, self.hidden_states_merge.dst_idx.repeat(1, 1, 3))) / 2
                        )
                        ], dim=1
                    ))
                    self.image_rotary_emb = image_rotary_emb
                else:
                    # Slice the rotary embeddings
                    # The first part of the rotary embeddings is the text part and the last part is the second part
                    # image_rotary_emb = torch.cat((image_rotary_emb[:, :, self.hidden_states_merge.unm_idx[0, :, 0]], ((image_rotary_emb[:, :, self.hidden_states_merge.dst_idx[0, :, 0]] + image_rotary_emb[:, :, self.hidden_states_merge.src_idx[0, :, 0]])/2), image_rotary_emb[:, :, -norm_encoder_hidden_states.shape[1]:]), dim=2)
                    if self.merge_method == "Random":
                        # if not hasattr(self, "image_rotary_emb"):
                        self.image_rotary_emb = torch.cat(
                            (
                                image_rotary_emb[:, :, :text_size], 
                                image_rotary_emb[:, :, text_size+self.hidden_states_merge.unm_idx[0, :, 0]], 
                                ((image_rotary_emb[:, :, text_size+self.hidden_states_merge.dst_idx[0, :, 0]] + image_rotary_emb[:, :, text_size+self.hidden_states_merge.src_idx[0, :, 0]])/2)
                            ), dim=2)
                        image_rotary_emb = self.image_rotary_emb
                            
                    elif self.merge_method == "BipartiteSoftMatching":
                        image_rotary_emb_txt = image_rotary_emb[:, :, :text_size]
                        image_rotary_emb = image_rotary_emb[:, :, text_size:]

                        def split(x):
                            B, N = x.shape[0], x.shape[2]
                            C1, C2, C3 = x.shape[3], x.shape[4], x.shape[5]
                            src = x[:, :, self.hidden_states_merge.a_idx.squeeze()]
                            dst = x[:, :, self.hidden_states_merge.b_idx.squeeze()]
                            return src, dst
                        src, dst = split(image_rotary_emb)
                        
                        unm = src[:, :, self.hidden_states_merge.unm_idx.squeeze()]
                        src = src[:, :, self.hidden_states_merge.src_idx.squeeze()]
                        C1, C2, C3 = src.shape[3], src.shape[4], src.shape[5]
                        dst = dst.scatter_reduce(2, self.hidden_states_merge.dst_idx[:, None, :, :, None, None].expand(1, 1, self.hidden_states_merge.r, C1, C2, C3), src, reduce="mean")

                        image_rotary_emb = torch.cat(
                            (
                                image_rotary_emb_txt, 
                                unm,
                                dst,
                            ), dim=2)
                        self.image_rotary_emb = image_rotary_emb
                    else:
                        raise ValueError("Invalid merge method")
                    



        # Attn
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # Unmerge
        if self.merge_attn:
            attn_output = torch.cat(
                (
                    attn_output[:, :text_size],
                    self.hidden_states_merge.unmerge(attn_output[:, text_size:])
                ), dim=1
            )




















        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)


        if self.merge_ff:
            # Calculate indices
            if not hasattr(self, "hidden_states_merge_ff"):
                generator = torch.Generator()#.manual_seed(42)
                r = int((hidden_states.shape[1]-text_size)*self.merge_per)
                self.hidden_states_merge_ff = BipartiteSoftMatching(hidden_states[:, text_size:], r, generator=generator, method=self.merge_method)

            # Merge the hidden states
            hidden_states = torch.cat(
                (
                    hidden_states[:, :text_size],
                    self.hidden_states_merge_ff.merge(hidden_states[:, text_size:], mode="mean")
                ), dim=1
            )

            # Merge the residual
            residual = torch.cat(
                (
                    residual[:, :text_size],
                    self.hidden_states_merge_ff.merge(residual[:, text_size:], mode="mean")
                ), dim=1
            )

        
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        if self.merge_ff:
            hidden_states = torch.cat(
                (
                    hidden_states[:, :text_size],
                    self.hidden_states_merge_ff.unmerge(hidden_states[:, text_size:])
                ), dim=1
            )

        return hidden_states
