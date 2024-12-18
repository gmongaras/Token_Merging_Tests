import torch
from diffusers import StableDiffusion3Pipeline








notes = """
Merge key and value
1 percent of tokens
mean merging
on key
"""















import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional
from diffusers.models.attention import Attention


class CustomProcessor:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        # attention
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)





        import math
        class BipartiteSoftMatching:
            def __init__(
                self, 
                metric: torch.Tensor, 
                r: int, 
                class_token: bool = False, 
                distill_token: bool = False
            ):
                """
                Applies ToMe with a balanced matching set (50%, 50%).

                Input size is [batch, tokens, channels].
                r indicates the number of tokens to remove (max 50% of tokens).

                Extra args:
                - class_token: Whether or not there's a class token.
                - distill_token: Whether or not there's also a distillation token.

                When enabled, the class token and distillation tokens won't get merged.
                """
                self.metric = metric
                self.class_token = class_token
                self.distill_token = distill_token
                self.protected = 0
                if class_token:
                    self.protected += 1
                if distill_token:
                    self.protected += 1

                # We can only reduce by a maximum of 50% tokens
                t = metric.shape[1]
                self.r = min(r, (t - self.protected) // 2)

                if self.r <= 0:
                    return

                with torch.no_grad():
                    # Normalize for scores between -1 and 1
                    metric = metric / metric.norm(dim=-1, keepdim=True)

                    # Compute the similarity matrix
                    a, b = metric[..., ::2, :], metric[..., 1::2, :]
                    scores = a @ b.transpose(-1, -2)

                    if self.class_token:
                        scores[..., 0, :] = -math.inf
                    if self.distill_token:
                        scores[..., :, 0] = -math.inf

                    # For each node in a, get the best match in b
                    node_max, node_idx = scores.max(dim=-1)
                    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

                    self.unm_idx = edge_idx[..., self.r:, :]  # Unmerged Tokens
                    self.src_idx = edge_idx[..., :self.r, :]  # Merged Tokens
                    self.dst_idx = node_idx[..., None].gather(dim=-2, index=self.src_idx)

                    if self.class_token:
                        # Sort to ensure the class token is at the start
                        self.unm_idx = self.unm_idx.sort(dim=1)[0]

            def merge(self, x: torch.Tensor, mode="mean") -> torch.Tensor:
                """
                Merge tokens based on precomputed indices.
                """
                src, dst = x[..., ::2, :], x[..., 1::2, :]
                n, t1, c = src.shape
                unm = src.gather(dim=-2, index=self.unm_idx.expand(n, t1 - self.r, c))
                src = src.gather(dim=-2, index=self.src_idx.expand(n, self.r, c))

                if mode == "MLERP":
                    # MLERP reduce
                    dst_ = dst.scatter_reduce(-2, self.dst_idx.expand(n, self.r, c), src, reduce="mean")
                    n = dst.norm(2, dim=-1, keepdim=True).scatter_reduce(-2, self.dst_idx.expand(n, self.r, 1), src.norm(2, dim=-1, keepdim=True), reduce="max")
                    dst = (dst_ / dst_.norm(2, dim=-1, keepdim=True)) * n
                else:
                    # Mean reduce
                    dst = dst.scatter_reduce(-2, self.dst_idx.expand(n, self.r, c), src, reduce=mode)

                if self.distill_token:
                    return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
                else:
                    return torch.cat([unm, dst], dim=1)

            def unmerge(self, x: torch.Tensor) -> torch.Tensor:
                """
                Unmerge tokens back to their original form based on precomputed indices.
                """
                unm_len = self.unm_idx.shape[1]
                unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
                n, _, c = unm.shape

                src = dst.gather(dim=-2, index=self.dst_idx.expand(n, self.r, c))

                out = torch.zeros(n, self.metric.shape[1], c, device=x.device, dtype=x.dtype)

                out[..., 1::2, :] = dst
                out.scatter_(dim=-2, index=(2 * self.unm_idx).expand(n, unm_len, c), src=unm)
                out.scatter_(dim=-2, index=(2 * self.src_idx).expand(n, self.r, c), src=src)

                return out





        ### ToME
        # Merge tokens
        numMerge = key.shape[1]//100
        key_merge = BipartiteSoftMatching(key, numMerge)
        key = key_merge.merge(key, mode="mean")
        value = key_merge.merge(value, mode="mean")
        # query_merge = BipartiteSoftMatching(query, 100)#query.shape[1]//8)
        # query = query_merge.merge(query, mode="mean")
        





        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)


        """ Merging on heads
        # Merge heads into batch dimension
        B, H, T, D = key.shape
        key = key.reshape(B * H, T, D)
        value = value.reshape(B * H, T, D)
        # Merge tokens
        num = 100
        key_merge = BipartiteSoftMatching(key, num)
        key = key_merge.merge(key, mode="mean")
        value = key_merge.merge(value, mode="sum")
        # Unmerge batch dimension
        key = key.reshape(B, H, T-num, D)
        value = value.reshape(B, H, T-num, D)
        """



        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Unmerge tokens
        # hidden_states = query_merge.unmerge(hidden_states)

        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states
















def get_model():
    # Read token from .env file
    with open(".env", "r") as f:
        token = f.read().strip()

    # Load model
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, cache_dir="./cache", token=token)
    pipe = pipe.to("cuda")

    old_processor = pipe.transformer.transformer_blocks[0].attn.processor.__class__

    # Get stable diffusion 3 config
    config = pipe.transformer.config

    # Replace attention in stable diffusion with the imported one
    num_skip = 0
    for i, layer in enumerate(pipe.transformer.transformer_blocks):
        if i < num_skip:
            layer.attn.set_processor(old_processor())
        elif i >= num_skip:
            layer.attn.set_processor(CustomProcessor())

    return pipe