import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional
from diffusers.models.attention import Attention

from Merge import BipartiteSoftMatching, calculate_indices




def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

class CustomProcessor:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
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

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            # YiYi to-do: update uising apply_rotary_emb
            # from ..embeddings import apply_rotary_emb
            # query = apply_rotary_emb(query, image_rotary_emb)
            # key = apply_rotary_emb(key, image_rotary_emb)
            query, key = apply_rope(query, key, image_rotary_emb)








        # Function to plot self-similarity matrix
        diag = torch.eye(query.shape[1], device=query.device).bool().unsqueeze(0).repeat(2, 1, 1)
        def plot_similarity_matrix(x):
            import matplotlib.pyplot as plt
            x = torch.nn.functional.normalize(x, dim=-1)
            similarity_matrix = x @ x.mT
            similarity_matrix = similarity_matrix.clamp(-1, 1)
            # Mask the diagonal
            similarity_matrix[diag] = 0
            # Plot distribution of similarity scores
            vals = similarity_matrix.max(-1)[0]
            vals = vals[0].cpu().flatten().numpy()
            return vals
        # Turn 1-D indices into 2-D indices
        def unflatten_indices(indices, n):
            row = indices // n
            col = indices % n
            return torch.stack((row, col), dim=0)
        def save_dist(metric, list_):
            metric_merge = BipartiteSoftMatching(metric, 1000, method="BipartiteSoftMatching")
            src = unflatten_indices(metric_merge.src_idx, (metric.shape[1]**0.5))
            src[0] = src[0]*2 # The src is every even index along the row
            dst = unflatten_indices(metric_merge.dst_idx, (metric.shape[1]**0.5))
            dst[0] = dst[0]*2+1 # The dst is every odd index along the row
            list_.append((((src - dst)**2).sum(0)**0.5).cpu().numpy().flatten())
        # Image distances
        # Look at 2d distances rather than 1D - Change to box plots
        save_dist(key, self.key_img_dist)
        save_dist(query, self.query_img_dist)
        save_dist(value, self.value_img_dist)
        save_dist(hidden_states, self.x_img_dist)

        # Similarity scores
        self.query_sim.append(plot_similarity_matrix(query))
        self.key_sim.append(plot_similarity_matrix(key))
        self.value_sim.append(plot_similarity_matrix(value))
        self.x_sim.append(plot_similarity_matrix(hidden_states))











        if self.merge:
            merge_method = "Random"
            min_idx =  0
            max_idx =  100
            merge_query = False

            # Merge batch and head dimensions
            B, H = key.shape[:2]
            key = key.view(B*H, key.shape[2], key.shape[3])
            value = value.view(B*H, value.shape[2], value.shape[3])
            query = query.view(B*H, query.shape[2], query.shape[3])

            # Only get the image part

            # Calculate indices
            if self.idx >= min_idx and self.idx < max_idx:
                # Check if self.key_merge exists
                if not hasattr(self, "key_merge"):
                    generator = torch.Generator()#.manual_seed(42)
                    self.key_merge = BipartiteSoftMatching(key, int(query.shape[1]*0.25), generator=generator, method=merge_method)
                    key_idx, key_idx_flat, key_idx_unmerged = calculate_indices(key, int(query.shape[1]*0.25))
                    self.key_merge.src_idx = key_idx
                    self.key_merge.dst_idx = key_idx_flat
                    self.key_merge.unm_idx = key_idx_unmerged
                if not hasattr(self, "query_merge"):
                    generator = torch.Generator()#.manual_seed(42)
                    self.query_merge = BipartiteSoftMatching(query, int(query.shape[1]*0.25), generator=generator, method=merge_method)
                    query_idx, query_idx_flat, query_idx_unmerged = calculate_indices(query, int(query.shape[1]*0.25))
                    self.query_merge.src_idx = query_idx
                    self.query_merge.dst_idx = query_idx_flat
                    self.query_merge.unm_idx = query_idx_unmerged

            # Merge the query and key
            if self.idx > min_idx and self.idx < max_idx:
                key = self.key_merge.merge(key, mode="mean")
                value = self.key_merge.merge(value, mode="mean")
                if merge_query:
                    query = self.query_merge.merge(query, mode="mean")

            # Unmerge batch and head dimensions
            key = key.view(B, H, key.shape[1], key.shape[2])
            value = value.view(B, H, value.shape[1], value.shape[2])
            query = query.view(B, H, query.shape[1], query.shape[2])




        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states