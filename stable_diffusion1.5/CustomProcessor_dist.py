import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional
from diffusers.models.attention import Attention

from merging_tests.Merge import BipartiteSoftMatching, calculate_indices


class CustomProcessor:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.merge = True

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






        if self.merge:
            min_idx =  2
            max_idx =  20
            merge_query = True

            # Calculate indices
            if self.idx > min_idx and self.idx < max_idx:
                # Check if self.key_merge exists
                if not hasattr(self, "key_merge"):
                    self.key_merge = BipartiteSoftMatching(key, int(query.shape[1]*0.25))
                    key_idx, key_idx_flat, key_idx_unmerged = calculate_indices(key, int(query.shape[1]*0.25))
                    self.key_merge.src_idx = key_idx
                    self.key_merge.dst_idx = key_idx_flat
                    self.key_merge.unm_idx = key_idx_unmerged
                if not hasattr(self, "query_merge"):
                    self.query_merge = BipartiteSoftMatching(query, int(query.shape[1]*0.25))
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



        # Concat
        pre_shape = query.shape
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)






        # Merge tokens
        # key_merge = BipartiteSoftMatching(key, 100)
        # key = key_merge.merge(key, mode="mean")
        # value = key_merge.merge(value, mode="sum")
        # with torch.no_grad():
        #     self.query_sim.append(plot_similarity_matrix(query))
        #     self.key_sim.append(plot_similarity_matrix(key))
        #     self.value_sim.append(plot_similarity_matrix(value))
        # query_merge = BipartiteSoftMatching(query, 100)
        # query = query_merge.merge(query, mode="MLERP")

        # matrix = torch.zeros(key.shape[1], key.shape[1], dtype=torch.bool)
        # # This is not inplace
        # # matrix[key_merge.src_idx[0, :, 0].cpu()][:, key_merge.dst_idx[0, :, 0].cpu()] = True
        # # This is inplace
        # matrix[key_merge.src_idx[0, :, 0].cpu(), key_merge.dst_idx[0, :, 0].cpu()] = True
        # # Plot the merged tokens
        # import matplotlib.pyplot as plt
        # plt.imshow(matrix.cpu())
        # plt.xlabel("Source")
        # plt.ylabel("Destination")
        # plt.colorbar()
        # plt.show()
        # plt.savefig("merged_tokens.png")
        # self.key_dist.append((key_merge.src_idx - key_merge.dst_idx).abs().float().cpu().numpy().flatten())
        # self.query_dist.append((query_merge.src_idx - query_merge.dst_idx).abs().float().cpu().numpy().flatten())
        # print(f"Key: {(key_merge.src_idx - key_merge.dst_idx).abs().float().mean().cpu().item()}")
        # print(f"Query: {(query_merge.src_idx - query_merge.dst_idx).abs().float().mean().cpu().item()}")

            
        





        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)








        # # Unmerge tokens by copying the dst to the src
        # for src, dst in zip(src_idx, dst_idx):
        #     hidden_states[:, src] = hidden_states[:, dst]










        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, :pre_shape[1]],
            hidden_states[:, pre_shape[1]:],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)




        # Unmerge tokens
        if self.merge:
            if self.idx > min_idx and self.idx < max_idx and merge_query:
                hidden_states = self.query_merge.unmerge(hidden_states)




        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states