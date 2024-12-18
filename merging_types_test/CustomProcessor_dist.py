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





        import math
        class BipartiteSoftMatching:
            # https://github.com/xjwu1024/PPT/blob/main/merge.py
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

                self.unm_idx = None
                self.src_idx = None
                self.dst_idx = None

            def merge(self, x: torch.Tensor, mode="mean") -> torch.Tensor:
                # Get self similarity
                self_sim = ((torch.nn.functional.normalize(x[0], dim=-1) @ torch.nn.functional.normalize(x[0], dim=-1).mT)*~torch.eye(x.shape[1]).bool().cuda()).cpu().numpy()
                # Plot the self similarity
                import matplotlib.pyplot as plt
                plt.imshow(self_sim)
                plt.colorbar()
                plt.show()
                plt.savefig("self_similarity.png")
                
                """
                Merge tokens based on precomputed indices.
                """
                #src, dst = x[..., ::2, :], x[..., 1::2, :]
                n, t1, c = x.shape
                unm = x.gather(dim=-2, index=self.unm_idx.expand(n, -1, c))
                src = x.gather(dim=-2, index=self.src_idx.expand(n, -1, c))
                # dst = src.scatter_reduce(-2, self.dst_idx.expand(n, -1, c).to(torch.int64), src, reduce=mode)
                dst = x.gather(dim=-2, index=self.dst_idx.expand(n, -1, c))
                dst = dst + src
                if mode == "mean":
                    dst = dst/2

                # if mode == "MLERP":
                #     # MLERP reduce
                #     dst_ = dst.scatter_reduce(-2, self.dst_idx.expand(n, self.r, c), src, reduce="mean")
                #     n = dst.norm(2, dim=-1, keepdim=True).scatter_reduce(-2, self.dst_idx.expand(n, self.r, 1), src.norm(2, dim=-1, keepdim=True), reduce="max")
                #     dst = (dst_ / dst_.norm(2, dim=-1, keepdim=True)) * n
                # else:
                #     # Mean reduce
                #     dst = dst.scatter_reduce(-2, self.dst_idx.expand(n, self.r, c), src, reduce=mode)

                if self.distill_token:
                    return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
                else:
                    return torch.cat([unm, dst], dim=1)

            def unmerge(self, x: torch.Tensor) -> torch.Tensor:
                """
                Unmerge tokens back to their original form based on precomputed indices.
                """
                # Get the numerged and merged tokens
                unm_len = self.unm_idx.shape[1]
                dst_len = self.dst_idx.shape[1]
                assert x.shape[-2] == unm_len + dst_len, f"Expected {unm_len + dst_len} tokens, got {x.shape[-2]} tokens."
                unm, dst = x[..., :unm_len, :], x[..., -dst_len:, :]
                n, _, c = unm.shape

                # src = dst.gather(dim=-2, index=self.dst_idx.expand(n, self.r, c))

                out = torch.zeros(n, self.metric.shape[1], c, device=x.device, dtype=x.dtype)

                # Put the unmerged tokens back in their respective places
                # Copy the unmerged tokens to where the source and dest tokens were taken from
                out.scatter_(dim=-2, index=(self.unm_idx).expand(n, unm_len, c), src=unm)
                out.scatter_(dim=-2, index=(self.src_idx).expand(n, dst_len, c), src=dst)
                out.scatter_(dim=-2, index=(self.dst_idx).expand(n, dst_len, c), src=dst)

                return out





        """
        def merge(tensor, n):
            # Calculate self cosine similarity
            metric = torch.nn.functional.normalize(tensor, dim=-1) @ torch.nn.functional.normalize(tensor, dim=-1).transpose(-1, -2)
            mask = (~torch.tril(torch.ones(metric.shape[1], metric.shape[1])).bool().mT).to(metric.device)
            # Make diag zero
            mask = mask.float() 
            metric = metric*mask
            seq_len = metric.shape[0]
            # Flatten the metric along the rows
            metric = metric.flatten(-2)
            indices = metric.argsort(descending=True, dim=-1)
            # Calculate the row and column
            row = indices // seq_len
            col = indices % seq_len
            # row = (indices/seq_len).floor()
            # col = indices - row*seq_len
            # Get n tokens to merge, skip tokens that are already merged
            n = 100
            merged = set()
            src_idx = []
            dst_idx = []
            out_indices = []
            i = 0
            while i < n:
                # Get the row and column of the most similar tokens
                r = row[i]
                c = col[i]
                # Skip tokens that are already merged
                if r.item() in merged or c.item() in merged:
                    continue
                # Merge the tokens by adding the column (src) to the row (dst)
                tensor[r] = (tensor[r] + tensor[c])/2
                # Mark the token as merged
                merged.add(r.item())
                merged.add(c.item())
                src_idx.append(c)
                dst_idx.append(r)
                i += 1
                
            # Turn into tensors
            src_idx = torch.stack(src_idx)
            dst_idx = torch.stack(dst_idx)
            # out_indices = torch.stack(out_indices).mT
            
            # Now we need to reduce the tokens
            reduced = torch.zeros(query.shape[0], seq_len-n, query.shape[2], device=query.device)
            for i in range(query.shape[0]):
                reduced[i] = torch.stack([query[i, tok] for tok in range(seq_len) if i not in src_idx[i]])
            # Get a subset of the queries, without the source tokens
            query = query[:, out_indices]

        value = torch.stack([
            merge(value[i], 100) for i in range(value.shape[0])
        ])
        """




        # Turn 1-D indices into 2-D indices
        def unflatten_indices(indices, n):
            row = indices // n
            col = indices % n
            return torch.stack((row, col), dim=0)
        # Turn 2-D indices into 1-D indices
        def flatten_indices(indices, n):
            return (indices[0]*n + indices[1]).int()
        


        def calculate_indices(tensor, num, structued=True):
            if structued:
                idxs_x = torch.arange(tensor.shape[1]**0.5, device="cpu")
                idxs_y = torch.arange(tensor.shape[1]**0.5 / 2, device="cpu") * 2
                # all combinations of indices
                idxs = torch.stack(torch.meshgrid(idxs_y, idxs_x), dim=-1).reshape(-1, 2)
                # Randomly sample num indices
                idxs_src = idxs[torch.randperm(idxs.shape[0])[:num]]
                # Add 1 to the column to get the destination indices
                idxs_dst = idxs_src + torch.tensor([0, 1], device="cpu")

                # Flatten the indices back to 1D
                idxs_src = flatten_indices(idxs_src.mT, tensor.shape[1]**0.5)
                idxs_dst = flatten_indices(idxs_dst.mT, tensor.shape[1]**0.5)

                # Get the unmerged indices
                idxs_unmerged = torch.arange(tensor.shape[1], device="cpu")
                idxs_unmerged = idxs_unmerged[~torch.isin(idxs_unmerged, idxs_src) & ~torch.isin(idxs_unmerged, idxs_dst)]

                idxs_src = idxs_src[None, :, None].expand(tensor.shape[0], -1, 1).to(torch.long).to(tensor.device)
                idxs_dst = idxs_dst[None, :, None].expand(tensor.shape[0], -1, 1).to(torch.long).to(tensor.device)
                idxs_unmerged = idxs_unmerged[None, :, None].expand(tensor.shape[0], -1, 1).to(torch.long).to(tensor.device)

                return idxs_src, idxs_dst, idxs_unmerged

            else:
                # Indices from 0 to the number of tokens
                idxs = torch.arange(tensor.shape[1], device=tensor.device)
                # Randomly sample num indices
                idxs = idxs[torch.randperm(idxs.shape[0])[:num]].clamp(0, tensor.shape[1]-65)
                # Unlatten the indices
                idxs_unf = unflatten_indices(idxs, tensor.shape[1]**0.5)

                # # Get the cloest tokens by randomly adding/subtracting 1
                # mask = torch.randint(0, 2, (2, idxs_unf.shape[1]), device=tensor.device)*2-1
                # Get the closest tokens by adding 1 to the row
                mask = torch.tensor([1, 0], device=tensor.device)[:, None]
                idxs_ = idxs_unf + mask

                # Flatten the indices
                idxs_flat = flatten_indices(idxs_, tensor.shape[1]**0.5)
                # Unmerged indices are all indices that are not in idxs or idxs_flat
                idxs_unmerged = torch.arange(tensor.shape[1], device=tensor.device)
                idxs_unmerged = idxs_unmerged[~torch.isin(idxs_unmerged, idxs) & ~torch.isin(idxs_unmerged, idxs_flat)]

                idxs = idxs[None, :, None].expand(tensor.shape[0], -1, 1)
                idxs_flat = idxs_flat[None, :, None].expand(tensor.shape[0], -1, 1).clamp(0, tensor.shape[1]-1).to(torch.long)
                idxs_unmerged = idxs_unmerged[None, :, None].expand(tensor.shape[0], -1, 1)

                return idxs, idxs_flat, idxs_unmerged



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
        if self.idx > min_idx and self.idx < max_idx and merge_query:
            hidden_states = self.query_merge.unmerge(hidden_states)




        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states