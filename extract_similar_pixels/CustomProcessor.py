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

                # We can only reduce by a maximum of 50% tokens
                t = metric.shape[1]
                self.r = min(r, (t - self.protected) // 2)

                if self.r <= 0:
                    return

                with torch.no_grad():
                    # Normalize for scores between -1 and 1
                    metric = metric / metric.norm(dim=-1, keepdim=True)
                    a, b = metric[..., ::2, :], metric[..., 1::2, :]

                    # Compute the similarity matrix
                    scores = a @ b.transpose(-1, -2)
                    # # Mask diagonal
                    # scores = scores * ~torch.eye(scores.shape[-1], device=scores.device).bool()

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
                dst = dst.scatter_reduce(-2, self.dst_idx.expand(n, self.r, c), src, reduce=mode)

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
                unm_len = self.unm_idx.shape[1]
                unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
                n, _, c = unm.shape

                src = dst.gather(dim=-2, index=self.dst_idx.expand(n, self.r, c))

                out = torch.zeros(n, self.metric.shape[1], c, device=x.device, dtype=x.dtype)

                out[..., 1::2, :] = dst
                out.scatter_(dim=-2, index=(2 * self.unm_idx).expand(n, unm_len, c), src=unm)
                out.scatter_(dim=-2, index=(2 * self.src_idx).expand(n, self.r, c), src=src)

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




        # Function to plot self-similarity matrix
        # mask__ = (~torch.tril(torch.ones(query.shape[1], query.shape[1])).bool().mT).to(query.device)
        diag = torch.eye(query.shape[1], device=query.device).bool().unsqueeze(0).repeat(2, 1, 1)
        def plot_similarity_matrix(x):
            import matplotlib.pyplot as plt
            x = torch.nn.functional.normalize(x, dim=-1)
            similarity_matrix = x @ x.mT
            similarity_matrix = similarity_matrix.clamp(-1, 1)

            # Mask the diagonal
            similarity_matrix[diag] = 0

            # Plot the similarity matrix
            # plt.imshow(similarity_matrix.cpu()[0])
            # plt.colorbar()
            # plt.show()
            # plt.savefig("similarity_matrix.png")

            # Q of shape (S x d)
            # (S x d) @ (d x S) = (S x S)

            # Plot distribution of similarity scores
            vals = similarity_matrix.max(-1)[0]
            vals = vals[0].cpu().flatten().numpy()
            # vals = vals[vals != 0] # Remove zero scores

            # # Get the top 10000 values
            # vals = vals[vals.argsort()[-10000:]]

            return vals
            plt.hist(vals, bins=100)
            plt.show()
            # plt.savefig("similarity_distribution.png")


        # Turn 1-D indices into 2-D indices
        def unflatten_indices(indices, n):
            row = indices // n
            col = indices % n
            return torch.stack((row, col), dim=0)


        # Image distances
        # Look at 2d distances rather than 1D - Change to box plots
        key_merge = BipartiteSoftMatching(key, 1000)
        key_src = unflatten_indices(key_merge.src_idx, (key.shape[1]**0.5))
        key_src[0] = key_src[0]*2 # The src is every even index along the row
        key_dst = unflatten_indices(key_merge.dst_idx, (key.shape[1]**0.5))
        key_dst[0] = key_dst[0]*2+1 # The dst is every odd index along the row
        self.key_img_dist.append((((key_src - key_dst)**2).sum(0)**0.5).cpu().numpy().flatten())
        query_merge = BipartiteSoftMatching(query, 1000)
        query_src = unflatten_indices(query_merge.src_idx, (query.shape[1]**0.5))
        query_src[0] = query_src[0]*2 # The src is every even index along the row
        query_dst = unflatten_indices(query_merge.dst_idx, (query.shape[1]**0.5))
        query_dst[0] = query_dst[0]*2+1 # The dst is every odd index along the row
        self.query_img_dist.append((((query_src - query_dst)**2).sum(0)**0.5).cpu().numpy().flatten())
        value_merge = BipartiteSoftMatching(value, 1000)
        value_src = unflatten_indices(value_merge.src_idx, (value.shape[1]**0.5))
        value_src[0] = value_src[0]*2 # The src is every even index along the row
        value_dst = unflatten_indices(value_merge.dst_idx, (value.shape[1]**0.5))
        value_dst[0] = value_dst[0]*2+1 # The dst is every odd index along the row
        self.value_img_dist.append((((value_src - value_dst)**2).sum(0)**0.5).cpu().numpy().flatten())
        x_merge = BipartiteSoftMatching(hidden_states, 1000)
        x_src = unflatten_indices(x_merge.src_idx, (hidden_states.shape[1]**0.5))
        x_src[0] = x_src[0]*2 # The src is every even index along the row
        x_dst = unflatten_indices(x_merge.dst_idx, (hidden_states.shape[1]**0.5))
        x_dst[0] = x_dst[0]*2+1 # The dst is every odd index along the row
        self.x_img_dist.append((((x_src - x_dst)**2).sum(0)**0.5).cpu().numpy().flatten())

        # Similarity scores
        self.query_sim.append(plot_similarity_matrix(query))
        self.key_sim.append(plot_similarity_matrix(key))
        self.value_sim.append(plot_similarity_matrix(value))
        self.x_sim.append(plot_similarity_matrix(hidden_states))

        # Text distances
        key_merge = BipartiteSoftMatching(encoder_hidden_states_key_proj, 100)
        self.key_text_dist.append((key_merge.src_idx - key_merge.dst_idx).abs().float().cpu().numpy().flatten())
        query_merge = BipartiteSoftMatching(encoder_hidden_states_query_proj, 100)
        self.query_text_dist.append((query_merge.src_idx - query_merge.dst_idx).abs().float().cpu().numpy().flatten())



        # Concat
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)






        # Merge tokens
        key_merge = BipartiteSoftMatching(key, 100)
        key = key_merge.merge(key, mode="mean")
        value = key_merge.merge(value, mode="sum")
        # with torch.no_grad():
        #     self.query_sim.append(plot_similarity_matrix(query))
        #     self.key_sim.append(plot_similarity_matrix(key))
        #     self.value_sim.append(plot_similarity_matrix(value))
        query_merge = BipartiteSoftMatching(query, 100)
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
        self.key_dist.append((key_merge.src_idx - key_merge.dst_idx).abs().float().cpu().numpy().flatten())
        self.query_dist.append((query_merge.src_idx - query_merge.dst_idx).abs().float().cpu().numpy().flatten())
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