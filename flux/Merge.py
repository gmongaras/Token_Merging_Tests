import torch


import torch
from typing import Tuple, Callable


def do_nothing(x: torch.Tensor, mode:str=None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    global gather
    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        sy = int(sy)
        sx = int(sx)
        hsy, wsx = h // sy, w // sx
        hsy = int(hsy)
        wsx = int(wsx)

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)
        
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :] # src
        b_idx = rand_idx[:, :num_dst, :] # dst

        global split
        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    return unm_idx, src_idx, dst_idx, r, num_dst, a_idx, b_idx

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out

    return merge, unmerge






class BipartiteSoftMatching:
    # https://github.com/xjwu1024/PPT/blob/main/merge.py
    def __init__(
        self, 
        metric: torch.Tensor, 
        r: int, 
        class_token: bool = False, 
        distill_token: bool = False,
        generator=torch.Generator(),
        method = "BipartiteSoftMatching"
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

        self.method = method


        if method == "BipartiteSoftMatching":
            self.unm_idx, self.src_idx, self.dst_idx, self.r, self.num_dst, self.a_idx, self.b_idx = bipartite_soft_matching_random2d(metric, metric.shape[1]**0.5, metric.shape[1]**0.5, 2, 2, r, generator=generator)
        elif method == "Random":
            idxs, idxs_flat, idxs_unmerged = calculate_indices(metric, r)
            self.src_idx = idxs
            self.dst_idx = idxs_flat
            self.unm_idx = idxs_unmerged
        else:
            raise NotImplementedError(f"Method {method} not implemented.")

    def merge(self, x: torch.Tensor, mode="mean") -> torch.Tensor:
        if self.method == "BipartiteSoftMatching":
            src, dst = split(x)
            n, t1, c = src.shape
            
            unm = gather(src, dim=-2, index=self.unm_idx.expand(n, t1 - self.r, c))
            src = gather(src, dim=-2, index=self.src_idx.expand(n, self.r, c))
            dst = dst.scatter_reduce(-2, self.dst_idx.expand(n, self.r, c), src, reduce=mode)

        elif self.method == "Random":
            # # Get self similarity
            # self_sim = ((torch.nn.functional.normalize(x[0], dim=-1) @ torch.nn.functional.normalize(x[0], dim=-1).mT)*~torch.eye(x.shape[1]).bool().cuda()).cpu().numpy()
            # # Plot the self similarity
            # import matplotlib.pyplot as plt
            # plt.imshow(self_sim)
            # plt.colorbar()
            # plt.show()
            # plt.savefig("self_similarity.png")
            
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
        else:
            raise NotImplementedError(f"Method {self.method} not implemented.")

        return torch.cat([unm, dst], dim=1)

    def unmerge(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unmerge tokens back to their original form based on precomputed indices.
        """
        if self.method == "BipartiteSoftMatching":
            B = x.shape[0]
            N = self.metric.shape[1]
            unm_len = self.unm_idx.shape[1]
            unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
            _, _, c = unm.shape

            src = gather(dst, dim=-2, index=self.dst_idx.expand(B, self.r, c))

            # Combine back to the original shape
            out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
            out.scatter_(dim=-2, index=self.b_idx.expand(B, self.num_dst, c), src=dst)
            out.scatter_(dim=-2, index=gather(self.a_idx.expand(B, self.a_idx.shape[1], 1), dim=1, index=self.unm_idx).expand(B, unm_len, c), src=unm)
            out.scatter_(dim=-2, index=gather(self.a_idx.expand(B, self.a_idx.shape[1], 1), dim=1, index=self.src_idx).expand(B, self.r, c), src=src)
        elif self.method == "Random":
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
        else:
            raise NotImplementedError(f"Method {self.method} not implemented.")

        return out




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
        # 50% of the time we want to merge horizontally, 50% vertically
        if torch.rand(1) > 0.5:
            idxs_x, idxs_y = idxs_y, idxs_x
        # all combinations of indices
        idxs = torch.stack(torch.meshgrid(idxs_x, idxs_y), dim=-1).reshape(-1, 2)
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