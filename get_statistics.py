import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from diffusers import StableDiffusion3Pipeline












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

        ### Save tokens before
        self.pre_tokens.append(hidden_states.detach().cpu())

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
                    scores = metric @ metric.transpose(-1, -2)
                    # Mask diagonal
                    scores = scores * ~torch.eye(scores.shape[-1], device=scores.device).bool()

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
                src, dst = x, x
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
        mask__ = (~torch.tril(torch.ones(query.shape[1], query.shape[1])).bool().mT).to(query.device)
        def plot_similarity_matrix(x):
            import matplotlib.pyplot as plt
            x = torch.nn.functional.normalize(x, dim=-1)
            similarity_matrix = x @ x.mT
            similarity_matrix = similarity_matrix.clamp(-1, 1)

            # Mask the diagonal
            similarity_matrix = similarity_matrix*mask__

            # Plot the similarity matrix
            # plt.imshow(similarity_matrix.cpu()[0])
            # plt.colorbar()
            # plt.show()
            # plt.savefig("similarity_matrix.png")

            # Plot distribution of similarity scores
            vals = similarity_matrix[0].cpu().flatten().numpy()
            vals = vals[vals != 0] # Remove zero scores
            return vals
            plt.hist(vals, bins=100)
            plt.show()
            # plt.savefig("similarity_distribution.png")






        # Merge tokens
        # key_merge = BipartiteSoftMatching(key, 100)
        # key = key_merge.merge(key, mode="mean")
        # value = key_merge.merge(value, mode="sum")
        with torch.no_grad():
            self.query_sim.append(plot_similarity_matrix(query))
            self.key_sim.append(plot_similarity_matrix(key))
            self.value_sim.append(plot_similarity_matrix(value))
        # query_merge = BipartiteSoftMatching(query, 100)
        # query = query_merge.merge(query, mode="MLERP")
            
            
        





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

        ### Save tokens after
        self.post_tokens.append(hidden_states.detach().cpu())

        return hidden_states, encoder_hidden_states





























# Load in model
with open(".env", "r") as f:
    token = f.read().strip()
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, cache_dir="./cache", token=token)
pipe = pipe.to("cuda")


# Get stable diffusion 3 config
config = pipe.transformer.config
# Replace attention in stable diffusion with the imported one
num_skip = 0
for i, layer in enumerate(pipe.transformer.transformer_blocks):
    if i >= num_skip:
        layer.attn.set_processor(CustomProcessor())


# Customer feed forward
import torch
from torch import nn
from typing import Optional
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, FP32SiLU, SwiGLU

class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "gelu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Save pre tokens
        self.pre_tokens.append(hidden_states.detach().cpu())

        for module in self.net:
            hidden_states = module(hidden_states)

        # Save post tokens
        self.post_tokens.append(hidden_states.detach().cpu())

        return hidden_states
    

# Replace feed forward in stable diffusion with the imported one
for i, layer in enumerate(pipe.transformer.transformer_blocks):
    if i >= num_skip:
        old_state_dict = layer.ff.state_dict().copy()
        device = layer.ff.net[0].proj.weight.device
        dtype = layer.ff.net[0].proj.weight.dtype
        layer.ff = FeedForward(
            dim=layer.ff.net[0].proj.weight.shape[1],
            dim_out=layer.ff.net[2].weight.shape[0],
            inner_dim=layer.ff.net[0].proj.weight.shape[0],
            activation_fn="gelu",
        )
        layer.ff.load_state_dict(old_state_dict)
        layer = layer.to("cuda").to(dtype)
        del old_state_dict









import os
def generate_image(prompt, base_dir, name, resolution):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    name = os.path.join(base_dir, name) + "/"
    if not os.path.exists(name):
        os.makedirs(name)

    # Reset all similarity scores
    for layer in pipe.transformer.transformer_blocks:
        layer.attn.processor.query_sim = []
        layer.attn.processor.key_sim = []
        layer.attn.processor.value_sim = []

    # Reset tokens in the layer
    for layer in pipe.transformer.transformer_blocks:
        layer.attn.processor.pre_tokens = []
        layer.attn.processor.post_tokens = []
        layer.ff.pre_tokens = []
        layer.ff.post_tokens = []

    generator = torch.Generator(device="cuda").manual_seed(0)
    image = pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=50,
        guidance_scale=7.0,
        generator=generator,
        height=resolution,
        width=resolution,
    ).images[0]
    
    # Save image
    image.save(f"{name}image.png")

    # Get similarity scores for all layers
    query_sim = []
    key_sim = []
    value_sim = []
    for layer in pipe.transformer.transformer_blocks:
        query_sim.append(layer.attn.processor.query_sim)
        key_sim.append(layer.attn.processor.key_sim)
        value_sim.append(layer.attn.processor.value_sim)




    ### Avg over time
    # Average the similarity scores over time
    query_avg_over_time = []
    key_avg_over_time = []
    value_avg_over_time = []
    for i in range(len(query_sim)):
        query_avg_over_time.append(np.concatenate(query_sim[i]).reshape(-1))
        key_avg_over_time.append(np.concatenate(key_sim[i]).reshape(-1))
        value_avg_over_time.append(np.concatenate(value_sim[i]).reshape(-1))

    # The sizes of the arrays are massive. Sample 10_000 elements from each
    def sample(arr):
        return np.random.choice(arr, 100_000)
    query_avg_over_time = [sample(arr) for arr in query_avg_over_time]
    key_avg_over_time = [sample(arr) for arr in key_avg_over_time]
    value_avg_over_time = [sample(arr) for arr in value_avg_over_time]

    # Box plot of similarity scores. The x axis ranges over the layers of the transformer, and the y axis ranges over the similarity scores.
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=pd.DataFrame(query_avg_over_time).T)
    plt.xlabel("Layer")
    plt.ylabel("Similarity score")
    plt.title("Query similarity over layers")
    plt.savefig(f"{name}query_similarity_over_layers.png")
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=pd.DataFrame(key_avg_over_time).T)
    plt.xlabel("Layer")
    plt.ylabel("Similarity score")
    plt.title("Key similarity over layers")
    plt.savefig(f"{name}key_similarity_over_layers.png")
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=pd.DataFrame(value_avg_over_time).T)
    plt.xlabel("Layer")
    plt.ylabel("Similarity score")
    plt.title("Value similarity over layers")
    plt.savefig(f"{name}value_similarity_over_layers.png")




    ### Avg over layerss
    query_avg_over_layers = [[] for i in range(0, len(query_sim[0]))]
    key_avg_over_layers = [[] for i in range(0, len(query_sim[0]))]
    value_avg_over_layers = [[] for i in range(0, len(query_sim[0]))]
    for i in range(len(query_sim)):
        for j in range(len(query_sim[i])):
            query_avg_over_layers[j].append(query_sim[i][j])
            key_avg_over_layers[j].append(key_sim[i][j])
            value_avg_over_layers[j].append(value_sim[i][j])

    # Combine over layers
    query_avg_over_layers = [np.concatenate(query_avg_over_layers[i]).reshape(-1) for i in range(len(query_avg_over_layers))]
    key_avg_over_layers = [np.concatenate(key_avg_over_layers[i]).reshape(-1) for i in range(len(key_avg_over_layers))]
    value_avg_over_layers = [np.concatenate(value_avg_over_layers[i]).reshape(-1) for i in range(len(value_avg_over_layers))]

    # Sample 10_000 elements from each
    query_avg_over_layers = [sample(arr) for arr in query_avg_over_layers]
    key_avg_over_layers = [sample(arr) for arr in key_avg_over_layers]
    value_avg_over_layers = [sample(arr) for arr in value_avg_over_layers]

    plt.figure(figsize=(20, 10))
    sns.boxplot(data=pd.DataFrame(query_avg_over_layers).T)
    plt.xlabel("Timestep")
    plt.ylabel("Similarity score")
    plt.title("Query similarity over time")
    plt.savefig(f"{name}query_similarity_over_time.png")
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=pd.DataFrame(key_avg_over_layers).T)
    plt.xlabel("Timestep")
    plt.ylabel("Similarity score")
    plt.title("Key similarity over time")
    plt.savefig(f"{name}key_similarity_over_time.png")
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=pd.DataFrame(value_avg_over_layers).T)
    plt.xlabel("Timestep")
    plt.ylabel("Similarity score")
    plt.title("Value similarity over time")
    plt.savefig(f"{name}value_similarity_over_time.png")














    # Get all pre and post tokens
    pre_tokens = [] # (num_layers, timesteps, seq_len, dim)
    post_tokens = [] # (num_layers, timesteps, seq_len, dim)
    pre_tokens_ff = [] # (num_layers, timesteps, seq_len, dim)
    post_tokens_ff = [] # (num_layers, timesteps, seq_len, dim)
    for layer in pipe.transformer.transformer_blocks:
        pre_tokens.append(torch.stack(layer.attn.processor.pre_tokens)[:, 0])
        post_tokens.append(torch.stack(layer.attn.processor.post_tokens)[:, 0])
        pre_tokens_ff.append(torch.stack(layer.ff.pre_tokens)[:, 0])
        post_tokens_ff.append(torch.stack(layer.ff.post_tokens)[:, 0])

    # Compute the difference between pre and post tokens
    similarity = []
    similarity_ff = []
    for i in range(len(pre_tokens)):
        similarity.append((torch.nn.functional.normalize(pre_tokens[i], p=2, dim=-1)*torch.nn.functional.normalize(post_tokens[i], p=2, dim=-1)).sum(-1))
        similarity_ff.append((torch.nn.functional.normalize(pre_tokens_ff[i], p=2, dim=-1)*torch.nn.functional.normalize(post_tokens_ff[i], p=2, dim=-1)).sum(-1))

    # Combine over layers (avg over time)
    similarity_over_time = torch.cat(similarity, dim=-1)
    similarity_over_time_ff = torch.cat(similarity_ff, dim=-1)

    # Combine over time (avg over layers)
    similarity_over_layers = []
    for i in range(len(similarity)):
        similarity_over_layers.append(similarity[i].reshape(-1))
    similarity_over_layers = torch.stack(similarity_over_layers)
    similarity_over_layers_ff = []
    for i in range(len(similarity_ff)):
        similarity_over_layers_ff.append(similarity_ff[i].reshape(-1))
    similarity_over_layers_ff = torch.stack(similarity_over_layers_ff)

    # Plot similarity over time
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=pd.DataFrame(similarity_over_time.mT.cpu().numpy()))
    plt.xlabel("Timestep")
    plt.ylabel("Similarity score")
    plt.title("Similarity over time")
    plt.savefig(f"{name}attn_tokens_similarity_over_time.png")

    # Plot similarity over layers
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=pd.DataFrame(similarity_over_layers.mT.cpu().numpy()))
    plt.xlabel("Layer")
    plt.ylabel("Similarity score")
    plt.title("Similarity over layers")
    plt.savefig(f"{name}attn_tokens_similarity_over_layers.png")


    # Plot similarity over time
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=pd.DataFrame(similarity_over_time_ff.mT.cpu().numpy()))
    plt.xlabel("Timestep")
    plt.ylabel("Similarity score")
    plt.title("Similarity over time")
    plt.savefig(f"{name}ff_tokens_similarity_over_time.png")

    # Plot similarity over layers
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=pd.DataFrame(similarity_over_layers_ff.mT.cpu().numpy()))
    plt.xlabel("Layer")
    plt.ylabel("Similarity score")
    plt.title("Similarity over layers")
    plt.savefig(f"{name}ff_tokens_similarity_over_layers.png")


dir = "images_50"
os.makedirs(dir, exist_ok=True)










# Load in dataset
from datasets import load_dataset
ds = load_dataset("daspartho/stable-diffusion-prompts")["train"]

# Dataloader
batch_size = 1
loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

# Iterate through dataset
for num, batch in enumerate(loader):
    prompt = batch["prompt"][0]
    num = str(num)

    # Random resolution between 256 and 512
    resolution = np.random.randint(256, 512)
    # Nearest multiple of 16
    resolution = resolution - resolution % 16

    # Make sure the prompt is only ASCII
    prompt = ascii(prompt)

    generate_image(prompt, dir, num, resolution)

    # Save prompt
    with open(f"{dir}/{num}/prompt.txt", "w") as f:
        f.write(prompt)
    # Save resolution
    with open(f"{dir}/{num}/resolution.txt", "w") as f:
        f.write(str(resolution))