"""ChronoGPT (He, Lv, Manela and Wu 2026, "Chronologically Consistent Large Language Models", JFE): the robustness encoder whose
pretraining data end at a fixed date (the vintage), so that its embeddings carry no information from after that date.
The model classes below are those of the vintage's ChronoGPT_inference.py on Hugging Face (MIT license), copied verbatim except
the hub mixin, the save / load helpers and the generation parts (key-value cache, output head in the forward pass: forward returns
the final normalized hidden state; lm_head stays for the weights); the weights load with torch.load(weights_only=True). Headlines are encoded with the
GPT-2 byte-pair encoding (the tokenizer the model was trained with), each sequence led by <|endoftext|> (the document start the
model saw in training, and the counterpart of the special tokens the other encoders' sequences start with), truncated to the
headline encoders' max length, right-padded (a causal model's real tokens never attend to the padding), and mean-pooled over the
sequence's tokens of the last layer (the final RMS norm), as Codes/1_data/03_embed.py pools the other encoders."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

REVISION = {"manelalab/chrono-gpt-v1-19991231": "45d6568340081add5c5b1444feb17a94b8f8199d"}   # the Hugging Face commits embedded
TOKENIZER = ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e")             # GPT-2 BPE, the same merges as tiktoken "gpt2"
EOT = 50256
POOLING = "mean over the tokens (a leading <|endoftext|> included) of the last layer (final RMS norm), bf16 network, fp32 accumulate, stored fp16"


def is_chrono(model):
    return model.split("/")[-1].startswith("chrono-gpt-")


def config(model):
    """the vintage's config (vocab_size, num_layers, num_heads, model_dim)"""
    cfg = json.loads(Path(hf_hub_download(model, "config.json", revision=REVISION[model])).read_text())
    cfg.pop("model_type", None)
    return cfg


def spec(model):
    """the encoder identity a headline chunk's manifest records (common/emb.py SPEC, with model and max_len set by 03_embed.py)"""
    return {"revision": REVISION[model], "tokenizer": "GPT2TokenizerFast " + "@".join(TOKENIZER), "pooling": POOLING,
            "hidden_size": config(model)["model_dim"]}


def load(model, device):
    """the vintage's network on device, in eval mode"""
    net = ChronoGPT(**config(model))
    net.load_state_dict(torch.load(hf_hub_download(model, "pytorch_model.bin", revision=REVISION[model]), map_location="cpu", weights_only=True))
    return net.to(device).eval()


def tokenize(texts, max_len):
    """<|endoftext|> + the GPT-2 tokens of each text, at most max_len in all. A text containing the literal "<|endoftext|>" is
    refused: the tokenizer would read it as the special token, the model's own tiktoken encoding as ordinary text"""
    from transformers import GPT2TokenizerFast
    texts = list(texts)
    bad = [t for t in texts if "<|endoftext|>" in t]
    if bad:
        raise ValueError(f"{len(bad)} texts contain the literal <|endoftext|>, e.g. {bad[0]!r}")
    tok = GPT2TokenizerFast.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    return [[EOT] + ids[:max_len - 1] for ids in tok(texts, add_special_tokens=False)["input_ids"]]


def pool(net, ids, batch_tokens, log=None):
    """mean over each sequence's tokens of the last layer, fp32; batches of similar length, sorted as 03_embed.py sorts them"""
    lens = np.array([len(x) for x in ids])
    order = np.argsort(lens)
    out = np.zeros((len(ids), net.embed.embedding_dim), np.float32)
    dev = next(net.parameters()).device
    i = nb = 0
    with torch.inference_mode():
        while i < len(order):
            bs = min(1024, max(1, batch_tokens // int(lens[order[min(i + 1023, len(order) - 1)]])))
            idx = order[i:i + bs]
            L = int(lens[idx].max())
            x = torch.full((len(idx), L), EOT, dtype=torch.long)
            for r, j in enumerate(idx):
                x[r, :lens[j]] = torch.tensor(ids[j])
            m = (torch.arange(L)[None, :] < torch.from_numpy(lens[idx])[:, None]).unsqueeze(-1).float().to(dev)
            h = net(x.to(dev)).float()
            out[idx] = ((h * m).sum(1) / m.sum(1)).cpu().numpy()
            i += bs
            nb += 1
            if log and nb % 100 == 0:
                log(i)
    return out


# ---- the model classes of ChronoGPT_inference.py (MIT license) -----------------------------------------------------------------
# Source: ChronoGPT_inference.py of https://huggingface.co/manelalab/chrono-gpt-v1-19991231 (revision
# 45d6568340081add5c5b1444feb17a94b8f8199d), by Songrun He, Linying Lv, Asaf Manela and Jimmy Wu, released under the MIT License
# (model card); its architecture is a modified modded-nanogpt (https://github.com/KellerJordan/modded-nanogpt).
#
# Copyright (c) 2025 Songrun He, Linying Lv, Asaf Manela, Jimmy Wu (ChronoGPT)
# Copyright (c) 2024 Keller Jordan (modded-nanogpt)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    @torch.inference_mode()
    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim, max_seq_len=65536):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum('i,j -> ij', t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    @torch.inference_mode()
    def forward(self, x):
        cos, sin = self.cos[None, :x.size(-3), None, :], self.sin[None, :x.size(-3), None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(self.head_dim)
        self.c_proj = CastedLinear(dim, dim)

    @torch.inference_mode()
    def forward(self, x, ve):
        B, T = x.size(0), x.size(1)
        
        # Generate Q, K, V
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.num_heads, self.head_dim)
        
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v
            
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)   # (B, heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_()

    @torch.inference_mode()
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, model_dim, num_heads, use_attn=True):
        super().__init__()
        self.attn = CausalSelfAttention(model_dim, num_heads) if use_attn else None
        self.mlp = MLP(model_dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    @torch.inference_mode()
    def forward(self, x, ve, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve)
        x = x + self.mlp(norm(x))
        return x

class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim, num_layers=52):
        super().__init__()
        self.num_layers = num_layers
        # We only have 3 distinct embedding modules, reused at beginning and end.
        self.embed = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])

    def forward(self, inputs):
        # Compute the base embeddings (a list of length 3)
        base = [emb(inputs).bfloat16() for emb in self.embed]
        L = self.num_layers
        half = L // 2  # number of encoder layers (assumes num_layers is even)
        # Build encoder: first 3 layers get embeddings, rest get None.
        encoder = [base[i] if i < 3 else None for i in range(half)]
        # Build decoder: last 3 layers get embeddings, others get None.
        # For decoder layers, if i is in [half-3, half-1] then assign base[0], base[1], base[2]
        decoder = [base[i - (half - 3)] if i >= (half - 3) else None for i in range(half)]
        return encoder + decoder


class ChronoGPT(nn.Module):
    def __init__(self, vocab_size, num_layers, num_heads, model_dim, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.vocab_size = vocab_size  # Store vocab_size as instance variable
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, use_attn=True) for i in range(num_layers)])
        self.value_embeds = ValueEmbedding(vocab_size, model_dim, num_layers=num_layers)
        self.lm_head = CastedLinear(model_dim, vocab_size)
        self.lm_head.weight.data.zero_()
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
    @torch.inference_mode()
    def forward(self, inputs):
        """the final normalized hidden state (B x T x d) of a batch of token ids"""
        x0 = norm(self.embed(inputs).bfloat16())
        x = x0
        ve = self.value_embeds(inputs)                                      # per layer: (B x T x d) or None
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]
        skip_connections = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0)
            skip_connections.append(x)
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0)
        return norm(x)
