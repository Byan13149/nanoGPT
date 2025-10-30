"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---- Rotary Embedding (with per-head base and content-adaptive scaling) ----
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, n_heads, base=10000.0, fraction=1.0, learned_base=False):
        super().__init__()
        assert 0.0 < fraction <= 1.0
        self.rotary_dim = int(head_dim * fraction) // 2 * 2  # even
        self.n_heads = n_heads
        self.base = float(base) 
        self.learned_base = bool(learned_base)
        if self.learned_base:
            # log-base per head -> base_h = exp(log_base_h)
            self.log_base = nn.Parameter(torch.full((n_heads,), math.log(self.base)))
        else:
            self.register_parameter('log_base', None)

    def _inv_freq(self, device):
        # shape: (H, D/2)
        if self.learned_base:
            base = torch.exp(self.log_base).to(device)
        else:
            base = torch.full((self.n_heads,), self.base, device=device)
        dim = self.rotary_dim // 2
        idx = torch.arange(0, dim, device=device, dtype=torch.float32) / max(dim, 1)
        inv = base.unsqueeze(1) ** (-idx)  # (H, D/2)
        return inv

    def forward(self, q, k, scale_per_head=None):
        # q,k: (B,H,T,Dh)
        if self.rotary_dim == 0:
            return q, k
        B, H, T, Dh = q.shape
        device = q.device
        inv = self._inv_freq(device)  # (H, D/2)
        pos = torch.arange(T, device=device, dtype=torch.float32)  # (T,)
        # angles: (H,T,D/2) → broadcast to (B,H,T,D/2)
        angles = inv.unsqueeze(1) * pos.view(1, T, 1)  # (H, T, D/2)
        if scale_per_head is not None:
            # scale_per_head: (B,H) -> broadcast to (B,H,T,D/2)
            angles = angles.unsqueeze(0) * scale_per_head.view(B, H, 1, 1).to(angles.dtype)
        else:
            angles = angles.unsqueeze(0)  # (1, H, T, D/2) → broadcasts to (B, H, T, D/2)
        cos = torch.cos(angles).to(q.dtype)
        sin = torch.sin(angles).to(q.dtype)

        def apply_rotary(x):
            x_ro, x_pass = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
            x_ro = x_ro.view(B, H, T, self.rotary_dim // 2, 2)
            x1, x2 = x_ro[..., 0], x_ro[..., 1]         # (B,H,T,D/2)
            y1 = x1 * cos - x2 * sin
            y2 = x1 * sin + x2 * cos
            y = torch.stack([y1, y2], dim=-1).reshape(B, H, T, self.rotary_dim)
            return torch.cat([y, x_pass], dim=-1)

        return apply_rotary(q), apply_rotary(k)

# ---- N-gram Adapter ----

class NgramAdapter(nn.Module):
    def __init__(self, n_embd, kernel_sizes=(2,3,4), dropout=0.0, bias=True):
        super().__init__()
        self.norm = LayerNorm(n_embd, bias=bias)
        self.kernel_sizes = kernel_sizes
        # depthwise conv with causal padding: padding=0 to avoid future leakage
        self.convs = nn.ModuleList([
            nn.Conv1d(n_embd, n_embd, k, groups=n_embd, padding=0, bias=bias)
            for k in kernel_sizes
        ])
        self.proj = nn.Linear(n_embd, 2 * n_embd, bias=bias)  # GLU
        self.out = nn.Linear(n_embd, n_embd, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: (B,T,C)
        B, T, C = x.shape
        y = self.norm(x).transpose(1, 2)  # (B,C,T)

        # Apply causal convolutions: pad left only, then trim to original length
        conv_outs = []
        for conv, k in zip(self.convs, self.kernel_sizes):
            # Pad (k-1) zeros on the left for causal masking
            y_padded = F.pad(y, (k-1, 0))
            conv_out = conv(y_padded)  # Now output has length T
            conv_outs.append(conv_out)

        y = sum(conv_outs) / len(conv_outs)
        y = y.transpose(1, 2)             # (B,T,C)
        a, b = self.proj(y).chunk(2, dim=-1)
        y = self.out(a * torch.sigmoid(b))
        return self.drop(y)
    
# ---- A Block Variant ----
class MemoryEnhancedBlock(nn.Module):
    def __init__(self, config, memory_size=128):
        super().__init__()
        self.memory_size = int(memory_size)
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        self.memory_keys = nn.Parameter(torch.randn(self.memory_size, config.n_embd))
        self.memory_values = nn.Parameter(torch.randn(self.memory_size, config.n_embd))
        self.memory_gate = nn.Linear(config.n_embd, 1)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        memory_scores = torch.matmul(self.ln_2(x), self.memory_keys.t())          # (B,T,M)
        memory_weights = F.softmax(memory_scores, dim=-1)                          # (B,T,M)
        retrieved_memory = torch.matmul(memory_weights, self.memory_values)        # (B,T,C)
        gate = torch.sigmoid(self.memory_gate(x))
        x_with_memory = x + gate * retrieved_memory
        return x_with_memory + self.mlp(self.ln_2(x_with_memory))

# ------------------------- core layers (kept compatible) ---------------------

class LayerNorm(nn.Module):
    """LayerNorm with optional bias (nanoGPT style)"""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CausalSelfAttention(nn.Module):
    """Full attention, flash if available (nanoGPT style)"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # causal mask buffer for non-flash path
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )

        self.head_dim = config.n_embd // config.n_head
        self.pos_encoding = getattr(config, 'pos_encoding', 'abs')
        if self.pos_encoding in ('rope', 'adaptive_rope'):
            self.rotary = RotaryEmbedding(
                head_dim=self.head_dim, n_heads=config.n_head,
                base=getattr(config, 'rope_base', 10000.0),
                fraction=getattr(config, 'rope_fraction', 1.0),
                learned_base=getattr(config, 'rope_learned_base', False),
            )
            self.rope_gate = None
            if self.pos_encoding == 'adaptive_rope':
                # 从序列平均表征生成每头缩放因子
                self.rope_gate = nn.Sequential(
                    LayerNorm(config.n_embd, bias=config.bias),
                    nn.Linear(config.n_embd, config.n_head)
                )
        else:
            self.rotary = None
            self.rope_gate = None


    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.rotary is not None:
            scale_per_head = None
            if self.rope_gate is not None:
                # x: (B,T,C) -> 先对 T 做平均，得到 (B,C)，再经 LN+Linear -> (B,H)
                g = self.rope_gate(x.mean(dim=1)).sigmoid()   # (B,H)
                scale_per_head = 0.5 + g
            else:
                scale_per_head = None

            q, k = self.rotary(q, k, scale_per_head=scale_per_head)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class StandardBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        self.ngram = NgramAdapter(config.n_embd,
                                  kernel_sizes=getattr(config, 'ngram_kernel_sizes', (2,3,4)),
                                  dropout=config.dropout,
                                  bias=config.bias) if getattr(config, 'use_ngram_adapter', False) else None

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        if self.ngram is not None:
            x = x + self.ngram(x)
        return x

def make_block(config):
    variant = getattr(config, 'block_variant', 'standard')
    if variant == 'memory':
        return MemoryEnhancedBlock(config, getattr(config, 'memory_size', 128))
    else:
        return StandardBlock(config)

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    # ---- New ----
    block_variant: str = 'standard'        # one of {'standard', 'memory'}

    # Memory block parameters (only used if block_variant == 'memory')
    memory_size: int = 128                 # number of memory slots (must be > 0)

    # Positional encoding options:
    #   - 'abs'           : absolute positional embedding or bias (RoPE disabled)
    #   - 'rope'          : rotary positional encoding
    #   - 'adaptive_rope' : RoPE with learnable base
    pos_encoding: str = 'abs'              # one of {'abs', 'rope', 'adaptive_rope'}

    # RoPE parameters (only relevant if pos_encoding is 'rope' or 'adaptive_rope')
    rope_base: float = 10000.0             # base frequency of RoPE (usually 10,000)
    rope_fraction: float = 1.0             # fraction [0.0–1.0] of each head’s dimension using RoPE
    rope_learned_base: bool = False        # whether to learn a separate RoPE base per head

    # N-gram adapter parameters (optional auxiliary module)
    use_ngram_adapter: bool = False        # enable n-gram adapter layer
    ngram_kernel_sizes: tuple = (2, 3, 4)  # kernel sizes used for n-gram filters

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        transformer_dict = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([make_block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        )
        # Only create wpe if using absolute positional encoding
        if getattr(config, 'pos_encoding', 'abs') == 'abs':
            transformer_dict['wpe'] = nn.Embedding(config.block_size, config.n_embd)

        self.transformer = nn.ModuleDict(transformer_dict)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # init
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    # ------- helpers kept from original -------
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module): 
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)

        # Use absolute positional encoding if configured, otherwise token embeddings only
        if getattr(self.config, 'pos_encoding', 'abs') == 'abs':
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            # RoPE/Adaptive RoPE: positional info is added in attention layer
            x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def crop_block_size(self, block_size):
        """shrink positional buffers & any attention-specific buffers to new block_size"""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if hasattr(self.transformer, 'wpe'):
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

        def _maybe_crop_attn(attn_mod):
            # full attention buffer
            if hasattr(attn_mod, 'bias'):  # (1,1,B,B)
                attn_mod.bias = attn_mod.bias[:, :, :block_size, :block_size]
            # dynamic sparse buffers/params
            if hasattr(attn_mod, 'causal_mask'):
                attn_mod.causal_mask = attn_mod.causal_mask[:block_size, :block_size]
            if hasattr(attn_mod, 'sparse_logits'):
                attn_mod.sparse_logits = nn.Parameter(attn_mod.sparse_logits[:block_size, :block_size])

        for block in self.transformer.h:
            # variant-aware cropping
            if isinstance(block, MemoryEnhancedBlock):
                _maybe_crop_attn(block.attn)
            elif isinstance(block, StandardBlock):
                _maybe_crop_attn(block.attn)
            else:
                # fallback for legacy Block (shouldn't happen here)
                if hasattr(block, 'attn'):
                    _maybe_crop_attn(block.attn)

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        Keep the original HF->nanoGPT loader; only allowed in vanilla setting:
        - block_variant == 'standard'
        - attention_type == 'full'
        - use_moe_mlp == False
        """
        override_args = override_args or {}
        # guard: only vanilla is supported for HF weight import
        if any(k in override_args for k in ('block_variant','attention_type','use_moe_mlp')):
            raise ValueError("from_pretrained only supports vanilla model; do not override variant knobs here.")
        # force vanilla
        if 'dropout' in override_args:
            pass  # ok
        # defer to original logic below (copied from your file) -----------------
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        # vanilla-only
        config_args.update(dict(attention_type='full', block_variant='standard', use_moe_mlp=False))
        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys()
                      if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad(): sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad(): sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
