# Transformer from Scratch (PyTorch)

An encoder–decoder Transformer, as described in *Attention Is All You Need*
(Vaswani et al., 2017), written module by module in PyTorch.

I built this to learn how the architecture fits together, following Umar Jamil's
[Coding a Transformer from scratch on PyTorch](https://github.com/hkproj/pytorch-transformer)
walkthrough. The comments are my own notes on what each part does and what shape
its tensors are.

## What's in `model.py`

| Module | What it does |
|---|---|
| `InputEmbeddings` | Token embedding scaled by √d_model |
| `PositionalEncoding` | Fixed sine/cosine position signal added to the embeddings, then dropout |
| `LayerNormalization` | Layer norm with learnable scale (`alpha`) and shift (`bias`) |
| `FeedForwardBlock` | Two linear layers with ReLU: d_model → d_ff → d_model |
| `MultiHeadAttentionBlock` | Scaled dot-product attention split across `h` heads, with an optional mask |
| `ResidualConnection` | Pre-norm residual: `x + dropout(sublayer(norm(x)))` |
| `EncoderBlock` / `Encoder` | Self-attention + feed-forward, stacked N times |
| `DecoderBlock` / `Decoder` | Masked self-attention + cross-attention over the encoder output + feed-forward, stacked N times |
| `ProjectionLayer` | Maps the decoder output to vocabulary logits |
| `build_transformer()` | Wires everything together (defaults: d_model=512, N=6, h=8, d_ff=2048, dropout=0.1) and applies Xavier initialization |

## Usage

```python
import torch
from model import build_transformer

model = build_transformer(src_vocab_size=100, tgt_vocab_size=120,
                          src_seq_len=16, tgt_seq_len=16,
                          d_model=64, N=2, h=4, d_ff=128)

src = torch.randint(0, 100, (2, 10))               # (batch, src_len)
tgt = torch.randint(0, 120, (2, 7))                # (batch, tgt_len)
causal_mask = torch.tril(torch.ones(1, 7, 7)).int()

encoder_out = model.encode(src, None)
logits = model.project(model.decode(encoder_out, None, tgt, causal_mask))
print(logits.shape)  # torch.Size([2, 7, 120])
```

Requires Python 3 and PyTorch.

## Scope

This is the model only. There is no tokenizer, dataset, or training loop yet.
