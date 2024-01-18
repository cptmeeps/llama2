import os
import time
import json
import math
from typing import Optional, Tuple, List, TypedDict
from dataclasses import dataclass
from sentencepiece import SentencePieceProcessor

import torch
from torch import nn
import torch.nn.functional as F

# utils

torch.set_default_tensor_type('torch.cuda.HalfTensor')

class Tokenizer:
  def __init__(self, model_path: str):
    # print(f"Tokenizer.init")
    assert os.path.isfile(model_path), model_path
    self.sp_model = SentencePieceProcessor(model_file=model_path)
    self.n_words: int = self.sp_model.vocab_size()
    self.bos_id: int = self.sp_model.bos_id()
    self.eos_id: int = self.sp_model.eos_id()
    self.pad_id: int = self.sp_model.pad_id()
    assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

  def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
    assert type(s) is str
    t = self.sp_model.encode(s)
    if bos:
      t = [self.bos_id] + t
    if eos:
      t = t + [self.eos_id]
    return t

  def decode(self, t: List[int]) -> str:
    return self.sp_model.decode(t)

# nn layers

@dataclass
class ModelArgs:
  dim = 4096
  n_layers = 32
  n_heads = 32
  n_kv_heads = None
  vocab_size = -1  # defined later by tokenizer
  multiple_of = 256  # make SwiGLU hidden layer size multiple of large power of 2
  ffn_dim_multiplier = None
  norm_eps = 1e-5
  max_batch_size = 1 # 32
  max_seq_len = 256 # 4096

class RMSNorm(torch.nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
  t = torch.arange(end, device=freqs.device)  # type: ignore
  freqs = torch.outer(t, freqs).float()  # type: ignore
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
  return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
  ndim = x.ndim
  assert 0 <= 1 < ndim
  assert freqs_cis.shape == (x.shape[1], x.shape[-1])
  shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
  return freqs_cis.view(*shape)  

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
  xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
  freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
  return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
  # torch.repeat_interleave(x, dim=2, repeats=n_rep)
  bs, slen, n_kv_heads, head_dim = x.shape
  if n_rep == 1:
    return x
  return (
    x[:, :, :, None, :]
    .expand(bs, slen, n_kv_heads, n_rep, head_dim)
    .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
  )

class Attention(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
    self.head_dim = args.dim // args.n_heads
    
    self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
    self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
    self.cache_k = torch.zeros(
      (
        args.max_batch_size,
        args.max_seq_len,
        self.n_kv_heads,
        self.head_dim,
      )
    )
    self.cache_v = torch.zeros(
      (
        args.max_batch_size,
        args.max_seq_len,
        self.n_kv_heads,
        self.head_dim,
      )
    )

  def forward(
      self,
      x: torch.Tensor,
      start_pos: int,
      freqs_cis: torch.Tensor,
      mask: Optional[torch.Tensor],
    ):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    xq = xq.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    self.cache_k = self.cache_k.to(xq)
    self.cache_v = self.cache_v.to(xq)

    self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
    self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

    keys = self.cache_k[:bsz, : start_pos + seqlen]
    values = self.cache_v[:bsz, : start_pos + seqlen]

    xq = xq.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)
    keys = keys.transpose(1, 2) # (bs, n_kv_heads, cache_len + seqlen, head_dim)
    values = values.transpose(1, 2) # (bs, n_kv_heads, cache_len + seqlen, head_dim)
    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
    if mask is not None:
      scores = scores + mask  # (bs, n_kv_heads, seqlen, cache_len + seqlen)
    scores = F.softmax(scores.float(), dim=-1).type_as(scores)
    output = torch.matmul(scores, values)  # (bs, n_kv_heads, seqlen, head_dim)
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)

class FeedForward(nn.Module):
  def __init__(
      self,
      dim: int,
      hidden_dim: int,
      multiple_of: int,
      ffn_dim_multiplier: Optional[float],
    ):
    super().__init__()
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_dim_multiplier is not None:
      hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    self.w1 = nn.Linear(dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    self.w3 = nn.Linear(dim, hidden_dim, bias=False)

  def forward(self, x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
  def __init__(self, layer_id: int, args: ModelArgs):
    super().__init__()
    self.n_heads = args.n_heads
    self.dim = args.dim
    self.head_dim = args.dim // args.n_heads
    self.attention = Attention(args)
    self.feed_forward = FeedForward(
      dim=args.dim,
      hidden_dim=4 * args.dim,
      multiple_of=args.multiple_of,
      ffn_dim_multiplier=args.ffn_dim_multiplier,
    )
    self.layer_id = layer_id
    self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
    self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

  def forward(
      self,
      x: torch.Tensor,
      start_pos: int,
      freqs_cis: torch.Tensor,
      mask: Optional[torch.Tensor],
    ):
    h = x + self.attention.forward(
      self.attention_norm(x), start_pos, freqs_cis, mask
    )
    out = h + self.feed_forward.forward(self.ffn_norm(h))
    return out

class Transformer(nn.Module):
  def __init__(self, params: ModelArgs):
    super().__init__()
    self.params = params
    self.vocab_size = params.vocab_size
    self.n_layers = params.n_layers

    self.tok_embeddings = nn.Embedding(
      params.vocab_size, params.dim
    )

    self.layers = torch.nn.ModuleList()
    for layer_id in range(params.n_layers):
      self.layers.append(TransformerBlock(layer_id, params))

    self.norm = RMSNorm(params.dim, eps=params.norm_eps)
    self.output = nn.Linear(
      params.dim, params.vocab_size, bias=False
    )

    self.freqs_cis = precompute_freqs_cis(
      self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
    )

  @torch.inference_mode()
  def forward(self, tokens: torch.Tensor, start_pos: int):
    _bsz, seqlen = tokens.shape
    h = self.tok_embeddings(tokens)
    self.freqs_cis = self.freqs_cis.to(h.device)
    freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

    mask = None
    if seqlen > 1:
      mask = torch.full(
        (seqlen, seqlen), float("-inf"), device=tokens.device
      )
      mask = torch.triu(mask, diagonal=1)
      mask = torch.hstack([
        torch.zeros((seqlen, start_pos), device=tokens.device),
        mask
      ]).type_as(h)

    for layer in self.layers:
      h = layer(h, start_pos, freqs_cis, mask)

    h = self.norm(h)
    output = h # skipping self.output (nn.Linear)    
    return output

# llama 

class Llama:
  def __init__(self):
    print("\n\nLlama.init")
    print(f"\tGPU mem avaialable:", torch.cuda.get_device_properties(0).total_memory)
    start_time = time.time()
    tokenizer = Tokenizer(model_path="tokenizer.model")
    model_args = ModelArgs()
    model_args.vocab_size = tokenizer.n_words
    checkpoint = torch.load("consolidated.00.pth", map_location="cuda")
    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)
    print(f"\tloaded in {time.time() - start_time:.2f} seconds")
    print(f"\tmemory alloc: {torch.cuda.memory_allocated(0)}")
    self.model = model
    self.tokenizer = tokenizer

  @torch.inference_mode()
  def encode_text(self, prompts):
    prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    bsz = len(prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    pad_id = self.tokenizer.pad_id
    tokens = torch.full((bsz, max_prompt_len), pad_id, dtype=torch.long, device="cuda")
    for k, t in enumerate(prompt_tokens):
      tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    encodings = self.model.forward(tokens, 0)
    return encodings

  def test_encode(self, prompt=None):
    start_time = time.time()
    if not prompt: 
      prompts = ["Tell me 3 facts about the color red"]
    encodings = self.encode_text(prompts)
    print(f"expected values:")
    print(f"\t1   [-0.0799,  1.6094,  3.1934,  ...,  1.3057, -0.8794,  1.9229]")
    print(f"\t2   [ 3.8477, -1.5674,  0.4629,  ..., -1.9082,  3.5605,  0.6763]")
    print(f"\t10  [ 1.1055, -2.2246,  0.3535,  ..., -2.5898,  2.4023, -1.1875]")
    print(f"actual encodings:\n\tshape: {encodings.shape}\n{encodings}")
    print(f"Llama.encode_text\n\tencode in {time.time() - start_time:.2f} seconds")

# workspace

model = Llama()
model.test_encode()
