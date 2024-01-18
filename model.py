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

def display_gpu_mem():
  print(torch.cuda.get_device_properties(0).total_memory)

def display_model_params():
  state_dict = torch.load('consolidated.00.pth', map_location='cuda')
  for key in state_dict.keys():
    layer = state_dict[key]
    shape = layer.shape
    device = layer.device
    dtype = layer.dtype
    if device.type != "cuda":
      print(device.type)
    # print(dtype)

class Tokenizer:
  def __init__(self, model_path: str):
    print(f"Tokenizer.init")
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

# model layers

@dataclass
class ModelArgs:
  dim: int = 4096
  n_layers: int = 32
  n_heads: int = 32
  n_kv_heads: Optional[int] = None
  vocab_size: int = -1  # defined later by tokenizer
  multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
  ffn_dim_multiplier: Optional[float] = None
  norm_eps: float = 1e-5
  max_batch_size: int = 1 # 32
  max_seq_len: int = 512 # 4096

class RMSNorm(torch.nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    # print(f"RMSNorm.init")
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    # print(f"RMSNorm.forward")
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
    # print(f"Attention.init")
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
    # print(f"Attention.forward")
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
    # print(f"FeedForward.init")
    hidden_dim = int(2 * hidden_dim / 3)
    # custom dim factor multiplier
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
    # print(f"TransformerBlock.init")
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
    # print(f"TransformerBlock.forward")
    h = x + self.attention.forward(
      self.attention_norm(x), start_pos, freqs_cis, mask
    )
    out = h + self.feed_forward.forward(self.ffn_norm(h))
    return out

class Transformer(nn.Module):
  def __init__(self, params: ModelArgs):
    super().__init__()
    print(f"Transformer.init")
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
    # print(f"Transformer.forward")
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

    output = self.output(h).float()
    return output

# generate

class CompletionPrediction(TypedDict, total=False):
  generation: str
  tokens: List[str]  # not required
  logprobs: List[float]  # not required

def sample_top_p(probs, p):
  probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
  probs_sum = torch.cumsum(probs_sort, dim=-1)
  mask = probs_sum - probs_sort > p
  probs_sort[mask] = 0.0
  probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
  next_token = torch.multinomial(probs_sort, num_samples=1)
  next_token = torch.gather(probs_idx, -1, next_token)
  return next_token

# llama 

class Llama:
  @staticmethod
  def build(
      max_seq_len: int,
      max_batch_size: int,
      seed: int = 1,
    ) -> "Llama":
    torch.manual_seed(seed)
    # torch.cuda.set_device(0)
    # torch.set_default_dtype(torch.float16)
    start_time = time.time()
    with open("params.json", "r") as f:
      params = json.loads(f.read())    
    tokenizer = Tokenizer(model_path="tokenizer.model")
    model_args: ModelArgs = ModelArgs(
      max_seq_len=max_seq_len,
      max_batch_size=max_batch_size,
      **params,
    )
    model_args.vocab_size = tokenizer.n_words
    print(f"Llama.build - load model")        
    checkpoint = torch.load("consolidated.00.pth", map_location="cuda")
    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)
    print(f"Llama.build - loaded in {time.time() - start_time:.2f} seconds")
    print(f"Llama.build - memory alloc: {torch.cuda.memory_allocated(0)}")
    return Llama(model, tokenizer)

  def __init__(self, model: Transformer, tokenizer: Tokenizer):
    self.model = model
    self.tokenizer = tokenizer

  @torch.inference_mode()
  def generate(
    self,
    prompt_tokens: List[List[int]],
    max_gen_len: int,
    temperature: float = 0.6,
    top_p: float = 0.9,
    logprobs: bool = False,
    echo: bool = False,
    ) -> Tuple[List[List[int]],
              Optional[List[List[float]]]]:
    params = self.model.params
    bsz = len(prompt_tokens)
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= params.max_seq_len
    total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

    pad_id = self.tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
    for k, t in enumerate(prompt_tokens):
      tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    if logprobs:
      token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz, device="cuda")
    input_text_mask = tokens != pad_id
    if min_prompt_len == total_len:
      logits = self.model.forward(tokens, prev_pos)
      token_logprobs = -F.cross_entropy(
        input=logits.transpose(1, 2),
        target=tokens,
        reduction="none",
        ignore_index=pad_id,
      )

    for cur_pos in range(min_prompt_len, total_len):
      logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
      if temperature > 0:
        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
      else:
        next_token = torch.argmax(logits[:, -1], dim=-1)

      next_token = next_token.reshape(-1)

      # only replace token if prompt has already been generated
      next_token = torch.where(
        input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
      )
      tokens[:, cur_pos] = next_token
      if logprobs:
        token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
          input=logits.transpose(1, 2),
          target=tokens[:, prev_pos + 1 : cur_pos + 1],
          reduction="none",
          ignore_index=pad_id,
        )
      eos_reached |= (~input_text_mask[:, cur_pos]) & (
        next_token == self.tokenizer.eos_id
      )
      prev_pos = cur_pos
      if all(eos_reached):
        break

    if logprobs:
      token_logprobs = token_logprobs.tolist()
    out_tokens, out_logprobs = [], []
    for i, toks in enumerate(tokens.tolist()):
      # cut to max gen len
      start = 0 if echo else len(prompt_tokens[i])
      toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
      probs = None
      if logprobs:
        probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
      # cut to eos tok if any
      if self.tokenizer.eos_id in toks:
        eos_idx = toks.index(self.tokenizer.eos_id)
        toks = toks[:eos_idx]
        probs = probs[:eos_idx] if logprobs else None
      out_tokens.append(toks)
      out_logprobs.append(probs)
    return (out_tokens, out_logprobs if logprobs else None)

  def text_completion(
      self,
      prompts: List[str],
      temperature: float = 0.6,
      top_p: float = 0.9,
      max_gen_len: Optional[int] = None,
      logprobs: bool = False,
      echo: bool = False,
    ) -> List[CompletionPrediction]:

    if max_gen_len is None:
      max_gen_len = self.model.params.max_seq_len - 1
    prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    generation_tokens, generation_logprobs = self.generate(
      prompt_tokens=prompt_tokens,
      max_gen_len=max_gen_len,
      temperature=temperature,
      top_p=top_p,
      logprobs=logprobs,
      echo=echo,
    )
    if logprobs:
      return [
        {
          "generation": self.tokenizer.decode(t),
          "tokens": [self.tokenizer.decode(x) for x in t],
          "logprobs": logprobs_i,
        }
        for t, logprobs_i in zip(generation_tokens, generation_logprobs)
      ]
    return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

# workspace

print("\n\n\n")
print(f"GPU mem avaialable:", torch.cuda.get_device_properties(0).total_memory)

generator = Llama.build(max_seq_len = 512, max_batch_size = 1)
prompts = ["Tell me 3 facts about the color red"]
results = generator.text_completion(prompts)
print(results)

print("\n\n\n")
