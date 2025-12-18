import os
import re
import time
import math
import wandb
import random
import argparse
import statistics
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Type

import torch
from paged_granite import PagedGranite
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset, DownloadConfig
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks import MMLU, TruthfulQA
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.benchmarks.tasks import TruthfulQATask
from deepeval.benchmarks.modes import TruthfulQAMode


# Hugging Face home Directory
#___________________
HF_CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_CACHE_DIR, "datasets"))

def load_dataset_with_retry(
    path: str,
    name: Optional[str] = None,
    split: Optional[str] = None,
    retries: int = 6,
    base_delay: float = 2.0,
    **kwargs,
):
    """
    Load hugging face dataset. Added try catch to prevent download/access errors
    """
    download_config = kwargs.pop("download_config", None) or DownloadConfig(
        resume_download=True,
        max_retries=1,
    )

    last_err = None
    for attempt in range(retries):
        try:
            return load_dataset(
                path,
                name=name,
                split=split,
                download_config=download_config,
                **kwargs,
            )
        except Exception as e:
            last_err = e
            msg = str(e)
            transient = any(s in msg for s in [
                " 502 ", " 503 ", " 504 ",
                "Bad Gateway", "Service Unavailable", "Gateway",
                "Read timed out", "Connection reset", "Temporary failure",
                "MaxRetryError", "HTTPSConnectionPool",
            ])
            if not transient and attempt >= 1:
                raise
            sleep_s = base_delay * (2 ** attempt) + random.random()
            print(f"[dataset retry] attempt {attempt+1}/{retries} failed: {e}\n -> sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise last_err


#helper functions
#_____________________
def _cuda_sync_if_needed(device: str):
    """
    use syncronize to ensure accuracy
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

def _percentile(xs: List[float], p: float) -> float:
    """
    calculate the percentile with linear interpolation
    """
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs_sorted[int(k)]
    return xs_sorted[f] * (c - k) + xs_sorted[c] * (k - f)

def _format_bytes(n: Optional[float]) -> str:
    """
    format bytes to to string representation
    """
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    u = 0
    n = float(n)
    while n >= 1024 and u < len(units) - 1:
        n /= 1024
        u += 1
    return f"{n:.2f} {units[u]}"


#metrics
#______________
@dataclass
class CallMetrics:
    latency_ms: float
    input_tokens: int
    output_tokens: int
    peak_alloc_bytes: Optional[int] = None
    peak_reserved_bytes: Optional[int] = None

@dataclass
class RunSummary:
    n_calls: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    qps: float
    tok_per_s_in: float
    tok_per_s_out: float
    peak_alloc_bytes_max: Optional[int]
    peak_reserved_bytes_max: Optional[int]

def summarize_calls(calls: List[CallMetrics]) -> RunSummary:
    n = len(calls)
    lats = [c.latency_ms for c in calls]
    in_toks = [c.input_tokens for c in calls]
    out_toks = [c.output_tokens for c in calls]
    total_s = sum(lats) / 1000.0 if lats else 0.0

    peak_allocs = [c.peak_alloc_bytes for c in calls if c.peak_alloc_bytes is not None]
    peak_resvs = [c.peak_reserved_bytes for c in calls if c.peak_reserved_bytes is not None]

    return RunSummary(
        n_calls=n,
        avg_latency_ms=(statistics.mean(lats) if lats else float("nan")),
        p50_latency_ms=_percentile(lats, 0.50),
        p95_latency_ms=_percentile(lats, 0.95),
        qps=(n / total_s if total_s > 0 else float("nan")),
        tok_per_s_in=(sum(in_toks) / total_s if total_s > 0 else float("nan")),
        tok_per_s_out=(sum(out_toks) / total_s if total_s > 0 else float("nan")),
        peak_alloc_bytes_max=(max(peak_allocs) if peak_allocs else None),
        peak_reserved_bytes_max=(max(peak_resvs) if peak_resvs else None),
    )


# =========================
# DeepEval "schema" return shim
# =========================
class AnswerObj:
    """Minimal object compatible with DeepEval expecting `.answer`."""
    def __init__(self, answer: str):
        self.answer = answer


# =========================
# Granite wrapper (structured schema + single-device model)
# =========================
class Granite4HF(DeepEvalBaseLLM):
    def __init__(
        self,
        model_path: str = "ibm-granite/granite-4.0-micro",
        device: str = "cuda",
        max_new_tokens: int = 16,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens

        self._is_cuda = device.startswith("cuda") and torch.cuda.is_available()

        if dtype is None:
            dtype = torch.float16 if self._is_cuda else torch.float32
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Prefer dtype=... (newer); fall back to torch_dtype=... (older)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=self.dtype)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=self.dtype)

        self.model.to(device)
        torch.compile(self.model)
        self.model.eval()

        self.calls: List[CallMetrics] = []
        if self._is_cuda:
            torch.cuda.reset_peak_memory_stats()

    def reset_metrics(self):
        self.calls.clear()
        if self._is_cuda:
            torch.cuda.reset_peak_memory_stats()

    def get_model_name(self) -> str:
        return self.model_path

    def load_model(self):
        return self.model

    def _extract_choice_token(self, text: str) -> str:
        """
        For MC tasks, DeepEval wants a compact option token.
        We try:
            - A-E
            - A-D (common)
            - otherwise return stripped text
        """
        t = text.strip()
        m = re.search(r"\b([A-E])\b", t)
        return m.group(1) if m else t

    @torch.inference_mode()
    def generate(self, prompt: Optional[str] = None, **kwargs):
        """
        DeepEval may call:
        generate(prompt=..., schema=MultipleChoiceSchema)
        If `schema` is provided, return an object with `.answer`.
        Otherwise return a plain string.
        """
        if prompt is None:
            prompt = kwargs.get("prompt", "")

        schema: Optional[Type] = kwargs.get("schema", None)

        if self._is_cuda:
            torch.cuda.reset_peak_memory_stats()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_len = int(inputs["input_ids"].shape[-1])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        _cuda_sync_if_needed(self.device)
        t0 = time.perf_counter()

        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        _cuda_sync_if_needed(self.device)
        t1 = time.perf_counter()

        gen_text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        output_len = int(out.shape[-1])
        new_tokens = max(0, output_len - input_len)

        completion = gen_text[len(prompt):] if len(gen_text) >= len(prompt) else gen_text
        answer = self._extract_choice_token(completion)

        latency_ms = (t1 - t0) * 1000.0
        peak_alloc = int(torch.cuda.max_memory_allocated()) if self._is_cuda else None
        peak_resv = int(torch.cuda.max_memory_reserved()) if self._is_cuda else None

        self.calls.append(
            CallMetrics(
                latency_ms=latency_ms,
                input_tokens=input_len,
                output_tokens=new_tokens,
                peak_alloc_bytes=peak_alloc,
                peak_reserved_bytes=peak_resv,
            )
        )

        # If DeepEval requested a schema, return an object with `.answer`
        if schema is not None:
            # Some DeepEval paths might accept a schema class instance; we keep it minimal.
            return AnswerObj(answer)

        return answer

    async def a_generate(self, prompt: str, **kwargs):
        return self.generate(prompt=prompt, **kwargs)
    
class PagedGranite4HF(Granite4HF):
    def __init__(
        self,
        model_path: str = "ibm-granite/granite-4.0-micro",
        device: str = "cuda",
        max_new_tokens: int = 16,
        dtype: Optional[torch.dtype] = None,
        num_pages: int = 128,
        page_size: int = 64,
        max_batch_size: int = 1,
    ):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens

        self._is_cuda = device.startswith("cuda") and torch.cuda.is_available()

        if dtype is None:
            dtype = torch.float16 if self._is_cuda else torch.float32
        self.dtype = dtype

        # Tokenizer stays the same
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load the model using PagedGranite.from_pretrained
        self.model = PagedGranite.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map="auto",
            num_pages=num_pages,
            page_size=page_size,
            max_batch_size=max_batch_size,
        )
        
        self.model.to(device)
        torch.compile(self.model)
        self.model.eval()

        self.calls: list[CallMetrics] = []
        if self._is_cuda:
            torch.cuda.reset_peak_memory_stats()

    # You inherit all methods from Granite4HF: generate, reset_metrics, etc.

    def reset_cache(self):
        """Reset paged KV cache for all attention layers."""
        # Delegate to PagedGranite’s method
        PagedGranite.reset_cache(self.model)


#MMLU
#__________________
class MMLU_Retry(MMLU):
    def load_benchmark_dataset(self, task: MMLUTask):
        self.dataset = load_dataset_with_retry("cais/mmlu", name=task.value)
        return super().load_benchmark_dataset(task)


@torch.inference_mode()
def bench_batching_efficiency(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    base_prompt: str,
    batch_sizes: List[int],
    max_new_tokens: int = 16,
    warmup: int = 1,
    iters: int = 5,
) -> List[Dict[str, Any]]:
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    results = []

    for bs in batch_sizes:
        PagedGranite.reset_cache(model)
        prompts = [base_prompt] * bs
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}

        for _ in range(warmup):
            _cuda_sync_if_needed(device)
            _ = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            _cuda_sync_if_needed(device)

        lat_ms_runs = []
        out_tokens_runs = []
        in_tokens_total = int(enc["input_ids"].numel())

        for _ in range(iters):
            if is_cuda:
                torch.cuda.reset_peak_memory_stats()

            _cuda_sync_if_needed(device)
            t0 = time.perf_counter()

            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            _cuda_sync_if_needed(device)
            t1 = time.perf_counter()

            lat_ms = (t1 - t0) * 1000.0
            lat_ms_runs.append(lat_ms)

            out_tokens_total = int(out.numel()) - int(enc["input_ids"].numel())
            out_tokens_runs.append(max(0, out_tokens_total))

        avg_lat_ms = statistics.mean(lat_ms_runs)
        total_s = avg_lat_ms / 1000.0

        results.append(
            {
                "batch_size": bs,
                "avg_batch_latency_ms": avg_lat_ms,
                "latency_ms_per_query": avg_lat_ms / bs,
                "qps": (bs / total_s) if total_s > 0 else float("nan"),
                "in_tokens_per_s": (in_tokens_total / total_s) if total_s > 0 else float("nan"),
                "out_tokens_per_s": (statistics.mean(out_tokens_runs) / total_s) if total_s > 0 else float("nan"),
                "peak_alloc": int(torch.cuda.max_memory_allocated()) if is_cuda else None,
                "peak_reserved": int(torch.cuda.max_memory_reserved()) if is_cuda else None,
            }
        )

    return results


@torch.inference_mode()
def bench_context_length_scalability(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    token_lengths: List[int],
    max_new_tokens: int = 8,
    warmup: int = 1,
    iters: int = 5,
) -> List[Dict[str, Any]]:
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    results = []
    chunk = " The quick brown fox jumps over the lazy dog."

    for target_len in token_lengths:
        PagedGranite.reset_cache(model)
        prompt = ""
        while len(tokenizer(prompt).input_ids) < target_len:
            prompt += chunk

        enc = tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        in_len = int(enc["input_ids"].shape[-1])

        for _ in range(warmup):
            _cuda_sync_if_needed(device)
            _ = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            _cuda_sync_if_needed(device)

        lats = []
        for _ in range(iters):
            if is_cuda:
                torch.cuda.reset_peak_memory_stats()

            _cuda_sync_if_needed(device)
            t0 = time.perf_counter()

            _ = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            _cuda_sync_if_needed(device)
            t1 = time.perf_counter()
            lats.append((t1 - t0) * 1000.0)

        results.append(
            {
                "input_tokens": in_len,
                "avg_latency_ms": statistics.mean(lats),
                "p95_latency_ms": _percentile(lats, 0.95),
                "ms_per_input_token": statistics.mean(lats) / max(1, in_len),
                "peak_alloc": int(torch.cuda.max_memory_allocated()) if is_cuda else None,
                "peak_reserved": int(torch.cuda.max_memory_reserved()) if is_cuda else None,
            }
        )

    return results

def print_run_report(title: str, overall_score: float, calls: List[CallMetrics]):
    s = summarize_calls(calls)
    print(f"\n=== {title} ===")
    print(f"Accuracy (overall_score): {overall_score:.4f}")
    print(f"Queries: {s.n_calls}")
    print(f"Latency (ms/query): avg={s.avg_latency_ms:.2f}, p50={s.p50_latency_ms:.2f}, p95={s.p95_latency_ms:.2f}")
    print(f"Throughput: {s.qps:.2f} queries/s")
    print(f"Token throughput: in={s.tok_per_s_in:.2f} tok/s, out={s.tok_per_s_out:.2f} tok/s")
    print(f"Peak GPU memory (allocated): {_format_bytes(s.peak_alloc_bytes_max)}")
    print(f"Peak GPU memory (reserved):  {_format_bytes(s.peak_reserved_bytes_max)}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Evaluate Granite models')
    parser.add_argument('--model_path', type=str, default='ibm-granite/granite-4.0-h-1b', help='Path to the Granite model')
    parser.add_argument('--mode', type=str, choices=['default', 'paged'], default='default', help='Model type: default or paged')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = args.model_path

    # WandB setup 
    wandb.login(key="d064fb57cc7dee8a7a9e21f1ebd59185be95ec43")

    if args.mode == 'default':
        granite = Granite4HF(model_path=model_path, device=device, max_new_tokens=16)
    else:
        granite = PagedGranite4HF(model_path=model_path, device=device, max_new_tokens=16, max_batch_size=16)

    config={
        "model_path": args.model_path.split("/")[-1],
        "mode": args.mode,
        "device": device
    }
    wandb.init(
        project="hpml-final-project",
        name=f"{args.model_path.split("/")[-1]}-{args.mode}",
        config=config
    )



    # -------- MMLU --------
    granite.reset_metrics()
    mmlu = MMLU_Retry(
        tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY],
        n_shots=3,
    )
    mmlu.evaluate(model=granite)
    print_run_report("MMLU Results + Metrics", mmlu.overall_score, granite.calls)

    # ----- TruthfulQA -----
    granite.reset_metrics()
    truthfulqa = TruthfulQA(
        tasks=[TruthfulQATask.SCIENCE, TruthfulQATask.FINANCE],
        mode=TruthfulQAMode.MC1,
    )
    truthfulqa.evaluate(model=granite)
    print_run_report("TruthfulQA Results + Metrics", truthfulqa.overall_score, granite.calls)

    # -------- Optional microbenchmarks --------
    hf_model = granite.model
    hf_tok = granite.tokenizer

    print("\n=== Batching Efficiency Microbenchmark ===")
    base_prompt = (
        "Answer with ONLY one letter: A, B, C, or D.\n"
        "Question: Which planet is known as the Red Planet?\n"
        "A) Venus\nB) Mars\nC) Jupiter\nD) Mercury\n"
        "Answer:"
    )
    batch_results = bench_batching_efficiency(
        model=hf_model,
        tokenizer=hf_tok,
        device=device,
        base_prompt=base_prompt,
        batch_sizes=[1, 2, 4, 8, 16],
        max_new_tokens=8,
        warmup=1,
        iters=5,
    )
    for r in batch_results:
        print(
            f"BS={r['batch_size']:>2} | ms/query={r['latency_ms_per_query']:.2f} | "
            f"qps={r['qps']:.2f} | in_tok/s={r['in_tokens_per_s']:.0f} | out_tok/s={r['out_tokens_per_s']:.0f} | "
            f"peak_alloc={_format_bytes(r['peak_alloc'])}"
        )
        wandb.log({
            "batch_size": r['batch_size'],                        # x-axis
            "latency_per_query_ms": r['latency_ms_per_query'],    # y-axis
            "qps": r['qps'],
            "in_tokens_per_s": r['in_tokens_per_s'],
            "out_tokens_per_s": r['out_tokens_per_s'],
            "peak_alloc": r['peak_alloc'] if r['peak_alloc'] is not None else 0,
        })

    print("\n=== Context Length Scalability Microbenchmark ===")
    max_ctx = getattr(getattr(hf_model, "config", None), "max_position_embeddings", None) or 4096
    token_lengths = [t for t in [128, 256, 512, 1024, 2048, 4096] if t <= max_ctx]

    ctx_results = bench_context_length_scalability(
        model=hf_model,
        tokenizer=hf_tok,
        device=device,
        token_lengths=token_lengths,
        max_new_tokens=8,
        warmup=1,
        iters=5,
    )
    print(f"Using max_position_embeddings={max_ctx}")
    for i, r in enumerate(ctx_results):
        print(
            f"token_length={token_lengths[i]:>4} | avg_ms={r['avg_latency_ms']:.2f} | p95_ms={r['p95_latency_ms']:.2f} | "
            f"ms/token={r['ms_per_input_token']:.4f} | peak_alloc={_format_bytes(r['peak_alloc'])}"
        )
        wandb.log({
            "token_length": token_lengths[i],                    # x-axis
            "avg_latency_ms": r['avg_latency_ms'],                # y-axis
            "p95_latency_ms": r['p95_latency_ms'],
            "ms_per_input_token": r['ms_per_input_token'],
            "peak_alloc": r['peak_alloc'] if r['peak_alloc'] is not None else 0,
        })