"""
app/system_profiler.py

System Profiler + Model Recommender.

Detects:
  - RAM (total, available)
  - GPU (vendor, VRAM via nvidia-smi / rocm-smi / Metal)
  - CPU (cores, architecture, AVX/AVX2/AVX512 support)
  - Disk (free space in models/ dir)
  - OS + Python info
  - Ollama availability + currently pulled models

Then recommends:
  - Which GGUF models to use for the LLM
  - Which quantisation tier (Q4, Q5, Q8)
  - Whether GPU offload is viable (n_gpu_layers)
  - Which Whisper model for STT
  - Whether to use Ollama or llama-cpp backend
  - Settings to put in .env

Run standalone:
    python -m app.system_profiler

Or call from code:
    from app.system_profiler import profile_system, recommend_models
    profile = profile_system()
    recommendations = recommend_models(profile)

Also registered as a tool:
    "system_profile": run_system_profile_tool
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── Data models ──────────────────────────────────────────────────────────────

@dataclass
class GPUInfo:
    vendor: str          # nvidia / amd / apple / intel / none
    name: str
    vram_mb: int         # 0 if unknown
    driver: str = ""
    cuda_version: str = ""
    compute_capability: str = ""


@dataclass
class SystemProfile:
    # CPU
    cpu_name: str = ""
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    cpu_arch: str = ""
    has_avx: bool = False
    has_avx2: bool = False
    has_avx512: bool = False

    # RAM
    ram_total_mb: int = 0
    ram_available_mb: int = 0

    # GPU
    gpus: list[GPUInfo] = field(default_factory=list)

    # Disk
    disk_free_gb: float = 0.0
    models_dir: str = ""

    # OS
    os_name: str = ""
    os_version: str = ""
    python_version: str = ""

    # Ollama
    ollama_available: bool = False
    ollama_version: str = ""
    ollama_models: list[str] = field(default_factory=list)

    # Derived
    primary_vram_mb: int = 0
    has_gpu: bool = False
    is_apple_silicon: bool = False

    def summary(self) -> str:
        gpu_str = f"{self.gpus[0].name} ({self.primary_vram_mb}MB VRAM)" if self.gpus else "No GPU"
        avx_str = " ".join(f for f, v in [
            ("AVX", self.has_avx), ("AVX2", self.has_avx2), ("AVX512", self.has_avx512)
        ] if v) or "none"
        return (
            f"CPU: {self.cpu_name} ({self.cpu_cores_physical}C/{self.cpu_cores_logical}T, {self.cpu_arch})\n"
            f"RAM: {self.ram_total_mb//1024}GB total, {self.ram_available_mb//1024}GB available\n"
            f"GPU: {gpu_str}\n"
            f"SIMD: {avx_str}\n"
            f"Disk free: {self.disk_free_gb:.1f}GB\n"
            f"OS: {self.os_name} {self.os_version}\n"
            f"Ollama: {'available (' + self.ollama_version + ')' if self.ollama_available else 'not found'}"
        )


@dataclass
class ModelRecommendation:
    tier: str                    # "high" / "mid" / "low" / "minimal"
    llm_model: str               # e.g. "qwen2.5:7b"
    llm_model_gguf: str          # e.g. "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    llm_quant: str               # e.g. "Q4_K_M"
    llm_size_gb: float
    n_gpu_layers: int            # 0 = CPU only, -1 = all layers on GPU
    n_ctx: int
    n_threads: int
    stt_model: str               # "tiny" / "base" / "small" / "medium" / "large-v3"
    backend: str                 # "ollama" / "llamacpp"
    reasoning: list[str]         # human-readable explanation per decision
    env_snippet: str             # ready-to-paste .env config
    warnings: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)


# ─── Profiling ────────────────────────────────────────────────────────────────

def _run(cmd: str, timeout: int = 5) -> str:
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _get_ram() -> tuple[int, int]:
    """Returns (total_mb, available_mb)."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total // (1024 * 1024), vm.available // (1024 * 1024)
    except ImportError:
        pass

    # Linux fallback
    if Path("/proc/meminfo").exists():
        info = Path("/proc/meminfo").read_text()
        total = int(re.search(r"MemTotal:\s+(\d+)", info).group(1)) // 1024
        avail_m = re.search(r"MemAvailable:\s+(\d+)", info)
        avail = int(avail_m.group(1)) // 1024 if avail_m else total // 2
        return total, avail

    return 0, 0


def _get_cpu() -> dict:
    info: dict = {
        "name": platform.processor() or "Unknown",
        "cores_physical": 1,
        "cores_logical": os.cpu_count() or 1,
        "arch": platform.machine(),
        "avx": False, "avx2": False, "avx512": False,
    }

    try:
        import psutil
        info["cores_physical"] = psutil.cpu_count(logical=False) or 1
        info["cores_logical"]  = psutil.cpu_count(logical=True)  or 1
    except ImportError:
        pass

    # CPU name
    if platform.system() == "Linux":
        cpuinfo = _run("grep -m1 'model name' /proc/cpuinfo")
        if cpuinfo:
            info["name"] = cpuinfo.split(":", 1)[-1].strip()
    elif platform.system() == "Darwin":
        info["name"] = _run("sysctl -n machdep.cpu.brand_string")

    # AVX flags
    if platform.system() == "Linux":
        flags = _run("grep -m1 flags /proc/cpuinfo")
        info["avx"]    = "avx"    in flags
        info["avx2"]   = "avx2"   in flags
        info["avx512"] = "avx512f" in flags
    elif platform.system() == "Darwin":
        info["avx"]  = bool(_run("sysctl -n hw.optional.avx1_0"))
        info["avx2"] = bool(_run("sysctl -n hw.optional.avx2_0"))

    return info


def _get_gpus() -> list[GPUInfo]:
    gpus = []

    # Apple Silicon — unified memory, treat as GPU with shared RAM
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        chip = _run("sysctl -n machdep.cpu.brand_string")
        ram_total, _ = _get_ram()
        gpus.append(GPUInfo(
            vendor="apple",
            name=chip or "Apple Silicon",
            vram_mb=ram_total,   # unified memory
        ))
        return gpus

    # NVIDIA
    nvidia_out = _run("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits")
    if nvidia_out:
        for line in nvidia_out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    vram = int(parts[1])
                except ValueError:
                    vram = 0
                gpus.append(GPUInfo(
                    vendor="nvidia",
                    name=parts[0],
                    vram_mb=vram,
                    driver=parts[2] if len(parts) > 2 else "",
                    compute_capability=parts[3] if len(parts) > 3 else "",
                ))

    # AMD ROCm
    if not gpus:
        rocm_out = _run("rocm-smi --showmeminfo vram --csv")
        if rocm_out:
            for line in rocm_out.strip().splitlines()[1:]:
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        vram = int(parts[1]) // (1024 * 1024)
                    except ValueError:
                        vram = 0
                    gpus.append(GPUInfo(vendor="amd", name=f"AMD GPU {len(gpus)}", vram_mb=vram))

    # Intel Arc (fallback via lspci)
    if not gpus:
        lspci = _run("lspci | grep -i 'vga\\|3d\\|display'")
        if "Intel" in lspci and "Arc" in lspci:
            gpus.append(GPUInfo(vendor="intel", name="Intel Arc", vram_mb=0))

    return gpus


def _get_disk_free(path: str) -> float:
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024 ** 3)
    except Exception:
        return 0.0


def _get_ollama() -> tuple[bool, str, list[str]]:
    version = _run("ollama --version")
    if not version:
        return False, "", []
    models_out = _run("ollama list")
    models = []
    for line in models_out.strip().splitlines()[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return True, version, models


def profile_system() -> SystemProfile:
    """Run all detectors and return a SystemProfile."""
    from app.config import settings

    profile = SystemProfile()

    # CPU
    cpu = _get_cpu()
    profile.cpu_name         = cpu["name"]
    profile.cpu_cores_physical = cpu["cores_physical"]
    profile.cpu_cores_logical  = cpu["cores_logical"]
    profile.cpu_arch         = cpu["arch"]
    profile.has_avx          = cpu["avx"]
    profile.has_avx2         = cpu["avx2"]
    profile.has_avx512       = cpu["avx512"]

    # RAM
    profile.ram_total_mb, profile.ram_available_mb = _get_ram()

    # GPU
    profile.gpus = _get_gpus()
    if profile.gpus:
        profile.has_gpu = True
        profile.primary_vram_mb = profile.gpus[0].vram_mb
        profile.is_apple_silicon = profile.gpus[0].vendor == "apple"

    # Disk
    models_dir = str(settings.models_dir)
    profile.models_dir = models_dir
    profile.disk_free_gb = _get_disk_free(models_dir)

    # OS
    profile.os_name    = platform.system()
    profile.os_version = platform.release()
    profile.python_version = sys.version.split()[0]

    # Ollama
    profile.ollama_available, profile.ollama_version, profile.ollama_models = _get_ollama()

    return profile


# ─── Model catalogue ──────────────────────────────────────────────────────────
# (name, size_gb, min_ram_gb, min_vram_gb_for_full_gpu, quality_tier, ollama_tag)

_LLM_MODELS = [
    # name                              size  min_ram  min_vram  tier      ollama_tag               quant
    ("Qwen2.5-0.5B-Instruct",          0.4,   2,       0,       "minimal", "qwen2.5:0.5b",          "Q4_K_M"),
    ("Qwen2.5-1.5B-Instruct",          0.9,   3,       0,       "minimal", "qwen2.5:1.5b",          "Q4_K_M"),
    ("Qwen2.5-3B-Instruct",            1.9,   4,       2,       "low",     "qwen2.5:3b",            "Q4_K_M"),
    ("Qwen2.5-7B-Instruct",            4.7,   8,       6,       "mid",     "qwen2.5:7b",            "Q4_K_M"),
    ("Qwen2.5-14B-Instruct",           8.7,   16,      10,      "high",    "qwen2.5:14b",           "Q4_K_M"),
    ("Qwen2.5-32B-Instruct",           19.0,  32,      22,      "high",    "qwen2.5:32b",           "Q4_K_M"),
    ("Llama-3.2-3B-Instruct",          2.0,   4,       2,       "low",     "llama3.2:3b",           "Q4_K_M"),
    ("Llama-3.1-8B-Instruct",          4.9,   8,       6,       "mid",     "llama3.1:8b",           "Q4_K_M"),
    ("Mistral-7B-Instruct-v0.3",       4.4,   8,       6,       "mid",     "mistral:7b",            "Q4_K_M"),
    ("Phi-3.5-mini-instruct",          2.2,   4,       2,       "low",     "phi3.5:mini",           "Q4_K_M"),
    ("Gemma-2-2B-it",                  1.6,   4,       2,       "low",     "gemma2:2b",             "Q4_K_M"),
    ("Gemma-2-9B-it",                  5.4,   10,      8,       "mid",     "gemma2:9b",             "Q4_K_M"),
    ("DeepSeek-R1-1.5B",               1.1,   3,       0,       "minimal", "deepseek-r1:1.5b",      "Q4_K_M"),
    ("DeepSeek-R1-7B",                 4.9,   8,       6,       "mid",     "deepseek-r1:7b",        "Q4_K_M"),
    ("DeepSeek-R1-14B",                8.9,   16,      10,      "high",    "deepseek-r1:14b",       "Q4_K_M"),
]

_STT_MODELS = [
    # (name, min_ram_mb, quality)
    ("tiny",      500,  "basic"),
    ("base",      700,  "good"),
    ("small",    1200,  "very good — recommended"),
    ("medium",   2500,  "excellent"),
    ("large-v3", 6000,  "best quality"),
]


# ─── Recommender ──────────────────────────────────────────────────────────────

def recommend_models(profile: SystemProfile) -> ModelRecommendation:
    """
    Given a system profile, return the best model configuration.
    """
    reasoning: list[str] = []
    warnings:  list[str] = []
    alternatives: list[str] = []

    ram_gb   = profile.ram_total_mb / 1024
    avail_gb = profile.ram_available_mb / 1024
    vram_mb  = profile.primary_vram_mb
    vram_gb  = vram_mb / 1024

    reasoning.append(f"System: {ram_gb:.0f}GB RAM ({avail_gb:.0f}GB free), "
                     f"{vram_gb:.0f}GB VRAM, {profile.cpu_cores_physical} CPU cores")

    # ── Backend choice ────────────────────────────────────────────────────────
    if profile.ollama_available:
        backend = "ollama"
        reasoning.append("Ollama is installed — using Ollama backend (easier model management)")
    else:
        backend = "llamacpp"
        reasoning.append("Ollama not found — using llama-cpp-python backend")
        warnings.append("Consider installing Ollama for easier model switching: https://ollama.com")

    # ── GPU offload ───────────────────────────────────────────────────────────
    if profile.is_apple_silicon:
        n_gpu_layers = -1   # Metal: offload everything
        reasoning.append("Apple Silicon detected — full Metal GPU acceleration (-1 layers)")
    elif profile.has_gpu and vram_mb >= 4000:
        # Estimate: ~1GB VRAM per 1.5B params at Q4
        n_gpu_layers = min(32, int(vram_mb / 200))
        reasoning.append(f"GPU with {vram_gb:.0f}GB VRAM — partial offload ({n_gpu_layers} layers)")
    else:
        n_gpu_layers = 0
        if profile.has_gpu:
            warnings.append(f"GPU detected but VRAM ({vram_mb}MB) too low for offload — using CPU")
        reasoning.append("CPU-only inference")

    # ── Thread count ──────────────────────────────────────────────────────────
    n_threads = max(1, profile.cpu_cores_physical - 2)
    reasoning.append(f"Using {n_threads} CPU threads (physical cores - 2 for OS/audio)")

    # ── LLM selection ─────────────────────────────────────────────────────────
    # Budget: use 60% of available RAM for the model (leave room for STT + embeddings)
    ram_budget_gb = avail_gb * 0.60
    reasoning.append(f"RAM budget for LLM: {ram_budget_gb:.1f}GB (60% of available)")

    # Filter models that fit in RAM budget
    viable = [
        m for m in _LLM_MODELS
        if m[1] <= ram_budget_gb   # size_gb fits
        and m[2] <= ram_gb         # min_ram fits
    ]

    if not viable:
        viable = [_LLM_MODELS[0]]  # absolute minimum fallback
        warnings.append("Very low RAM — using smallest possible model. Performance will be limited.")

    # Pick the best quality tier that fits
    tier_order = ["high", "mid", "low", "minimal"]
    chosen = None
    for tier in tier_order:
        tier_models = [m for m in viable if m[4] == tier]
        if tier_models:
            # Within a tier, pick largest that fits
            chosen = sorted(tier_models, key=lambda m: m[1], reverse=True)[0]
            break

    if not chosen:
        chosen = viable[-1]

    name, size_gb, min_ram, min_vram, tier, ollama_tag, quant = chosen
    reasoning.append(f"Selected model: {name} ({size_gb:.1f}GB, {tier} tier)")

    # Alternatives
    for m in viable:
        if m[5] != ollama_tag:
            alternatives.append(f"{m[5]} ({m[1]:.1f}GB, {m[4]} tier)")
    alternatives = alternatives[:4]

    # ── Context length ────────────────────────────────────────────────────────
    if ram_gb >= 32:
        n_ctx = 8192
    elif ram_gb >= 16:
        n_ctx = 6144
    elif ram_gb >= 8:
        n_ctx = 4096
    else:
        n_ctx = 2048
    reasoning.append(f"Context length: {n_ctx} tokens")

    # ── Quantisation advice ───────────────────────────────────────────────────
    if avail_gb >= size_gb * 1.5:
        quant = "Q5_K_M"
        reasoning.append("Sufficient RAM for Q5_K_M — better quality than Q4")
    elif avail_gb < size_gb * 1.1:
        quant = "Q3_K_M"
        reasoning.append("Tight RAM — using Q3_K_M to reduce memory pressure")
        warnings.append("RAM is tight. Close other applications for better performance.")

    gguf_name = f"{name}-{quant}.gguf"

    # ── STT selection ─────────────────────────────────────────────────────────
    # STT needs ~600MB RAM on top of the LLM
    ram_after_llm = avail_gb - size_gb
    stt_model = "tiny"
    for stt_name, stt_min_mb, stt_quality in reversed(_STT_MODELS):
        if ram_after_llm * 1024 >= stt_min_mb + 300:
            stt_model = stt_name
            reasoning.append(f"STT: Whisper {stt_name} ({stt_quality})")
            break

    # ── AVX warning ───────────────────────────────────────────────────────────
    if not profile.has_avx2 and backend == "llamacpp":
        warnings.append(
            "No AVX2 detected — llama-cpp-python will be slow. "
            "Use the generic build: pip install llama-cpp-python --force-reinstall"
        )

    # ── .env snippet ──────────────────────────────────────────────────────────
    env_lines = [
        f"# ── Generated by Tzar System Profiler ──",
        f"LLM_BACKEND={backend}",
        f"LLM_MODEL={ollama_tag}" if backend == "ollama" else f"LLM_MODEL_PATH=models/{gguf_name}",
        f"LLM_N_GPU_LAYERS={n_gpu_layers}",
        f"LLM_CONTEXT_LENGTH={n_ctx}",
        f"LLM_THREADS={n_threads}",
        f"STT_MODEL={stt_model}",
    ]
    if profile.is_apple_silicon:
        env_lines.append("# Apple Silicon: Metal acceleration is automatic")
    env_snippet = "\n".join(env_lines)

    return ModelRecommendation(
        tier=tier,
        llm_model=ollama_tag,
        llm_model_gguf=gguf_name,
        llm_quant=quant,
        llm_size_gb=size_gb,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_threads=n_threads,
        stt_model=stt_model,
        backend=backend,
        reasoning=reasoning,
        env_snippet=env_snippet,
        warnings=warnings,
        alternatives=alternatives,
    )


# ─── Formatted output ─────────────────────────────────────────────────────────

def format_report(profile: SystemProfile, rec: ModelRecommendation) -> str:
    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║       Tzar System Profile + Model Advisor       ║",
        "╚══════════════════════════════════════════════════╝",
        "",
        "── Hardware ─────────────────────────────────────────",
        profile.summary(),
        "",
        "── Recommendation ───────────────────────────────────",
        f"Tier:       {rec.tier.upper()}",
        f"Backend:    {rec.backend}",
        f"LLM:        {rec.llm_model}",
        f"GGUF:       {rec.llm_model_gguf}",
        f"Quant:      {rec.llm_quant}",
        f"Model size: {rec.llm_size_gb:.1f}GB",
        f"GPU layers: {rec.n_gpu_layers} ({'full offload' if rec.n_gpu_layers == -1 else 'CPU only' if rec.n_gpu_layers == 0 else 'partial'})",
        f"Context:    {rec.n_ctx} tokens",
        f"Threads:    {rec.n_threads}",
        f"STT:        Whisper {rec.stt_model}",
        "",
        "── Reasoning ────────────────────────────────────────",
    ]
    for r in rec.reasoning:
        lines.append(f"  • {r}")

    if rec.alternatives:
        lines += ["", "── Alternatives ─────────────────────────────────────"]
        for a in rec.alternatives:
            lines.append(f"  • {a}")

    if rec.warnings:
        lines += ["", "── Warnings ─────────────────────────────────────────"]
        for w in rec.warnings:
            lines.append(f"  ⚠  {w}")

    lines += [
        "",
        "── Setup commands ───────────────────────────────────",
    ]
    if rec.backend == "ollama":
        lines.append(f"  ollama pull {rec.llm_model}")
    else:
        lines.append(f"  # Download {rec.llm_model_gguf} to your models/ directory")
        lines.append(f"  # e.g. from https://huggingface.co/bartowski/{rec.llm_model_gguf.split('-Q')[0]}-GGUF")

    lines += [
        "",
        "── .env snippet ─────────────────────────────────────",
        rec.env_snippet,
        "",
    ]

    return "\n".join(lines)


# ─── Async tool wrapper ───────────────────────────────────────────────────────

async def run_system_profile_tool() -> str:
    """Tool function — registered in TOOL_REGISTRY as 'system_profile'."""
    loop = asyncio.get_running_loop()
    profile = await loop.run_in_executor(None, profile_system)
    rec     = recommend_models(profile)
    return format_report(profile, rec)


# ─── Standalone CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Profiling your system…\n")
    profile = profile_system()
    rec     = recommend_models(profile)
    print(format_report(profile, rec))
