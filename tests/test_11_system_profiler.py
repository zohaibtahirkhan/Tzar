"""
Tests for the System Profiler.

Run: pytest tests/test_11_system_profiler.py -v

100% offline — no LLM, no GPU, no Ollama, no network.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# System Profiler
# ═══════════════════════════════════════════════════════════════════════════════

class TestSystemProfilerDetectors:
    def test_get_ram_returns_two_ints(self):
        from app.system_profiler import _get_ram
        total, avail = _get_ram()
        assert isinstance(total, int)
        assert isinstance(avail, int)
        assert total >= 0
        assert avail >= 0

    def test_get_cpu_returns_dict(self):
        from app.system_profiler import _get_cpu
        cpu = _get_cpu()
        assert "name" in cpu
        assert "cores_logical" in cpu
        assert "avx" in cpu
        assert isinstance(cpu["cores_logical"], int)
        assert cpu["cores_logical"] >= 1

    def test_get_disk_free_returns_float(self):
        from app.system_profiler import _get_disk_free
        free = _get_disk_free("/tmp")
        assert isinstance(free, float)
        assert free >= 0.0

    def test_get_disk_free_bad_path(self):
        from app.system_profiler import _get_disk_free
        free = _get_disk_free("/nonexistent/path/xyz")
        assert free == 0.0

    def test_get_ollama_returns_tuple(self):
        from app.system_profiler import _get_ollama
        available, version, models = _get_ollama()
        assert isinstance(available, bool)
        assert isinstance(version, str)
        assert isinstance(models, list)


class TestProfileSystem:
    def test_profile_system_returns_profile(self):
        from app.system_profiler import profile_system, SystemProfile
        with patch("app.system_profiler._get_ollama", return_value=(False, "", [])):
            profile = profile_system()
        assert isinstance(profile, SystemProfile)
        assert profile.cpu_cores_logical >= 1
        assert profile.ram_total_mb >= 0
        assert isinstance(profile.gpus, list)

    def test_profile_summary_is_string(self):
        from app.system_profiler import profile_system
        with patch("app.system_profiler._get_ollama", return_value=(False, "", [])):
            profile = profile_system()
        summary = profile.summary()
        assert isinstance(summary, str)
        assert "CPU" in summary
        assert "RAM" in summary


class TestModelRecommender:
    """Test the recommender with synthetic profiles."""

    def _make_profile(
        self,
        ram_total_gb=16, ram_avail_gb=10,
        vram_mb=0, has_gpu=False,
        is_apple=False, ollama=True,
        cpu_cores=8,
    ):
        from app.system_profiler import SystemProfile, GPUInfo
        p = SystemProfile()
        p.ram_total_mb     = ram_total_gb * 1024
        p.ram_available_mb = ram_avail_gb * 1024
        p.primary_vram_mb  = vram_mb
        p.has_gpu          = has_gpu
        p.is_apple_silicon = is_apple
        p.cpu_cores_physical = cpu_cores
        p.cpu_cores_logical  = cpu_cores * 2
        p.has_avx2         = True
        p.ollama_available  = ollama
        p.ollama_version    = "0.5.0" if ollama else ""
        p.disk_free_gb      = 50.0
        p.os_name           = "Linux"
        p.os_version        = "6.8"
        p.python_version    = "3.12.0"
        if has_gpu and not is_apple:
            p.gpus = [GPUInfo(vendor="nvidia", name="RTX 3080", vram_mb=vram_mb)]
        elif is_apple:
            p.gpus = [GPUInfo(vendor="apple", name="M2 Pro", vram_mb=ram_total_gb*1024)]
        return p

    def test_high_ram_system_gets_high_tier(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(ram_total_gb=64, ram_avail_gb=48)
        rec = recommend_models(profile)
        assert rec.tier in ("high", "mid")
        assert rec.llm_size_gb >= 4.0

    def test_low_ram_system_gets_small_model(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(ram_total_gb=4, ram_avail_gb=2)
        rec = recommend_models(profile)
        assert rec.llm_size_gb <= 2.0

    def test_ollama_backend_when_available(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(ollama=True)
        rec = recommend_models(profile)
        assert rec.backend == "ollama"

    def test_llamacpp_backend_when_ollama_absent(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(ollama=False)
        rec = recommend_models(profile)
        assert rec.backend == "llamacpp"
        assert any("Ollama" in w for w in rec.warnings)

    def test_apple_silicon_gets_full_gpu_offload(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(ram_total_gb=32, ram_avail_gb=24,
                                     is_apple=True, has_gpu=True, vram_mb=32*1024)
        rec = recommend_models(profile)
        assert rec.n_gpu_layers == -1

    def test_cpu_only_system_gets_zero_gpu_layers(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(has_gpu=False, vram_mb=0)
        rec = recommend_models(profile)
        assert rec.n_gpu_layers == 0

    def test_thread_count_is_cores_minus_two(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(cpu_cores=8)
        rec = recommend_models(profile)
        assert rec.n_threads == 6

    def test_thread_count_minimum_one(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(cpu_cores=2)
        rec = recommend_models(profile)
        assert rec.n_threads >= 1

    def test_stt_model_is_valid(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile()
        rec = recommend_models(profile)
        assert rec.stt_model in ("tiny", "base", "small", "medium", "large-v3")

    def test_high_ram_gets_better_stt(self):
        from app.system_profiler import recommend_models
        low  = recommend_models(self._make_profile(ram_total_gb=4,  ram_avail_gb=2))
        high = recommend_models(self._make_profile(ram_total_gb=32, ram_avail_gb=24))
        stt_order = ["tiny", "base", "small", "medium", "large-v3"]
        assert stt_order.index(high.stt_model) >= stt_order.index(low.stt_model)

    def test_env_snippet_contains_backend(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile()
        rec = recommend_models(profile)
        assert "LLM_BACKEND" in rec.env_snippet

    def test_env_snippet_contains_threads(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile()
        rec = recommend_models(profile)
        assert "LLM_THREADS" in rec.env_snippet

    def test_reasoning_is_nonempty(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile()
        rec = recommend_models(profile)
        assert len(rec.reasoning) > 0

    def test_alternatives_list(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(ram_total_gb=32, ram_avail_gb=20)
        rec = recommend_models(profile)
        assert isinstance(rec.alternatives, list)

    def test_format_report_is_string(self):
        from app.system_profiler import recommend_models, format_report
        with patch("app.system_profiler._get_ollama", return_value=(False, "", [])):
            from app.system_profiler import profile_system
            profile = profile_system()
        rec = recommend_models(profile)
        report = format_report(profile, rec)
        assert isinstance(report, str)
        assert "Tzar System Profile" in report
        assert "Recommendation" in report
        assert "env snippet" in report.lower() or ".env" in report

    def test_very_low_ram_does_not_crash(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(ram_total_gb=2, ram_avail_gb=1)
        rec = recommend_models(profile)   # should not raise
        assert rec.llm_size_gb > 0

    def test_nvidia_gpu_partial_offload(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(has_gpu=True, vram_mb=8192)
        rec = recommend_models(profile)
        assert rec.n_gpu_layers > 0


class TestSystemProfilerTool:
    @pytest.mark.asyncio
    async def test_tool_returns_string(self):
        from app.system_profiler import run_system_profile_tool
        with patch("app.system_profiler._get_ollama", return_value=(False, "", [])):
            result = await run_system_profile_tool()
        assert isinstance(result, str)
        assert len(result) > 100
