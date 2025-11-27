"""
NeurIPS TP and MoE Configuration Testing

This test runs batch-1 benchmarks with different TP sizes and MoE backends
to find optimal configurations for each model.

Models tested:
- DeepSeek V3 0324 (MoE)
- Qwen 235B (MoE)
- Qwen 480B Coder (MoE)
- Minimax M2 (MoE)
- Kimi K2 Thinking (MoE)
- GLM 4.5-Air
- Llama 3.2

For each model, we test:
- TP4 and TP8
- For MoE models: flashinfer_trtllm and flashinfer_cutlass backends
"""

import unittest

from nightly_utils import NightlyBenchmarkRunner

from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

# Profile directory for all results
PROFILE_DIR = "performance_profiles_neurips_tp_moe"

# Batch size 1 only (as requested)
BATCH_SIZES = [1]
INPUT_LENS = (4096,)
OUTPUT_LENS = (512,)

# Models to test (all using FP8 quantization)
MODELS = {
    "deepseek-v3": {
        "path": "deepseek-ai/DeepSeek-V3",
        "is_moe": True,
        "extra_args": ["--trust-remote-code", "--quantization", "fp8"],
    },
    "qwen3-235b": {
        "path": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        "is_moe": True,
        "extra_args": ["--trust-remote-code", "--quantization", "fp8"],
    },
    "qwen3-coder-480b": {
        "path": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "is_moe": True,
        "extra_args": ["--trust-remote-code", "--quantization", "fp8"],
    },
    "minimax-m2": {
        "path": "MiniMaxAI/Minimax-M2",
        "is_moe": True,
        "extra_args": ["--trust-remote-code", "--quantization", "fp8"],
    },
    "kimi-k2": {
        "path": "moonshotai/Kimi-K2-Thinking",
        "is_moe": True,
        "extra_args": [
            "--trust-remote-code",
            "--quantization",
            "fp8",
            "--tool-call-parser",
            "kimi_k2",
            "--reasoning-parser",
            "kimi_k2",
        ],
    },
    "glm-4-6": {
        "path": "zai-org/GLM-4.6",
        "is_moe": False,
        "extra_args": ["--trust-remote-code", "--quantization", "fp8"],
    },
    "llama-32": {
        "path": "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "is_moe": False,
        "extra_args": ["--trust-remote-code", "--quantization", "fp8"],
    },
}

# TP sizes to test
TP_SIZES = [4, 8]

# MoE backends to test (for MoE models only)
MOE_BACKENDS = ["flashinfer_trtllm", "flashinfer_cutlass"]


class TestNeurIPSTPMoEConfigs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.runner = NightlyBenchmarkRunner(
            PROFILE_DIR, "NeurIPS TP and MoE Configuration Tests", cls.base_url
        )
        cls.runner.setup_profile_directory()

    def test_all_tp_moe_configs(self):
        """Run batch-1 benchmarks for all models with different TP/MoE configs."""

        all_results = []
        overall_success = True

        for model_key, model_info in MODELS.items():
            model_path = model_info["path"]
            is_moe = model_info["is_moe"]
            base_extra_args = model_info["extra_args"]

            print(f"\n{'='*80}")
            print(f"Testing {model_key}: {model_path}")
            print(f"MoE Model: {is_moe}")
            print(f"{'='*80}\n")

            for tp_size in TP_SIZES:
                # Build base server args with TP
                server_args = ["--tp", str(tp_size)] + base_extra_args

                if is_moe:
                    # Test each MoE backend for MoE models
                    for moe_backend in MOE_BACKENDS:
                        variant = f"TP{tp_size}_{moe_backend}"
                        moe_server_args = server_args + [
                            "--moe-runner-backend",
                            moe_backend,
                        ]

                        print(f"\nRunning {model_key} with {variant}...")

                        results, success = self.runner.run_benchmark_for_model(
                            model_path=model_path,
                            batch_sizes=BATCH_SIZES,
                            input_lens=INPUT_LENS,
                            output_lens=OUTPUT_LENS,
                            other_args=moe_server_args,
                            variant=variant,
                        )

                        if success and results:
                            all_results.extend(results)
                            self.runner.add_report(results)
                        else:
                            overall_success = False
                            print(f"⚠️  Failed: {model_key} {variant}")
                else:
                    # Test without MoE backend for non-MoE models
                    variant = f"TP{tp_size}"

                    print(f"\nRunning {model_key} with {variant}...")

                    results, success = self.runner.run_benchmark_for_model(
                        model_path=model_path,
                        batch_sizes=BATCH_SIZES,
                        input_lens=INPUT_LENS,
                        output_lens=OUTPUT_LENS,
                        other_args=server_args,
                        variant=variant,
                    )

                    if success and results:
                        all_results.extend(results)
                        self.runner.add_report(results)
                    else:
                        overall_success = False
                        print(f"⚠️  Failed: {model_key} {variant}")

        # Write final report to GitHub summary
        self.runner.write_final_report()

        print(f"\n{'='*80}")
        print(f"Total configurations tested: {len(all_results)}")
        print(f"Overall success: {overall_success}")
        print(f"{'='*80}\n")

        if not overall_success:
            self.fail("Some benchmark configurations failed. Check logs for details.")


if __name__ == "__main__":
    unittest.main()
