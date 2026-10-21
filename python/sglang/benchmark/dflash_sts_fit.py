from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path
from collections.abc import Sequence
from typing import Optional

import torch

from sglang.benchmark.dspark_sts_fit import expected_calibration_error
from sglang.srt.speculative.dflash_sts import (
    dflash_sts_temperature_grid,
    partition_sums_over_grid,
)
from sglang.srt.speculative.dspark_components.dspark_sts import DSparkStsCalibration

logger = logging.getLogger(__name__)


def fit_dflash_sts_temperatures(
    *,
    partitions: torch.Tensor,
    prefix_mask: torch.Tensor,
    grid: torch.Tensor,
    num_bins: int = 15,
) -> dict[str, list[float]]:
    """[n, gamma, len(grid)] partition sums + [n, gamma] correctness labels ->
    the per-position temperature whose survival probability is best calibrated.

    DFLASH's confidence at temperature grid[k] is 1 / partitions[..., k], so
    the search is a lookup along the collected grid rather than a closed-form
    rescaling; the collector pays for that by recording the whole grid.
    """
    partitions = partitions.to(torch.float64)
    prefix_mask = prefix_mask.to(torch.float64)
    num_samples, gamma, num_grid = partitions.shape
    if num_samples == 0:
        raise ValueError("fit_dflash_sts_temperatures requires at least one sample.")
    if prefix_mask.shape != (num_samples, gamma):
        raise ValueError(
            f"prefix_mask shape {tuple(prefix_mask.shape)} does not match the "
            f"partitions' [{num_samples}, {gamma}]."
        )
    if int(grid.numel()) != num_grid:
        raise ValueError(
            f"The shards carry {num_grid} grid points but the grid has "
            f"{int(grid.numel())}."
        )
    grid_values = grid.to(torch.float64).tolist()
    probs = partitions.reciprocal()
    # The uncalibrated confidence is the T=1 column; it is what the runtime
    # reports without a table, so it is the only honest "before" baseline.
    identity_index = min(range(num_grid), key=lambda k: abs(grid_values[k] - 1.0))
    # Ties are the common case once the survival has collapsed (every grid
    # point then scores a perfect ECE against an all-reject label), so the
    # search visits the grid from the identity outwards and only replaces on a
    # strict improvement. The arbitrary end of the grid would win otherwise.
    search_order = sorted(
        range(num_grid), key=lambda k: (abs(grid_values[k] - 1.0), k)
    )

    temperatures: list[float] = []
    ece_before: list[float] = []
    ece_after: list[float] = []

    survival_at_one = torch.ones(num_samples, dtype=torch.float64)
    survival_fitted = torch.ones(num_samples, dtype=torch.float64)
    for position in range(gamma):
        position_target = prefix_mask[:, position]
        survival_at_one = survival_at_one * probs[:, position, identity_index]
        ece_before.append(
            expected_calibration_error(
                probs=survival_at_one, targets=position_target, num_bins=num_bins
            )
        )

        best_index = search_order[0]
        best_survival = survival_fitted * probs[:, position, best_index]
        best_ece = expected_calibration_error(
            probs=best_survival, targets=position_target, num_bins=num_bins
        )
        for index in search_order[1:]:
            candidate_survival = survival_fitted * probs[:, position, index]
            candidate_ece = expected_calibration_error(
                probs=candidate_survival, targets=position_target, num_bins=num_bins
            )
            if candidate_ece < best_ece:
                best_ece = candidate_ece
                best_index = index
                best_survival = candidate_survival

        temperatures.append(float(grid_values[best_index]))
        ece_after.append(float(best_ece))
        survival_fitted = best_survival

    return {
        "temperatures": temperatures,
        "ece_before": ece_before,
        "ece_after": ece_after,
    }


def synthesize_dflash_sts_data(
    *,
    num_samples: int,
    planted_temperatures: Sequence[float],
    vocab: int = 2000,
    sharpness: tuple[float, float] = (6.0, 14.0),
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A self-consistent draft whose calibrated temperature is known.

    Each sample's logits are drawn at a random sharpness, so the confidence
    varies across samples -- a draft whose every row is equally peaked carries
    no signal about which temperature is right. Position j is then accepted
    with exactly the probability the planted temperature assigns it, which
    makes planted_temperatures the answer the fit has to recover.

    The identification degrades once the survival probability collapses: with
    few surviving positives left, every temperature scores the same
    calibration error. Keep the planted acceptance rate realistic.
    """
    generator = torch.Generator().manual_seed(seed)
    grid = dflash_sts_temperature_grid()
    gamma = len(planted_temperatures)
    planted_index = [
        int((grid - temperature).abs().argmin())
        for temperature in planted_temperatures
    ]

    partitions = torch.empty(num_samples, gamma, int(grid.numel()))
    planted_probs = torch.empty(num_samples, gamma)
    low, high = sharpness
    for position in range(gamma):
        scale = low + torch.rand(num_samples, 1, generator=generator) * (high - low)
        logits = torch.randn(num_samples, vocab, generator=generator) * scale
        sums, _ = partition_sums_over_grid(logits=logits, grid=grid)
        partitions[:, position, :] = sums
        planted_probs[:, position] = sums[:, planted_index[position]].reciprocal()

    alive = torch.ones(num_samples, dtype=torch.bool)
    prefix_mask = torch.zeros(num_samples, gamma)
    for position in range(gamma):
        drawn = torch.rand(num_samples, generator=generator)
        alive = alive & (drawn < planted_probs[:, position])
        prefix_mask[:, position] = alive.to(torch.float32)

    return partitions, prefix_mask, grid


def self_test(
    *,
    num_samples: int,
    planted_temperatures: Sequence[float],
    seed: int,
    num_bins: int,
) -> None:
    """Run collect -> fit on synthetic data and print what came back."""
    partitions, prefix_mask, grid = synthesize_dflash_sts_data(
        num_samples=num_samples,
        planted_temperatures=planted_temperatures,
        seed=seed,
    )
    result = fit_dflash_sts_temperatures(
        partitions=partitions, prefix_mask=prefix_mask, grid=grid, num_bins=num_bins
    )
    print(
        f"Synthetic closed loop over {num_samples} samples "
        f"(gamma={len(planted_temperatures)}, seed={seed})"
    )
    print("pos  planted  nearest  fitted   ratio  accept rate  ece_before  ece_after")
    rates = prefix_mask.mean(dim=0).tolist()
    for position, planted in enumerate(planted_temperatures):
        nearest = float(grid[int((grid - planted).abs().argmin())])
        fitted = result["temperatures"][position]
        print(
            f"{position:>3}  {planted:>7.4f}  {nearest:>7.4f}  {fitted:>6.4f}  "
            f"{fitted / nearest:>6.3f}  {rates[position]:>11.3f}  "
            f"{result['ece_before'][position]:>10.4f}  "
            f"{result['ece_after'][position]:>9.4f}"
        )
    print(
        "  ratio is against the nearest grid point, not the planted value: the "
        "fit can only return a temperature the grid carries."
    )


def load_collected_shards(
    *, data_glob: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shard_paths = sorted(glob.glob(data_glob))
    if not shard_paths:
        raise ValueError(f"No DFLASH STS data shards matched {data_glob!r}.")

    partition_shards: list[torch.Tensor] = []
    prefix_mask_shards: list[torch.Tensor] = []
    grid: Optional[torch.Tensor] = None
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu")
        shard_partitions = shard["partitions"]
        shard_prefix_mask = shard["prefix_mask"]
        shard_grid = shard["temperature_grid"]
        if shard_partitions.shape[:2] != shard_prefix_mask.shape:
            raise ValueError(
                f"Shard {shard_path!r} partitions / prefix_mask shape mismatch: "
                f"{tuple(shard_partitions.shape)} vs "
                f"{tuple(shard_prefix_mask.shape)}."
            )
        if grid is None:
            grid = shard_grid
        elif not torch.equal(shard_grid, grid):
            raise ValueError(
                f"Shard {shard_path!r} was collected on a different temperature "
                "grid than the earlier shards; a fit may only mix shards whose "
                "grids agree."
            )
        partition_shards.append(shard_partitions)
        prefix_mask_shards.append(shard_prefix_mask)

    return (
        torch.cat(partition_shards, dim=0),
        torch.cat(prefix_mask_shards, dim=0),
        grid,
    )


def fit(
    *,
    data_glob: str,
    out: Path,
    num_bins: int = 15,
    gamma: Optional[int] = None,
) -> None:
    partitions, prefix_mask, grid = load_collected_shards(data_glob=data_glob)
    resolved_gamma = int(partitions.shape[1])
    if gamma is not None and gamma != resolved_gamma:
        raise ValueError(
            f"Collected shards have gamma={resolved_gamma} but --gamma={gamma}."
        )
    num_samples = int(partitions.shape[0])

    result = fit_dflash_sts_temperatures(
        partitions=partitions,
        prefix_mask=prefix_mask,
        grid=grid,
        num_bins=num_bins,
    )
    calibration = DSparkStsCalibration(
        temperatures=result["temperatures"],
        dataset=data_glob,
        num_samples=num_samples,
        ece_before=result["ece_before"],
        ece_after=result["ece_after"],
    )
    out.write_text(calibration.to_json(), encoding="utf-8")

    print(
        f"Fit DFLASH STS temperatures over {num_samples} samples "
        f"(gamma={resolved_gamma}) -> {out}"
    )
    print("pos  temperature  ece_before  ece_after")
    for position in range(resolved_gamma):
        print(
            f"{position:>3}  {result['temperatures'][position]:>11.4f}  "
            f"{result['ece_before'][position]:>10.4f}  "
            f"{result['ece_after'][position]:>9.4f}"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Fit DFLASH Sequential Temperature Scaling (STS) calibration "
        "temperatures from collected draft-confidence partition sums. The table "
        "is loaded by --speculative-dspark-confidence-sts-path, but it is not "
        "interchangeable with a table fit for DSpark's confidence head: DFLASH "
        "applies the temperature inside a softmax, DSpark inside a sigmoid."
    )
    parser.add_argument(
        "--data-glob",
        default=None,
        help="Glob of collected .pt shards, each a dict with [n, gamma, grid] "
        "'partitions', [n, gamma] 'prefix_mask' and [grid] 'temperature_grid'.",
    )
    parser.add_argument(
        "--out",
        default=None,
        type=Path,
        help="Output STS calibration JSON path.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=15,
        help="Number of equal-width ECE bins.",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=None,
        help="Optional gamma override to validate the shards against.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Skip the shards and run the fit against synthetic data whose "
        "calibrated temperatures are known, to check the loop end to end.",
    )
    parser.add_argument(
        "--planted",
        type=float,
        nargs="+",
        default=[0.5, 0.8, 1.26, 2.0],
        help="--self-test only: the temperatures to plant, one per position.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4000,
        help="--self-test only: how many synthetic samples to draw.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="--self-test only: the draw's seed."
    )
    args = parser.parse_args()

    if args.self_test:
        self_test(
            num_samples=args.num_samples,
            planted_temperatures=args.planted,
            seed=args.seed,
            num_bins=args.num_bins,
        )
        return

    if args.data_glob is None or args.out is None:
        parser.error("--data-glob and --out are required without --self-test.")

    fit(
        data_glob=args.data_glob,
        out=args.out,
        num_bins=args.num_bins,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
