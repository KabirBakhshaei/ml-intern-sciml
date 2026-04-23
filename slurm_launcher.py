"""
slurm_launcher.py — SLURM job launcher for ml-intern-sciml
===========================================================
Replaces Hugging Face Jobs in the original ml-intern.
Submits PINN experiments to hpcsrv, polls until completion,
and returns the metrics.json output.

Usage (as an ml-intern tool):
    from slurm_launcher import SlurmPinnLauncher
    launcher = SlurmPinnLauncher()
    result = await launcher.run(config_path="configs/phase2/phase2_mystrategy.yaml")

Direct test:
    python slurm_launcher.py --config configs/phase2/phase2_functional.yaml
"""

import asyncio
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ─── Cluster configuration ───────────────────────────────────────────────────
CLUSTER_USER     = "your_username"
CONDA_ENV        = "pinns_cuda"
PROJECT_DIR      = "<your_storage>/Collocation"
RESULTS_DIR      = os.path.join(PROJECT_DIR, "Results")
LOGS_DIR         = os.path.join(PROJECT_DIR, "logs")
MAIN_SCRIPT      = os.path.join(PROJECT_DIR, "main_poisson.py")

# SLURM defaults for a single PINN run (~30s on H100 for 3k epochs)
SLURM_DEFAULTS = {
    "ntasks":         1,
    "cpus_per_task":  4,
    "gpus":           1,
    "time":           "00:15:00",   # 15 min: generous for 3k–8k epochs
    "mem":            "16G",
}

# Polling interval and timeout
POLL_INTERVAL_SEC = 15
JOB_TIMEOUT_SEC   = 900   # 15 minutes hard limit


# ─── Result dataclass ─────────────────────────────────────────────────────────
@dataclass
class PinnResult:
    """Parsed output of a completed PINN run."""
    strategy:               str
    run_tag:                str
    rel_l2_grid:            float
    rel_linf_grid:          float
    best_rel_l2:            float
    best_epoch:             int
    n_collocation_final:    int
    train_wallclock_seconds: float
    metrics_path:           str
    beats_functional:       bool       # rel_l2_grid < 8.578e-4
    beats_random:           bool       # rel_l2_grid < 1.001e-3

    def summary(self) -> str:
        status = "✅ BEATS FUNCTIONAL" if self.beats_functional \
            else ("✓ beats random" if self.beats_random else "✗ worse than random")
        return (
            f"Strategy:    {self.strategy}\n"
            f"Rel L²:      {self.rel_l2_grid:.3e}   {status}\n"
            f"Rel L∞:      {self.rel_linf_grid:.3e}\n"
            f"Best val L²: {self.best_rel_l2:.3e}  (epoch {self.best_epoch})\n"
            f"N_c final:   {self.n_collocation_final}\n"
            f"Time:        {self.train_wallclock_seconds:.1f}s\n"
            f"Metrics:     {self.metrics_path}"
        )


# ─── SLURM launcher ──────────────────────────────────────────────────────────
class SlurmPinnLauncher:
    """
    Submits a PINN config to SLURM, waits for completion,
    and returns a parsed PinnResult.
    """

    def __init__(
        self,
        project_dir: str = PROJECT_DIR,
        conda_env:   str = CONDA_ENV,
        slurm_opts:  Optional[dict] = None,
    ):
        self.project_dir = project_dir
        self.conda_env   = conda_env
        self.opts        = {**SLURM_DEFAULTS, **(slurm_opts or {})}

    # ── public API ────────────────────────────────────────────────────────────

    async def run(self, config_path: str) -> PinnResult:
        """
        Submit a job for config_path, poll until done, return PinnResult.
        Raises RuntimeError if the job fails or times out.
        """
        config_path = os.path.abspath(config_path)
        strategy    = self._strategy_from_config(config_path)
        job_name    = f"pinn_{strategy}"
        log_out     = os.path.join(LOGS_DIR, f"{job_name}_%j.out")
        log_err     = os.path.join(LOGS_DIR, f"{job_name}_%j.err")

        os.makedirs(LOGS_DIR, exist_ok=True)

        wrap_cmd = (
            f"source ~/.bashrc && "
            f"conda activate {self.conda_env} && "
            f"cd {self.project_dir} && "
            f"python {MAIN_SCRIPT} --config {config_path}"
        )

        sbatch_cmd = [
            "sbatch",
            f"--job-name={job_name}",
            f"--ntasks={self.opts['ntasks']}",
            f"--cpus-per-task={self.opts['cpus_per_task']}",
            f"--gpus={self.opts['gpus']}",
            f"--time={self.opts['time']}",
            f"--mem={self.opts['mem']}",
            f"--output={log_out}",
            f"--error={log_err}",
            f"--wrap={wrap_cmd}",
        ]

        # Submit
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"sbatch failed:\n{result.stderr.strip()}"
            )

        job_id = self._parse_job_id(result.stdout)
        print(f"[slurm_launcher] Submitted job {job_id} for strategy '{strategy}'")

        # Poll
        await self._wait_for_job(job_id)

        # Read result
        metrics_path = self._find_metrics(strategy)
        return self._parse_metrics(metrics_path)

    def run_sync(self, config_path: str) -> PinnResult:
        """Synchronous wrapper around run() for non-async callers."""
        return asyncio.get_event_loop().run_until_complete(self.run(config_path))

    # ── internals ─────────────────────────────────────────────────────────────

    def _strategy_from_config(self, config_path: str) -> str:
        """Extract strategy name from config filename."""
        stem = Path(config_path).stem   # e.g. "phase2_mystrategy"
        return stem.replace("phase2_", "")

    def _parse_job_id(self, sbatch_stdout: str) -> str:
        """Parse 'Submitted batch job 12345' → '12345'."""
        m = re.search(r"(\d+)", sbatch_stdout)
        if not m:
            raise RuntimeError(f"Could not parse job ID from: {sbatch_stdout!r}")
        return m.group(1)

    async def _wait_for_job(self, job_id: str):
        """Poll squeue until the job is no longer running."""
        deadline = time.time() + JOB_TIMEOUT_SEC
        while time.time() < deadline:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True, text=True,
            )
            state = result.stdout.strip()
            if state == "":
                # Job no longer in queue → completed (or failed)
                print(f"[slurm_launcher] Job {job_id} finished.")
                return
            print(f"[slurm_launcher] Job {job_id} state: {state} — waiting...")
            await asyncio.sleep(POLL_INTERVAL_SEC)
        raise RuntimeError(
            f"Job {job_id} did not complete within {JOB_TIMEOUT_SEC}s."
        )

    def _find_metrics(self, strategy: str) -> str:
        """
        Find metrics.json for the strategy.
        Searches Results/ for a directory whose name contains the strategy.
        """
        results_root = Path(self.project_dir) / "Results"
        candidates   = sorted(
            results_root.glob(f"*{strategy}*/metrics.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(
                f"No metrics.json found for strategy '{strategy}' in {results_root}.\n"
                f"Check SLURM logs in {LOGS_DIR}/"
            )
        return str(candidates[0])

    def _parse_metrics(self, metrics_path: str) -> PinnResult:
        """Load metrics.json and return a structured PinnResult."""
        with open(metrics_path) as f:
            m = json.load(f)

        l2 = float(m["rel_l2_grid"])
        return PinnResult(
            strategy               = m.get("strategy", "unknown"),
            run_tag                = m.get("run_tag", "unknown"),
            rel_l2_grid            = l2,
            rel_linf_grid          = float(m.get("rel_linf_grid", float("nan"))),
            best_rel_l2            = float(m.get("best_rel_l2", float("nan"))),
            best_epoch             = int(m.get("best_epoch", -1)),
            n_collocation_final    = int(m.get("N_COLLOCATION_FINAL",
                                              m.get("N_COLLOCATION", -1))),
            train_wallclock_seconds = float(m.get("train_wallclock_seconds", 0)),
            metrics_path           = metrics_path,
            beats_functional       = l2 < 8.578e-4,
            beats_random           = l2 < 1.001e-3,
        )


# ─── CLI test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test SLURM launcher")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    launcher = SlurmPinnLauncher()
    result   = launcher.run_sync(args.config)
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(result.summary())
