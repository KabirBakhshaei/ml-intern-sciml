"""
strategy_writer.py — Utilities for writing strategy configs and reading results
===============================================================================
Used by agent_sciml.py tools to:
  - Generate Phase 2 YAML configs for new strategies
  - Patch main_poisson.py with new strategy implementations
  - Read and compare all metrics.json files in Results/
"""

import json
import os
import textwrap
import yaml
from pathlib import Path
from typing import Optional


# ─── Shared Phase 2 base config (matches thesis sweet-spot) ──────────────────
PHASE2_BASE = {
    "reproducibility": {
        "torch_seed": 0,
        "numpy_seed": 0,
    },
    "model": {
        "name":       "MLP",
        "hidden":     128,
        "layers":     6,
        "activation": "tanh",
    },
    "optimizer": {
        "name":                   "SOAP",
        "lr":                     1e-3,
        "betas":                  [0.95, 0.95],
        "weight_decay":           0.01,
        "precondition_frequency": 10,
    },
    "sampling": {
        "n_data":        5000,
        "n_collocation": 200,
        "n_bc":          400,
        "n_test":        10000,
    },
    "training": {
        "epochs":      3000,
        "w_data":      1.0,
        "w_pde":       1.0,
        "w_bc":        50.0,
        "print_every": 500,
    },
    "evaluation": {
        "grid_res": 400,
    },
}

# Known baselines for comparison
BASELINES = {
    "functional":  8.578e-4,
    "random":      1.001e-3,
    "pacmann":     1.082e-3,
    "disagreement":1.179e-3,
    "rar":         1.633e-3,
    "qrdeim":      2.662e-3,
    "pareto":      2.699e-3,
    "soptdeim":    3.085e-3,
    "rad":         3.702e-3,
}


class StrategyWriter:

    def __init__(
        self,
        project_dir:  str = "/storage/home/k.bakhshaei/Collocation",
        configs_dir:  str = "configs/phase2",
        results_dir:  str = "Results",
        main_script:  str = "main_poisson.py",
    ):
        self.project_dir = Path(project_dir)
        self.configs_dir = self.project_dir / configs_dir
        self.results_dir = self.project_dir / results_dir
        self.main_script = self.project_dir / main_script

    # ── Write YAML config ─────────────────────────────────────────────────────

    def write_config(
        self,
        strategy_name:  str,
        strategy_block: dict,
        notes:          str = "",
    ) -> str:
        """
        Write a Phase 2 YAML config for a new strategy.
        Returns the path to the created file.
        """
        self.configs_dir.mkdir(parents=True, exist_ok=True)

        tag      = f"phase2_{strategy_name}"
        out_path = self.configs_dir / f"{tag}.yaml"

        cfg = {
            "run": {
                "tag":   tag,
                "notes": notes or f"Phase 2: {strategy_name} strategy",
            },
            **PHASE2_BASE,
            "strategy": strategy_block,
        }

        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        print(f"[strategy_writer] Written config: {out_path}")
        return str(out_path)

    # ── Patch main_poisson.py ─────────────────────────────────────────────────

    def write_implementation(
        self,
        strategy_name: str,
        python_code:   str,
        description:   str = "",
    ) -> str:
        """
        Append a new strategy branch to main_poisson.py.
        Inserts after the last elif strategy_name == "..." block.
        Returns a status message.
        """
        with open(self.main_script, "r") as f:
            source = f.read()

        # Safety: don't add the same strategy twice
        if f'strategy_name == "{strategy_name}"' in source:
            return (
                f"Strategy '{strategy_name}' already exists in main_poisson.py. "
                "Edit it manually if you want to change the implementation."
            )

        # Find the insertion point: after the last elif strategy_name block
        # in the training loop update section
        insert_marker = "# ── END OF STRATEGY BLOCKS ──"
        if insert_marker not in source:
            return (
                f"Could not find insertion marker '{insert_marker}' in "
                f"main_poisson.py. Add it manually after the last strategy block."
            )

        indent = "        "  # 8 spaces (inside epoch loop)
        code_lines = textwrap.indent(
            python_code.strip(),
            indent + "    "
        )

        new_block = (
            f"\n{indent}# ── {strategy_name.upper()} ({'─'*40})\n"
            f"{indent}# {description}\n"
            f"{indent}elif strategy_name == \"{strategy_name}\":\n"
            f"{code_lines}\n"
        )

        patched = source.replace(insert_marker, new_block + "\n" + insert_marker)

        # Write backup first
        backup = str(self.main_script) + ".bak"
        with open(backup, "w") as f:
            f.write(source)

        with open(self.main_script, "w") as f:
            f.write(patched)

        return (
            f"Patched main_poisson.py with '{strategy_name}' strategy block. "
            f"Backup saved to {backup}."
        )

    # ── Compare results ───────────────────────────────────────────────────────

    def compare_results(self, tag_prefix: Optional[str] = None) -> str:
        """
        Read all metrics.json under Results/ and return a formatted
        comparison table sorted by rel_l2_grid.
        """
        runs = []
        for path in sorted(self.results_dir.rglob("metrics.json")):
            try:
                with open(path) as f:
                    m = json.load(f)
                if tag_prefix and not m.get("run_tag", "").startswith(tag_prefix):
                    continue
                runs.append(m)
            except (json.JSONDecodeError, KeyError):
                pass

        if not runs:
            return f"No metrics.json files found in {self.results_dir}/"

        runs.sort(key=lambda r: r.get("rel_l2_grid", float("inf")))

        lines = [
            f"{'Strategy':<20} {'Rel L²':>12} {'Rel L∞':>12} "
            f"{'Best L²':>12} {'N_c':>6} {'Time(s)':>8}  Status",
            "─" * 90,
        ]

        functional_l2 = BASELINES["functional"]
        random_l2     = BASELINES["random"]

        for r in runs:
            tag   = r.get("run_tag", r.get("strategy", "?"))
            l2    = r.get("rel_l2_grid",      float("nan"))
            linf  = r.get("rel_linf_grid",    float("nan"))
            bl2   = r.get("best_rel_l2",      float("nan"))
            nc    = r.get("N_COLLOCATION_FINAL", r.get("N_COLLOCATION", -1))
            t     = r.get("train_wallclock_seconds", 0)

            if l2 < functional_l2:
                status = "✅ NEW BEST"
            elif l2 < random_l2:
                status = "✓ > random"
            else:
                status = "✗"

            lines.append(
                f"{tag:<20} {l2:>12.3e} {linf:>12.3e} "
                f"{bl2:>12.3e} {nc:>6} {t:>8.1f}  {status}"
            )

        lines.append("─" * 90)
        lines.append(f"Total: {len(runs)} runs")
        lines.append(f"\nBaseline reference:")
        lines.append(f"  Functional: {functional_l2:.3e} (target to beat)")
        lines.append(f"  Random:     {random_l2:.3e}")

        return "\n".join(lines)
