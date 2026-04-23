# 🧪 ml-intern-sciml

An adaptation of [huggingface/ml-intern](https://github.com/huggingface/ml-intern)
for Physics-Informed Neural Network (PINN) collocation strategy research on HPC clusters.

> Developed as part of PhD research at **SMARTLab, BioRobotics Institute (Sant'Anna School of Advanced Studies / University of Pisa)**, within the **ERC DANTE project**, supervised by Prof. [Giovanni Stabile](https://www.giovannistabile.com).

---

## What This Is

The original `ml-intern` automates LLM post-training: it reads papers, finds datasets, trains models, and benchmarks on GPQA. This fork replaces that loop with a domain-specific version for PINNs research:

| Original ml-intern | ml-intern-sciml (this fork) |
|---|---|
| Reads arXiv for RLHF / fine-tuning papers | Reads arXiv for PINN collocation strategy papers |
| Finds datasets on HF Hub | No equivalent (data from OpenFOAM snapshots) |
| Trains LLMs via HF Jobs | Implements strategies in codebase, submits via SLURM |
| Benchmarks on GPQA | Benchmarks on relative L² error (Poisson + CFD) |
| Targets GPQA > 32% | Poisson: beat Functional (8.578×10⁻⁴, sanity check). CFD: beat Functional on Navier–Stokes — the real scientific target |

---

## Research Context

The host PhD research compares 9 collocation point selection strategies for PINNs across three stages:

- Phase 1: 1,728-configuration hyperparameter sweep on a Poisson benchmark
- Phase 2: 9-strategy comparison (Random, RAR, RAD, QR-DEIM, S-OPT-DEIM, PACMANN, Functional α-β-γ, Pareto-front, KL-Disagreement)
- Phase 3 (upcoming): Transfer to 2D urban CFD (incompressible Navier–Stokes, flow around buildings)

Current best: Functional strategy, rel L² = 8.578×10⁻⁴.

This agent searches recent literature for novel strategies that could outperform it, implements them, and tests them automatically.

---

## Architecture

```
ml-intern-sciml/
├── agent_sciml.py       — Entry point, system prompt, tool registration
├── domain_context.md    — Research background, codebase interface, success criteria
├── slurm_launcher.py    — Replaces HF Jobs: submits sbatch, polls squeue, reads metrics
├── strategy_writer.py   — Generates YAML configs, patches main_poisson.py, compares results
└── README.md
```

The PINN codebase (separate repo) lives at `<your_storage>/Collocation/` on the HPC cluster.

The PINN codebase will be available at **KabirBakhshaei/pinns-collocation** (coming soon) 
and currently lives at `<your_storage>/Collocation/` on the HPC cluster.

---

## Installation

```bash
git clone https://github.com/KabirBakhshaei/ml-intern-sciml.git
cd ml-intern-sciml
conda activate pinns_cuda
pip install smolagents litellm
```

---

## LLM Backend

The agent uses Gemma 4 31B via vLLM. Start the server before running the agent:

```bash
srun --ntasks=1 --cpus-per-task=4 --gpus=2 --time=02:00:00 --pty bash
conda activate pinns_cuda
vllm serve <your_storage>/models/gemma-4-31b-it \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --tool-call-parser pythonic \
    --port 8000
```

Wait for `Application startup complete.` then open a second terminal to run the agent.

---

## Usage

```bash
# Find and test a novel collocation strategy
python agent_sciml.py "Find a curvature-aware collocation strategy from 2025 PINN
                       literature and test it on the Poisson benchmark"

# Target the CFD transfer case
python agent_sciml.py "Search for strategies designed for boundary-layer problems
                       or flow with sharp gradients — implement the best one"

# Compare all current results
python agent_sciml.py --evaluate
```

---

## How It Works

```
1. SEARCH    arXiv query for PINN collocation papers (2023–2026)
      ↓
2. READ      Fetch abstracts and method sections; extract algorithm
      ↓
3. ASSESS    Is this strategy novel? Will it help on CFD?
      ↓
4. IMPLEMENT write_strategy_config() + write_strategy_implementation()
      ↓
5. SUBMIT    run_pinn_experiment() → sbatch → poll squeue
      ↓
6. EVALUATE  Read metrics.json → compare_all_results()
      ↓
7. REPORT    Scientific interpretation: why did it work or not?
```

---

## Success Criteria

Evaluation is two-tiered, reflecting the two-phase experimental design.

### Phase 2 — Poisson Benchmark (sanity check)

The Poisson problem has a smooth, globally uniform solution with no localised features. On this benchmark, the fixed random baseline is remarkably competitive (rel L² = 1.001×10⁻³, ranked 2nd out of 9 strategies). Beating it is non-trivial but not the primary scientific goal.

| Result | Interpretation |
|---|---|
| rel L² < 8.578×10⁻⁴ | ✅ Beats Functional — candidate for CFD transfer |
| rel L² < 1.001×10⁻³ | ✓ Beats random baseline — worth investigating |
| rel L² > 1.001×10⁻³ | ✗ Worse than random on smooth problem |

A strategy that wins here has passed a proof-of-implementation check. It does not yet prove scientific value.

### Phase 3 — CFD Benchmark (the real test)

The urban flow case (2D incompressible Navier–Stokes, flow around building obstacles) has sharp gradients at wall boundaries, recirculation zones in the wake, and thin boundary layers near solid surfaces. These are precisely the conditions where adaptive collocation is theoretically expected to outperform random sampling.

A strategy that merely won on Poisson may fail here, and vice versa. The CFD benchmark is where the scientific contribution lives.

Target: beat Functional on CFD (baseline TBD once Phase 3 experiments run).

---

## Differences from Original ml-intern

- SLURM instead of HF Jobs: `slurm_launcher.py` replaces `hf_jobs_launcher.py`
- Domain context: `domain_context.md` replaces the generic LLM post-training context
- Metrics: rel L² error replaces GPQA score
- No dataset search: PINN training data is generated from the PDE, not downloaded
- vLLM as orchestrator: the same Gemma 4 31B used for PaperRadar is reused here
- Strategy YAML interface: new strategies are YAML configs + Python branches, not fine-tuning scripts

---

## Citation

```bibtex
@software{bakhshaei2025mlintern_sciml,
  author    = {Bakhshaei, Kabir},
  title     = {ml-intern-sciml: Autonomous PINN Collocation Strategy Research Agent},
  year      = {2025},
  url       = {https://github.com/KabirBakhshaei/ml-intern-sciml},
  note      = {Adapted from huggingface/ml-intern.
               Developed at SMARTLab, BioRobotics Institute,
               Sant'Anna School of Advanced Studies / University of Pisa,
               within the ERC DANTE project (Grant 101115741)}
}
```

---

## Acknowledgements

- [huggingface/ml-intern](https://github.com/huggingface/ml-intern), original agent
- [smolagents](https://github.com/huggingface/smolagents), agent framework
- [vLLM](https://github.com/vllm-project/vllm), local LLM inference
- ERC DANTE project, research funding
