"""
agent_sciml.py — ml-intern-sciml entry point
=============================================
Adapted from huggingface/ml-intern for Physics-Informed Neural Network
collocation strategy research on an HPC cluster.

Original: reads arXiv → finds HF datasets → trains LLMs → benchmarks on GPQA
This fork: reads arXiv → extracts PINN strategies → implements in codebase
           → submits SLURM jobs → benchmarks on Poisson / CFD L² error

Usage:
    python agent_sciml.py "Find a novel collocation strategy from 2025 PINN
                           literature that could beat Functional"

    python agent_sciml.py --evaluate   # compare all strategies in Results/
"""

import json
import os
from pathlib import Path

# Load domain context
_DOMAIN_CONTEXT_PATH = Path(__file__).parent / "domain_context.md"
with open(_DOMAIN_CONTEXT_PATH) as f:
    DOMAIN_CONTEXT = f.read()


# ─── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""
IMPORTANT: You must always write valid Python code inside <code>...</code> tags.
To call a tool, call it as a Python function. For example:
    result = search_arxiv(query="PINN collocation curvature 2025")
    print(result)
Never use formats like call:tool_name{{...}} — always use Python function calls.

You are ml-intern-sciml, an autonomous research agent specialising in
Physics-Informed Neural Networks (PINNs) for computational fluid dynamics.

Your capabilities:
1. Search arXiv for recent PINN collocation strategy papers using search_arxiv
2. Read and understand algorithm descriptions in papers
3. Implement new strategies as YAML configs + Python code additions
4. Submit experiments to the SLURM cluster via run_pinn_experiment
5. Read metrics.json output files to evaluate results
6. Compare against existing baselines and explain findings scientifically

Your research context:
{'='*70}
{DOMAIN_CONTEXT}
{'='*70}

Workflow for each task:
1. SEARCH: call search_arxiv with relevant keywords
2. READ: extract the algorithm from the abstract
3. ASSESS: is this strategy novel (not already in the 9 existing ones)?
4. IMPLEMENT: call write_strategy_config then write_strategy_implementation
5. SUBMIT: call run_pinn_experiment with the config path
6. EVALUATE: compare rel_l2_grid vs Functional (8.578e-4)
7. REPORT: explain the result scientifically

Constraints:
- Always use the Phase 2 shared base config (128x6 MLP, SOAP, lr=1e-3, 3000 epochs)
- Point budget B=800 for add-based strategies; fixed N_c=200 for replace-based
- Think about CFD transferability: will this strategy handle heterogeneous residuals?
- Never use open() to write files directly — always use write_strategy_config and write_strategy_implementation tools
"""


# ─── Tool definitions ─────────────────────────────────────────────────────────

def create_builtin_tools():
    from smolagents import Tool
    from slurm_launcher import SlurmPinnLauncher
    from strategy_writer import StrategyWriter

    launcher = SlurmPinnLauncher()
    writer   = StrategyWriter()

    class SearchArxiv(Tool):
        name = "search_arxiv"
        description = (
            "Search arXiv for recent papers on PINN collocation strategies. "
            "Returns titles, abstracts and IDs of the most relevant results."
        )
        inputs = {
            "query": {
                "type": "string",
                "description": "Search query, e.g. 'adaptive collocation PINN curvature 2025'"
            }
        }
        output_type = "string"
        

        def forward(self, query: str) -> str:
            import requests
            import xml.etree.ElementTree as ET
            url = (
                f"http://export.arxiv.org/api/query"
                f"?search_query=abs:{query.replace(' ', '+AND+abs:')}"
                f"&start=0&max_results=8"
                f"&sortBy=relevance&sortOrder=descending"
            )
            resp = requests.get(url, timeout=30)
            root = ET.fromstring(resp.content)
            ns = "{http://www.w3.org/2005/Atom}"
            results = []
            for entry in root.findall(f"{ns}entry"):
                title    = entry.find(f"{ns}title").text.strip()
                summary  = entry.find(f"{ns}summary").text.strip()[:500]
                arxiv_id = entry.find(f"{ns}id").text.strip()
                results.append(f"ID: {arxiv_id}\nTitle: {title}\nAbstract: {summary}")
            return "\n---\n".join(results) if results else "No results found."


    class RunPinnExperiment(Tool):
        name = "run_pinn_experiment"
        description = (
            "Submit a PINN strategy experiment to the SLURM cluster. "
            "Provide the path to a YAML config file. "
            "Returns rel_l2_grid, time, and comparison against baselines."
        )
        inputs = {
            "config_path": {
                "type": "string",
                "description": "Path to the YAML config file, e.g. configs/phase2/phase2_mystrategy.yaml"
            }
        }
        output_type = "string"

        def forward(self, config_path: str) -> str:
            result = launcher.run_sync(config_path)
            return result.summary()

    class WriteStrategyConfig(Tool):
        name = "write_strategy_config"

        description = (
            "Write a Phase 2 YAML config for a new collocation strategy. "
            "Call it like this: write_strategy_config("
            "strategy_name='hessian_adaptive', "
            "config='{\"name\": \"hessian_adaptive\", \"refine_every\": 100, \"n_candidates\": 20000, \"n_add\": 200, \"point_budget\": 800}', "
            "notes='Hessian norm adaptive sampling'). "
            "config must be a JSON string or dict. Returns the path to the created YAML file."
        )

        inputs = {
            "strategy_name": {
                "type": "string",
                "description": "Short name, e.g. 'curvature_hessian'"
            },
            "config": {
                "type": "string",
                "description": "JSON string of strategy params. If unsure, pass '{}'",
                "nullable": True
            },

            "notes": {
                "type": "string",
                "description": "One-line description of this strategy",
                "nullable": True
            }
        }
        output_type = "string"


        def forward(self, strategy_name: str, config: str = "{}", notes: str = "") -> str:
        
            import json
            import yaml as _yaml
            try:
                block = json.loads(config)
            except (json.JSONDecodeError, ValueError):
                try:
                    block = _yaml.safe_load(config)
                except Exception:
                    block = {}
            if not block:
                block = {"name": strategy_name, "refine_every": 100,
                        "n_candidates": 20000, "n_add": 200, "point_budget": 800}
            if "name" not in block:
                block["name"] = strategy_name
            return writer.write_config(strategy_name, block, notes)
    
    class WriteStrategyImplementation(Tool):
        name = "write_strategy_implementation"
        description = (
            "Append a new strategy to main_poisson.py. "
            "Call it like this: write_strategy_implementation("
            "strategy_name='hessian_adaptive', "
            "python_code='...the elif block code...', "
            "description='Hessian norm adaptive sampling'). "
            "The argument is python_code, not code or implementation_code."
        )
        inputs = {
            "strategy_name": {
                "type": "string",
                "description": "Short name matching the YAML strategy block name"
            },
            "implementation_code": {
                "type": "string",
                "description": "Python code block for the strategy update step"
            },
            "notes": {
                "type": "string",
                "description": "Scientific description of the strategy",
                "nullable": True
            }
        }

        output_type = "string"

        def forward(self, strategy_name: str, implementation_code: str, notes: str = "") -> str:
            return writer.write_implementation(strategy_name, implementation_code, notes)

    class CompareAllResults(Tool):
        name = "compare_all_results"
        description = (
            "Read all metrics.json files in Results/ and return a "
            "comparison table sorted by rel_l2_grid (lower is better)."
        )
        inputs = {
            "tag_prefix": {
                "type": "string",
                "description": "Optional: filter to runs whose tag starts with this prefix",
                "nullable": True
            }
        }
        output_type = "string"

        def forward(self, tag_prefix: str = "") -> str:
            return writer.compare_results(tag_prefix or None)

    return [
        SearchArxiv(),
        RunPinnExperiment(),
        WriteStrategyConfig(),
        WriteStrategyImplementation(),
        CompareAllResults(),
    ]


# ─── Main CLI ─────────────────────────────────────────────────────────────────


def main():
    import argparse
    from smolagents import CodeAgent, LiteLLMModel
    
    class PatchedLiteLLMModel(LiteLLMModel):
        def generate(self, *args, **kwargs):
            response = super().generate(*args, **kwargs)
            if hasattr(response, 'content') and response.content:
                content = response.content
                content = content.replace("<code >", "<code>")
                content = content.replace("<code\n", "<code>\n")
                # Fix call: format → proper code block
                import re
                content = re.sub(
                    r'call:(\w+)\(([^)]*)\)',
                    r'<code>\n\1(\2)\n</code>',
                    content
                )
                response.content = content
            return response

    parser = argparse.ArgumentParser(description="ml-intern-sciml: PINN research agent")
    parser.add_argument("task", nargs="?", help="Research task to perform")
    parser.add_argument("--evaluate", action="store_true",
                        help="Compare all existing results and exit")
    parser.add_argument("--model", default="vllm/gemma-4-31b-it",
                        help="LLM backend label (display only)")
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="vLLM server URL")
    args = parser.parse_args()

    if args.evaluate:
        from strategy_writer import StrategyWriter
        print(StrategyWriter().compare_results())
        return

    if not args.task:
        print(__doc__)
        return

    model = PatchedLiteLLMModel(
        model_id="openai//storage/home/k.bakhshaei/models/gemma-4-31b-it",
        api_base=args.base_url,
        api_key="dummy",
    )

    tools = create_builtin_tools()

    agent = CodeAgent(
        tools=tools,
        model=model,
        max_steps=20,
        additional_authorized_imports=[
            "requests", "os", "json", "subprocess",
            "yaml", "torch", "numpy", "scipy",
            "xml", "xml.etree", "xml.etree.ElementTree",
        ],
    )
    agent.prompt_templates["system_prompt"] = SYSTEM_PROMPT

    print(f"\n{'='*60}")
    print("ml-intern-sciml")
    print(f"{'='*60}")
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"{'='*60}\n")

    result = agent.run(args.task)
    print("\n" + "="*60)
    print("AGENT RESULT")
    print("="*60)
    print(result)


if __name__ == "__main__":
    main()
