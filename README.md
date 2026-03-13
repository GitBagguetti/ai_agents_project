# Teaching Large Language Models how to Code

**Author:** Pierre Loertscher  
**Course:** MACS 37005 – AI Agents for Social Science and Society, University of Chicago  
**Date:** March 13, 2026

---

## Overview

This project investigates how multi-agent LLM systems and automated prompt optimisation techniques can teach large language models to perform structured information extraction — specifically, subject-verb relationship ("motif") extraction from social-media texts about Donald Trump.

The pipeline proceeds through four stages:

1. **AutoGen multi-agent prompt engineering** (Stages 1–3): Four specialised prompt-engineer agents each analyse an exclusive slice of annotated data, produce extraction prompts, then converge (with human-in-the-loop feedback) on a single best prompt via a coordinator agent.  
2. **GRPO reinforcement learning** (separate notebook): A Llama-3.1-8B-Instruct model is fine-tuned with Group Relative Policy Optimisation to reward structured JSON output and penalise verbose completions.  
3. **DSPy optimisation**: The AutoGen-derived prompt seeds BootstrapFewShot, MIPROv2, BootstrapFinetune, and BetterTogether optimisers on both OpenAI and the RL-trained Llama backend.  
4. **SpaCy MCP tools**: NER and dependency-parsing tools are exposed to the LLM via a ReAct loop (2-call-per-document safety limit, full audit logging), allowing the model to ground extraction decisions in syntactic structure.

---

## Repository Structure

```
ai_agents_project/
├── full_project.ipynb       # Main notebook: Stages A–D, evaluation, and visualisations
├── llama_RL.ipynb           # Standalone GRPO RL training (run on a separate Colab A100 instance)
├── modules/
│   ├── __init__.py
│   ├── autogen_pipeline.py  # AutoGen multi-agent pipeline (Stages 1–3) + SpaCy tool-assisted
│   │                        #   extraction, audit logging, and all related visualisations
│   ├── dspy_pipeline.py     # DSPy pure-optimisation pipeline (Part A) + DSPy + MCP ReAct
│   │                        #   pipeline with SpaCy tools (Part B) and MCP visualisations
│   └── llama_RL.py          # GRPO RL training utilities (data loading, reward function,
│                            #   trainer setup, evaluation); used exclusively by llama_RL.ipynb
└── README.md
```

### Module responsibilities

| Module | Used by | Key exports |
|--------|---------|-------------|
| `autogen_pipeline.py` | `full_project.ipynb` | `run_stage1/2/3`, `run_stage3_with_tools`, `run_stage3_with_tools_standalone`, visualisation helpers, `WORKER_BEHAVIORAL_PROMPT` |
| `dspy_pipeline.py` | `full_project.ipynb` | `setup_dspy`, `run_bootstrap_fewshot`, `run_mipro`, `run_mcp_*`, `plot_*`, `MotifReActModule` |
| `llama_RL.py` | `llama_RL.ipynb` | `load_motifs_for_rl`, `make_motif_reward_func`, `RLConfig`, GRPO training utilities |

---

## Running the Project

Both notebooks are designed for **Google Colab** (GPU recommended for `llama_RL.ipynb`).

### `full_project.ipynb`

1. Mount Google Drive and place `original_content_trump_motifs_en_10k.csv` at `drive/MyDrive/`.
2. Upload the `modules/` directory via the VS Code ↔ Colab extension (or the Colab file browser).
3. Run cells in order. Section 0 installs all dependencies.
4. Enter your OpenAI API key when prompted.

### `llama_RL.ipynb`

1. Run on a separate Colab instance with an A100 GPU.
2. Paste the final prompt produced by Stage 2 of `full_project.ipynb` when prompted.
3. After training, the model is saved to `drive/MyDrive/llama_motif_grpo`.
4. Update `rl_model_path` in `full_project.ipynb` to load the RL model for Stage C evaluation.

---

## Data

The dataset (`original_content_trump_motifs_en_10k.csv`) contains ~10 000 social-media posts with hand-coded entity/action annotations. It is available via Google Drive; the notebooks download it automatically if not already present.

---

## Dependencies

```
pyautogen / autogen-agentchat~=0.2
openai
dspy-ai
trl
transformers
datasets
accelerate
spacy + en_core_web_lg
sentence-transformers
scikit-learn
matplotlib
pandas
numpy
gdown
```

Install in Colab via the `!pip install` cell at the top of each notebook.
