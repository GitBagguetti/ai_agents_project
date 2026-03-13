"""
Llama GRPO RL Module for Information Extraction
================================================
Reinforcement learning pipeline using GRPO (Group Relative Policy Optimization)
to train Llama-3.1-8B-Instruct for information extraction with JSON output format.
Rewards valid JSON and penalizes text outside JSON.

Designed for Google Colab. All logic lives here; notebook imports and runs in segments.
"""

import os
import re
import json
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset

# TRL / Transformers (optional - fail gracefully if not installed)
try:
    from trl import GRPOTrainer, GRPOConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRL = True
except ImportError:
    HAS_TRL = False

# ============================================================================
# CONFIGURATION
# ============================================================================

# Import shared worker behavioral prompt (accuracy + efficiency)
try:
    from .prompts import WORKER_BEHAVIORAL_PROMPT
except ImportError:
    WORKER_BEHAVIORAL_PROMPT = (
        "You are a highly efficient information detection and extraction engine, specialized in analyzing natural language data. "
        "You value accuracy and efficiency. Follow formatting conventions in the extraction prompt."
    )

DEFAULT_PROMPT_TEMPLATE = (
    "Extract the subject-verb relationships (entity, action) from this text. "
    "Output ONLY valid JSON with keys: entity, action. Example: {\"entity\": \"Trump\", \"action\": \"sign\"}. "
    "No text outside JSON.\n\nText: {text}"
)


def _extract_json_from_completion(completion: str, output_columns: List[str]) -> tuple:
    """
    Extract valid JSON from completion. Returns (parsed_records, json_str_len).
    parsed_records: list of dicts with keys from output_columns.
    json_str_len: character length of the JSON substring.
    """
    if not isinstance(completion, str):
        return [], 0
    try:
        match = re.search(r'\{.*\}', completion, re.DOTALL)
        if not match:
            return [], 0
        s = match.group()
        # Try to parse; if nested, find shortest valid JSON from start
        data = None
        for end in range(len(s), 0, -1):
            try:
                data = json.loads(s[:end])
                s = s[:end]
                break
            except json.JSONDecodeError:
                continue
        if data is None:
            return [], 0
        records = data.get("motifs", data.get("items", []))
        if isinstance(data, dict) and not records:
            if any(k in data for k in output_columns):
                records = [data]
        if not isinstance(records, list):
            records = []
        result = []
        for m in records:
            if not isinstance(m, dict):
                continue
            rec = {}
            for col in output_columns:
                val = m.get(col, m.get("actor" if col == "entity" else col, ""))
                rec[col] = str(val).strip().lower() if val is not None else ""
            if any(rec.values()):
                result.append(rec)
        return result, len(s)
    except (json.JSONDecodeError, TypeError, AttributeError):
        return [], 0


def _normalize(s: str) -> str:
    """Normalize string for comparison."""
    return str(s).lower().strip() if s else ""


# ============================================================================
# REWARD FUNCTION
# ============================================================================

def make_motif_reward_func(
    output_columns: Optional[List[str]] = None,
    format_payoff: float = 1.0,
    penalty_per_char: float = 0.01,
    content_bonus: float = 0.5,
) -> Callable:
    """
    Create a reward function for JSON information extraction.

    - Format payoff: fixed reward when output contains valid JSON with output_columns keys
    - Outside-length penalty: penalty scaling with text length outside the JSON block
    - Content bonus (optional): bonus when extracted values match ground truth

    Returns a function compatible with TRL GRPOTrainer reward_funcs.
    """
    out_cols = output_columns or ["entity", "action"]

    _first_batch_done = [False]  # mutable to allow assignment in closure

    def motif_reward_func(completions: List, **kwargs) -> List[float]:
        rewards = []
        # TRL passes dataset columns as kwargs; ensure lists
        gt_values = {col: kwargs.get(col, []) for col in out_cols}
        for col in out_cols:
            v = gt_values[col]
            if not isinstance(v, (list, tuple)):
                gt_values[col] = [v] * len(completions) if v is not None else []

        # Diagnostic: print on first batch to confirm gt_values is populated (content_bonus silently fails if empty)
        if not _first_batch_done[0]:
            n = len(completions)
            samples_with_gt = sum(
                1 for i in range(n)
                if any(_normalize(gt_values[col][i]) if i < len(gt_values[col]) else "" for col in out_cols)
            )
            print(f"[motif_reward] First batch (n={n}): {samples_with_gt} samples have non-empty ground truth. "
                  f"content_bonus={'active' if samples_with_gt > 0 else 'INACTIVE (empty gt — check TRL passes dataset columns)'}")
            _first_batch_done[0] = True

        for i, completion in enumerate(completions):
            if isinstance(completion, (list, tuple)):
                content = ""
                for msg in completion:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        content += msg.get("content", "")
                completion = content
            elif not isinstance(completion, str):
                completion = str(completion) if completion else ""

            records, json_len = _extract_json_from_completion(completion, out_cols)
            outside_len = len(completion) - json_len

            # Format payoff: at least one valid JSON record
            fmt_ok = len(records) >= 1
            r = format_payoff if fmt_ok else 0.0

            # Outside-length penalty (punish text outside JSON)
            r -= penalty_per_char * max(0, outside_len)

            # Optional content bonus when extraction matches ground truth
            if content_bonus > 0 and records:
                gt = {col: _normalize(gt_values[col][i]) if i < len(gt_values[col]) else "" for col in out_cols}
                for rec in records:
                    if all(_normalize(rec.get(col, "")) == gt.get(col, "") for col in out_cols):
                        r += content_bonus
                        break

            rewards.append(float(r))

        return rewards

    return motif_reward_func


# ============================================================================
# PROMPT CONVERSION (AutoGen -> RL JSON format)
# ============================================================================

def build_prompt_from_autogen(
    final_prompt: str,
    output_columns: Optional[List[str]] = None,
) -> str:
    """
    Convert AutoGen prompt to RL template with JSON output format.
    Keeps extraction instruction, adds JSON output requirement. Must contain {text}.
    Escapes literal curly braces so only {text} is a format placeholder.
    """
    out_cols = output_columns or ["entity", "action"]
    if not final_prompt or not str(final_prompt).strip():
        return DEFAULT_PROMPT_TEMPLATE
    fp = str(final_prompt).strip()
    # Escape literal { } so str.format() only substitutes {text}; JSON examples like
    # {"entity": "...", "action": "..."} would otherwise be parsed as format keys
    _placeholder = "\x00TEXT_PLACEHOLDER\x00"
    fp = fp.replace("{text}", _placeholder)
    fp = fp.replace("{", "{{").replace("}", "}}")
    fp = fp.replace(_placeholder, "{text}")
    example = json.dumps({col: "..." for col in out_cols})
    example_escaped = example.replace("{", "{{").replace("}", "}}")
    out_instruction = (
        f"Output ONLY valid JSON with keys: {out_cols}. "
        f"Example: {example_escaped}. No text outside JSON."
    )
    if "{text}" in fp:
        return f"{fp}\n\n{out_instruction}"
    return f"{fp}\n\n{out_instruction}\n\nText: {{text}}"


# ============================================================================
# DATA LOADING
# ============================================================================

def load_motifs_for_rl(
    path: str,
    n_rows: int = 500,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    prompt_from_autogen: Optional[str] = None,
    text_col: str = "text",
    output_columns: Optional[List[str]] = None,
    use_system_prompt: bool = True,
) -> Dataset:
    """
    Load CSV with text + output_columns and prepare HuggingFace Dataset for GRPO.

    Supports Colab paths: /content/drive/MyDrive/... or /content/...
    Returns Dataset with columns: prompt (conversational), text, and one per output_column.
    """
    out_cols = output_columns or ["entity", "action"]

    # Resolve path (Colab Drive vs gdown)
    if not os.path.exists(path):
        alt = path.replace("/content/drive/MyDrive", "/content").replace("/drive/MyDrive", "/content/drive/MyDrive")
        if os.path.exists(alt):
            path = alt

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty dataframe from {path}")

    required = {text_col} | set(out_cols)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {list(missing)}. Expected: text + {out_cols}. Found: {list(df.columns)}")

    df = df.head(n_rows)
    df = df.drop_duplicates(subset=[text_col], keep="first")

    if prompt_from_autogen:
        prompt_template = build_prompt_from_autogen(prompt_from_autogen, out_cols)

    prompts = []
    texts = []
    col_data = {col: [] for col in out_cols}

    for _, row in df.iterrows():
        text = str(row[text_col])
        format_kw = {"text": text}

        for col in out_cols:
            val = str(row.get(col, ""))
            format_kw[col] = val
            col_data[col].append(val)
        format_kw["entity"] = format_kw.get("entity", "")  # backward compat for canonical
        format_kw["canonical"] = format_kw.get("entity", "")  # backward compat
        format_kw["action"] = format_kw.get("action", "")

        content = prompt_template.format(**format_kw)
        if use_system_prompt:
            prompt = [
                {"role": "system", "content": WORKER_BEHAVIORAL_PROMPT},
                {"role": "user", "content": content},
            ]
        else:
            prompt = [{"role": "user", "content": content}]
        prompts.append(prompt)
        texts.append(text)

    result = {"prompt": prompts, "text": texts, **col_data}
    return Dataset.from_dict(result)


def plot_dataset_overview(dataset: Dataset) -> None:
    """Visualize dataset size and text length distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(["RL Dataset"], [len(dataset)], color="#2196F3")
    ax1.set_ylabel("Count")
    ax1.set_title("Dataset Size")
    lengths = [len(row["text"]) for row in dataset]
    ax2.hist(lengths, bins=30, color="#4CAF50", alpha=0.7)
    ax2.set_xlabel("Text length (chars)")
    ax2.set_ylabel("Count")
    ax2.set_title("Text Length Distribution")
    plt.tight_layout()
    plt.show()


def plot_reward_accuracy(
    dataset: Dataset,
    reward_func: Callable,
    output_columns: Optional[List[str]] = None,
    n_samples: int = 100,
) -> None:
    """
    Plot reward discrimination: chosen (valid JSON only) vs rejected (non-JSON).
    Mirrors the RL reward: format payoff for valid JSON, penalty for text outside.
    Computes ranking accuracy and histograms, matching Week_7 style.
    """
    out_cols = output_columns or ["entity", "action"]
    n_samples = min(n_samples, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    chosen_completions = []
    rejected_completions = []
    entity_list = []
    action_list = []

    for idx in indices:
        row = dataset[int(idx)]
        entity_list.append(row.get("entity", ""))
        action_list.append(row.get("action", ""))
        chosen_completions.append(json.dumps({out_cols[0]: row.get("entity", ""), out_cols[1]: row.get("action", "")}))
        rejected_completions.append(
            f"Entity: {row.get('entity', '')}, Action: {row.get('action', '')}"
        )

    chosen_scores = reward_func(
        completions=chosen_completions,
        entity=entity_list,
        action=action_list,
    )
    rejected_scores = reward_func(
        completions=rejected_completions,
        entity=entity_list,
        action=action_list,
    )

    accuracy = np.mean(np.array(chosen_scores) > np.array(rejected_scores))
    print(f"Ranking accuracy: {accuracy:.1%}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(chosen_scores, bins=30, alpha=0.6, label=f"Chosen (mean={np.mean(chosen_scores):.2f})", color="#2ecc71")
    ax.hist(rejected_scores, bins=30, alpha=0.6, label=f"Rejected (mean={np.mean(rejected_scores):.2f})", color="#e74c3c")
    ax.set_xlabel("Reward Score")
    ax.set_ylabel("Count")
    ax.set_title(f"RM Score Distribution — Accuracy: {accuracy:.1%}")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================================
# GRPO TRAINING
# ============================================================================

def run_grpo_train(
    train_dataset: Dataset,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    output_dir: str = "./grpo_motif_output",  # Local dir for training; use save_model_to_drive for Drive
    reward_funcs: Optional[Callable] = None,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    n_rows: int = 500,
    num_generation: int = 4,
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 1e-6,
    logging_steps: int = 10,
    **kwargs,
) -> "GRPOTrainer":
    """
    Create and return GRPOTrainer for motif extraction. Call trainer.train() to run.

    Requires: trl, transformers, datasets, accelerate.
    """
    if not HAS_TRL:
        raise ImportError("Install trl, transformers, datasets, accelerate: pip install trl transformers datasets accelerate")

    if reward_funcs is None:
        reward_funcs = make_motif_reward_func()

    # GRPO requires generation_batch_size (from per_device_train_batch_size) divisible by num_generations
    if per_device_train_batch_size % num_generation != 0:
        per_device_train_batch_size = num_generation * (
            (per_device_train_batch_size + num_generation - 1) // num_generation
        )

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        num_generations=num_generation,  # GRPOConfig expects num_generations (plural)
        remove_unused_columns=False,  # Keep output columns for reward
        gradient_checkpointing=False,  # Reduce memory (trades compute for memory)
        **kwargs,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
    )

    return trainer


# ============================================================================
# MODEL SAVING (Colab Drive)
# ============================================================================

def save_model_to_drive(
    trainer: "GRPOTrainer",
    output_dir: str = "MyDrive/llama_motif_grpo",
) -> str:
    """
    Save trained model to Google Drive when in Colab.
    Falls back to local output_dir if not in Colab.
    """
    try:
        from google.colab import drive
        drive_path = "/content/drive"
        if not os.path.exists(drive_path):
            raise RuntimeError("Drive not mounted. Run: from google.colab import drive; drive.mount('/content/drive')")
        full_path = f"/content/drive/{output_dir.lstrip('/')}"
    except ImportError:
        full_path = output_dir

    os.makedirs(full_path, exist_ok=True)
    trainer.save_model(full_path)
    if hasattr(trainer, "processing_class") and trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(full_path)
    print(f"Model saved to {full_path}")
    return full_path


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_reward_over_training(trainer: "GRPOTrainer") -> None:
    """Plot reward over training steps from trainer state."""
    if not hasattr(trainer, "state") or not trainer.state.log_history:
        print("No training logs available.")
        return

    rewards = []
    steps = []
    for entry in trainer.state.log_history:
        if "reward" in entry:
            rewards.append(entry["reward"])
            steps.append(entry.get("step", len(steps)))
        elif "rewards" in entry:
            r = entry["rewards"]
            rewards.extend(r if isinstance(r, (list, tuple)) else [r])
            steps.extend(range(len(rewards) - len(steps), len(rewards)))

    if not rewards:
        # Fallback: try to get from any scalar
        for i, entry in enumerate(trainer.state.log_history):
            for k, v in entry.items():
                if "reward" in k.lower() and isinstance(v, (int, float)):
                    rewards.append(v)
                    steps.append(entry.get("step", i))
                    break

    if not rewards:
        print("No reward data in logs.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps if steps else range(len(rewards)), rewards, color="#2196F3", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Over Training")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_loss_curve(trainer: "GRPOTrainer") -> None:
    """Plot training loss from trainer state."""
    if not hasattr(trainer, "state") or not trainer.state.log_history:
        print("No training logs available.")
        return

    losses = []
    steps = []
    for entry in trainer.state.log_history:
        if "loss" in entry:
            losses.append(entry["loss"])
            steps.append(entry.get("step", len(steps)))

    if not losses:
        print("No loss data in logs.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, color="#FF6B6B", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_format_compliance(
    dataset: Dataset,
    model,
    tokenizer,
    n_samples: int = 20,
    output_columns: Optional[List[str]] = None,
) -> None:
    """
    Plot format compliance (% outputs with valid JSON) over a sample.
    Requires model and tokenizer for inference.
    """
    import torch

    out_cols = output_columns or ["entity", "action"]
    n_samples = min(n_samples, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    matches = 0
    for idx in indices:
        row = dataset[int(idx)]
        prompt = row["prompt"]
        if isinstance(prompt, list):
            content = prompt[0]["content"] if prompt else ""
        else:
            content = str(prompt)
        inputs = tokenizer(content, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        records, _ = _extract_json_from_completion(completion, out_cols)
        if records:
            matches += 1

    pct = 100 * matches / n_samples
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Format match", "No match"], [matches, n_samples - matches], color=["#4CAF50", "#FF6B6B"])
    ax.set_ylabel("Count")
    ax.set_title(f"Format Compliance (n={n_samples}): {pct:.0f}%")
    plt.tight_layout()
    plt.show()


def get_sample_outputs(dataset: Dataset, model, tokenizer, n_samples: int = 5) -> List[Dict[str, str]]:
    """Generate sample outputs from model for given dataset. Returns list of {text, output, entity, action}."""
    import torch

    samples = []
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    for idx in indices:
        row = dataset[int(idx)]
        prompt = row["prompt"]
        content = prompt[0]["content"] if isinstance(prompt, list) and prompt else str(prompt)
        inputs = tokenizer(content, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        samples.append({
            "text": row.get("text", "")[:100],
            "output": completion,
            "entity": row.get("entity", ""),
            "action": row.get("action", ""),
        })
    return samples


def plot_pre_post_comparison(
    pre_samples: List[Dict[str, str]],
    post_samples: List[Dict[str, str]],
    title: str = "Pre vs Post RL Sample Comparison",
) -> None:
    """
    Side-by-side comparison of sample outputs before and after RL.
    pre_samples, post_samples: list of {"text": ..., "output": ..., "entity": ..., "action": ...}
    """
    n = min(len(pre_samples), len(post_samples), 5)
    if n == 0:
        print("No samples to compare.")
        return

    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        pre = pre_samples[i]
        post = post_samples[i]
        pre_txt = (pre.get("output", "") or "")[:300]
        post_txt = (post.get("output", "") or "")[:300]
        axes[i, 0].set_title(f"Sample {i+1} (Pre-RL)")
        axes[i, 0].text(0.05, 0.95, pre_txt, transform=axes[i, 0].transAxes, fontsize=8, verticalalignment="top", wrap=True)
        axes[i, 0].axis("off")
        axes[i, 1].set_title(f"Sample {i+1} (Post-RL)")
        axes[i, 1].text(0.05, 0.95, post_txt, transform=axes[i, 1].transAxes, fontsize=8, verticalalignment="top", wrap=True)
        axes[i, 1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_training_summary(trainer: "GRPOTrainer") -> None:
    """Combined plot: loss and any reward-like metrics from logs."""
    if not hasattr(trainer, "state") or not trainer.state.log_history:
        print("No training logs available.")
        return

    steps, losses, rewards = [], [], []
    for entry in trainer.state.log_history:
        s = entry.get("step")
        if "loss" in entry:
            losses.append((s, entry["loss"]))
        if "reward" in entry:
            rewards.append((s, entry["reward"]))

    n_plots = sum(1 for x in [losses, rewards] if x)
    if n_plots == 0:
        print("No plottable data.")
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    idx = 0
    if losses:
        steps_l, vals_l = zip(*losses)
        axes[idx].plot(steps_l, vals_l, color="#FF6B6B")
        axes[idx].set_title("Loss")
        axes[idx].set_xlabel("Step")
        axes[idx].grid(alpha=0.3)
        idx += 1
    if rewards:
        steps_r, vals_r = zip(*rewards)
        axes[idx].plot(steps_r, vals_r, color="#2196F3")
        axes[idx].set_title("Reward")
        axes[idx].set_xlabel("Step")
        axes[idx].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
