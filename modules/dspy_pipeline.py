"""
DSPy Pipeline for Motif Extraction
====================================
Unified DSPy pipeline for structured motif extraction from social-media texts.

Part A – Pure DSPy: BootstrapFewShot, MIPROv2, BootstrapFinetune, BetterTogether
          optimizers on a ChainOfThought extraction module. Supports dual model
          backends (OpenAI gpt-4o-mini / Llama-3.1-8B via SGLang).

Part B – DSPy + MCP Tools: extends Part A with SpaCy NER and dependency-parsing
          tools via a ReAct module. Safety guardrails: 2 tool calls / document,
          full audit logging.

Designed for Google Colab. All logic lives here; notebook imports and calls in segments.
"""

# ============================================================================
# SETUP AND IMPORTS
# ============================================================================

# !pip install -q dspy-ai openai sentence-transformers scikit-learn matplotlib pandas numpy spacy
# !python -m spacy download en_core_web_lg

import os
import re
import ast
import json
from contextvars import ContextVar
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from getpass import getpass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import dspy
from dspy.evaluate import Evaluate

# Optional imports (sentence_transformers for prompt evolution)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# Teleprompt imports - MIPROv2, BootstrapFinetune, BetterTogether may not exist in older DSPy
BootstrapFinetune = None
BetterTogether = None
try:
    from dspy.teleprompt import BootstrapFewShot, MIPROv2
    HAS_MIPRO = True
except ImportError:
    BootstrapFewShot = getattr(dspy, "BootstrapFewShot", None)
    MIPROv2 = getattr(dspy, "MIPROv2", None)
    HAS_MIPRO = MIPROv2 is not None

try:
    from dspy.teleprompt import BootstrapFinetune, BetterTogether
    HAS_BOOTSTRAP_FINETUNE = BootstrapFinetune is not None
    HAS_BETTER_TOGETHER = BetterTogether is not None
except ImportError:
    try:
        import dspy.teleprompt as tp
        BootstrapFinetune = getattr(tp, "BootstrapFinetune", None)
        BetterTogether = getattr(tp, "BetterTogether", None)
    except (ImportError, AttributeError):
        pass
    HAS_BOOTSTRAP_FINETUNE = BootstrapFinetune is not None
    HAS_BETTER_TOGETHER = BetterTogether is not None

# SpaCy (optional; required for MCP tool sections)
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    nlp = None

# ============================================================================
# SHARED PROMPT
# ============================================================================

WORKER_BEHAVIORAL_PROMPT = """You are a highly efficient information detection and extraction engine, specialized in analyzing natural language data.
You value accuracy: when the user asks you to extract certain information from given text data, you will try your best to adhere to what is directly mentioned in the text and the extraction criteria.
You value efficiency: your responses will be very concise, because they will be stored as values in a dataset. These responses will also strictly follow formatting conventions specified in the extraction prompt."""

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration for DSPy motif extraction."""
    api_key: str = ""
    model: str = "gpt-4o-mini"
    target_entity: str = "Donald Trump"
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 4
    num_candidates: int = 8
    validation_split: float = 0.35
    random_seed: int = 42
    temperature: float = 0.3
    embedding_model: str = "all-MiniLM-L6-v2"
    # Train/val/test split ratios (used when train_ratio/val_ratio/test_ratio provided)
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    n_samples: int = 1000
    # Token/cost tracking
    usage_history: List[Dict[str, Any]] = field(default_factory=list)


config = Config()


# ============================================================================
# DATA LOADING
# ============================================================================

def _normalize_entity(s: str) -> str:
    """Normalize entity for matching (lowercase, strip)."""
    if not s:
        return ""
    return str(s).lower().strip()


def _entity_matches(pred_actor: str, truth_entity: str) -> bool:
    """Check if predicted actor matches ground-truth entity (fuzzy)."""
    p = _normalize_entity(pred_actor)
    t = _normalize_entity(truth_entity)
    if not t:
        return False
    return t in p or p in t or (t.split()[-1] in p if t.split() else False)


def load_from_drive_path() -> str:
    """Resolve data path for Colab: drive/MyDrive/original_content_trump_motifs_en_10k.csv."""
    candidates = [
        "/content/drive/MyDrive/original_content_trump_motifs_en_10k.csv",
        "/content/original_content_trump_motifs_en_10k.csv",
        "original_content_trump_motifs_en_10k.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def load_and_prepare_data(
    path: str,
    target_entity: str = "Donald Trump",
    n_samples: int = None,
    entity_col: str = "entity",
    action_col: str = "action",
    text_col: str = "text",
    id_col: str = "posts_id",
    train_ratio: float = None,
    val_ratio: float = None,
    test_ratio: float = None,
) -> Tuple[pd.DataFrame, List, List, Optional[List]]:
    """
    Load CSV and prepare train/val (and optionally test) MotifExamples.
    CSV columns: text, entity (varying name in text), canonical, action, posts_id.
    entity_col defaults to "entity" for ground truth per spec.
    If train_ratio, val_ratio, test_ratio are provided, returns (prep_df, trainset, valset, testset).
    Otherwise uses validation_split for train/val only and returns (prep_df, trainset, valset, None).
    """
    n_samples = n_samples if n_samples is not None else config.n_samples
    train_ratio = train_ratio if train_ratio is not None else config.train_ratio
    val_ratio = val_ratio if val_ratio is not None else config.val_ratio
    test_ratio = test_ratio if test_ratio is not None else config.test_ratio

    if not os.path.exists(path):
        path = load_from_drive_path()
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty dataframe from {path}")

    # Map columns - handle both 'entity' and 'canonical'
    if entity_col not in df.columns:
        entity_col = "canonical" if "canonical" in df.columns else "entity"
    if entity_col not in df.columns:
        raise ValueError(f"Entity column not found. Available: {list(df.columns)}")

    # Sample
    df = df.head(n_samples) if len(df) > n_samples else df
    df = df.drop_duplicates(subset=[text_col], keep="first")

    # Build motifs: each row = one text, one motif {"actor": canonical, "action": action}
    rows = []
    for _, row in df.iterrows():
        content = row[text_col]
        entity = row[entity_col]
        action = row[action_col]
        pid = row.get(id_col, len(rows))
        motifs = [{"actor": str(entity), "action": str(action)}]
        rows.append({"content": content, "motifs": motifs, "posts_id": pid})

    prep_df = pd.DataFrame(rows)
    examples = create_dspy_examples(prep_df)

    if test_ratio and test_ratio > 0 and (train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        # Three-way split: train / val / test
        n = len(examples)
        np.random.seed(config.random_seed)
        idx = np.random.permutation(n)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
        trainset = [examples[i] for i in train_idx]
        valset = [examples[i] for i in val_idx]
        testset = [examples[i] for i in test_idx]
        print(f"Loaded {len(prep_df)} examples. Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")
        return prep_df, trainset, valset, testset
    else:
        # Two-way split (backward compatible)
        trainset, valset = train_test_split(
            examples,
            test_size=config.validation_split,
            random_state=config.random_seed,
        )
        print(f"Loaded {len(prep_df)} examples. Train: {len(trainset)}, Val: {len(valset)}")
        return prep_df, trainset, valset, None


# ============================================================================
# DSPY EXAMPLE AND MODULE
# ============================================================================

class MotifExample(dspy.Example):
    """DSPy Example for motif extraction."""

    def __init__(self, content: str = None, motifs: List[Dict[str, str]] = None, posts_id: int = None, base=None, **kwargs):
        if base is not None:
            super().__init__(base=base, **kwargs)
        else:
            super().__init__(content=content, motifs=motifs, posts_id=posts_id)


def create_dspy_examples(df: pd.DataFrame) -> List[MotifExample]:
    """Convert DataFrame to DSPy Examples."""
    examples = []
    for _, row in df.iterrows():
        ex = MotifExample(
            content=row["content"],
            motifs=row["motifs"],
            posts_id=row.get("posts_id", 0),
        )
        ex = ex.with_inputs("content")
        examples.append(ex)
    return examples


def _make_signature_class(target_entity: str, instruction: Optional[str] = None):
    """Create entity-specific MotifExtractionSignature. instruction overrides default docstring."""

    default_doc = (
        f"Extract subject-verb relationships from social media text where the actor refers to {target_entity} "
        f"(including co-references like 'he', 'President {target_entity.split()[-1] if target_entity else ''}', etc.). "
        "Return the actual text span as actor. "
        "Output: Python list of dicts with 'actor' and 'action' keys. "
        "Example: [{'actor': 'Trump', 'action': 'sign'}]. Return [] if no subject-verb relationships found."
    )
    task_doc = instruction if instruction else default_doc
    doc = f"{WORKER_BEHAVIORAL_PROMPT}\n\n{task_doc}"

    class MotifExtractionSignature(dspy.Signature):
        __doc__ = doc
        content = dspy.InputField(desc=f"Social media post text to analyze for {target_entity}-related subject-verb relationships")
        motifs = dspy.OutputField(
            desc=f"List of {target_entity}-related subject-verb relationships as Python list of dicts with 'actor' and 'action'. "
            "Return [] if none found."
        )

    return MotifExtractionSignature


def create_module_from_prompt(
    instruction: str,
    target_entity: str = None,
) -> "MotifExtractorModule":
    """Create DSPy module with custom instruction (e.g. from AutoGen or DSPy-optimized prompt)."""
    entity = target_entity or config.target_entity
    Sig = _make_signature_class(entity, instruction=instruction)

    class Mod(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.ChainOfThought(Sig)

        def forward(self, content: str) -> dspy.Prediction:
            return self.predictor(content=content)

    return Mod()


def create_motif_module(target_entity: str = None) -> "MotifExtractorModule":
    """Create MotifExtractorModule with entity-specific signature."""
    entity = target_entity or config.target_entity
    Sig = _make_signature_class(entity)

    class MotifExtractorModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.ChainOfThought(Sig)

        def forward(self, content: str) -> dspy.Prediction:
            return self.predictor(content=content)

    return MotifExtractorModule()


class MotifExtractorModule(dspy.Module):
    def __init__(self, signature_cls=None):
        super().__init__()
        self.predictor = dspy.ChainOfThought(
            signature_cls or _make_signature_class(config.target_entity)
        )

    def forward(self, content: str) -> dspy.Prediction:
        return self.predictor(content=content)


# ============================================================================
# PARSING
# ============================================================================

def parse_motifs(motifs_str) -> List[Dict[str, str]]:
    """Parse LLM output string to list of dicts. Accepts both 'actor' and 'entity' keys."""
    if motifs_str is None:
        return []
    try:
        s = str(motifs_str)
        list_match = re.search(r"\[.*?\]", s, re.DOTALL)
        if list_match:
            parsed = ast.literal_eval(list_match.group())
            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    if isinstance(item, dict) and ("actor" in item or "entity" in item) and "action" in item:
                        a = item.get("actor") or item.get("entity", "")
                        result.append({"actor": str(a), "action": str(item.get("action", ""))})
                return result
        # Try JSON {"motifs": [...]}
        json_match = re.search(r'\{.*\}', s, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            motifs = data.get("motifs", [])
            return [
                {"actor": str(m.get("actor", m.get("entity", ""))), "action": str(m.get("action", ""))}
                for m in motifs if isinstance(m, dict)
            ]
        return []
    except Exception:
        return []


# ============================================================================
# METRICS
# ============================================================================

def motif_accuracy_metric(example: MotifExample, prediction, trace=None) -> float:
    """Jaccard-like accuracy for motif sets."""
    try:
        pred_motifs = parse_motifs(getattr(prediction, "motifs", prediction))
        pred_set = set()
        for item in pred_motifs:
            if isinstance(item, dict):
                pred_set.add((str(item.get("actor", "")), str(item.get("action", ""))))

        truth_set = set()
        for item in (example.motifs or []):
            if isinstance(item, dict):
                truth_set.add((str(item.get("actor", "")), str(item.get("action", ""))))

        if len(truth_set) == 0:
            return 1.0 if len(pred_set) == 0 else 0.0
        inter = len(pred_set & truth_set)
        union = len(pred_set | truth_set)
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def compute_recall_f1(
    examples: List[MotifExample],
    predictions: List,
) -> Dict[str, float]:
    """Compute entity/action/motif recall and F1 across examples."""
    entity_tp = entity_fp = entity_fn = 0
    action_tp = action_fp = action_fn = 0
    motif_tp = motif_fp = motif_fn = 0

    for ex, pred in zip(examples, predictions):
        pred_motifs = parse_motifs(getattr(pred, "motifs", pred))
        pred_actors = {_normalize_entity(m.get("actor", "")) for m in pred_motifs if isinstance(m, dict)}
        pred_actions = {_normalize_entity(m.get("action", "")) for m in pred_motifs if isinstance(m, dict)}
        pred_pairs = set()
        for m in pred_motifs:
            if isinstance(m, dict):
                pred_pairs.add((_normalize_entity(m.get("actor", "")), _normalize_entity(m.get("action", ""))))

        truth_motifs = ex.motifs or []
        truth_actors = set()
        truth_actions = set()
        truth_pairs = set()
        for m in truth_motifs:
            if isinstance(m, dict):
                a, b = _normalize_entity(m.get("actor", "")), _normalize_entity(m.get("action", ""))
                truth_actors.add(a)
                truth_actions.add(b)
                truth_pairs.add((a, b))

        # Entity: match predicted actors to truth actors (fuzzy)
        for ta in truth_actors:
            if any(_entity_matches(pa, ta) for pa in pred_actors):
                entity_tp += 1
            else:
                entity_fn += 1
        for pa in pred_actors:
            if not any(_entity_matches(pa, ta) for ta in truth_actors):
                entity_fp += 1

        # Action: exact set match
        inter_a = len(truth_actions & pred_actions)
        action_tp += inter_a
        action_fn += len(truth_actions) - inter_a
        action_fp += len(pred_actions - truth_actions)

        # Motif: match (actor, action) with fuzzy entity matching
        for (ta, tb) in truth_pairs:
            found = any(
                _entity_matches(pa, ta) and _normalize_entity(pb) == _normalize_entity(tb)
                for (pa, pb) in pred_pairs
            )
            if found:
                motif_tp += 1
            else:
                motif_fn += 1
        for (pa, pb) in pred_pairs:
            found = any(
                _entity_matches(pa, ta) and _normalize_entity(pb) == _normalize_entity(tb)
                for (ta, tb) in truth_pairs
            )
            if not found:
                motif_fp += 1

    def _f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    def _recall(tp, fn):
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        "entity_recall": _recall(entity_tp, entity_fn),
        "entity_f1": _f1(entity_tp, entity_fp, entity_fn),
        "action_recall": _recall(action_tp, action_fn),
        "action_f1": _f1(action_tp, action_fp, action_fn),
        "motif_recall": _recall(motif_tp, motif_fn),
        "motif_f1": _f1(motif_tp, motif_fp, motif_fn),
    }


def aggregate_lm_usage(lm=None) -> Dict[str, Any]:
    """Aggregate token usage from lm.history. Returns dict with prompt_tokens, completion_tokens, total_tokens, cost."""
    lm = lm or dspy.settings.lm
    if lm is None or not hasattr(lm, "history"):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}
    total_p, total_c, total_t, total_cost = 0, 0, 0, 0.0
    for h in lm.history:
        u = h.get("usage") or {}
        if isinstance(u, dict):
            total_p += u.get("prompt_tokens", 0) or u.get("input_tokens", 0)
            total_c += u.get("completion_tokens", 0) or u.get("output_tokens", 0)
            total_t += u.get("total_tokens", total_p + total_c)
        c = h.get("cost")
        if c is not None:
            total_cost += float(c)
    return {
        "prompt_tokens": total_p,
        "completion_tokens": total_c,
        "total_tokens": total_t or (total_p + total_c),
        "cost": total_cost,
    }


# ============================================================================
# DSPY SETUP
# ============================================================================

def _check_sglang_reachable(sglang_base: str) -> bool:
    """Verify SGLang server is reachable before configuring. Returns True if OK."""
    try:
        import urllib.request
        base = sglang_base.rstrip("/").replace("/v1", "").rstrip("/")
        url = f"{base}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as _:
            return True
    except Exception as e:
        print(f"⚠ SGLang may not be running at {sglang_base}: {e}")
        print("  Start with: !CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --port 7501 --model-path <path> &")
        return False


def _suppress_dspy_verbose_warnings():
    """Suppress repetitive DSPy warnings (json_adapter structured output fallback)."""
    import logging
    logging.getLogger("dspy.adapters.json_adapter").setLevel(logging.ERROR)


def _suppress_litellm_warnings_for_sglang():
    """Suppress LiteLLM 'Missing dependency' warnings when using SGLang. Do NOT set OPENAI_API_KEY here."""
    import logging
    os.environ.setdefault("LITELLM_LOG", "ERROR")
    for name in ("litellm", "LiteLLM", "litellm.llms.openai", "dspy.utils.parallelizer"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def setup_dspy(
    api_key: str = None,
    model: str = "openai/gpt-4o-mini",
    use_llama: bool = False,
    sglang_base: str = "http://localhost:7501/v1",
    temperature: float = 0.3,
) -> bool:
    """
    Configure DSPy LM. For OpenAI: pass api_key or it will prompt.
    For Llama: use_llama=True, ensure SGLang server is running at sglang_base.
    """
    if use_llama:
        _suppress_litellm_warnings_for_sglang()
        _check_sglang_reachable(sglang_base)
        lm = dspy.LM(
            "openai/meta-llama/Llama-3.1-8B-Instruct",
            api_base=sglang_base,
            api_key="sglang",  # SGLang ignores; OpenAI client requires non-empty
            model_type="chat",
            temperature=temperature,
            timeout=300,  # Local SGLang can be slow; avoid connection/timeout errors
        )
        dspy.configure(lm=lm)
        print("✓ DSPy configured with Llama-3.1-8B (SGLang)")
        return True

    key = api_key or config.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("OpenAI API Key Required. Enter when prompted.")
        try:
            key = getpass("OpenAI API Key: ")
        except Exception:
            key = input("OpenAI API Key: ")
    if not key:
        raise ValueError("API key is required.")
    config.api_key = key
    os.environ["OPENAI_API_KEY"] = key  # Override any stale value (e.g. from prior SGLang setup)
    model_name = model if model.startswith("openai/") else f"openai/{model}"
    lm = dspy.LM(model_name, api_key=key, temperature=temperature)
    dspy.configure(lm=lm)
    print(f"✓ DSPy configured with OpenAI ({model})")
    return True


def setup_dspy_with_model_path(
    model_path: str,
    sglang_base: str = "http://localhost:7501/v1",
    temperature: float = 0.3,
) -> bool:
    """
    Configure DSPy for RL-trained model. SGLang must be started with:
    --model-path <model_path> (e.g. /content/drive/MyDrive/llama_motif_grpo)
    """
    _suppress_litellm_warnings_for_sglang()
    _check_sglang_reachable(sglang_base)
    lm = dspy.LM(
        "openai/meta-llama/Llama-3.1-8B-Instruct",
        api_base=sglang_base,
        api_key="sglang",  # SGLang ignores; OpenAI client requires non-empty
        model_type="chat",
        temperature=temperature,
        timeout=300,  # Local SGLang can be slow; avoid connection/timeout errors
    )
    dspy.configure(lm=lm)
    print(f"✓ DSPy configured for RL model. Ensure SGLang runs with: --model-path {model_path}")
    return True


# ============================================================================
# OPTIMIZATION RUNNERS
# ============================================================================

def get_prompt_from_module(module) -> str:
    """Extract prompt/signature from a DSPy module for qualitative comparison."""
    if module is None:
        return ""
    if hasattr(module, "predictor") and hasattr(module.predictor, "signature"):
        sig = module.predictor.signature
        return str(sig) if sig else ""
    if hasattr(module, "react") and hasattr(module.react, "signature"):
        return str(module.react.signature) if module.react.signature else ""
    return ""


def print_qualitative_prompt_comparison(original: str, optimized: str, title: str = "Prompt Comparison"):
    """Print side-by-side original vs optimized prompt for qualitative review."""
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)
    print("\n--- ORIGINAL (from autogen) ---")
    print((original or "(none)")[:2000])
    if len(original or "") > 2000:
        print("... [truncated]")
    print("\n--- OPTIMIZED (from DSPy) ---")
    print((optimized or "(none)")[:2000])
    if len(optimized or "") > 2000:
        print("... [truncated]")
    print("=" * 70)


def run_bootstrap_fewshot(
    trainset: List[MotifExample],
    valset: List[MotifExample],
    target_entity: str = None,
    instruction: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], str, str]:
    """Run BootstrapFewShot. Returns (optimized_module, metrics, usage, original_prompt, optimized_prompt)."""
    if BootstrapFewShot is None:
        raise ImportError("BootstrapFewShot not available. Install dspy-ai.")
    entity = target_entity or config.target_entity
    original_prompt = instruction or ""

    if instruction:
        baseline = create_module_from_prompt(instruction, target_entity)
    else:
        Sig = _make_signature_class(entity)
        class Mod(dspy.Module):
            def __init__(self):
                super().__init__()
                self.predictor = dspy.ChainOfThought(Sig)
            def forward(self, content: str):
                return self.predictor(content=content)
        baseline = Mod()

    lm = dspy.settings.lm
    if lm and hasattr(lm, "history"):
        lm.history = []
    evaluator = Evaluate(devset=valset, metric=motif_accuracy_metric, num_threads=1, display_progress=True, display_table=0)
    with dspy.settings.context(track_usage=True, cache=False):
        baseline_score = evaluator(baseline)
    baseline_val = float(baseline_score) / 100 if hasattr(baseline_score, "__float__") else baseline_score
    usage_baseline = aggregate_lm_usage()

    optimizer = BootstrapFewShot(
        metric=motif_accuracy_metric,
        max_bootstrapped_demos=config.max_bootstrapped_demos,
        max_labeled_demos=config.max_labeled_demos,
        max_rounds=1,
    )
    with dspy.settings.context(track_usage=True, cache=False):
        optimized = optimizer.compile(baseline, trainset=trainset)
        opt_score = evaluator(optimized)
    opt_val = float(opt_score) / 100 if hasattr(opt_score, "__float__") else opt_score
    usage_full = aggregate_lm_usage()

    optimized_prompt = get_prompt_from_module(optimized)
    if not original_prompt and instruction is None:
        original_prompt = get_prompt_from_module(baseline)

    metrics = {
        "baseline_score": baseline_val,
        "optimized_score": opt_val,
        "improvement": opt_val - baseline_val,
        "trainset_size": len(trainset),
        "valset_size": len(valset),
    }
    usage = {"baseline": usage_baseline, "full": usage_full}
    return optimized, metrics, usage, original_prompt, optimized_prompt


def run_mipro(
    trainset: List[MotifExample],
    valset: List[MotifExample],
    target_entity: str = None,
    auto: str = "light",
    instruction: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], str, str]:
    """Run MIPROv2. Returns (optimized_module, metrics, usage, original_prompt, optimized_prompt)."""
    if not HAS_MIPRO or MIPROv2 is None:
        raise ImportError("MIPROv2 not available. Upgrade dspy-ai.")
    _suppress_dspy_verbose_warnings()
    entity = target_entity or config.target_entity
    original_prompt = instruction or ""

    if instruction:
        baseline = create_module_from_prompt(instruction, target_entity)
    else:
        Sig = _make_signature_class(entity)
        class Mod(dspy.Module):
            def __init__(self):
                super().__init__()
                self.predictor = dspy.ChainOfThought(Sig)
            def forward(self, content: str):
                return self.predictor(content=content)
        baseline = Mod()

    lm = dspy.settings.lm
    if lm and hasattr(lm, "history"):
        lm.history = []
    evaluator = Evaluate(devset=valset, metric=motif_accuracy_metric, num_threads=1, display_progress=True, display_table=0)
    with dspy.settings.context(track_usage=True, cache=False):
        try:
            baseline_score = evaluator(baseline)
        except Exception as e:
            if "Execution cancelled" in str(e) and valset:
                try:
                    ex0 = valset[0]
                    _ = baseline(content=ex0.content)
                    _ = motif_accuracy_metric(ex0, _)
                except Exception as inner:
                    raise RuntimeError(f"DSPy evaluation failed. First-example error: {inner}") from inner
            raise
    baseline_val = float(baseline_score) / 100 if hasattr(baseline_score, "__float__") else baseline_score
    usage_baseline = aggregate_lm_usage()

    tp = MIPROv2(metric=motif_accuracy_metric, auto=auto)
    with dspy.settings.context(track_usage=True, cache=False):
        optimized = tp.compile(baseline, trainset=trainset, valset=valset)
        opt_score = evaluator(optimized)
    opt_val = float(opt_score) / 100 if hasattr(opt_score, "__float__") else opt_score
    usage_full = aggregate_lm_usage()

    optimized_prompt = get_prompt_from_module(optimized)
    if not original_prompt:
        original_prompt = get_prompt_from_module(baseline)

    metrics = {
        "baseline_score": baseline_val,
        "optimized_score": opt_val,
        "improvement": opt_val - baseline_val,
        "trainset_size": len(trainset),
        "valset_size": len(valset),
    }
    usage = {"baseline": usage_baseline, "full": usage_full}
    return optimized, metrics, usage, original_prompt, optimized_prompt


def run_bootstrap_finetune(
    trainset: List[MotifExample],
    valset: List[MotifExample],
    target_entity: str = None,
    instruction: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], str, str]:
    """Run BootstrapFinetune. Returns (optimized_module, metrics, usage, original_prompt, optimized_prompt)."""
    if not HAS_BOOTSTRAP_FINETUNE or BootstrapFinetune is None:
        raise ImportError("BootstrapFinetune not available. Upgrade dspy-ai.")
    entity = target_entity or config.target_entity
    original_prompt = instruction or ""

    if instruction:
        baseline = create_module_from_prompt(instruction, target_entity)
    else:
        baseline = create_motif_module(target_entity)

    lm = dspy.settings.lm
    if lm is None:
        raise ValueError("DSPy LM not configured. Call setup_dspy() first.")
    baseline.set_lm(lm)
    if hasattr(lm, "history"):
        lm.history = []

    evaluator = Evaluate(devset=valset, metric=motif_accuracy_metric, num_threads=1, display_progress=True, display_table=0)
    with dspy.settings.context(track_usage=True, cache=False):
        baseline_score = evaluator(baseline)
    baseline_val = float(baseline_score) / 100 if hasattr(baseline_score, "__float__") else baseline_score
    usage_baseline = aggregate_lm_usage()

    optimizer = BootstrapFinetune(metric=motif_accuracy_metric)
    with dspy.settings.context(track_usage=True, cache=False):
        optimized = optimizer.compile(baseline, trainset=trainset, valset=valset)
        opt_score = evaluator(optimized)
    opt_val = float(opt_score) / 100 if hasattr(opt_score, "__float__") else opt_score
    usage_full = aggregate_lm_usage()

    optimized_prompt = get_prompt_from_module(optimized)
    if not original_prompt:
        original_prompt = get_prompt_from_module(baseline)

    metrics = {
        "baseline_score": baseline_val,
        "optimized_score": opt_val,
        "improvement": opt_val - baseline_val,
        "trainset_size": len(trainset),
        "valset_size": len(valset),
    }
    usage = {"baseline": usage_baseline, "full": usage_full}
    return optimized, metrics, usage, original_prompt, optimized_prompt


def run_better_together(
    trainset: List[MotifExample],
    valset: List[MotifExample],
    target_entity: str = None,
    instruction: Optional[str] = None,
    auto: str = "light",
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], str, str]:
    """Run BetterTogether (MIPROv2 -> BootstrapFinetune). Returns (optimized_module, metrics, usage, original_prompt, optimized_prompt)."""
    if not HAS_BETTER_TOGETHER or BetterTogether is None:
        raise ImportError("BetterTogether not available. Upgrade dspy-ai.")
    if not HAS_MIPRO or MIPROv2 is None:
        raise ImportError("MIPROv2 required for BetterTogether.")
    if not HAS_BOOTSTRAP_FINETUNE or BootstrapFinetune is None:
        raise ImportError("BootstrapFinetune required for BetterTogether.")

    entity = target_entity or config.target_entity
    original_prompt = instruction or ""

    if instruction:
        baseline = create_module_from_prompt(instruction, target_entity)
    else:
        baseline = create_motif_module(target_entity)

    lm = dspy.settings.lm
    if lm is None:
        raise ValueError("DSPy LM not configured. Call setup_dspy() first.")
    baseline.set_lm(lm)
    if hasattr(lm, "history"):
        lm.history = []

    evaluator = Evaluate(devset=valset, metric=motif_accuracy_metric, num_threads=1, display_progress=True, display_table=0)
    with dspy.settings.context(track_usage=True, cache=False):
        baseline_score = evaluator(baseline)
    baseline_val = float(baseline_score) / 100 if hasattr(baseline_score, "__float__") else baseline_score
    usage_baseline = aggregate_lm_usage()

    optimizer = BetterTogether(
        metric=motif_accuracy_metric,
        p=MIPROv2(metric=motif_accuracy_metric, auto=auto),
        w=BootstrapFinetune(metric=motif_accuracy_metric),
    )
    with dspy.settings.context(track_usage=True, cache=False):
        optimized = optimizer.compile(baseline, trainset=trainset, valset=valset, strategy="p -> w")
        opt_score = evaluator(optimized)
    opt_val = float(opt_score) / 100 if hasattr(opt_score, "__float__") else opt_score
    usage_full = aggregate_lm_usage()

    optimized_prompt = get_prompt_from_module(optimized)
    if not original_prompt:
        original_prompt = get_prompt_from_module(baseline)

    metrics = {
        "baseline_score": baseline_val,
        "optimized_score": opt_val,
        "improvement": opt_val - baseline_val,
        "trainset_size": len(trainset),
        "valset_size": len(valset),
    }
    usage = {"baseline": usage_baseline, "full": usage_full}
    return optimized, metrics, usage, original_prompt, optimized_prompt


def run_comparison_pipeline(
    trainset: List[MotifExample],
    valset: List[MotifExample],
    target_entity: str = None,
) -> Dict[str, Any]:
    """
    Run baseline eval, BootstrapFewShot, and MIPROv2. Return aggregated results.
    """
    entity = target_entity or config.target_entity
    Sig = _make_signature_class(entity)

    class Mod(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.ChainOfThought(Sig)
        def forward(self, content: str):
            return self.predictor(content=content)

    evaluator = Evaluate(devset=valset, metric=motif_accuracy_metric, num_threads=1, display_progress=True, display_table=0)
    results = {"baseline": {}, "bootstrap": {}, "mipro": {}, "baseline_module": None, "bootstrap_module": None, "mipro_module": None}
    lm = dspy.settings.lm

    # Baseline
    baseline = Mod()
    if lm and hasattr(lm, "history"):
        lm.history = []
    with dspy.settings.context(track_usage=True, cache=False):
        bs = evaluator(baseline)
    results["baseline"]["score"] = float(bs) / 100 if hasattr(bs, "__float__") else bs
    results["baseline"]["usage"] = aggregate_lm_usage()
    results["baseline_module"] = baseline

    # BootstrapFewShot
    if BootstrapFewShot:
        if lm and hasattr(lm, "history"):
            lm.history = []
        opt_bfs = BootstrapFewShot(metric=motif_accuracy_metric, max_bootstrapped_demos=config.max_bootstrapped_demos, max_labeled_demos=config.max_labeled_demos, max_rounds=1)
        with dspy.settings.context(track_usage=True, cache=False):
            mod_bfs = opt_bfs.compile(Mod(), trainset=trainset)
            s_bfs = evaluator(mod_bfs)
        results["bootstrap"]["score"] = float(s_bfs) / 100 if hasattr(s_bfs, "__float__") else s_bfs
        results["bootstrap"]["usage"] = aggregate_lm_usage()
        results["bootstrap_module"] = mod_bfs

    # MIPROv2
    if HAS_MIPRO and MIPROv2:
        if lm and hasattr(lm, "history"):
            lm.history = []
        tp = MIPROv2(metric=motif_accuracy_metric, auto="light")
        with dspy.settings.context(track_usage=True, cache=False):
            mod_mipro = tp.compile(Mod(), trainset=trainset, valset=valset)
            s_mipro = evaluator(mod_mipro)
        results["mipro"]["score"] = float(s_mipro) / 100 if hasattr(s_mipro, "__float__") else s_mipro
        results["mipro"]["usage"] = aggregate_lm_usage()
        results["mipro_module"] = mod_mipro

    results["trainset"] = trainset
    results["valset"] = valset
    return results


# ============================================================================
# VISUALIZATIONS (Pure DSPy)
# ============================================================================

def plot_dataset_split(trainset: List[MotifExample], valset: List[MotifExample]):
    """Motif distribution across train/val split."""
    train_counts = defaultdict(int)
    val_counts = defaultdict(int)
    for ex in trainset:
        for m in ex.motifs or []:
            k = (m.get("actor", ""), m.get("action", "")) if isinstance(m, dict) else m
            train_counts[k] += 1
    for ex in valset:
        for m in ex.motifs or []:
            k = (m.get("actor", ""), m.get("action", "")) if isinstance(m, dict) else m
            val_counts[k] += 1
    all_motifs = sorted(set(train_counts) | set(val_counts))[:30]
    x = np.arange(len(all_motifs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, [train_counts[m] for m in all_motifs], w, label="Train", color="#2196F3")
    ax.bar(x + w/2, [val_counts[m] for m in all_motifs], w, label="Val", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m[0]}:{m[1]}"[:20] for m in all_motifs], rotation=45, ha="right")
    ax.set_title(f"Motif Distribution (Train: {len(trainset)}, Val: {len(valset)})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_optimization_process(metrics: Dict[str, Any]):
    """Baseline vs Optimized (2-panel)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    methods = ["Baseline\n(Zero-shot)", "Optimized\n(Few-shot)"]
    scores = [metrics["baseline_score"], metrics["optimized_score"]]
    colors = ["#FF6B6B", "#4CAF50"]
    bars = ax1.bar(methods, scores, color=colors, edgecolor="black", linewidth=2)
    for b, s in zip(bars, scores):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f"{s:.1%}", ha="center", va="bottom", fontsize=12)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Before vs After")
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis="y", alpha=0.3)
    imp = metrics.get("improvement", 0) * 100
    ax2.barh(["Improvement"], [imp], color="#2196F3")
    ax2.axvline(0, color="black")
    ax2.set_xlabel("Percentage Points")
    ax2.set_title("Absolute Improvement")
    ax2.text(imp/2, 0, f"{imp:+.1f}%", ha="center", va="center", fontsize=14, color="white")
    plt.tight_layout()
    plt.show()


def plot_error_analysis(module, valset: List[MotifExample], title: str):
    """Error breakdown: false positives, false negatives, perfect matches, parse errors."""
    errors = {"false_positives": 0, "false_negatives": 0, "perfect_matches": 0, "parse_errors": 0}
    for ex in valset:
        try:
            pred = module(content=ex.content)
            pred_motifs = parse_motifs(getattr(pred, "motifs", pred))
            pred_set = {(m.get("actor", ""), m.get("action", "")) for m in pred_motifs if isinstance(m, dict)}
            truth_set = {(m.get("actor", ""), m.get("action", "")) for m in (ex.motifs or []) if isinstance(m, dict)}
            if pred_set == truth_set:
                errors["perfect_matches"] += 1
            else:
                errors["false_positives"] += len(pred_set - truth_set)
                errors["false_negatives"] += len(truth_set - pred_set)
        except Exception:
            errors["parse_errors"] += 1
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = list(errors.keys())
    vals = list(errors.values())
    colors = ["#4CAF50", "#FF6B6B", "#FFA726", "#9C27B0"]
    ax.bar(cats, vals, color=colors)
    for b, v in zip(ax.patches, vals):
        if v > 0:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2, str(v), ha="center", fontsize=11)
    ax.set_title(f"Error Analysis: {title}")
    ax.set_xticklabels(cats, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(f"Perfect: {errors['perfect_matches']}/{len(valset)}, FP: {errors['false_positives']}, FN: {errors['false_negatives']}, Parse: {errors['parse_errors']}")


def visualize_prompt_evolution(baseline_module, optimized_module, embedding_model: str = None):
    """Prompt similarity analysis (embeddings-based)."""
    if not HAS_SENTENCE_TRANSFORMERS:
        print("sentence-transformers not installed. Skipping prompt evolution viz.")
        return
    try:
        model = SentenceTransformer(embedding_model or config.embedding_model)
        br = str(getattr(baseline_module.predictor, "signature", ""))
        or_ = str(getattr(optimized_module.predictor, "signature", ""))
        if hasattr(optimized_module.predictor, "demos") and optimized_module.predictor.demos:
            or_ += " " + " ".join(str(d)[:200] for d in optimized_module.predictor.demos)
        emb = model.encode([br, or_])
        sim = np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]))
        print(f"Prompt similarity (cosine): {sim:.3f}")
    except Exception as e:
        print(f"Prompt evolution viz: {e}")


def plot_optimization_comparison(results: Dict[str, Any]):
    """Bar chart: Baseline vs BootstrapFewShot vs MIPROv2."""
    methods, scores = [], []
    if "baseline" in results and "score" in results["baseline"]:
        methods.append("Baseline")
        scores.append(results["baseline"]["score"])
    if "bootstrap" in results and "score" in results["bootstrap"]:
        methods.append("BootstrapFewShot")
        scores.append(results["bootstrap"]["score"])
    if "mipro" in results and "score" in results["mipro"]:
        methods.append("MIPROv2")
        scores.append(results["mipro"]["score"])
    if not methods:
        print("No comparison data.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#FF6B6B", "#4CAF50", "#2196F3"][:len(methods)]
    bars = ax.bar(methods, scores, color=colors)
    for b, s in zip(bars, scores):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f"{s:.1%}", ha="center", va="bottom")
    ax.set_ylabel("Accuracy")
    ax.set_title("Optimization Comparison")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_recall_vs_f1_by_aspect(recall_f1: Dict[str, float]):
    """Grouped bar: Entity, Action, Motif x Recall/F1."""
    aspects = ["Entity", "Action", "Motif"]
    recalls = [recall_f1.get(f"{a.lower()}_recall", 0) for a in aspects]
    f1s = [recall_f1.get(f"{a.lower()}_f1", 0) for a in aspects]
    x = np.arange(len(aspects))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, recalls, w, label="Recall", color="#2196F3")
    ax.bar(x + w/2, f1s, w, label="F1", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels(aspects)
    ax.set_ylabel("Score")
    ax.set_title("Recall vs F1 by Aspect")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_token_cost_comparison(results: Dict[str, Any]):
    """Bar chart of total tokens per run."""
    methods, tokens, costs = [], [], []
    for name, key in [("Baseline", "baseline"), ("BootstrapFewShot", "bootstrap"), ("MIPROv2", "mipro")]:
        if key in results and "usage" in results[key]:
            u = results[key]["usage"]
            methods.append(name)
            tokens.append(u.get("total_tokens", 0))
            costs.append(u.get("cost", 0))
    if not methods:
        print("No usage data.")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(methods, tokens, color=["#FF6B6B", "#4CAF50", "#2196F3"][:len(methods)])
    ax1.set_ylabel("Total Tokens")
    ax1.set_title("Token Usage by Method")
    ax1.tick_params(axis="x", rotation=15)
    ax2.bar(methods, costs, color=["#FF6B6B", "#4CAF50", "#2196F3"][:len(methods)])
    ax2.set_ylabel("Cost ($)")
    ax2.set_title("Cost by Method")
    ax2.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.show()


def plot_metrics_heatmap(results: Dict[str, Any] = None, recall_f1: Dict[str, float] = None):
    """Heatmap: methods x metrics. Pass recall_f1 for aspect metrics, or results for method scores."""
    if recall_f1:
        metrics = ["Entity Recall", "Entity F1", "Action Recall", "Action F1", "Motif Recall", "Motif F1"]
        keys = ["entity_recall", "entity_f1", "action_recall", "action_f1", "motif_recall", "motif_f1"]
        arr = np.array([[recall_f1.get(k, 0) for k in keys]])
        row_labels = ["Combined"]
    elif results:
        row_labels = []
        scores = []
        for key, label in [("baseline", "Baseline"), ("bootstrap", "BootstrapFewShot"), ("mipro", "MIPROv2")]:
            if key in results and "score" in results[key]:
                row_labels.append(label)
                scores.append(results[key]["score"])
        if not row_labels:
            print("No heatmap data.")
            return
        arr = np.array(scores).reshape(-1, 1)
        metrics = ["Score"]
    else:
        print("No heatmap data.")
        return
    fig, ax = plt.subplots(figsize=(10, max(3, len(row_labels) * 0.5)))
    im = ax.imshow(arr, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    plt.colorbar(im, ax=ax)
    ax.set_title("Metrics Heatmap")
    plt.tight_layout()
    plt.show()


def plot_tokens_vs_accuracy(results: Dict[str, Any]):
    """Scatter: tokens vs accuracy."""
    tokens, scores, labels = [], [], []
    for key, label in [("baseline", "Baseline"), ("bootstrap", "BootstrapFewShot"), ("mipro", "MIPROv2")]:
        if key in results and "score" in results[key] and "usage" in results[key]:
            tokens.append(results[key]["usage"].get("total_tokens", 0))
            scores.append(results[key]["score"])
            labels.append(label)
    if len(tokens) < 2:
        print("Need at least 2 runs for scatter.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, l in enumerate(labels):
        ax.scatter(tokens[i], scores[i], s=150, label=l)
        ax.annotate(l, (tokens[i], scores[i]), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Total Tokens")
    ax.set_ylabel("Accuracy")
    ax.set_title("Tokens vs Accuracy (Efficiency)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(
    openai_results: Dict = None,
    llama_results: Dict = None,
    use_unseen: bool = True,
):
    """
    Compare gpt-4o-mini vs Llama-8b-instruct on entity, action, and motif F1 (unseen by default).
    """
    OPENAI_COLOR = "#81C784"
    LLAMA_COLOR = "#64B5F6"

    def _get_metrics(d: Dict, prefer_unseen: bool) -> Dict:
        if d is None:
            return None
        if "baseline" in d:
            return d["baseline"]
        if "unseen" in d and prefer_unseen:
            return d["unseen"]
        if "seen" in d:
            return d["seen"]
        return d if isinstance(d, dict) else None

    openai_m = _get_metrics(openai_results, use_unseen)
    llama_m = _get_metrics(llama_results, use_unseen)
    if openai_m is None and llama_m is None:
        print("No model comparison data.")
        return

    metrics_names = ["Entity F1", "Action F1", "Motif F1"]
    metric_keys = ["entity_f1", "action_f1", "motif_f1"]

    def _val(m: Dict, k: str) -> float:
        if m is None:
            return 0.0
        v = m.get(k)
        return float(v) if v is not None else 0.0

    x = np.arange(len(metrics_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    if openai_m is not None:
        openai_vals = [_val(openai_m, k) for k in metric_keys]
        ax.bar(x - w / 2, openai_vals, w, label="OpenAI", color=OPENAI_COLOR)
    if llama_m is not None:
        llama_vals = [_val(llama_m, k) for k in metric_keys]
        ax.bar(x + w / 2, llama_vals, w, label="Llama", color=LLAMA_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel("F1 Score (Unseen)" if use_unseen else "F1 Score")
    ax.set_title("gpt-4o-mini vs Llama-8b-instruct Comparison")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_full_comparison_whisker(
    results: Dict[str, Any],
    autogen_gpt4o: Dict[str, float] = None,
    autogen_llama: Dict[str, float] = None,
):
    """
    Whisker/box plot comparing all methods.
    results: dict with keys like 'mipro_basic', 'mipro_tool', etc.
    """
    OPENAI_COLOR = "#81C784"
    LLAMA_COLOR = "#64B5F6"
    DSPY_COLOR = "#FFB74D"

    def _to_values(d: Dict) -> List[float]:
        if d is None:
            return []
        if "score" in d and d["score"] is not None:
            return [float(d["score"])] * 3
        vals = []
        for k in ["entity_f1", "action_f1", "motif_f1"]:
            v = d.get(k)
            vals.append(float(v) if v is not None else 0.0)
        return vals if vals else [0.0, 0.0, 0.0]

    labels = []
    data = []
    colors = []

    if autogen_gpt4o is not None:
        labels.append("gpt-4o\n(autogen)")
        data.append(_to_values(autogen_gpt4o))
        colors.append(OPENAI_COLOR)
    if autogen_llama is not None:
        labels.append("llama\n(autogen)")
        data.append(_to_values(autogen_llama))
        colors.append(LLAMA_COLOR)

    for opt in ["mipro", "bootstrap_finetune", "better_together"]:
        for suffix in ["basic", "tool"]:
            key = f"{opt}_{suffix}"
            if key in results and results[key]:
                labels.append(f"{opt}\n({suffix})")
                data.append(_to_values(results[key]))
                colors.append(DSPY_COLOR)

    if not data:
        print("No comparison data.")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i] if i < len(colors) else DSPY_COLOR)
    ax.set_ylabel("Score (F1 / Accuracy)")
    ax.set_title("Full Comparison: Autogen vs DSPy Optimizers")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()


def print_optimization_summary(metrics: Dict[str, Any], trainset: List, valset: List):
    """Print optimization summary."""
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"Train: {len(trainset)}, Val: {len(valset)}")
    print(f"Baseline: {metrics.get('baseline_score', 0):.2%}")
    print(f"Optimized: {metrics.get('optimized_score', 0):.2%}")
    print(f"Improvement: {metrics.get('improvement', 0):+.2%}")
    print("="*60)


def analyze_few_shot_examples(module, trainset: List[MotifExample]):
    """Print few-shot demos selected by optimizer."""
    print("\n--- Few-Shot Examples Selected ---")
    if hasattr(module, "predictor") and hasattr(module.predictor, "demos") and module.predictor.demos:
        for i, d in enumerate(module.predictor.demos):
            print(f"Demo {i+1}: {str(d)[:200]}...")
    else:
        print("No demos found.")


def qualitative_comparison(baseline_module, optimized_module, valset: List[MotifExample], num_samples: int = 5):
    """Compare baseline vs optimized on a few examples."""
    n = min(num_samples, len(valset))
    for i in range(n):
        ex = valset[i]
        print(f"\n--- Example {i+1} ---")
        print(f"Content: {ex.content[:150]}...")
        print(f"Truth: {ex.motifs}")
        try:
            bp = baseline_module(content=ex.content)
            print(f"Baseline: {parse_motifs(getattr(bp, 'motifs', bp))}")
        except Exception as e:
            print(f"Baseline: ERROR {e}")
        try:
            op = optimized_module(content=ex.content)
            print(f"Optimized: {parse_motifs(getattr(op, 'motifs', op))}")
        except Exception as e:
            print(f"Optimized: ERROR {e}")


# ============================================================================
# DSPy + MCP TOOL SECTION
# Extends Part A with SpaCy NER/dep-parse tools via ReAct, safety audit logging.
# ============================================================================

# Context for current posts_id (used by safe tool wrapper)
_current_posts_id: ContextVar[int] = ContextVar("current_posts_id", default=0)

# Audit log and call counts (shared across tool executions)
_audit_log: List[Dict[str, Any]] = []
_posts_id_call_counts: Dict[int, int] = {}
MAX_CALLS_PER_POSTS_ID = 2


def _get_nlp():
    """Load SpaCy model. Lazy load to avoid import-time download."""
    global nlp
    if not HAS_SPACY:
        raise ImportError("spacy not installed. Run: pip install spacy && python -m spacy download en_core_web_lg")
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_lg")
        except OSError:
            raise RuntimeError(
                "SpaCy model en_core_web_lg not found. Run: python -m spacy download en_core_web_lg"
            )
    return nlp


def _ner_impl(text: str) -> str:
    """Named entity recognition. Returns JSON string of entities with entity, label, start, end."""
    if not text or not str(text).strip():
        return json.dumps({"error": "Text cannot be empty", "entities": []})
    nlp_model = _get_nlp()
    doc = nlp_model(text[:100000])
    entities = [
        {"entity": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]
    return json.dumps({"entities": entities})


def _dep_parse_impl(text: str) -> str:
    """Dependency parsing. Returns JSON string of tokens with token, pos, dep, head."""
    if not text or not str(text).strip():
        return json.dumps({"error": "Text cannot be empty", "tokens": []})
    nlp_model = _get_nlp()
    doc = nlp_model(text[:100000])
    tokens = [
        {
            "token": tok.text,
            "pos": tok.pos_,
            "dep": tok.dep_,
            "head": tok.head.text if tok.head else "",
        }
        for tok in doc
    ]
    return json.dumps({"tokens": tokens})


def _safe_execute(
    tool_name: str,
    impl: Callable[[str], str],
    text: str,
) -> str:
    """
    Execute tool with safety: useful errors, 2 calls/posts_id limit, audit logging.
    Returns JSON string with success/error and result.
    """
    posts_id = _current_posts_id.get()
    timestamp = datetime.now().isoformat()

    count = _posts_id_call_counts.get(posts_id, 0)
    if count >= MAX_CALLS_PER_POSTS_ID:
        result = json.dumps({
            "success": False,
            "error": "Max 2 tool calls per document exceeded. Cannot call more tools for this document.",
        })
        _audit_log.append({
            "timestamp": timestamp,
            "posts_id": posts_id,
            "tool_name": tool_name,
            "arguments": {"text": text[:100] + "..." if len(text) > 100 else text},
            "success": False,
            "error": "Max 2 tool calls per document exceeded",
            "result_summary": "BLOCKED",
        })
        return result

    try:
        raw_result = impl(text)
        _posts_id_call_counts[posts_id] = count + 1
        _audit_log.append({
            "timestamp": timestamp,
            "posts_id": posts_id,
            "tool_name": tool_name,
            "arguments": {"text": text[:100] + "..." if len(text) > 100 else text},
            "success": True,
            "error": None,
            "result_summary": str(raw_result)[:200] + ("..." if len(str(raw_result)) > 200 else ""),
        })
        return json.dumps({"success": True, "result": raw_result})
    except Exception as e:
        err_msg = str(e)
        if "en_core_web_lg" in err_msg or "model" in err_msg.lower():
            err_msg = "SpaCy model not loaded. Run: python -m spacy download en_core_web_lg"
        elif len(text) > 100000:
            err_msg = "Text too long. Maximum 100,000 characters allowed."
        _audit_log.append({
            "timestamp": timestamp,
            "posts_id": posts_id,
            "tool_name": tool_name,
            "arguments": {"text": text[:100] + "..." if len(text) > 100 else text},
            "success": False,
            "error": err_msg,
            "result_summary": "FAILED",
        })
        return json.dumps({"success": False, "error": err_msg})


def ner_tool(text: str) -> str:
    """
    Perform named entity recognition on text.
    Returns JSON with entities: [{entity, label, start, end}, ...].
    Use this to identify which entities (people, orgs, places) are present and where.
    """
    return _safe_execute("ner_tool", _ner_impl, text)


def dep_parse_tool(text: str) -> str:
    """
    Perform dependency parsing on text.
    Returns JSON with tokens: [{token, pos, dep, head}, ...].
    Use this to understand sentence structure and who does what (subject-verb-object).
    """
    return _safe_execute("dep_parse_tool", _dep_parse_impl, text)


def get_audit_log() -> List[Dict[str, Any]]:
    """Return the audit log for qualitative review."""
    return list(_audit_log)


def export_audit_log(path: str) -> None:
    """Export audit log to JSON file for review."""
    with open(path, "w") as f:
        json.dump(_audit_log, f, indent=2)
    print(f"Audit log exported to {path}")


def reset_audit_state() -> None:
    """Reset audit log and posts_id call counts (call before each evaluation run)."""
    global _audit_log, _posts_id_call_counts
    _audit_log = []
    _posts_id_call_counts = {}


def format_tool_use_log(audit_log: List[Dict[str, Any]] = None, max_entries: int = 50) -> str:
    """Format audit log for human-readable inspection of tool use."""
    log = audit_log if audit_log is not None else get_audit_log()
    if not log:
        return "No tool use log entries."
    lines = []
    for i, e in enumerate(log[:max_entries]):
        ts = e.get("timestamp", "?")
        tool = e.get("tool_name", "?")
        success = "OK" if e.get("success") else "FAIL"
        err = e.get("error", "")
        summary = (e.get("result_summary") or "")[:80]
        lines.append(f"[{i+1}] {ts} | {tool} | {success} | {summary}")
        if err:
            lines.append(f"    Error: {err}")
    if len(log) > max_entries:
        lines.append(f"... ({len(log) - max_entries} more entries)")
    return "\n".join(lines)


def create_mcp_examples(df) -> List[MotifExample]:
    """Create examples with content and posts_id for MCP pipeline (enables posts_id limit)."""
    examples = create_dspy_examples(df)
    return [ex.with_inputs("content", "posts_id") for ex in examples]


def prepare_mcp_trainset_valset(
    trainset: List[MotifExample],
    valset: List[MotifExample],
) -> Tuple[List[MotifExample], List[MotifExample]]:
    """Convert standard trainset/valset to MCP format (with_inputs content, posts_id)."""
    mcp_trainset = [ex.with_inputs("content", "posts_id") for ex in trainset]
    mcp_valset = [ex.with_inputs("content", "posts_id") for ex in valset]
    return mcp_trainset, mcp_valset


def prepare_mcp_testset(testset: List[MotifExample]) -> List[MotifExample]:
    """Convert testset to MCP format (with_inputs content, posts_id)."""
    if testset is None:
        return []
    return [ex.with_inputs("content", "posts_id") for ex in testset]


def _make_motif_react_signature(target_entity: str, instruction: Optional[str] = None):
    """Create ReAct signature for motif extraction with tools. instruction overrides default docstring."""
    default_doc = (
        f"Extract subject-verb relationships from social media text where the actor refers to {target_entity} "
        f"(including co-references like 'he', 'President {target_entity.split()[-1] if target_entity else ''}', etc.). "
        "You may use ner_tool to find entities and dep_parse_tool to understand sentence structure. "
        "Return the actual text span as actor. "
        "Output: Python list of dicts with 'actor' and 'action' keys. "
        "Example: [{'actor': 'Trump', 'action': 'sign'}]. Return [] if no subject-verb relationships found."
    )
    task_doc = instruction if instruction else default_doc
    doc = f"{WORKER_BEHAVIORAL_PROMPT}\n\n{task_doc}"

    class MotifReActSignature(dspy.Signature):
        __doc__ = doc
        content = dspy.InputField(desc=f"Social media post text to analyze for {target_entity}-related subject-verb relationships")
        motifs = dspy.OutputField(
            desc=f"List of {target_entity}-related subject-verb relationships as Python list of dicts with 'actor' and 'action'. "
            "Return [] if none found."
        )
    return MotifReActSignature


def create_motif_react_module_with_prompt(
    instruction: str,
    target_entity: str = None,
    max_iters: int = 5,
) -> "MotifReActModule":
    """Create MotifReActModule with custom instruction (e.g. from DSPy-optimized prompt)."""
    entity = target_entity or config.target_entity
    Sig = _make_motif_react_signature(entity, instruction=instruction)

    class ModWithPrompt(dspy.Module):
        def __init__(self):
            super().__init__()
            self.target_entity = entity
            self.react = dspy.ReAct(Sig, tools=[ner_tool, dep_parse_tool], max_iters=max_iters)

        def forward(self, content: str, posts_id: int = 0) -> dspy.Prediction:
            _current_posts_id.set(posts_id)
            return self.react(content=content)

    return ModWithPrompt()


def create_motif_react_module(
    target_entity: str = None,
    max_iters: int = 5,
) -> dspy.Module:
    """Create ReAct-based motif extractor with NER and dep_parse tools."""
    entity = target_entity or config.target_entity
    Sig = _make_motif_react_signature(entity)
    tools = [ner_tool, dep_parse_tool]
    return dspy.ReAct(Sig, tools=tools, max_iters=max_iters)


class MotifReActModule(dspy.Module):
    """
    ReAct module that accepts content and posts_id, sets context for safety, and delegates to ReAct.
    """

    def __init__(self, target_entity: str = None, max_iters: int = 5):
        super().__init__()
        self.target_entity = target_entity or config.target_entity
        self.max_iters = max_iters
        Sig = _make_motif_react_signature(self.target_entity)
        self.react = dspy.ReAct(
            Sig,
            tools=[ner_tool, dep_parse_tool],
            max_iters=max_iters,
        )

    def forward(self, content: str, posts_id: int = 0) -> dspy.Prediction:
        """Run ReAct with posts_id context for safety limits."""
        _current_posts_id.set(posts_id)
        return self.react(content=content)


def motif_accuracy_metric_mcp(example: MotifExample, prediction, trace=None) -> float:
    """Same as motif_accuracy_metric but works with ReAct output (may have 'answer' or 'motifs')."""
    return motif_accuracy_metric(example, prediction, trace)


def run_mcp_baseline(
    valset: List[MotifExample],
    target_entity: str = None,
) -> Tuple[Any, float, Dict[str, Any]]:
    """Run MCP ReAct baseline (no optimization). Returns (module, score, usage)."""
    reset_audit_state()
    module = MotifReActModule(target_entity=target_entity)
    evaluator = Evaluate(
        devset=valset,
        metric=motif_accuracy_metric_mcp,
        num_threads=1,
        display_progress=True,
        display_table=0,
    )
    lm = dspy.settings.lm
    if lm and hasattr(lm, "history"):
        lm.history = []
    with dspy.settings.context(track_usage=True, cache=False):
        score = evaluator(module)
    score_val = float(score) / 100 if hasattr(score, "__float__") else score
    usage = aggregate_lm_usage()
    return module, score_val, usage


def run_mcp_bootstrap_fewshot(
    trainset: List[MotifExample],
    valset: List[MotifExample],
    target_entity: str = None,
    instruction: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], str, str, List[Dict[str, Any]]]:
    """Run BootstrapFewShot on MCP ReAct module. Returns (optimized_module, metrics, usage, original_prompt, optimized_prompt, tool_log)."""
    if not HAS_BOOTSTRAP or BootstrapFewShot is None:
        raise ImportError("BootstrapFewShot not available. Install dspy-ai.")
    reset_audit_state()
    entity = target_entity or config.target_entity
    original_prompt = instruction or ""

    if instruction:
        baseline = create_motif_react_module_with_prompt(instruction, target_entity=entity)
    else:
        baseline = MotifReActModule(target_entity=entity)

    evaluator = Evaluate(
        devset=valset,
        metric=motif_accuracy_metric_mcp,
        num_threads=1,
        display_progress=True,
        display_table=0,
    )
    lm = dspy.settings.lm
    if lm and hasattr(lm, "history"):
        lm.history = []
    with dspy.settings.context(track_usage=True, cache=False):
        baseline_score = evaluator(baseline)
    baseline_val = float(baseline_score) / 100 if hasattr(baseline_score, "__float__") else baseline_score
    usage_baseline = aggregate_lm_usage()

    optimizer = BootstrapFewShot(
        metric=motif_accuracy_metric_mcp,
        max_bootstrapped_demos=config.max_bootstrapped_demos,
        max_labeled_demos=config.max_labeled_demos,
        max_rounds=1,
    )
    reset_audit_state()
    with dspy.settings.context(track_usage=True, cache=False):
        optimized = optimizer.compile(baseline, trainset=trainset)
        opt_score = evaluator(optimized)
    opt_val = float(opt_score) / 100 if hasattr(opt_score, "__float__") else opt_score
    usage_full = aggregate_lm_usage()

    optimized_prompt = get_prompt_from_module(optimized)
    if not original_prompt:
        original_prompt = get_prompt_from_module(baseline)
    tool_log = get_audit_log()

    metrics = {
        "baseline_score": baseline_val,
        "optimized_score": opt_val,
        "improvement": opt_val - baseline_val,
        "trainset_size": len(trainset),
        "valset_size": len(valset),
    }
    usage = {"baseline": usage_baseline, "full": usage_full}
    return optimized, metrics, usage, original_prompt, optimized_prompt, tool_log


def run_mcp_mipro(
    trainset: List[MotifExample],
    valset: List[MotifExample],
    target_entity: str = None,
    instruction: Optional[str] = None,
    auto: str = "light",
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], str, str, List[Dict[str, Any]]]:
    """Run MIPROv2 on MCP ReAct module. Returns (optimized_module, metrics, usage, original_prompt, optimized_prompt, tool_log)."""
    if not HAS_MIPRO or MIPROv2 is None:
        raise ImportError("MIPROv2 not available. Upgrade dspy-ai.")
    reset_audit_state()
    entity = target_entity or config.target_entity
    original_prompt = instruction or ""

    if instruction:
        baseline = create_motif_react_module_with_prompt(instruction, target_entity=entity)
    else:
        baseline = MotifReActModule(target_entity=entity)

    evaluator = Evaluate(
        devset=valset,
        metric=motif_accuracy_metric_mcp,
        num_threads=1,
        display_progress=True,
        display_table=0,
    )
    lm = dspy.settings.lm
    if lm and hasattr(lm, "history"):
        lm.history = []
    with dspy.settings.context(track_usage=True, cache=False):
        baseline_score = evaluator(baseline)
    baseline_val = float(baseline_score) / 100 if hasattr(baseline_score, "__float__") else baseline_score
    usage_baseline = aggregate_lm_usage()

    optimizer = MIPROv2(metric=motif_accuracy_metric_mcp, auto=auto)
    reset_audit_state()
    with dspy.settings.context(track_usage=True, cache=False):
        optimized = optimizer.compile(baseline, trainset=trainset, valset=valset)
        opt_score = evaluator(optimized)
    opt_val = float(opt_score) / 100 if hasattr(opt_score, "__float__") else opt_score
    usage_full = aggregate_lm_usage()

    optimized_prompt = get_prompt_from_module(optimized)
    if not original_prompt:
        original_prompt = get_prompt_from_module(baseline)
    tool_log = get_audit_log()

    metrics = {
        "baseline_score": baseline_val,
        "optimized_score": opt_val,
        "improvement": opt_val - baseline_val,
        "trainset_size": len(trainset),
        "valset_size": len(valset),
    }
    usage = {"baseline": usage_baseline, "full": usage_full}
    return optimized, metrics, usage, original_prompt, optimized_prompt, tool_log


def run_mcp_bootstrap_finetune(
    trainset: List[MotifExample],
    valset: List[MotifExample],
    target_entity: str = None,
    instruction: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], str, str, List[Dict[str, Any]]]:
    """Run BootstrapFinetune on MCP ReAct module. Returns (optimized_module, metrics, usage, original_prompt, optimized_prompt, tool_log)."""
    if not HAS_BOOTSTRAP_FINETUNE or BootstrapFinetune is None:
        raise ImportError("BootstrapFinetune not available. Upgrade dspy-ai.")
    reset_audit_state()
    entity = target_entity or config.target_entity
    original_prompt = instruction or ""

    if instruction:
        baseline = create_motif_react_module_with_prompt(instruction, target_entity=entity)
    else:
        baseline = MotifReActModule(target_entity=entity)

    lm = dspy.settings.lm
    if lm is None:
        raise ValueError("DSPy LM not configured. Call setup_dspy() first.")
    baseline.set_lm(lm)
    if hasattr(lm, "history"):
        lm.history = []

    evaluator = Evaluate(
        devset=valset,
        metric=motif_accuracy_metric_mcp,
        num_threads=1,
        display_progress=True,
        display_table=0,
    )
    with dspy.settings.context(track_usage=True, cache=False):
        baseline_score = evaluator(baseline)
    baseline_val = float(baseline_score) / 100 if hasattr(baseline_score, "__float__") else baseline_score
    usage_baseline = aggregate_lm_usage()

    optimizer = BootstrapFinetune(metric=motif_accuracy_metric_mcp)
    reset_audit_state()
    with dspy.settings.context(track_usage=True, cache=False):
        optimized = optimizer.compile(baseline, trainset=trainset, valset=valset)
        opt_score = evaluator(optimized)
    opt_val = float(opt_score) / 100 if hasattr(opt_score, "__float__") else opt_score
    usage_full = aggregate_lm_usage()

    optimized_prompt = get_prompt_from_module(optimized)
    if not original_prompt:
        original_prompt = get_prompt_from_module(baseline)
    tool_log = get_audit_log()

    metrics = {
        "baseline_score": baseline_val,
        "optimized_score": opt_val,
        "improvement": opt_val - baseline_val,
        "trainset_size": len(trainset),
        "valset_size": len(valset),
    }
    usage = {"baseline": usage_baseline, "full": usage_full}
    return optimized, metrics, usage, original_prompt, optimized_prompt, tool_log


def run_mcp_better_together(
    trainset: List[MotifExample],
    valset: List[MotifExample],
    target_entity: str = None,
    instruction: Optional[str] = None,
    auto: str = "light",
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], str, str, List[Dict[str, Any]]]:
    """Run BetterTogether (MIPROv2 -> BootstrapFinetune) on MCP ReAct module. Returns (optimized_module, metrics, usage, original_prompt, optimized_prompt, tool_log)."""
    if not HAS_BETTER_TOGETHER or BetterTogether is None:
        raise ImportError("BetterTogether not available. Upgrade dspy-ai.")
    if not HAS_MIPRO or MIPROv2 is None:
        raise ImportError("MIPROv2 required for BetterTogether.")
    if not HAS_BOOTSTRAP_FINETUNE or BootstrapFinetune is None:
        raise ImportError("BootstrapFinetune required for BetterTogether.")
    reset_audit_state()
    entity = target_entity or config.target_entity
    original_prompt = instruction or ""

    if instruction:
        baseline = create_motif_react_module_with_prompt(instruction, target_entity=entity)
    else:
        baseline = MotifReActModule(target_entity=entity)

    lm = dspy.settings.lm
    if lm is None:
        raise ValueError("DSPy LM not configured. Call setup_dspy() first.")
    baseline.set_lm(lm)
    if hasattr(lm, "history"):
        lm.history = []

    evaluator = Evaluate(
        devset=valset,
        metric=motif_accuracy_metric_mcp,
        num_threads=1,
        display_progress=True,
        display_table=0,
    )
    with dspy.settings.context(track_usage=True, cache=False):
        baseline_score = evaluator(baseline)
    baseline_val = float(baseline_score) / 100 if hasattr(baseline_score, "__float__") else baseline_score
    usage_baseline = aggregate_lm_usage()

    optimizer = BetterTogether(
        metric=motif_accuracy_metric_mcp,
        p=MIPROv2(metric=motif_accuracy_metric_mcp, auto=auto),
        w=BootstrapFinetune(metric=motif_accuracy_metric_mcp),
    )
    reset_audit_state()
    with dspy.settings.context(track_usage=True, cache=False):
        optimized = optimizer.compile(baseline, trainset=trainset, valset=valset, strategy="p -> w")
        opt_score = evaluator(optimized)
    opt_val = float(opt_score) / 100 if hasattr(opt_score, "__float__") else opt_score
    usage_full = aggregate_lm_usage()

    optimized_prompt = get_prompt_from_module(optimized)
    if not original_prompt:
        original_prompt = get_prompt_from_module(baseline)
    tool_log = get_audit_log()

    metrics = {
        "baseline_score": baseline_val,
        "optimized_score": opt_val,
        "improvement": opt_val - baseline_val,
        "trainset_size": len(trainset),
        "valset_size": len(valset),
    }
    usage = {"baseline": usage_baseline, "full": usage_full}
    return optimized, metrics, usage, original_prompt, optimized_prompt, tool_log


# ============================================================================
# MCP VISUALIZATIONS
# ============================================================================

def plot_audit_log_summary(audit_log: List[Dict[str, Any]] = None) -> None:
    """Bar chart of tool calls by tool, success/failure counts, calls per posts_id distribution."""
    log = audit_log if audit_log is not None else get_audit_log()
    if not log:
        print("No audit log entries.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    tool_counts = {}
    for e in log:
        t = e.get("tool_name", "unknown")
        tool_counts[t] = tool_counts.get(t, 0) + 1
    ax1 = axes[0]
    tools = list(tool_counts.keys())
    counts = list(tool_counts.values())
    ax1.bar(tools, counts, color=["#2196F3", "#4CAF50"][:len(tools)])
    ax1.set_ylabel("Count")
    ax1.set_title("Tool Calls by Tool")
    ax1.tick_params(axis="x", rotation=15)

    success_count = sum(1 for e in log if e.get("success"))
    fail_count = len(log) - success_count
    ax2 = axes[1]
    ax2.bar(["Success", "Failure"], [success_count, fail_count], color=["#4CAF50", "#FF6B6B"])
    ax2.set_ylabel("Count")
    ax2.set_title("Success vs Failure")

    posts_calls = {}
    for e in log:
        pid = e.get("posts_id", 0)
        posts_calls[pid] = posts_calls.get(pid, 0) + 1
    call_dist = list(posts_calls.values())
    ax3 = axes[2]
    if call_dist:
        ax3.hist(call_dist, bins=min(10, max(call_dist) + 1), color="#9C27B0", edgecolor="black")
    ax3.set_xlabel("Calls per posts_id")
    ax3.set_ylabel("Number of docs")
    ax3.set_title("Calls per Document Distribution")
    plt.tight_layout()
    plt.show()


def plot_combined_token_cost(
    mcp_usage: Dict[str, Any],
    pure_results: Dict[str, Any],
) -> None:
    """Bar chart of tokens and cost for MCP and Pure DSPy methods."""
    methods, tokens, costs = [], [], []
    if mcp_usage:
        mcp_base = mcp_usage.get("baseline", {})
        mcp_full = mcp_usage.get("full", mcp_base)
        methods.extend(["MCP Baseline", "MCP Optimized"])
        tokens.extend([mcp_base.get("total_tokens", 0), mcp_full.get("total_tokens", 0)])
        costs.extend([mcp_base.get("cost", 0), mcp_full.get("cost", 0)])
    for name, key in [("Pure DSPy Baseline", "baseline"), ("Pure DSPy Bootstrap", "bootstrap")]:
        if pure_results and key in pure_results and pure_results[key]:
            u = pure_results[key].get("usage", {})
            methods.append(name)
            tokens.append(u.get("total_tokens", 0))
            costs.append(u.get("cost", 0))
    if not methods:
        print("No usage data.")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#2196F3", "#2196F3", "#FF9800", "#FF9800"][:len(methods)]
    ax1.bar(methods, tokens, color=colors)
    ax1.set_ylabel("Total Tokens")
    ax1.set_title("Token Usage by Method")
    ax1.tick_params(axis="x", rotation=15)
    ax2.bar(methods, costs, color=colors)
    ax2.set_ylabel("Cost ($)")
    ax2.set_title("Cost by Method")
    ax2.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.show()


def plot_mcp_vs_pure_dspy(
    mcp_results: Dict[str, Any],
    pure_results: Dict[str, Any],
) -> None:
    """Side-by-side MCP vs pure DSPy on accuracy and cost."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    methods = ["MCP (ReAct)", "Pure DSPy"]
    mcp_score = mcp_results.get("optimized_score") or mcp_results.get("baseline_score", 0)
    pure_score = (
        pure_results.get("bootstrap", {}).get("score")
        or pure_results.get("baseline", {}).get("score")
        or 0
    )
    scores = [mcp_score, pure_score]
    ax1.bar(methods, scores, color=["#2196F3", "#FF9800"])
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy Comparison")
    ax1.set_ylim(0, 1.1)
    for i, s in enumerate(scores):
        ax1.text(i, s + 0.02, f"{s:.1%}", ha="center", va="bottom")

    mcp_usage = mcp_results.get("usage", {})
    mcp_cost = mcp_usage.get("full", {}).get("cost") or mcp_usage.get("baseline", {}).get("cost", 0)
    pure_usage = pure_results.get("bootstrap", {}).get("usage") or pure_results.get("baseline", {}).get("usage", {})
    pure_cost = pure_usage.get("cost", 0)
    costs = [mcp_cost, pure_cost]
    ax2.bar(methods, costs, color=["#2196F3", "#FF9800"])
    ax2.set_ylabel("Cost ($)")
    ax2.set_title("Cost Comparison")
    for i, c in enumerate(costs):
        ax2.text(i, c + 0.0001, f"${c:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_error_analysis_mcp(module, valset: List[MotifExample], title: str) -> None:
    """Error analysis for MCP module (passes posts_id)."""
    errors = {"false_positives": 0, "false_negatives": 0, "perfect_matches": 0, "parse_errors": 0}
    for ex in valset:
        try:
            content = ex.content
            posts_id = getattr(ex, "posts_id", 0)
            pred = module(content=content, posts_id=posts_id)
            pred_motifs = parse_motifs(getattr(pred, "motifs", getattr(pred, "answer", pred)))
            pred_set = {(m.get("actor", ""), m.get("action", "")) for m in pred_motifs if isinstance(m, dict)}
            truth_set = {(m.get("actor", ""), m.get("action", "")) for m in (ex.motifs or []) if isinstance(m, dict)}
            if pred_set == truth_set:
                errors["perfect_matches"] += 1
            else:
                errors["false_positives"] += len(pred_set - truth_set)
                errors["false_negatives"] += len(truth_set - pred_set)
        except Exception:
            errors["parse_errors"] += 1
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = list(errors.keys())
    vals = list(errors.values())
    colors = ["#4CAF50", "#FF6B6B", "#FFA726", "#9C27B0"]
    ax.bar(cats, vals, color=colors)
    for b, v in zip(ax.patches, vals):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, str(v), ha="center", fontsize=11)
    ax.set_title(f"Error Analysis: {title}")
    ax.set_xticklabels(cats, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(f"Perfect: {errors['perfect_matches']}/{len(valset)}, FP: {errors['false_positives']}, FN: {errors['false_negatives']}, Parse: {errors['parse_errors']}")


# Alias for backward compatibility with code that references HAS_BOOTSTRAP
HAS_BOOTSTRAP = BootstrapFewShot is not None
