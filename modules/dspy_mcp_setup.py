"""
DSPy MCP Motif Extraction Setup Module
========================================
Extends the DSPy motif extraction pipeline with SpaCy-based MCP tools (NER, dependency parsing),
safety measures (error messages, 2 calls/posts_id limit, audit logging), and ReAct integration.

Supports dual model backends (same pipeline, parameter-regulated):
- OpenAI (default): use_llama=False in setup_dspy()
- Llama-3.1-8B via SGLang: use_llama=True, ensure SGLang server at sglang_base (default port 7501)

Designed for Google Colab. All logic lives here; notebook imports and runs in segments.
"""

import json
from contextvars import ContextVar
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt

# Import from dspy_setup
try:
    from .prompts import WORKER_BEHAVIORAL_PROMPT
except ImportError:
    WORKER_BEHAVIORAL_PROMPT = (
        "You are a highly efficient information detection and extraction engine. "
        "You value accuracy and efficiency. Follow formatting conventions in the extraction prompt."
    )

from .dspy_setup import (
    config,
    MotifExample,
    create_dspy_examples,
    parse_motifs,
    motif_accuracy_metric,
    compute_recall_f1,
    aggregate_lm_usage,
    setup_dspy,
    load_and_prepare_data,
    plot_dataset_split,
    plot_optimization_process,
    plot_error_analysis,
    plot_optimization_comparison,
    plot_recall_vs_f1_by_aspect,
    plot_token_cost_comparison,
    plot_tokens_vs_accuracy,
    plot_metrics_heatmap,
    plot_full_comparison_whisker,
    get_prompt_from_module,
    print_qualitative_prompt_comparison,
)

import dspy
from dspy.evaluate import Evaluate

try:
    from dspy.teleprompt import BootstrapFewShot
    HAS_BOOTSTRAP = True
except ImportError:
    BootstrapFewShot = None
    HAS_BOOTSTRAP = False

try:
    from dspy.teleprompt import MIPROv2, BootstrapFinetune, BetterTogether
    HAS_MIPRO = MIPROv2 is not None
    HAS_BOOTSTRAP_FINETUNE = BootstrapFinetune is not None
    HAS_BETTER_TOGETHER = BetterTogether is not None
except (ImportError, AttributeError):
    MIPROv2 = None
    BootstrapFinetune = None
    BetterTogether = None
    HAS_MIPRO = False
    HAS_BOOTSTRAP_FINETUNE = False
    HAS_BETTER_TOGETHER = False

# SpaCy
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    nlp = None

# Context for current posts_id (used by safe tool wrapper)
_current_posts_id: ContextVar[int] = ContextVar("current_posts_id", default=0)

# Audit log and call counts (shared across tool executions)
_audit_log: List[Dict[str, Any]] = []
_posts_id_call_counts: Dict[int, int] = {}
MAX_CALLS_PER_POSTS_ID = 2


# ============================================================================
# SPACY TOOLS
# ============================================================================

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
    doc = nlp_model(text[:100000])  # Limit length to avoid OOM
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


# ============================================================================
# SAFE TOOL WRAPPER
# ============================================================================

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

    # Check call limit
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

    # Execute
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


# ============================================================================
# AUDIT LOG HELPERS
# ============================================================================

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


# ============================================================================
# MCP EXAMPLES (with posts_id for safety)
# ============================================================================

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


# ============================================================================
# MOTIF REACT MODULE
# ============================================================================

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


# ============================================================================
# MCP METRIC (handles posts_id)
# ============================================================================

def motif_accuracy_metric_mcp(example: MotifExample, prediction, trace=None) -> float:
    """Same as motif_accuracy_metric but works with ReAct output (may have 'answer' or 'motifs')."""
    return motif_accuracy_metric(example, prediction, trace)


# ============================================================================
# RUN MCP BASELINE AND BOOTSTRAP
# ============================================================================

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
# VISUALIZATIONS
# ============================================================================

def plot_audit_log_summary(audit_log: List[Dict[str, Any]] = None) -> None:
    """Bar chart of tool calls by tool, success/failure counts, calls per posts_id distribution."""
    log = audit_log if audit_log is not None else get_audit_log()
    if not log:
        print("No audit log entries.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # 1. Calls by tool
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

    # 2. Success vs failure
    success_count = sum(1 for e in log if e.get("success"))
    fail_count = len(log) - success_count
    ax2 = axes[1]
    ax2.bar(["Success", "Failure"], [success_count, fail_count], color=["#4CAF50", "#FF6B6B"])
    ax2.set_ylabel("Count")
    ax2.set_title("Success vs Failure")

    # 3. Calls per posts_id distribution
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
