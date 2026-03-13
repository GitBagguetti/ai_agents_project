"""
SpaCy Tools for Information Extraction
======================================
Adds SpaCy-based NER and dependency parsing tools to an LLM (gpt-4o-mini) for
structured information extraction. Works with AutoGen Stage 3, outside DSPy.

Includes safety measures (2 calls per document, audit logging), visualization
functions, and run_stage3_with_tools for tool-assisted extraction.

Designed for Google Colab.
"""

import json
import os
from contextvars import ContextVar
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Callable

import numpy as np

# #region agent log
_DEBUG_LOG_PATH: Optional[str] = None

def _dbg(data: dict) -> None:
    global _DEBUG_LOG_PATH
    candidates = [
        os.path.join(os.getcwd(), "debug-584311.log"),  # Colab: cwd is /content
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "debug-584311.log"),
    ]
    payload = {"sessionId": "584311", "timestamp": datetime.now().timestamp() * 1000, **data}
    line = json.dumps(payload, default=str) + "\n"
    for p in candidates:
        try:
            with open(p, "a", encoding="utf-8") as f:
                f.write(line)
            globals()["_DEBUG_LOG_PATH"] = p
            return
        except Exception:
            continue
# #endregion
import pandas as pd
import matplotlib.pyplot as plt

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
        raise ImportError(
            "spacy not installed. Run: pip install spacy && python -m spacy download en_core_web_lg"
        )
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_lg")
        except OSError:
            raise RuntimeError(
                "SpaCy model en_core_web_lg not found. Run: python -m spacy download en_core_web_lg"
            )
    return nlp


def _ner_impl(text: str) -> str:
    """Named entity recognition. Returns JSON string of entities."""
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
    """Dependency parsing. Returns JSON string of tokens."""
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


def _clear_autogen_globals() -> None:
    """Clear AutoGen globals to ensure runs are completely unaffected by prior state."""
    try:
        import autogen
        if hasattr(autogen, "runtime_logging"):
            try:
                autogen.runtime_logging.stop()
            except Exception:
                pass
    except Exception:
        pass
    for mod_name, attr in [
        ("autogen.oai.client", "ChatCompletion"),
        ("autogen.oai.openai_utils", "OpenAIWrapper"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            obj = getattr(mod, attr, None)
            if obj is not None and hasattr(obj, "clear_usage_summary"):
                obj.clear_usage_summary()
        except Exception:
            continue


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
# PACKAGE CHECK: autogen-agentchat nested chats
# ============================================================================

def _has_nested_chats_support() -> bool:
    """Detect whether autogen-agentchat~=0.2 is available (register_nested_chats)."""
    try:
        from autogen import ConversableAgent
        return hasattr(ConversableAgent, "register_nested_chats")
    except Exception:
        return False


# ============================================================================
# STAGE 3 WITH TOOLS
# ============================================================================

def _is_extraction_done(msg: dict) -> bool:
    """
    Termination check for UserProxy/ToolProxy (executor agents).
    Returns True when:
      (a) the worker explicitly included "TERMINATE" anywhere in its reply, OR
      (b) the worker returned a self-contained JSON object/array — meaning the
          extraction is complete but the model forgot to append TERMINATE.
    This prevents the 20-round echo loop that occurs when the worker returns
    correct JSON without TERMINATE and UserProxy bounces back an empty string.
    """
    content = str(msg.get("content") or "").strip()
    if not content:
        return False
    if "TERMINATE" in content:
        return True
    if content.startswith("{") or content.startswith("["):
        try:
            json.loads(content)
            return True
        except Exception:
            pass
    return False


WORKER_TOOLS_SYSTEM_APPEND = """

You have access to optional tools:
- ner_tool(text): Perform named entity recognition. Use to identify people, orgs, places in the text.
- dep_parse_tool(text): Perform dependency parsing. Use to understand sentence structure and subject-verb-object relationships.

You may call these tools to help with extraction, but you are not required to. When you have finished extraction, include TERMINATE in your final message. Return ONLY valid JSON with your extraction results."""


def run_stage3_with_tools(
    final_prompt: str,
    seen_df: pd.DataFrame,
    unseen_df: pd.DataFrame,
    llm_config: Optional[dict] = None,
) -> Tuple[Dict, Dict, List, List, List[Dict[str, Any]]]:
    """
    Run Stage 3 with SpaCy tools: Worker (gpt-4o-mini) extracts motifs using optional
    ner_tool and dep_parse_tool. Returns (metrics_seen, metrics_unseen, preds_seen, preds_unseen, audit_log).
    """
    from autogen import ConversableAgent
    from .autogen_setup import (
        get_llm_config,
        config,
        WORKER_SYSTEM_PROMPT,
        _format_sample_prompts_block,
        _parse_worker_response,
        _ground_truth_to_motifs,
        compute_recall_f1,
    )

    print(f"\n{'='*60}")
    print("STAGE 3: Tool-Assisted Extraction (gpt-4o-mini + SpaCy)")
    print(f"{'='*60}")

    worker_llm_config = llm_config or get_llm_config(config.worker_temp)

    # User proxy executes tools; only the executor checks for termination
    user_proxy = ConversableAgent(
        name="UserProxy",
        llm_config=False,
        human_input_mode="NEVER",
        is_termination_msg=_is_extraction_done,
    )

    # Worker with tools — no is_termination_msg so the incoming prompt (which mentions
    # the word TERMINATE) does not cause immediate self-termination before a reply is sent
    worker_system = WORKER_SYSTEM_PROMPT + WORKER_TOOLS_SYSTEM_APPEND
    worker = ConversableAgent(
        name="WorkerLLM",
        system_message=worker_system,
        llm_config=worker_llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=20,
    )

    # Register tools per AutoGen 0.2: register_for_llm (caller) + register_for_execution (executor)
    try:
        worker.register_for_llm(name="ner_tool", description="Perform named entity recognition on text. Returns JSON with entities. Use to identify people, orgs, places.")(ner_tool)
        user_proxy.register_for_execution(name="ner_tool")(ner_tool)
        worker.register_for_llm(name="dep_parse_tool", description="Perform dependency parsing on text. Returns JSON with tokens. Use to understand sentence structure and who does what.")(dep_parse_tool)
        user_proxy.register_for_execution(name="dep_parse_tool")(dep_parse_tool)
    except AttributeError:
        from autogen import register_function
        register_function(ner_tool, caller=worker, executor=user_proxy, name="ner_tool",
            description="Perform named entity recognition on text. Returns JSON with entities. Use to identify people, orgs, places.")
        register_function(dep_parse_tool, caller=worker, executor=user_proxy, name="dep_parse_tool",
            description="Perform dependency parsing on text. Returns JSON with tokens. Use to understand sentence structure and who does what.")

    sample_block = _format_sample_prompts_block()

    def _extract_from_chat(chat_result, initiator, recipient, captured_msgs: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """Parse Worker's JSON extraction from chat. Prefer captured_msgs from register_reply hook."""
        # #region agent log
        _dbg({"hypothesisId": "A", "location": "spacy_tools:_extract_from_chat", "message": "chat_result_in", "data": {"cr_is_none": chat_result is None, "captured_len": len(captured_msgs) if captured_msgs else 0}})
        # #endregion
        if chat_result is None and not captured_msgs:
            return []
        def _is_prompt(c: str) -> bool:
            c = c.strip().lower()
            return (c.startswith("you are tasked with") or "text to analyze:" in c[:300] or
                    c.startswith("example prompts"))

        # Prefer captured messages from Worker (register_reply hook)
        if captured_msgs:
            for content in reversed(captured_msgs):
                if content and not _is_prompt(content):
                    parsed = _parse_worker_response(content)
                    if parsed:
                        return parsed
        if chat_result is None:
            return []
        history = getattr(chat_result, "chat_history", None) or []
        # Prefer agent chat_messages: ChatResult.chat_history can be wrong/truncated with tools
        for agent, other in ((initiator, recipient), (recipient, initiator)):
            if agent is None or other is None:
                continue
            for attr in ("chat_messages", "_oai_messages"):
                msgs = getattr(agent, attr, None)
                if not isinstance(msgs, dict):
                    continue
                for k, v in msgs.items():
                    cand = v if isinstance(v, list) else []
                    if k is other or (getattr(k, "name", None) == getattr(other, "name", None)):
                        if len(cand) > len(history):
                            history = cand
                        break
                if len(history) > 1:
                    break
            if len(history) > 1:
                break
        # #region agent log
        _dbg({"hypothesisId": "B", "location": "spacy_tools:_extract_from_chat", "message": "history_structure", "data": {"len": len(history), "msg_roles": [m.get("role", "?") for m in history[:12]], "content_previews": [str(m.get("content") or "")[:80] for m in history[:6]]}})
        # #endregion
        # Scan ALL messages for valid extraction JSON (roles are unreliable with tools).
        for msg in reversed(history):
            content = msg.get("content") or ""
            if not content or _is_prompt(content):
                continue
            parsed = _parse_worker_response(str(content))
            # #region agent log
            _dbg({"hypothesisId": "C", "location": "spacy_tools:_extract_from_chat", "message": "msg_parsed", "data": {"content_preview": str(content)[:200], "parsed": parsed}})
            # #endregion
            if parsed:
                return parsed
        summary = getattr(chat_result, "summary", None)
        if summary:
            parsed = _parse_worker_response(str(summary))
            if parsed:
                return parsed
        # #region agent log
        _dbg({"hypothesisId": "D", "location": "spacy_tools:_extract_from_chat", "message": "no_parse", "data": {}})
        # #endregion
        return []

    # Capture last message from Worker via reply hook (ChatResult.chat_history can be wrong with tools)
    _last_worker_msg: List[str] = []

    def _capture_worker_reply(recipient, messages, sender, config):
        # Capture messages from the assistant (Worker) in this 2-agent chat
        for m in (messages or []):
            c = m.get("content") or ""
            if c:
                _last_worker_msg.append(str(c))
        return False, None  # Continue normal flow

    try:
        user_proxy.register_reply(ConversableAgent, _capture_worker_reply)
    except Exception:
        pass

    # Fallback worker (no tools) for when chat extraction fails - ChatResult.chat_history is unreliable with tools
    _fallback_worker = ConversableAgent(
        name="WorkerLLM",
        system_message=WORKER_SYSTEM_PROMPT,
        llm_config=worker_llm_config,
        human_input_mode="NEVER",
    )

    def run_on_df(df: pd.DataFrame, label: str) -> Tuple[List[List[Dict]], Dict]:
        preds = []
        for idx, row in df.iterrows():
            text = row["text"]
            posts_id = int(row.get("posts_id", idx))
            _current_posts_id.set(posts_id)

            prompt_prefix = f"{sample_block}\n" if sample_block else ""
            msg = f"""{prompt_prefix}{final_prompt}

TEXT TO ANALYZE:
{text}

You may use ner_tool or dep_parse_tool to help with extraction. Perform the extraction as specified above. Return ONLY valid JSON following the specified schema. When done, include TERMINATE in your message."""

            try:
                _last_worker_msg.clear()
                chat_result = user_proxy.initiate_chat(worker, message=msg, clear_history=True)
                # #region agent log
                _dbg({"hypothesisId": "E", "location": "spacy_tools:run_on_df", "message": "initiate_chat_returned", "data": {"idx": int(idx), "cr_type": type(chat_result).__name__, "captured_len": len(_last_worker_msg)}})
                # #endregion
                pred = _extract_from_chat(chat_result, user_proxy, worker, _last_worker_msg)
                # Fallback: ChatResult.chat_history is unreliable with tools; use worker without tools for direct extraction
                if not pred:
                    try:
                        resp = _fallback_worker.generate_reply(messages=[{"role": "user", "content": msg}])
                        pred = _parse_worker_response(resp) if resp else []
                    except Exception:
                        pass
                # #region agent log
                _dbg({"hypothesisId": "E", "location": "spacy_tools:run_on_df", "message": "extract_result", "data": {"idx": int(idx), "pred": pred}})
                # #endregion
            except Exception as e:
                # #region agent log
                _dbg({"hypothesisId": "E", "location": "spacy_tools:run_on_df", "message": "exception", "data": {"idx": int(idx), "exc_type": type(e).__name__, "exc_msg": str(e)[:300]}})
                # #endregion
                print(f"  Warning: row {idx} failed: {e}")
                pred = []
            preds.append(pred)

        gt = [_ground_truth_to_motifs(row) for _, row in df.iterrows()]
        metrics = compute_recall_f1(preds, gt)
        if "entity_f1" in metrics and "action_f1" in metrics:
            print(f"  {label}: {len(preds)} samples, entity F1={metrics['entity_f1']:.3f}, action F1={metrics['action_f1']:.3f}, motif F1={metrics['motif_f1']:.3f}")
        else:
            print(f"  {label}: {len(preds)} samples, motif F1={metrics.get('motif_f1', 0):.3f}")
        return preds, metrics

    reset_audit_state()
    preds_seen, metrics_seen = run_on_df(seen_df, "Seen")

    # Fresh agents for unseen to avoid accumulated state
    user_proxy = ConversableAgent(
        name="UserProxy",
        llm_config=False,
        human_input_mode="NEVER",
        is_termination_msg=_is_extraction_done,
    )
    worker = ConversableAgent(
        name="WorkerLLM",
        system_message=worker_system,
        llm_config=worker_llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=20,
    )
    _last_worker_msg.clear()
    try:
        user_proxy.register_reply(ConversableAgent, _capture_worker_reply)
    except Exception:
        pass
    try:
        worker.register_for_llm(name="ner_tool", description="Perform named entity recognition on text. Returns JSON with entities. Use to identify people, orgs, places.")(ner_tool)
        user_proxy.register_for_execution(name="ner_tool")(ner_tool)
        worker.register_for_llm(name="dep_parse_tool", description="Perform dependency parsing on text. Returns JSON with tokens. Use to understand sentence structure and who does what.")(dep_parse_tool)
        user_proxy.register_for_execution(name="dep_parse_tool")(dep_parse_tool)
    except AttributeError:
        from autogen import register_function
        register_function(ner_tool, caller=worker, executor=user_proxy, name="ner_tool",
            description="Perform named entity recognition on text. Returns JSON with entities. Use to identify people, orgs, places.")
        register_function(dep_parse_tool, caller=worker, executor=user_proxy, name="dep_parse_tool",
            description="Perform dependency parsing on text. Returns JSON with tokens. Use to understand sentence structure and who does what.")

    _fallback_worker = ConversableAgent(
        name="WorkerLLM",
        system_message=WORKER_SYSTEM_PROMPT,
        llm_config=worker_llm_config,
        human_input_mode="NEVER",
    )
    preds_unseen, metrics_unseen = run_on_df(unseen_df, "Unseen")
    audit_log = get_audit_log()
    # #region agent log
    _lp = globals().get("_DEBUG_LOG_PATH")
    if _lp and os.path.isfile(_lp):
        print("\n--- DEBUG LOG (copy for analysis) ---")
        with open(_lp, "r", encoding="utf-8") as f:
            print(f.read())
        print("--- END DEBUG LOG ---")
    # #endregion
    return metrics_seen, metrics_unseen, preds_seen, preds_unseen, audit_log


def _run_stage3_nested_chats(
    final_prompt: str,
    seen_df: pd.DataFrame,
    unseen_df: pd.DataFrame,
    llm_config: Optional[dict] = None,
) -> Tuple[Dict, Dict, List, List, List[Dict[str, Any]]]:
    """
    Nested-chats flow (autogen-agentchat 0.2): Extractor + ToolProxy.
    Extractor has nested chat with ToolProxy for tool execution; last_msg becomes reply.
    """
    from autogen import ConversableAgent, register_function
    from .autogen_setup import (
        get_llm_config,
        config,
        WORKER_SYSTEM_PROMPT,
        _format_sample_prompts_block,
        _parse_worker_response,
        _ground_truth_to_motifs,
        compute_recall_f1,
    )

    print(f"\n{'='*60}")
    print("STAGE 3: Tool-Assisted Extraction (nested chats, gpt-4o-mini + SpaCy)")
    print(f"{'='*60}")

    worker_llm_config = llm_config or get_llm_config(config.worker_temp)
    worker_system = WORKER_SYSTEM_PROMPT + WORKER_TOOLS_SYSTEM_APPEND
    sample_block = _format_sample_prompts_block()

    def _make_agents():
        """Create fresh UserProxy, Extractor (ToolExtractor), ToolProxy."""
        user_proxy = ConversableAgent(
            name="UserProxy",
            llm_config=False,
            human_input_mode="NEVER",
            is_termination_msg=_is_extraction_done,
        )
        extractor = ConversableAgent(
            name="ToolExtractor",
            system_message=worker_system,
            llm_config=worker_llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=20,
        )
        tool_proxy = ConversableAgent(
            name="ToolProxy",
            llm_config=False,
            human_input_mode="NEVER",
            is_termination_msg=_is_extraction_done,
        )
        register_function(
            ner_tool,
            caller=extractor,
            executor=tool_proxy,
            name="ner_tool",
            description="Perform named entity recognition on text. Returns JSON with entities. Use to identify people, orgs, places.",
        )
        register_function(
            dep_parse_tool,
            caller=extractor,
            executor=tool_proxy,
            name="dep_parse_tool",
            description="Perform dependency parsing on text. Returns JSON with tokens. Use to understand sentence structure and who does what.",
        )
        extractor.register_nested_chats(
            trigger=user_proxy,
            chat_queue=[{"sender": tool_proxy, "recipient": extractor, "summary_method": "last_msg"}],
        )
        return user_proxy, extractor, tool_proxy

    def _extract_from_chat(chat_result) -> List[Dict[str, str]]:
        """Parse Extractor's JSON from chat_result.summary or last assistant message."""
        if chat_result is None:
            return []
        summary = getattr(chat_result, "summary", None)
        if summary:
            parsed = _parse_worker_response(str(summary))
            if parsed:
                return parsed
        history = getattr(chat_result, "chat_history", None) or []
        for msg in reversed(history):
            content = msg.get("content") or ""
            if not content or ("text to analyze:" in content[:300].lower() or content.strip().lower().startswith("you are tasked with")):
                continue
            parsed = _parse_worker_response(str(content))
            if parsed:
                return parsed
        return []

    def run_on_df(df: pd.DataFrame, label: str) -> Tuple[List[List[Dict]], Dict]:
        user_proxy, extractor, _ = _make_agents()
        preds = []
        for idx, row in df.iterrows():
            text = row["text"]
            posts_id = int(row.get("posts_id", idx))
            _current_posts_id.set(posts_id)
            prompt_prefix = f"{sample_block}\n" if sample_block else ""
            msg = f"""{prompt_prefix}{final_prompt}

TEXT TO ANALYZE:
{text}

You may use ner_tool or dep_parse_tool to help with extraction. Perform the extraction as specified above. Return ONLY valid JSON following the specified schema. When done, include TERMINATE in your message."""
            try:
                chat_result = user_proxy.initiate_chat(extractor, message=msg, clear_history=True)
                pred = _extract_from_chat(chat_result)
                if not pred:
                    pred = []
            except Exception as e:
                print(f"  Warning: row {idx} failed: {e}")
                pred = []
            preds.append(pred)
        gt = [_ground_truth_to_motifs(row) for _, row in df.iterrows()]
        metrics = compute_recall_f1(preds, gt)
        if "entity_f1" in metrics and "action_f1" in metrics:
            print(f"  {label}: {len(preds)} samples, entity F1={metrics['entity_f1']:.3f}, action F1={metrics['action_f1']:.3f}, motif F1={metrics['motif_f1']:.3f}")
        else:
            print(f"  {label}: {len(preds)} samples, motif F1={metrics.get('motif_f1', 0):.3f}")
        return preds, metrics

    reset_audit_state()
    preds_seen, metrics_seen = run_on_df(seen_df, "Seen")
    preds_unseen, metrics_unseen = run_on_df(unseen_df, "Unseen")
    return metrics_seen, metrics_unseen, preds_seen, preds_unseen, get_audit_log()


def run_stage3_with_tools_standalone(
    final_prompt: str,
    seen_df: pd.DataFrame,
    unseen_df: pd.DataFrame,
    llm_config: Optional[dict] = None,
) -> Tuple[Dict, Dict, List, List, List[Dict[str, Any]]]:
    """
    Standalone tool-assisted extraction: clears AutoGen globals before/after to ensure
    runs are completely unaffected by prior extraction or AutoGen runs (gpt-4o-mini, Llama).
    Always uses the standard 2-agent flow (UserProxy initiates, WorkerLLM calls tools),
    which matches the AutoGen 0.2 tutorial pattern exactly and avoids nested-chat
    state accumulation that can interfere with fresh runs.
    Returns (metrics_seen, metrics_unseen, preds_seen, preds_unseen, audit_log).
    """
    _clear_autogen_globals()
    reset_audit_state()
    try:
        return run_stage3_with_tools(final_prompt, seen_df, unseen_df, llm_config=llm_config)
    finally:
        _clear_autogen_globals()


# ============================================================================
# VISUALIZATION FUNCTIONS
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


def plot_combined_token_cost(usage_dicts: Dict[str, Dict[str, Any]]) -> None:
    """
    Bar chart of tokens and cost for each method.
    usage_dicts: {"openai": {"total_tokens": N, "cost": C}, "llama": {...}, "tools": {...}}
    """
    if not usage_dicts:
        print("No usage data.")
        return
    methods = list(usage_dicts.keys())
    tokens = [usage_dicts[m].get("total_tokens", 0) for m in methods]
    costs = [usage_dicts[m].get("cost", 0) for m in methods]
    if not any(tokens) and not any(costs):
        print("No token or cost data in usage_dicts.")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#81C784", "#64B5F6", "#FFB74D"][:len(methods)]
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


def plot_tool_aspect_performance(results: Dict[str, Any], output_columns: Optional[List[str]] = None) -> None:
    """Plot recall and F1 by aspect (entity, action, motif) for seen vs unseen."""
    print("\n--- Tool-Assisted: Recall and F1 by Aspect (Seen vs Unseen) ---")
    ms = results.get("metrics_seen", {})
    mu = results.get("metrics_unseen", {})
    out_cols = output_columns or ["entity", "action"]

    if "entity_f1" in ms and "action_f1" in ms:
        aspects = ["entity", "action", "motif"]
    else:
        aspects = ["output", "motif"]

    x = np.arange(len(aspects))
    width = 0.2

    def _get_vals(metrics: Dict, aspect: str) -> Tuple[float, float]:
        if aspect == "output":
            return metrics.get("motif_recall", 0), metrics.get("output_f1", metrics.get("motif_f1", 0))
        return metrics.get(f"{aspect}_recall", 0), metrics.get(f"{aspect}_f1", 0)

    seen_recall = [_get_vals(ms, a)[0] for a in aspects]
    seen_f1 = [_get_vals(ms, a)[1] for a in aspects]
    unse_recall = [_get_vals(mu, a)[0] for a in aspects]
    unse_f1 = [_get_vals(mu, a)[1] for a in aspects]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5*width, seen_recall, width, label="Seen Recall")
    ax.bar(x - 0.5*width, seen_f1, width, label="Seen F1")
    ax.bar(x + 0.5*width, unse_recall, width, label="Unseen Recall")
    ax.bar(x + 1.5*width, unse_f1, width, label="Unseen F1")
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in aspects])
    ax.set_ylabel("Score")
    ax.set_title("Tool-Assisted: Recall and F1 by Aspect (Seen vs Unseen)")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_full_model_comparison(
    openai_metrics: Optional[Dict] = None,
    llama_metrics: Optional[Dict] = None,
    tools_metrics: Optional[Dict] = None,
    use_unseen: bool = True,
) -> None:
    """
    Compare gpt-4o-mini, Llama, and gpt-4o-mini+tools on entity, action, motif F1.
    Each metrics dict can have 'baseline', 'seen', or 'unseen' key, or be the metrics directly.
    """
    OPENAI_COLOR = "#81C784"
    LLAMA_COLOR = "#64B5F6"
    TOOLS_COLOR = "#FFB74D"

    def _resolve(d: Optional[Dict]) -> Optional[Dict]:
        if d is None:
            return None
        if "baseline" in d:
            return d["baseline"]
        if "unseen" in d and use_unseen:
            return d["unseen"]
        if "seen" in d:
            return d["seen"]
        return d if isinstance(d, dict) else None

    openai_m = _resolve(openai_metrics)
    llama_m = _resolve(llama_metrics)
    tools_m = _resolve(tools_metrics)
    if openai_m is None and llama_m is None and tools_m is None:
        print("No model comparison data.")
        return

    metrics_names = ["Entity F1", "Action F1", "Motif F1"]
    metric_keys = ["entity_f1", "action_f1", "motif_f1"]

    def _val(m: Optional[Dict], k: str) -> float:
        if m is None:
            return 0.0
        v = m.get(k)
        return float(v) if v is not None else 0.0

    x = np.arange(len(metrics_names))
    n_models = sum(m is not None for m in [openai_m, llama_m, tools_m])
    w = 0.8 / max(n_models, 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    offset = -w * (n_models - 1) / 2
    idx = 0
    if openai_m is not None:
        vals = [_val(openai_m, k) for k in metric_keys]
        ax.bar(x + offset + idx * w, vals, w, label="gpt-4o-mini", color=OPENAI_COLOR)
        idx += 1
    if llama_m is not None:
        vals = [_val(llama_m, k) for k in metric_keys]
        ax.bar(x + offset + idx * w, vals, w, label="Llama", color=LLAMA_COLOR)
        idx += 1
    if tools_m is not None:
        vals = [_val(tools_m, k) for k in metric_keys]
        ax.bar(x + offset + idx * w, vals, w, label="gpt-4o-mini+tools", color=TOOLS_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel("F1 Score (Unseen)" if use_unseen else "F1 Score")
    ax.set_title("Model Comparison: gpt-4o-mini vs Llama vs gpt-4o-mini+tools")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# QUALITATIVE EXAMPLE SELECTION
# ============================================================================

def _entity_match(pred_list: List[Dict], gt_list: List[Dict]) -> bool:
    """Check if predicted entity matches ground truth (fuzzy)."""
    def norm(s):
        return str(s).strip().lower() if s else ""
    gt_entities = {norm(m.get("entity", m.get("actor", ""))) for m in (gt_list or []) if m}
    if not gt_entities:
        return True
    pred_entities = {norm(m.get("entity", m.get("actor", ""))) for m in (pred_list or []) if m}
    for ge in gt_entities:
        if not ge:
            continue
        for pe in pred_entities:
            if ge in pe or pe in ge or (ge.split()[-1] in pe if ge.split() else False):
                return True
    return False


def _action_match(pred_list: List[Dict], gt_list: List[Dict]) -> bool:
    """Check if predicted action matches ground truth (fuzzy)."""
    def norm(s):
        return str(s).strip().lower() if s else ""
    gt_actions = {norm(m.get("action", "")) for m in (gt_list or []) if m}
    if not gt_actions:
        return True
    pred_actions = {norm(m.get("action", "")) for m in (pred_list or []) if m}
    for ga in gt_actions:
        if not ga:
            continue
        for pa in pred_actions:
            if ga in pa or pa in ga or ga == pa:
                return True
    return False


def select_qualitative_examples(
    preds_openai: List[List[Dict]],
    preds_tools: List[List[Dict]],
    df: pd.DataFrame,
    preds_llama: Optional[List[List[Dict]]] = None,
    n: int = 5,
) -> Dict[str, List[Tuple[int, Any, List[Dict], List[Dict], List[Dict], List[Dict]]]]:
    """
    Select examples where tool-assisted extraction improved or worsened vs baselines.
    Returns: {
        "entity_improved": [(idx, row, gt, pred_openai, pred_llama, pred_tools), ...],
        "entity_worse": [...],
        "action_improved": [...],
        "action_worse": [...],
    }
    preds_llama can be None if Llama section was skipped.
    """
    from .autogen_setup import _ground_truth_to_motifs

    preds_llama = preds_llama or []
    result = {
        "entity_improved": [],
        "entity_worse": [],
        "action_improved": [],
        "action_worse": [],
    }
    n_rows = min(len(df), len(preds_openai or []), len(preds_tools or []))
    if preds_llama:
        n_rows = min(n_rows, len(preds_llama))
    for i in range(n_rows):
        row = df.iloc[i]
        gt = _ground_truth_to_motifs(row)
        pred_o = (preds_openai or [])[i] if i < len(preds_openai or []) else []
        pred_l = (preds_llama or [])[i] if i < len(preds_llama or []) else []
        pred_t = (preds_tools or [])[i] if i < len(preds_tools or []) else []
        e_o = _entity_match(pred_o, gt)
        e_l = _entity_match(pred_l, gt) if pred_l else False
        e_t = _entity_match(pred_t, gt)
        a_o = _action_match(pred_o, gt)
        a_l = _action_match(pred_l, gt) if pred_l else False
        a_t = _action_match(pred_t, gt)
        baseline_entity_ok = e_o or e_l
        baseline_entity_fail = not e_o and not e_l
        baseline_action_ok = a_o or a_l
        baseline_action_fail = not a_o and not a_l
        if e_t and baseline_entity_fail and len(result["entity_improved"]) < n:
            result["entity_improved"].append((i, row, gt, pred_o, pred_l, pred_t))
        if not e_t and baseline_entity_ok and len(result["entity_worse"]) < n:
            result["entity_worse"].append((i, row, gt, pred_o, pred_l, pred_t))
        if a_t and baseline_action_fail and len(result["action_improved"]) < n:
            result["action_improved"].append((i, row, gt, pred_o, pred_l, pred_t))
        if not a_t and baseline_action_ok and len(result["action_worse"]) < n:
            result["action_worse"].append((i, row, gt, pred_o, pred_l, pred_t))
    return result
