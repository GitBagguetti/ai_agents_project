"""
Multi-LLM Simulation for Prompt Engineering and Information Extraction
=======================================================================
This code implements a 4-stage multi-LLM pipeline for generating optimal prompts
for information extraction tasks. The agents infer the extraction task from annotated
sample data and create prompts that replicate the observed annotation patterns.

Designed for Google Colab environment.
"""

# ============================================================================
# SETUP AND IMPORTS
# ============================================================================

# !pip install -q pyautogen openai sentence-transformers scikit-learn matplotlib pandas numpy

import os
import json
import re
import ast
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# AutoGen imports (install if missing, e.g. when running cells out of order)
try:
    import autogen
    from autogen import ConversableAgent
except ModuleNotFoundError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyautogen"])
    import autogen
    from autogen import ConversableAgent

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration for the multi-LLM simulation."""
    # OpenAI API configuration
    api_key: str = ""  # Set via environment variable or directly
    model: str = "gpt-4o-mini"  # Default model for all agents

    # Data paths (Colab default)
    data_path: str = "/content/original_content_trump_motifs_en_10k.csv"

    # Column mapping: text = input; output_columns = fields to extract; posts_id optional
    col_text: str = "text"
    output_columns: List[str] = field(default_factory=lambda: ["entity", "action"])
    posts_id_column: Optional[str] = "posts_id"

    # Stage 2 convergence settings
    max_rounds_per_cycle: int = 12  # Maximum rounds before forced voting
    rounds_before_human: int = 3  # Human-in-the-loop after N rounds without consensus
    min_approvals_for_consensus: int = 4  # All 4 agents must approve

    # Human input function (use input for Colab)
    human_input_func: Any = None  # Default: builtins.input

    # Cycling settings (single run by default)
    num_cycles: int = 1

    # Temperature settings
    prompt_engineer_temp: float = 0.7
    critic_temp: float = 0.5
    deliberation_temp: float = 0.6
    coordinator_temp: float = 0.5
    worker_temp: float = 0.3

    # Embedding model for visualization
    embedding_model: str = "all-MiniLM-L6-v2"

    # User-supplied sample prompts for information extraction (engineers, critic, worker use as examples)
    sample_prompts: List[str] = field(default_factory=list)


# Initialize config
config = Config()

# Set human input function (for Colab: input() works in notebook cells)
if config.human_input_func is None:
    import builtins
    config.human_input_func = builtins.input

# Set your API key here or via environment variable
# config.api_key = "your-api-key-here"
if not config.api_key:
    config.api_key = os.environ.get("OPENAI_API_KEY", "")

# LLM configuration for AutoGen
def get_llm_config(
    temperature: float = 0.7,
    use_llama: bool = False,
    sglang_base: str = "http://localhost:7501/v1",
    model_override: Optional[str] = None,
) -> dict:
    """Generate LLM config for AutoGen agents. Set use_llama=True for Llama-3.1-8B via SGLang."""
    if use_llama:
        return {
            "config_list": [
                {
                    "model": model_override or "meta-llama/Llama-3.1-8B-Instruct",
                    "base_url": sglang_base,
                    "api_key": "sglang",
                    "price": [0, 0],  # Suppress "Model not found, cost will be 0" warning
                }
            ],
            "temperature": temperature,
            "timeout": 120,
        }
    return {
        "config_list": [
            {
                "model": model_override or config.model,
                "api_key": config.api_key,
            }
        ],
        "temperature": temperature,
        "timeout": 120,
    }


# ============================================================================
# DATA LOADING (Colab: run !pip and !gdown in a notebook cell before the rest)
# ============================================================================

def load_motifs_data(path: Optional[str] = None, output_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load CSV with text + output columns. Returns DataFrame with columns:
    text, posts_id (if present), and all output_columns.
    """
    p = path or config.data_path
    df = pd.read_csv(p)
    out_cols = output_columns if output_columns is not None else config.output_columns

    # Require text + all output columns
    required = {config.col_text} | set(out_cols)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Data missing required columns: {list(missing)}. "
            f"Expected: text + {out_cols}. Found: {list(df.columns)}"
        )

    # Select and optionally add posts_id
    cols = [config.col_text] + out_cols
    rename = {config.col_text: "text"}
    if config.posts_id_column and config.posts_id_column in df.columns:
        cols.append(config.posts_id_column)
        rename[config.posts_id_column] = "posts_id"
    df = df[cols].copy()
    df = df.rename(columns=rename)
    # Ensure posts_id exists for prepare_stage1_data (use index if absent)
    if "posts_id" not in df.columns:
        df["posts_id"] = df.index
    return df


# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

@dataclass
class AgentMemory:
    """Memory store for individual agents across cycles."""
    agent_name: str
    prompts_generated: List[str] = field(default_factory=list)
    thinking_history: List[str] = field(default_factory=list)
    feedback_received: List[str] = field(default_factory=list)
    evaluation_results: List[Dict] = field(default_factory=list)
    
    def add_prompt(self, prompt: str, thinking: str, feedback: Optional[str] = None):
        """Add a generated prompt and associated context to memory."""
        self.prompts_generated.append(prompt)
        self.thinking_history.append(thinking)
        if feedback:
            self.feedback_received.append(feedback)
    
    def add_evaluation(self, accuracy: float, extracted_motifs: List, cycle: int):
        """Add evaluation results from a cycle."""
        self.evaluation_results.append({
            "cycle": cycle,
            "accuracy": accuracy,
            "sample_extractions": extracted_motifs[:3]  # Store first 3 for reference
        })
    
    def get_context_for_refinement(self) -> str:
        """Generate context string for prompt refinement in subsequent cycles."""
        if not self.prompts_generated:
            return ""
        
        context = "\n\n=== YOUR PREVIOUS WORK ===\n"
        for i, (prompt, thinking) in enumerate(zip(self.prompts_generated, self.thinking_history)):
            context += f"\n--- Cycle {i+1} ---\n"
            context += f"Your thinking: {thinking[:500]}...\n" if len(thinking) > 500 else f"Your thinking: {thinking}\n"
            context += f"Your prompt: {prompt[:300]}...\n" if len(prompt) > 300 else f"Your prompt: {prompt}\n"
            
            if i < len(self.feedback_received):
                context += f"Feedback received: {self.feedback_received[i][:300]}...\n" if len(self.feedback_received[i]) > 300 else f"Feedback received: {self.feedback_received[i]}\n"
            
            if i < len(self.evaluation_results):
                eval_result = self.evaluation_results[i]
                context += f"Accuracy achieved: {eval_result['accuracy']:.2%}\n"
        
        return context


@dataclass  
class SharedMemory:
    """Shared memory for group deliberation across cycles."""
    deliberation_history: List[Dict] = field(default_factory=list)
    final_prompts: List[str] = field(default_factory=list)
    convergence_rounds: List[int] = field(default_factory=list)
    
    def add_deliberation(self, cycle: int, transcript: str, final_prompt: str, rounds: int):
        """Record a deliberation session."""
        self.deliberation_history.append({
            "cycle": cycle,
            "transcript_summary": transcript[:2000] if len(transcript) > 2000 else transcript,
        })
        self.final_prompts.append(final_prompt)
        self.convergence_rounds.append(rounds)
    
    def get_context_for_deliberation(self) -> str:
        """Generate context for subsequent deliberation sessions."""
        if not self.final_prompts:
            return ""
        
        context = "\n\n=== PREVIOUS DELIBERATION OUTCOMES ===\n"
        for i, (prompt, rounds) in enumerate(zip(self.final_prompts, self.convergence_rounds)):
            context += f"\n--- Cycle {i+1} ---\n"
            context += f"Rounds to converge: {rounds}\n"
            context += f"Final agreed prompt: {prompt[:400]}...\n" if len(prompt) > 400 else f"Final agreed prompt: {prompt}\n"
        
        return context


# Initialize memory stores
agent_memories: Dict[str, AgentMemory] = {}
shared_memory = SharedMemory()


@dataclass
class UsageTracker:
    """Track token usage and cost by stage."""
    stage1_prompt_tokens: int = 0
    stage1_completion_tokens: int = 0
    stage2_prompt_tokens: int = 0
    stage2_completion_tokens: int = 0
    stage3_prompt_tokens: int = 0
    stage3_completion_tokens: int = 0
    total_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage1": {"prompt_tokens": self.stage1_prompt_tokens, "completion_tokens": self.stage1_completion_tokens},
            "stage2": {"prompt_tokens": self.stage2_prompt_tokens, "completion_tokens": self.stage2_completion_tokens},
            "stage3": {"prompt_tokens": self.stage3_prompt_tokens, "completion_tokens": self.stage3_completion_tokens},
            "total_prompt_tokens": self.stage1_prompt_tokens + self.stage2_prompt_tokens + self.stage3_prompt_tokens,
            "total_completion_tokens": self.stage1_completion_tokens + self.stage2_completion_tokens + self.stage3_completion_tokens,
            "total_cost": self.total_cost,
        }


usage_tracker = UsageTracker()


# ============================================================================
# STAGE 1: INDIVIDUAL PROMPT ENGINEERING
# ============================================================================

# System prompts for Stage 1 prompt engineers (task-agnostic)
PROMPT_ENGINEER_SYSTEM = """You are an expert prompt engineer specializing in information extraction from text data.
Your expertise lies in crafting precise, effective prompts that guide LLMs to extract structured information from unstructured text.

Your task is to analyze sample data with handcoded annotations and infer the information extraction task, 
then create a prompt for a worker LLM that REPRODUCES THE EXACT EXTRACTION PATTERNS seen in the annotations.

You will receive examples showing: Text (the source data) and annotated fields representing the extracted information.

FIRST, analyze the samples to understand:
1. What KIND of information is being extracted? (relationships, entities, attributes, events, etc.)
2. What SPECIFIC PATTERNS and CONVENTIONS are used in the annotations?
   - How are entities/concepts identified and labeled?
   - What level of specificity or abstraction is used?
   - Are there normalization rules (e.g., lowercase, lemmatization)?
3. What is the EXACT STRUCTURE of the annotations? (field names, data types, nesting)
4. What are the EDGE CASES and how are they handled in the annotations?
5. What OUTPUT FORMAT is used? (JSON schema, field names, value formats)

CRITICAL: Your prompt must teach the worker LLM to REPLICATE the annotation style, not invent its own interpretation.

THEN, design a prompt for a worker LLM that:
- Clearly explains the extraction task based on the observed patterns
- Specifies the EXACT output format (matching annotation field names and structure)
- Provides explicit guidance on MATCHING the annotation conventions
- Includes specific instructions on replicating the extraction patterns you identified
- Handles edge cases consistently with the annotations
- Follows prompting best practices (clear, specific, structured)

The worker must output structured JSON data matching the annotation schema exactly.

FORMAT YOUR RESPONSE AS:
<thinking>
[Your detailed analysis of the annotation patterns and extraction conventions]
</thinking>

<key_criteria>
[1-5 key criteria that must be included in any final prompt]
</key_criteria>

<prompt>
[Your final prompt for the worker LLM that ensures pattern replication]
</prompt>"""


PROMPT_ENGINEER_WITH_CRITIC_SYSTEM = """You are an expert prompt engineer specializing in information extraction from text data.
Your expertise lies in crafting precise, effective prompts that guide LLMs to extract structured information from unstructured text.

Your task is to analyze sample data with handcoded annotations and infer the information extraction task, then create a prompt for a worker LLM that REPRODUCES THE EXACT EXTRACTION PATTERNS seen in the annotations.

You will receive examples showing: Text (the source data) and annotated fields representing the extracted information.

FIRST, analyze the samples to understand:
- What kind of information is being extracted
- What SPECIFIC PATTERNS and CONVENTIONS are used in the annotations
- How to replicate the annotation style (specificity, normalization, labeling conventions)
- What the exact output structure should be
- What edge cases exist and how they're handled

CRITICAL: Your prompt must teach the worker LLM to REPLICATE the annotation style, not invent its own interpretation.

THEN, design a prompt that ensures the worker LLM matches the observed extraction patterns exactly.

IMPORTANT: You will receive feedback from a critic. You MUST incorporate their feedback before finalizing your prompt.

FORMAT YOUR RESPONSE AS:
<thinking>
[Your detailed analysis of the annotation patterns and extraction conventions]
</thinking>

<key_criteria>
[1-5 key criteria that must be included in any final prompt]
</key_criteria>

<prompt>
[Your final prompt for the worker LLM that ensures pattern replication]
</prompt>"""


CRITIC_SYSTEM = """You are an expert critic and reviewer of prompts for LLM-based information extraction systems.
Your role is to provide constructive, specific feedback on prompts designed for extracting structured information from text.

CRITICAL FOCUS: The prompt must ensure the worker LLM REPLICATES the exact extraction patterns and conventions seen in the annotated training data, not invent its own interpretation.

Evaluate prompts on these criteria:
1. TASK CLARITY: Is the extraction task clearly and unambiguously defined based on the annotation patterns?
2. PATTERN REPLICATION: Does the prompt explicitly teach the worker to match the annotation style?
   - Are specific conventions identified (e.g., normalization, labeling, specificity level)?
   - Does it guide the worker to replicate observed patterns?
3. FORMAT SPECIFICATION: Is the output format (JSON schema, field names, data types) exactly specified to match annotations?
4. EXTRACTION GUIDANCE: Does it provide clear, specific guidance on HOW to identify and extract information AS DONE IN THE ANNOTATIONS?
5. CONSISTENCY: Will the prompt produce extractions consistent with the annotation conventions?
6. EDGE CASE HANDLING: Does it specify how to handle edge cases consistently with the annotated examples?
7. COMPLETENESS: Does it cover all aspects of the extraction task evident in the data?
8. ACTIONABILITY: Can a worker LLM follow this prompt and reproduce the annotation patterns?

Provide specific, actionable feedback. Be constructive but thorough.

FORMAT YOUR RESPONSE AS:
<evaluation>
[Detailed evaluation of each criterion, focusing on pattern replication]
</evaluation>

<feedback>
[Specific suggestions for improvement to ensure pattern matching]
</feedback>"""


def format_sample_for_prompt_engineering(sample_df: pd.DataFrame) -> str:
    """Format the sample data for prompt engineering agents. Shows text and all output columns."""
    out_cols = config.output_columns
    examples = []
    for _, row in sample_df.iterrows():
        text = row['text']
        content = text[:500] + "..." if len(str(text)) > 500 else str(text)
        parts = [f"Text: {content}"]
        for col in out_cols:
            if col in row:
                parts.append(f"{col.capitalize()}: {row[col]}")
        examples.append("\n".join(parts))
    return "\n\n---\n\n".join(examples)


def _format_sample_prompts_block() -> str:
    """Format user-supplied sample prompts for inclusion in agent messages."""
    if not config.sample_prompts:
        return ""
    lines = ["EXAMPLE PROMPTS (use as inspiration for information extraction):"]
    for i, p in enumerate(config.sample_prompts, 1):
        lines.append(f"\n--- Example {i} ---\n{p.strip()}")
    return "\n".join(lines) + "\n\n"


def prepare_stage1_data(df: pd.DataFrame, n_per_agent: int = 100) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """
    Split data into: 4 mutually exclusive chunks for engineers, 100 seen (from train), 100 unseen (held out).
    Returns (agent_data, seen_df, unseen_df).
    """
    np.random.seed(42)
    unique_ids = df['posts_id'].unique()
    np.random.shuffle(unique_ids)

    n_engineers = 4
    n_train = n_per_agent * n_engineers  # 400
    if len(unique_ids) < n_train + 100:
        raise ValueError(f"Need at least {n_train + 100} unique posts_ids, got {len(unique_ids)}")

    unseen_ids = set(unique_ids[:100])
    train_ids = unique_ids[100:100 + n_train]
    seen_ids = np.random.choice(train_ids, size=min(100, len(train_ids)), replace=False)

    chunks = np.array_split(train_ids, n_engineers)
    agent_names = ["PromptEngineer_Alpha", "PromptEngineer_Beta", "PromptEngineer_Gamma", "PromptEngineer_Delta"]
    agent_data = {
        name: df[df['posts_id'].isin(chunk)].head(n_per_agent).reset_index(drop=True)
        for name, chunk in zip(agent_names, chunks)
    }
    seen_df = df[df['posts_id'].isin(seen_ids)].head(100).reset_index(drop=True)
    unseen_df = df[df['posts_id'].isin(unseen_ids)].head(100).reset_index(drop=True)
    return agent_data, seen_df, unseen_df


def parse_prompt_response(response: str) -> Tuple[str, str, str]:
    """Parse thinking, key_criteria, and prompt from agent response."""
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
    criteria_match = re.search(r'<key_criteria>(.*?)</key_criteria>', response, re.DOTALL)
    prompt_match = re.search(r'<prompt>(.*?)</prompt>', response, re.DOTALL)

    thinking = thinking_match.group(1).strip() if thinking_match else ""
    key_criteria = criteria_match.group(1).strip() if criteria_match else ""
    prompt = prompt_match.group(1).strip() if prompt_match else response

    return thinking, key_criteria, prompt


def parse_critic_response(response: str) -> Tuple[str, str]:
    """Parse evaluation and feedback from critic response."""
    eval_match = re.search(r'<evaluation>(.*?)</evaluation>', response, re.DOTALL)
    feedback_match = re.search(r'<feedback>(.*?)</feedback>', response, re.DOTALL)
    
    evaluation = eval_match.group(1).strip() if eval_match else ""
    feedback = feedback_match.group(1).strip() if feedback_match else response
    
    return evaluation, feedback


def run_stage1(agent_data: Dict[str, pd.DataFrame], cycle: int = 1) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Run Stage 1: Individual prompt engineering. Each agent gets 100 mutually exclusive rows.
    Returns (prompts, key_criteria) - both dicts mapping agent names to their outputs.
    """
    print(f"\n{'='*60}")
    print(f"STAGE 1 - CYCLE {cycle}: Individual Prompt Engineering")
    print(f"{'='*60}")

    agent_configs = [
        {"name": "PromptEngineer_Alpha", "has_critic": False},
        {"name": "PromptEngineer_Beta", "has_critic": False},
        {"name": "PromptEngineer_Gamma", "has_critic": True},
        {"name": "PromptEngineer_Delta", "has_critic": True},
    ]

    prompts = {}
    key_criteria = {}

    for agent_config in agent_configs:
        agent_name = agent_config["name"]
        has_critic = agent_config["has_critic"]
        sample_df = agent_data[agent_name]
        formatted_data = format_sample_for_prompt_engineering(sample_df)

        print(f"\n--- {agent_name} {'(with critic)' if has_critic else ''} ---")

        if agent_name not in agent_memories:
            agent_memories[agent_name] = AgentMemory(agent_name=agent_name)
        memory = agent_memories[agent_name]
        memory_context = memory.get_context_for_refinement() if cycle > 1 else ""

        system_prompt = PROMPT_ENGINEER_WITH_CRITIC_SYSTEM if has_critic else PROMPT_ENGINEER_SYSTEM
        engineer = ConversableAgent(
            name=agent_name,
            system_message=system_prompt,
            llm_config=get_llm_config(config.prompt_engineer_temp),
            human_input_mode="NEVER",
        )

        sample_block = _format_sample_prompts_block()
        user_message = f"""Please analyze the following sample data with handcoded annotations and infer the information extraction task.
{sample_block}
SAMPLE DATA WITH ANNOTATIONS:
{formatted_data}

{memory_context}

INSTRUCTIONS:
1. Examine the sample data to understand what kind of information is being extracted
2. Identify SPECIFIC PATTERNS and CONVENTIONS in how the annotations are done:
   - What level of detail/specificity is used?
   - Are there normalization rules (lowercase, lemmatization, etc.)?
   - How are entities/concepts labeled?
   - How is the information structured?
3. Determine the EXACT output format (JSON schema matching annotation field names)
4. Design a prompt for a worker LLM that will REPRODUCE these exact annotation patterns

CRITICAL: Your prompt must teach the worker to REPLICATE the annotation style seen in the examples, not invent its own interpretation.

Create your prompt now, ensuring it enables precise pattern replication."""

        response = engineer.generate_reply(messages=[{"role": "user", "content": user_message}])
        thinking, criteria, prompt = parse_prompt_response(response)
        feedback = None

        if has_critic:
            critic = ConversableAgent(
                name=f"Critic_for_{agent_name}",
                system_message=CRITIC_SYSTEM,
                llm_config=get_llm_config(config.critic_temp),
                human_input_mode="NEVER",
            )
            sample_block = _format_sample_prompts_block()
            critic_message = f"""Please review this prompt for information extraction.
{sample_block}
PROMPT TO REVIEW:
{prompt}

CONTEXT - Sample data with annotations it should replicate:
{formatted_data[:1000]}...

Evaluate whether this prompt will enable the worker LLM to REPLICATE the exact annotation patterns seen in the sample data. Specifically assess:
- Does it identify and teach the specific conventions used in the annotations?
- Does it specify the exact output format matching the annotation schema?
- Will it produce extractions consistent with the annotation style?

Provide your evaluation and specific feedback for improving pattern replication."""

            critic_response = critic.generate_reply(messages=[{"role": "user", "content": critic_message}])
            evaluation, feedback = parse_critic_response(critic_response)
            print(f"  Critic feedback received: {feedback[:200]}...")

            revision_message = f"""Your critic has provided the following feedback:

EVALUATION:
{evaluation}

FEEDBACK:
{feedback}

Please revise your prompt. Maintain the same format with <thinking>, <key_criteria>, and <prompt> tags."""

            revised_response = engineer.generate_reply(
                messages=[
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": revision_message}
                ]
            )
            thinking, criteria, prompt = parse_prompt_response(revised_response)

        memory.add_prompt(prompt, thinking, feedback)
        prompts[agent_name] = prompt
        key_criteria[agent_name] = criteria
        print(f"  Prompt generated ({len(prompt)} chars)")

    return prompts, key_criteria


# ============================================================================
# STAGE 2: COORDINATOR-LED DISCUSSION WITH HUMAN-IN-THE-LOOP
# ============================================================================

COORDINATOR_SYSTEM = """You are the Discussion Coordinator. You receive prompts and key criteria from 4 prompt engineers.
Your role: Synthesize an ideal prompt that incorporates the best elements from all proposals and satisfies the key criteria.

The engineers have analyzed sample data to infer an information extraction task. Your job is to create a unified prompt that ENSURES THE WORKER LLM REPLICATES THE EXACT EXTRACTION PATTERNS seen in the annotations.

Your synthesized prompt must:
- Clearly define the extraction task (based on engineer consensus about observed patterns)
- Specify the EXACT output format (structured JSON matching annotation field names and schema)
- Include explicit instructions for MATCHING the annotation conventions (normalization, specificity, labeling style)
- Incorporate the strongest pattern-replication insights from each engineer's approach
- Provide clear guidance on reproducing the annotation style, not inventing a new interpretation
- Follow prompting best practices

CRITICAL: The prompt must teach pattern replication, not just task understanding.

When engineers request changes (up to 5 specific items), incorporate them into the prompt.
When the human provides feedback, implement it precisely.

Output format: When you propose a prompt, wrap it in <proposed_prompt>...</proposed_prompt>.
The worker LLM must output structured JSON data matching the annotation schema exactly."""

ENGINEER_RESPONSE_SYSTEM = """You are an engineer participating in prompt deliberation.
You will see a proposed prompt from the coordinator.

Evaluate whether the prompt will enable a worker LLM to REPLICATE the exact annotation patterns from the training data.

Respond with EXACTLY one of:
1. "AGREE" - if the prompt adequately teaches pattern replication
2. Or list up to 5 specific changes in <changes>...</changes>:
   <changes>
   1. [specific change to improve pattern replication]
   2. [specific change to improve pattern replication]
   ...
   </changes>

Focus on: Does it identify annotation conventions? Does it specify the exact output format? Will it produce consistent extractions?

Be brief and specific."""


def _parse_engineer_response(response: str) -> str:
    """Check if engineer said AGREE or provided changes."""
    r = response.strip().upper()
    if r == "AGREE" or "AGREE" in r.split():
        return "AGREE"
    changes_match = re.search(r'<changes>(.*?)</changes>', response, re.DOTALL | re.IGNORECASE)
    return changes_match.group(1).strip() if changes_match else "AGREE" if "AGREE" in r else response


def _parse_coordinator_prompt(response: str) -> str:
    """Extract proposed prompt from coordinator response."""
    match = re.search(r'<proposed_prompt>(.*?)</proposed_prompt>', response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else response


def run_stage2(prompts: Dict[str, str], key_criteria: Dict[str, str], cycle: int = 1) -> Tuple[str, int, int]:
    """
    Run Stage 2: Coordinator-led discussion with human-in-the-loop.
    Returns (final_prompt, rounds, human_interventions).
    """
    print(f"\n{'='*60}")
    print(f"STAGE 2 - CYCLE {cycle}: Coordinator-Led Discussion with HITL")
    print(f"{'='*60}")

    coordinator = ConversableAgent(
        name="Coordinator",
        system_message=COORDINATOR_SYSTEM,
        llm_config=get_llm_config(config.coordinator_temp),
        human_input_mode="NEVER",
    )

    engineer_agents = {}
    for name in prompts.keys():
        engineer_agents[name] = ConversableAgent(
            name=name,
            system_message=ENGINEER_RESPONSE_SYSTEM,
            llm_config=get_llm_config(config.deliberation_temp),
            human_input_mode="NEVER",
        )

    # Build initial input for coordinator
    sample_block = _format_sample_prompts_block()
    coordinator_input = f"{sample_block}PROMPTS AND KEY CRITERIA FROM ENGINEERS:\n\n"
    for name in prompts.keys():
        coordinator_input += f"--- {name} ---\nKey criteria: {key_criteria.get(name, '')}\n\nPrompt:\n{prompts[name]}\n\n"

    coordinator_input += "\nSynthesize an ideal prompt that incorporates the best pattern-replication insights from all engineers. The prompt must enable the worker LLM to REPRODUCE the exact annotation patterns seen in the training data. Output it in <proposed_prompt>...</proposed_prompt>."

    max_rounds = config.max_rounds_per_cycle
    rounds_before_human = config.rounds_before_human
    human_interventions = 0
    final_prompt = ""
    human_feedback = None
    engineer_changes = ""

    for round_num in range(max_rounds):
        print(f"\n  Round {round_num + 1}/{max_rounds}")

        # Coordinator proposes
        if round_num == 0:
            msg = coordinator_input
        else:
            if human_feedback:
                msg = f"""HUMAN FEEDBACK (implement this): {human_feedback}

Revise the prompt accordingly. Output the revised prompt in <proposed_prompt>...</proposed_prompt>."""
                human_feedback = None
            else:
                msg = f"""Engineers requested changes:

{engineer_changes}

Incorporate these changes into the prompt. Output the revised prompt in <proposed_prompt>...</proposed_prompt>."""

        coord_response = coordinator.generate_reply(messages=[{"role": "user", "content": msg}])
        final_prompt = _parse_coordinator_prompt(coord_response)
        if not final_prompt:
            final_prompt = coord_response

        # Engineers respond
        engineer_response_msg = f"""The coordinator proposes this prompt:

{final_prompt}

Respond with AGREE or list up to 5 specific changes in <changes>...</changes>."""

        responses = {}
        for name, agent in engineer_agents.items():
            r = agent.generate_reply(messages=[{"role": "user", "content": engineer_response_msg}])
            responses[name] = _parse_engineer_response(r)

        all_agree = all(v == "AGREE" for v in responses.values())
        if all_agree:
            print(f"  All engineers agreed after round {round_num + 1}")
            break

        engineer_changes = "\n".join([f"{n}: {v}" for n, v in responses.items() if v != "AGREE"])

        # Human-in-the-loop every N rounds
        if (round_num + 1) % rounds_before_human == 0:
            human_interventions += 1
            print(f"\n  --- Human review (intervention {human_interventions}) ---")
            print(f"  Proposed prompt:\n{final_prompt[:500]}...")
            user_input = config.human_input_func(
                "Review the prompt above. Type AGREE to accept, or provide specific feedback: "
            ).strip()
            if user_input.upper() == "AGREE":
                print("  Human approved.")
                break
            human_feedback = user_input

    # Final human review (always)
    print("\n  --- Final human review (required) ---")
    print(f"  Final prompt:\n{final_prompt[:800]}...")
    while True:
        user_input = config.human_input_func(
            "Approve this prompt? Type AGREE or provide specific feedback for the coordinator: "
        ).strip()
        if user_input.upper() == "AGREE":
            break
        human_interventions += 1
        coord_response = coordinator.generate_reply(messages=[
            {"role": "user", "content": f"HUMAN FEEDBACK: {user_input}\n\nRevise the prompt. Output in <proposed_prompt>...</proposed_prompt>."}
        ])
        final_prompt = _parse_coordinator_prompt(coord_response) or coord_response
        print(f"  Revised prompt:\n{final_prompt[:500]}...")

    print(f"  Deliberation completed in {round_num + 1} rounds, {human_interventions} human interventions")
    print(f"  Final prompt length: {len(final_prompt)} chars")
    shared_memory.add_deliberation(cycle, "", final_prompt, round_num + 1)
    return final_prompt, round_num + 1, human_interventions


# ============================================================================
# STAGE 3: TESTING WITH WORKER LLM
# ============================================================================

try:
    from .prompts import WORKER_BEHAVIORAL_PROMPT
except ImportError:
    WORKER_BEHAVIORAL_PROMPT = (
        "You are a highly efficient information detection and extraction engine, specialized in analyzing natural language data. "
        "You value accuracy and efficiency. Follow formatting conventions in the extraction prompt."
    )

WORKER_SYSTEM_PROMPT = f"""{WORKER_BEHAVIORAL_PROMPT}

You perform information extraction from text, following the EXACT patterns and conventions specified in the task prompt.
Your goal is to REPLICATE the annotation style described in the prompt, not to invent your own interpretation.
Output MUST be valid JSON only, following the exact schema specified in the extraction prompt.
If no relevant information is found, return an empty result following the specified format."""


def _parse_worker_response(response: str, output_columns: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Parse worker JSON output into list of dicts with keys from output_columns."""
    out_cols = output_columns or config.output_columns
    try:
        cleaned = response.strip()
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        s = match.group() if match else cleaned
        data = json.loads(s)
        # Support motifs, items, or single record
        records = data.get("motifs", data.get("items", []))
        if isinstance(data, dict) and not records:
            # Single record: {entity: "...", action: "..."}
            if any(k in data for k in out_cols):
                records = [data]
        if not isinstance(records, list):
            records = []
        result = []
        for m in records:
            if not isinstance(m, dict):
                continue
            rec = {}
            for col in out_cols:
                val = m.get(col, m.get("actor" if col == "entity" else col, ""))
                rec[col] = str(val).strip().lower() if val is not None else ""
            if any(rec.values()):
                result.append(rec)
        return result
    except (json.JSONDecodeError, TypeError, AttributeError):
        return []


def _ground_truth_to_motifs(row, output_columns: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Convert a row to list of record dicts with keys from output_columns."""
    out_cols = output_columns or config.output_columns
    vals = [str(row.get(col, "")).strip().lower() for col in out_cols]
    if not any(v and v not in ("na", "nan") for v in vals):
        return []
    return [{col: (vals[i] or "na") for i, col in enumerate(out_cols)}]


def run_stage3(
    final_prompt: str,
    seen_df: pd.DataFrame,
    unseen_df: pd.DataFrame,
    llm_config: Optional[dict] = None,
) -> Tuple[Dict, Dict, List, List]:
    """
    Run Stage 3: Worker extracts motifs on 100 seen and 100 unseen messages.
    Pass llm_config to use custom worker (e.g. Llama via SGLang). Returns (metrics_seen, metrics_unseen, preds_seen, preds_unseen).
    """
    print(f"\n{'='*60}")
    print("STAGE 3: Testing with Worker LLM (Seen + Unseen)")
    print(f"{'='*60}")

    worker_llm_config = llm_config or get_llm_config(config.worker_temp)
    worker = ConversableAgent(
        name="WorkerLLM",
        system_message=WORKER_SYSTEM_PROMPT,
        llm_config=worker_llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1000,  # Avoid "Maximum number of consecutive auto-replies reached" when processing many samples
    )

    sample_block = _format_sample_prompts_block()
    def run_on_df(df: pd.DataFrame, label: str) -> Tuple[List[List[Dict]], Dict]:
        preds = []
        for _, row in df.iterrows():
            text = row["text"]
            prompt_prefix = f"{sample_block}\n" if sample_block else ""
            msg = f"""{prompt_prefix}{final_prompt}

TEXT TO ANALYZE:
{text}

Perform the extraction as specified above, following the exact patterns and conventions described. Return ONLY valid JSON following the specified schema."""
            resp = worker.generate_reply(messages=[{"role": "user", "content": msg}])
            preds.append(_parse_worker_response(resp))
        gt = [_ground_truth_to_motifs(row) for _, row in df.iterrows()]
        metrics = compute_recall_f1(preds, gt)
        if "entity_f1" in metrics and "action_f1" in metrics:
            print(f"  {label}: {len(preds)} samples, entity F1={metrics['entity_f1']:.3f}, action F1={metrics['action_f1']:.3f}, motif F1={metrics['motif_f1']:.3f}")
        else:
            print(f"  {label}: {len(preds)} samples, output F1={metrics.get('output_f1', metrics.get('motif_f1', 0)):.3f}")
        return preds, metrics

    preds_seen, metrics_seen = run_on_df(seen_df, "Seen")
    # Create fresh worker for unseen to avoid accumulated chat state triggering auto-reply limit
    worker = ConversableAgent(
        name="WorkerLLM",
        system_message=WORKER_SYSTEM_PROMPT,
        llm_config=worker_llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1000,
    )
    preds_unseen, metrics_unseen = run_on_df(unseen_df, "Unseen")
    return metrics_seen, metrics_unseen, preds_seen, preds_unseen


def compute_recall_f1(
    predictions: List[List[Dict[str, str]]],
    ground_truth: List[List[Dict[str, str]]],
    output_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute recall and F1 for entities, actions, and full motifs. Uses metrics_utils when available."""
    try:
        from .metrics_utils import compute_recall_f1_unified
        return compute_recall_f1_unified(predictions, ground_truth, output_columns or config.output_columns)
    except ImportError:
        pass
    # Fallback: inline implementation
    def norm(s):
        return str(s).strip().lower() if s else ""
    def motif_key(m):
        return (norm(m.get("entity", m.get("actor", ""))), norm(m.get("action", "")))
    tp_entity, tp_action, tp_motif = 0, 0, 0
    total_gt_entity, total_gt_action, total_gt_motif = 0, 0, 0
    total_pred_entity, total_pred_action, total_pred_motif = 0, 0, 0
    for pred_list, truth_list in zip(predictions, ground_truth):
        pred_motifs = [motif_key(m) for m in (pred_list or []) if m.get("entity") or m.get("actor") or m.get("action")]
        truth_motifs = [motif_key(m) for m in (truth_list or []) if m.get("entity") or m.get("action")]
        pred_entities, pred_actions = {m[0] for m in pred_motifs}, {m[1] for m in pred_motifs}
        pred_motif_set = set(pred_motifs)
        truth_entities, truth_actions = {m[0] for m in truth_motifs}, {m[1] for m in truth_motifs}
        truth_motif_set = set(truth_motifs)
        for m in truth_motifs:
            if m[0] in pred_entities:
                tp_entity += 1
            if m[1] in pred_actions:
                tp_action += 1
            if m in pred_motif_set:
                tp_motif += 1
        total_gt_entity += len(truth_entities)
        total_gt_action += len(truth_actions)
        total_gt_motif += len(truth_motif_set)
        total_pred_entity += len(pred_entities)
        total_pred_action += len(pred_actions)
        total_pred_motif += len(pred_motif_set)
    def recall_precision_f1(tp, n_gt, n_pred):
        recall = tp / n_gt if n_gt > 0 else 1.0
        precision = tp / n_pred if n_pred > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return recall, precision, f1
    er, ep, ef = recall_precision_f1(tp_entity, total_gt_entity, total_pred_entity)
    ar, ap, af = recall_precision_f1(tp_action, total_gt_action, total_pred_action)
    mr, mp, mf = recall_precision_f1(tp_motif, total_gt_motif, total_pred_motif)
    return {
        "entity_recall": er, "entity_precision": ep, "entity_f1": ef,
        "action_recall": ar, "action_precision": ap, "action_f1": af,
        "motif_recall": mr, "motif_precision": mp, "motif_f1": mf,
        "true_positives": tp_motif,
        "false_negatives": total_gt_motif - tp_motif,
    }


# ============================================================================
# STAGE 4: CYCLING AND EVALUATION
# ============================================================================

def run_full_pipeline(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run the complete pipeline: prepare data, Stage 1 (4 engineers, 100 rows each),
    Stage 2 (coordinator + HITL), Stage 3 (worker on seen + unseen).
    """
    print("\n" + "="*70)
    print("STARTING MULTI-LLM PROMPT ENGINEERING PIPELINE")
    print("="*70)

    agent_data, seen_df, unseen_df = prepare_stage1_data(df)
    print(f"  Data split: 4x100 train, 100 seen eval, 100 unseen eval")

    prompts, key_criteria = run_stage1(agent_data, cycle=1)
    final_prompt, discussion_rounds, human_interventions = run_stage2(prompts, key_criteria, cycle=1)
    metrics_seen, metrics_unseen, preds_seen, preds_unseen = run_stage3(final_prompt, seen_df, unseen_df)

    try:
        from autogen.oai.client import ChatCompletion
        usage = ChatCompletion.get_usage_summary()
        if usage:
            usage_tracker.stage1_prompt_tokens = usage.get("prompt_tokens", 0)
            usage_tracker.stage1_completion_tokens = usage.get("completion_tokens", 0)
    except Exception:
        pass

    results = {
        "stage1_prompts": [prompts],
        "stage2_final_prompt": final_prompt,
        "stage2_rounds": discussion_rounds,
        "stage2_human_interventions": human_interventions,
        "metrics_seen": metrics_seen,
        "metrics_unseen": metrics_unseen,
        "preds_seen": preds_seen,
        "preds_unseen": preds_unseen,
        "seen_df": seen_df,
        "unseen_df": unseen_df,
        "usage": usage_tracker.to_dict(),
    }
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    return results


# ============================================================================
# EVALUATION AND VISUALIZATION
# ============================================================================

def visualize_recall_f1_by_aspect(results: Dict[str, Any], output_columns: Optional[List[str]] = None):
    """Plot A: Recall and F1 by aspect for seen vs unseen. Uses entity/action/motif when available, else output_f1."""
    print("\n--- Plot A: Recall and F1 by Aspect (Seen vs Unseen) ---")
    ms = results["metrics_seen"]
    mu = results["metrics_unseen"]
    out_cols = output_columns or config.output_columns

    # Use entity/action/motif when available (entity/action task); else use output/motif
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
    ax.set_title("Recall and F1 by Aspect (Seen vs Unseen)")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_token_usage(results: Dict[str, Any]):
    """Plot B: Token usage and cost by stage."""
    print("\n--- Plot B: Token Usage and Cost by Stage ---")
    u = results.get("usage", {})
    stages = ["Stage 1", "Stage 2", "Stage 3"]
    prompt_tokens = [
        u.get("stage1", {}).get("prompt_tokens", 0),
        u.get("stage2", {}).get("prompt_tokens", 0),
        u.get("stage3", {}).get("prompt_tokens", 0),
    ]
    completion_tokens = [
        u.get("stage1", {}).get("completion_tokens", 0),
        u.get("stage2", {}).get("completion_tokens", 0),
        u.get("stage3", {}).get("completion_tokens", 0),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(stages))
    w = 0.35
    ax.bar(x - w/2, prompt_tokens, w, label="Prompt tokens")
    ax.bar(x + w/2, completion_tokens, w, label="Completion tokens")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Tokens")
    ax.set_title("Token Usage by Stage")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    total = u.get("total_prompt_tokens", 0) + u.get("total_completion_tokens", 0)
    print(f"  Total tokens: {total}")


def visualize_discussion_dynamics(results: Dict[str, Any]):
    """Plot C: Discussion dynamics (rounds, human interventions, final prompt length)."""
    print("\n--- Plot C: Discussion Dynamics ---")
    rounds = results.get("stage2_rounds", 0)
    interventions = results.get("stage2_human_interventions", 0)
    prompt_len = len(results.get("stage2_final_prompt", ""))

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Rounds", "Human\nInterventions", "Prompt\nLength (chars)"]
    vals = [rounds, interventions, min(prompt_len, 5000)]
    bars = ax.bar(labels, vals)
    ax.set_ylabel("Count")
    ax.set_title("Discussion Dynamics")
    for b, v in zip(bars, [rounds, interventions, prompt_len]):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + max(vals)*0.02, str(v), ha="center", fontsize=11)
    plt.tight_layout()
    plt.show()


def _compute_entity_specific_metrics(
    predictions: List[List[Dict[str, str]]],
    ground_truth: List[List[Dict[str, str]]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute recall and F1 per entity. Returns dict mapping entity -> {recall, precision, f1, count}.
    """
    def norm(s):
        return str(s).strip().lower() if s else ""

    entity_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"tp": 0, "gt_count": 0, "pred_count": 0})

    for pred_list, truth_list in zip(predictions, ground_truth):
        pred_entities = {norm(m.get("entity", m.get("actor", ""))) for m in (pred_list or []) if m.get("entity") or m.get("actor") or m.get("action")}
        truth_entities = {norm(m.get("entity", "")) for m in (truth_list or []) if m.get("entity") or m.get("action")}

        for ent in truth_entities:
            if ent and ent not in ("na", "nan", ""):
                entity_stats[ent]["gt_count"] += 1
                if ent in pred_entities:
                    entity_stats[ent]["tp"] += 1

        for ent in pred_entities:
            if ent and ent not in ("na", "nan", ""):
                entity_stats[ent]["pred_count"] += 1

    out = {}
    for ent, s in entity_stats.items():
        recall = s["tp"] / s["gt_count"] if s["gt_count"] > 0 else 1.0
        precision = s["tp"] / s["pred_count"] if s["pred_count"] > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[ent] = {"recall": recall, "precision": precision, "f1": f1, "count": s["gt_count"]}
    return out


def visualize_entity_specific_recall_f1(
    results: Dict[str, Any],
    seen_df: pd.DataFrame,
    unseen_df: pd.DataFrame,
    min_samples: int = 2,
    top_n: int = 15,
) -> None:
    """
    Plot entity-specific recall and F1 by entity name (from ground truth).
    Groups by entity, computes recall/F1 per entity, shows top entities by frequency.
    """
    print("\n--- Entity-Specific Recall and F1 by Entity Name ---")
    preds_seen = results.get("preds_seen", [])
    preds_unseen = results.get("preds_unseen", [])
    if not preds_seen and not preds_unseen:
        print("  No predictions in results. Run Stage 3 first.")
        return

    gt_seen = [_ground_truth_to_motifs(row) for _, row in seen_df.iterrows()]
    gt_unseen = [_ground_truth_to_motifs(row) for _, row in unseen_df.iterrows()]

    def _plot_one(preds: List, gt: List, label: str) -> None:
        metrics = _compute_entity_specific_metrics(preds, gt)
        filtered = [(e, m) for e, m in metrics.items() if m["count"] >= min_samples]
        filtered.sort(key=lambda x: -x[1]["count"])
        filtered = filtered[:top_n]
        if not filtered:
            print(f"  {label}: No entities with >={min_samples} samples.")
            return
        entities = [(e[:20] + "...") if len(e) > 20 else e for e, _ in filtered]
        recalls = [m["recall"] for _, m in filtered]
        f1s = [m["f1"] for _, m in filtered]
        x = np.arange(len(entities))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(10, len(entities) * 0.6), 5))
        ax.bar(x - width / 2, recalls, width, label="Recall")
        ax.bar(x + width / 2, f1s, width, label="F1")
        ax.set_xticks(x)
        ax.set_xticklabels(entities, rotation=45, ha="right")
        ax.set_ylabel("Score")
        ax.set_title(f"Entity-Specific Recall and F1 ({label})")
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    if preds_seen:
        _plot_one(preds_seen, gt_seen, "Seen")
    if preds_unseen:
        _plot_one(preds_unseen, gt_unseen, "Unseen")


def visualize_qualitative_extraction_samples(
    results: Dict[str, Any],
    seen_df: pd.DataFrame,
    unseen_df: pd.DataFrame,
    n_per_category: int = 10,
    output_columns: Optional[List[str]] = None,
) -> None:
    """
    Display qualitative samples for evaluation: 10 successful + 10 unsuccessful extractions
    for entity and for action each (40 total when entity/action are in output_columns).
    Only runs when output_columns includes 'entity' and 'action'.
    """
    out_cols = output_columns or config.output_columns
    if "entity" not in out_cols or "action" not in out_cols:
        print("\n--- Qualitative Extraction Samples (skipped: requires entity and action columns) ---")
        return

    preds_seen = results.get("preds_seen", [])
    preds_unseen = results.get("preds_unseen", [])
    gt_seen = [_ground_truth_to_motifs(row) for _, row in seen_df.iterrows()]
    gt_unseen = [_ground_truth_to_motifs(row) for _, row in unseen_df.iterrows()]

    def _norm(s):
        return str(s).strip().lower() if s else ""

    def _collect_samples(
        df: pd.DataFrame,
        preds: List[List[Dict[str, str]]],
        gt_list: List[List[Dict[str, str]]],
    ) -> List[Dict[str, Any]]:
        samples = []
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= len(preds) or i >= len(gt_list):
                break
            text = str(row.get("text", ""))[:500]
            gt = gt_list[i] or []
            pred = preds[i] or []
            gt_entities = {_norm(m.get("entity", "")) for m in gt if m.get("entity") and _norm(m.get("entity")) not in ("na", "nan", "")}
            gt_actions = {_norm(m.get("action", "")) for m in gt if m.get("action") and _norm(m.get("action")) not in ("na", "nan", "")}
            pred_entities = {_norm(m.get("entity", m.get("actor", ""))) for m in pred if m.get("entity") or m.get("actor")}
            pred_actions = {_norm(m.get("action", "")) for m in pred if m.get("action")}

            entity_ok = bool(gt_entities and gt_entities <= pred_entities) if gt_entities else None
            action_ok = bool(gt_actions and gt_actions <= pred_actions) if gt_actions else None

            samples.append({
                "text": text,
                "gt_entity": ", ".join(gt_entities) if gt_entities else "",
                "gt_action": ", ".join(gt_actions) if gt_actions else "",
                "pred_entity": ", ".join(pred_entities) if pred_entities else "(none)",
                "pred_action": ", ".join(pred_actions) if pred_actions else "(none)",
                "entity_ok": entity_ok,
                "action_ok": action_ok,
            })
        return samples

    all_samples = _collect_samples(seen_df, preds_seen, gt_seen) + _collect_samples(unseen_df, preds_unseen, gt_unseen)

    entity_success = [s for s in all_samples if s["entity_ok"] is True]
    entity_fail = [s for s in all_samples if s["entity_ok"] is False and s["gt_entity"]]
    action_success = [s for s in all_samples if s["action_ok"] is True]
    action_fail = [s for s in all_samples if s["action_ok"] is False and s["gt_action"]]

    def _print_samples(samples: List[Dict], label: str, n: int):
        for i, s in enumerate(samples[:n]):
            print(f"\n  [{i+1}] {label}")
            print(f"      Text: {s['text'][:200]}...")
            print(f"      GT entity: {s['gt_entity']} | Pred entity: {s['pred_entity']}")
            print(f"      GT action: {s['gt_action']} | Pred action: {s['pred_action']}")

    print("\n" + "="*70)
    print("QUALITATIVE EXTRACTION SAMPLES (for manual evaluation)")
    print("="*70)

    print("\n--- Entity: 10 SUCCESSFUL extractions ---")
    _print_samples(entity_success, "Entity SUCCESS", n_per_category)

    print("\n--- Entity: 10 UNSUCCESSFUL extractions ---")
    _print_samples(entity_fail, "Entity FAIL", n_per_category)

    print("\n--- Action: 10 SUCCESSFUL extractions ---")
    _print_samples(action_success, "Action SUCCESS", n_per_category)

    print("\n--- Action: 10 UNSUCCESSFUL extractions ---")
    _print_samples(action_fail, "Action FAIL", n_per_category)

    print(f"\n(Pool: {len(all_samples)} samples; entity success={len(entity_success)}, fail={len(entity_fail)}; action success={len(action_success)}, fail={len(action_fail)})")
    print("="*70)


def visualize_stage1_embeddings(results: Dict[str, Any]):
    """Stage 1: Prompt embeddings in 2D PCA. Engineer prompts in green (dark→light alpha→delta); final prompt in red."""
    print("\n--- Stage 1: Prompt Embeddings ---")
    prompts_dict = results["stage1_prompts"][0]
    final_prompt = results.get("stage2_final_prompt", "")
    prompts = list(prompts_dict.values())
    if len(prompts) < 2:
        print("  Skipping (need 2+ prompts)")
        return

    # Order: Alpha, Beta, Gamma, Delta (dark green → light green)
    agent_order = ["PromptEngineer_Alpha", "PromptEngineer_Beta", "PromptEngineer_Gamma", "PromptEngineer_Delta"]
    green_shades = ["#1b5e20", "#388e3c", "#66bb6a", "#a5d6a7"]  # dark → light
    ordered_names = [k for k in agent_order if k in prompts_dict]
    ordered_prompts = [prompts_dict[k] for k in ordered_names]

    all_prompts = ordered_prompts + ([final_prompt] if final_prompt else [])
    model = SentenceTransformer(config.embedding_model)
    embeddings = model.encode(all_prompts)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7, 6))
    n_engine = len(ordered_names)
    for i in range(n_engine):
        ax.scatter(coords[i, 0], coords[i, 1], c=green_shades[i % 4], s=150, label=ordered_names[i].split("_")[1], zorder=2)
    if final_prompt:
        ax.scatter(coords[n_engine, 0], coords[n_engine, 1], c="#c62828", s=200, label="Final prompt", zorder=3, marker="*")
    ax.legend()
    ax.set_title("Stage 1: Prompt Engineer Outputs & Final Prompt")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def generate_full_evaluation_report(
    results: Dict[str, Any],
    seen_df: Optional[pd.DataFrame] = None,
    unseen_df: Optional[pd.DataFrame] = None,
):
    """Generate evaluation visualizations and summary."""
    print("\n" + "="*70)
    print("GENERATING EVALUATION REPORT")
    print("="*70)

    seen_df = seen_df if seen_df is not None else results.get("seen_df")
    unseen_df = unseen_df if unseen_df is not None else results.get("unseen_df")

    visualize_recall_f1_by_aspect(results)
    if seen_df is not None and unseen_df is not None:
        visualize_entity_specific_recall_f1(results, seen_df, unseen_df)
        visualize_qualitative_extraction_samples(results, seen_df, unseen_df)
    visualize_token_usage(results)
    visualize_discussion_dynamics(results)
    if results.get("stage1_prompts") and len(results["stage1_prompts"][0]) >= 2:
        visualize_stage1_embeddings(results)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    ms = results["metrics_seen"]
    mu = results["metrics_unseen"]
    if "entity_f1" in ms and "action_f1" in ms:
        print(f"\nSeen  - Entity F1: {ms['entity_f1']:.3f}, Action F1: {ms['action_f1']:.3f}, Motif F1: {ms['motif_f1']:.3f}")
        print(f"Unseen - Entity F1: {mu['entity_f1']:.3f}, Action F1: {mu['action_f1']:.3f}, Motif F1: {mu['motif_f1']:.3f}")
    else:
        print(f"\nSeen  - Output F1: {ms.get('output_f1', ms.get('motif_f1', 0)):.3f}")
        print(f"Unseen - Output F1: {mu.get('output_f1', mu.get('motif_f1', 0)):.3f}")
    print(f"Discussion: {results.get('stage2_rounds', 0)} rounds, {results.get('stage2_human_interventions', 0)} human interventions")


def display_final_prompt(prompt: str) -> str:
    """
    Display the final/ideal prompt passed to the worker LLM for qualitative examination.
    Returns the prompt string for programmatic use.
    """
    print("\n" + "="*70)
    print("FINAL PROMPT (passed to worker LLM) - for qualitative examination")
    print("="*70)
    print(prompt)
    print("="*70)
    return prompt


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(df: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
    """
    Main execution. Loads data if not provided, runs pipeline, generates report.
    In Colab: config.api_key = getpass("Enter your OpenAI API key: ")
    """
    if df is None:
        df = load_motifs_data()
    if df is None or len(df) == 0:
        print("ERROR: No data loaded. Call load_motifs_data(path) or pass a DataFrame.")
        return None

    if not config.api_key:
        print("ERROR: OpenAI API key not set!")
        print("Set via: config.api_key = getpass('Enter your OpenAI API key: ')")
        print("Or: config.api_key = 'your-key'")
        return None

    print(f"Data size: {len(df)} rows")
    print(f"Model: {config.model}")

    results = run_full_pipeline(df)
    generate_full_evaluation_report(results)
    return results


# ============================================================================
# COLAB EXECUTION
# ============================================================================
# Run !pip and !gdown in a cell first, then:
# config.api_key = getpass("Enter your OpenAI API key: ")
# df = load_motifs_data()
# results = main(df)
