# finetune.py
# Fine-tuning notebook for the NIFTY options signal pod.
# Designed to run on Kaggle free-tier GPU (T4, 16GB VRAM).
# All MLflow tracking is initialised before the first training run —
# nothing is retrofitted after seeing results.
#
# Base model  : TinyLlama-1.1B-Chat-v1.0
# Fine-tuning : LoRA only, rank 8 primary / rank 4 comparison
# Inference   : CPU, 4-bit quantization via bitsandbytes
#
# Model choice rationale (Section 3 of report):
# I chose TinyLlama-1.1B over Phi-2 for three reasons:
# (1) The task is structured JSON output, not open-ended reasoning —
#     a 1.1B model with LoRA is sufficient for schema-following behaviour.
# (2) TinyLlama fits comfortably on T4 with headroom for multiple runs,
#     which lets me compare rank configurations within the 30hr weekly budget.
# (3) CPU inference speed matters — the pod runs on CPU in production and
#     TinyLlama is meaningfully faster than Phi-2 at 4-bit quantization.
#
# LoRA rank rationale (Section 3 of report):
# I run rank 8 as my primary configuration and rank 4 as a comparison.
# Rank 8 adds ~0.6M trainable parameters — enough expressiveness to learn
# conviction variation across market regimes without overfitting on 255
# training examples. Rank 4 is the comparison baseline: if it performs
# similarly, that tells me the conviction signal is robust and does not
# require additional capacity. If it degrades, that confirms rank 8 was
# the right choice.

# ── Imports ───────────────────────────────────────────────────────────────────

import json
import os
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from eval_suite import build_clean_dataset, DATA_AUDIT_FINDINGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finetune")

# ── Paths (Kaggle layout) ─────────────────────────────────────────────────────
# On Kaggle, input files live under /kaggle/input/<dataset-name>/.
# I keep all paths in one place so nothing breaks if the directory
# structure changes between runs.

BASE_DIR        = Path("/kaggle/working")
INPUT_DIR       = Path("/kaggle/input/nifty-slm-data")
INSTRUCTIONS_PATH = INPUT_DIR / "finetune_instructions.jsonl"
OUTPUT_DIR      = BASE_DIR / "outputs"
ADAPTER_DIR     = BASE_DIR / "lora_adapter"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ── MLflow setup ──────────────────────────────────────────────────────────────
# MLflow is initialised here, before any training code runs.
# I log every hyperparameter decision and its rationale so the experiment
# table in Section 3 is fully reproducible from the run history alone.

mlflow.set_experiment("nifty-signal-pod")

# ── Prompt template ───────────────────────────────────────────────────────────
# The prompt template is the interface between the market state and the model.
# I designed it to:
# (1) Give the model a clear role definition up front
# (2) Present market features in a consistent key=value format so the model
#     learns to associate specific feature patterns with signal outputs
# (3) Remind the model of the exact output schema on every call, reducing
#     the chance of schema drift after fine-tuning
# (4) End with an unambiguous completion cue ("Signal:") so the model
#     knows exactly where its output begins
#
# Worked example (for Section 3 of report):
#
# Input market state:
#   {"nifty_spot": 22859.61, "atm_iv": 13.41, "iv_skew_25d": 3.88,
#    "pcr": 1.13, "adx_14": 29.35, "realized_vol_5d": 13.61,
#    "vix_india": 14.08, "dte_nearest": 2, "moneyness_band": "ATM"}
#
# Constructed prompt → see format_prompt() below
#
# Expected output:
#   {"direction": "PE", "conviction": 0.47, "horizon": "intraday",
#    "signal_id": "17ece277-...", "timestamp": "2024-10-01T09:15:00+05:30"}

SYSTEM_PROMPT = (
    "You are a trading signal generator for NIFTY 50 options. "
    "Analyse the market state snapshot and return ONLY valid JSON "
    "matching this schema exactly: "
    '{\"direction\": \"CE\"|\"PE\"|\"NEUTRAL\", '
    '\"conviction\": float 0.0-1.0, '
    '\"horizon\": \"intraday\"|\"next_session\", '
    '\"signal_id\": string, '
    '\"timestamp\": string ISO8601}. '
    "No explanation. No markdown. JSON only."
)


def format_prompt(market_state: dict, output: str | None = None) -> str:
    """
    Formats a market state dict into the instruction prompt the model sees.
    During training, output is appended so the model learns the full
    input→output mapping. During inference, output is omitted and the
    model completes from the Signal: cue.
    """
    ms = market_state if isinstance(market_state, dict) else json.loads(market_state)

    user_content = (
        f"Market snapshot:\n"
        f"  nifty_spot={ms.get('nifty_spot', 'N/A')}\n"
        f"  atm_iv={ms.get('atm_iv', 'N/A')}\n"
        f"  iv_skew_25d={ms.get('iv_skew_25d', 'N/A')}\n"
        f"  pcr={ms.get('pcr', 'N/A')}\n"
        f"  adx_14={ms.get('adx_14', 'N/A')}\n"
        f"  realized_vol_5d={ms.get('realized_vol_5d', 'N/A')}\n"
        f"  vix_india={ms.get('vix_india', 'N/A')}\n"
        f"  dte_nearest={ms.get('dte_nearest', 'N/A')}\n"
        f"  moneyness_band={ms.get('moneyness_band', 'N/A')}\n"
        f"Signal:"
    )

    # TinyLlama uses the ChatML format internally
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{user_content}</s>\n"
        f"<|assistant|>\n"
    )

    if output is not None:
        prompt += output + "</s>"

    return prompt


# ── RAG prompt template ───────────────────────────────────────────────────────
# For the RAG experiment I prepend retrieved historical episodes to the
# user content before the market snapshot. The hypothesis is that similar
# historical outcomes give the model grounding for its conviction score —
# if three retrieved episodes with similar ADX/VIX/PCR all resolved as PE,
# the model should produce a higher conviction PE signal.
#
# I keep the retrieved context concise (summary only, not full market state)
# to avoid exceeding the context window on a 1.1B model.

def format_prompt_with_rag(market_state: dict, episodes: list, output: str | None = None) -> str:
    """
    RAG variant of format_prompt. Prepends up to 3 retrieved episode
    summaries before the market snapshot. Used in the RAG ablation
    experiment — the non-RAG baseline uses format_prompt() instead.
    """
    context_lines = ["Retrieved similar historical episodes:"]
    for i, ep in enumerate(episodes[:3], 1):
        context_lines.append(
            f"  [{i}] {ep.get('summary', '')} → outcome: {ep.get('outcome', 'unknown')}"
        )
    context_block = "\n".join(context_lines)

    ms = market_state if isinstance(market_state, dict) else json.loads(market_state)

    user_content = (
        f"{context_block}\n\n"
        f"Current market snapshot:\n"
        f"  nifty_spot={ms.get('nifty_spot', 'N/A')}\n"
        f"  atm_iv={ms.get('atm_iv', 'N/A')}\n"
        f"  iv_skew_25d={ms.get('iv_skew_25d', 'N/A')}\n"
        f"  pcr={ms.get('pcr', 'N/A')}\n"
        f"  adx_14={ms.get('adx_14', 'N/A')}\n"
        f"  realized_vol_5d={ms.get('realized_vol_5d', 'N/A')}\n"
        f"  vix_india={ms.get('vix_india', 'N/A')}\n"
        f"  dte_nearest={ms.get('dte_nearest', 'N/A')}\n"
        f"  moneyness_band={ms.get('moneyness_band', 'N/A')}\n"
        f"Signal:"
    )

    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{user_content}</s>\n"
        f"<|assistant|>\n"
    )

    if output is not None:
        prompt += output + "</s>"

    return prompt


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_hf_dataset(clean_records: list) -> Dataset:
    """
    Converts the cleaned training records into a HuggingFace Dataset
    with a single 'text' column containing the full formatted prompt.
    SFTTrainer expects this format.
    """
    texts = []
    for r in clean_records:
        market_state = json.loads(r['input'])
        output       = r['output']
        texts.append(format_prompt(market_state, output=output))
    return Dataset.from_dict({"text": texts})


# ── LoRA configuration ────────────────────────────────────────────────────────
# I target the attention projection layers (q_proj, v_proj) because these
# are where the model learns to associate input patterns with output tokens.
# Adding k_proj and o_proj would increase expressiveness but also parameter
# count — I keep it to q and v for the primary run to stay conservative
# with 255 training examples.
#
# alpha=16 with rank=8 gives an effective scaling factor of 2.0 (alpha/rank).
# This is the standard initialisation — it means LoRA weights start small
# and scale up to a reasonable magnitude as training progresses.
#
# dropout=0.05 is a light regularisation. With only 255 examples I want
# some noise to prevent the model from memorising the training outputs,
# but not so much that it fails to converge.

def make_lora_config(rank: int) -> LoraConfig:
    """
    Returns a LoraConfig for the given rank.
    I call this twice — once for rank 8 (primary) and once for rank 4
    (comparison experiment) — so the only variable between runs is rank.
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=16,          # scaling factor = alpha/r. Fixed at 16 so
                                # the effective scale changes with rank,
                                # which is what I want to measure.
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],  # attention projections only
    )


# ── Training arguments ────────────────────────────────────────────────────────
# I use a small batch size with gradient accumulation to simulate a larger
# effective batch on the T4's 16GB VRAM without OOM errors.
# effective_batch = per_device_train_batch_size × gradient_accumulation_steps
#                = 4 × 4 = 16

def make_training_args(run_name: str, output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,   # effective batch size = 16
        learning_rate=2e-4,              # standard LoRA learning rate
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=True,                       # T4 supports fp16, not bf16
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",        # no separate val split —
                                         # I evaluate via walk-forward only
        report_to="none",                # MLflow logging is handled manually
        dataloader_num_workers=2,
        seed=42,
    )


# ── Conviction field design ───────────────────────────────────────────────────
# This is addressed explicitly because the brief flags it as a design problem.
#
# The conviction score (0.0–1.0) appears in the model output as text — e.g.
# the token sequence "0", ".", "7", "2". It is NOT a softmax probability.
#
# Softmax over the next token gives the probability that "0" (or "0.72" as
# a subword token) follows the preceding context. That is a measure of how
# confidently the model predicts that specific token string — not a measure
# of how reliable the directional signal is.
#
# What makes conviction meaningful in my implementation:
# The training data contains conviction values that were assigned alongside
# directional labels. After dropping the string-valued rows, the remaining
# 255 records have float convictions in [0.31, 0.79]. The model learns to
# associate specific market state patterns (high ADX, low VIX, strong skew)
# with higher conviction outputs. This is a form of implicit calibration —
# the conviction value reflects the strength of the market-state pattern
# relative to training examples, not a probabilistic confidence estimate.
#
# I validate this in eval_suite.conviction_calibration(): if higher conviction
# bins show higher directional accuracy, the value is informative. If not,
# I report that honestly in Section 4.


# ── Main training function ────────────────────────────────────────────────────

def train(rank: int, run_name: str):
    """
    Full training pipeline for a single LoRA rank configuration.
    Called twice — once for rank 8, once for rank 4.
    """

    logger.info(f"Starting run: {run_name} | rank={rank}")

    # ── Step 1: Load and clean training data ──────────────────────────────────
    # I apply the same cleaning function used in eval_suite so the training
    # data preparation is auditable from a single definition.
    clean_records = build_clean_dataset(str(INSTRUCTIONS_PATH))
    logger.info(f"Clean training records: {len(clean_records)}")

    dataset = build_hf_dataset(clean_records)
    logger.info(f"Dataset size: {len(dataset)} examples")

    # ── Step 2: Load base model with 4-bit quantization ───────────────────────
    # I load in 4-bit for training too (QLoRA pattern) to fit within T4 VRAM.
    # The quantization config is the same one used at inference time so
    # there is no mismatch between training and production behaviour.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",       # NormalFloat4 — best for LLM weights
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # nested quantization saves ~0.4GB
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token  # TinyLlama has no pad token
    tokenizer.padding_side = "right"           # pad on right for causal LM

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    # ── Step 3: Apply LoRA ────────────────────────────────────────────────────
    lora_config = make_lora_config(rank)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Step 4: MLflow run ────────────────────────────────────────────────────
    with mlflow.start_run(run_name=run_name):

        # Log everything that could affect results
        mlflow.log_params({
            "base_model":          BASE_MODEL_ID,
            "lora_rank":           rank,
            "lora_alpha":          16,
            "lora_dropout":        0.05,
            "target_modules":      "q_proj,v_proj",
            "num_train_epochs":    3,
            "per_device_batch":    4,
            "grad_accum_steps":    4,
            "effective_batch":     16,
            "learning_rate":       2e-4,
            "lr_scheduler":        "cosine",
            "warmup_ratio":        0.05,
            "quantization":        "4bit_nf4_double",
            "clean_train_records": len(clean_records),
            "dropped_records":     300 - len(clean_records),
            "seed":                42,
        })

        mlflow.log_text(DATA_AUDIT_FINDINGS, "data_audit_findings.txt")

        # ── Step 5: Train ─────────────────────────────────────────────────────
        adapter_path = str(ADAPTER_DIR / run_name)
        training_args = make_training_args(run_name, adapter_path)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=512,        # well within TinyLlama's 2048 limit;
                                       # our prompts are ~200 tokens typically
            args=training_args,
            peft_config=lora_config,
        )

        train_result = trainer.train()

        # Log training metrics
        mlflow.log_metrics({
            "train_loss":           train_result.training_loss,
            "train_runtime_sec":    train_result.metrics.get("train_runtime", 0),
            "train_samples_per_sec": train_result.metrics.get("train_samples_per_second", 0),
        })

        # ── Step 6: Save adapter ──────────────────────────────────────────────
        trainer.model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        mlflow.log_artifacts(adapter_path, artifact_path="lora_adapter")

        logger.info(f"Adapter saved to {adapter_path}")
        mlflow.log_param("adapter_path", adapter_path)

    return adapter_path


# ── Inference pod ─────────────────────────────────────────────────────────────
# This is the pod the orchestrator wraps. It loads the fine-tuned adapter
# on top of the 4-bit quantized base model and runs inference on CPU.
# I keep the pod stateless — it accepts a market state dict and returns
# a raw string. The orchestrator handles everything else.

class SignalPod:
    """
    The fine-tuned signal pod. Loaded once and reused across all
    walk-forward inference calls. Running on CPU with 4-bit quantization
    as required by the brief.

    I deliberately do not parse or validate the output here — that is
    the orchestrator's job. The pod's only responsibility is to produce
    a string. Whether that string is valid JSON is the orchestrator's concern.
    """

    def __init__(self, adapter_path: str):
        logger.info(f"Loading pod from adapter: {adapter_path}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="cpu",           # CPU inference as required
            trust_remote_code=False,
        )

        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()

        logger.info("Pod loaded and ready.")

    def __call__(self, market_state: dict, use_rag: bool = False,
                 episodes: list | None = None) -> str:
        """
        Runs inference for a single market state.
        Returns the raw model output string — unparsed, unvalidated.

        use_rag: if True, prepends retrieved episodes to the prompt.
                 The orchestrator passes episodes from retrieve() when
                 running the RAG ablation experiment.
        """
        if use_rag and episodes:
            prompt = format_prompt_with_rag(market_state, episodes)
        else:
            prompt = format_prompt(market_state)

        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=120,     # enough for the JSON schema
                do_sample=False,        # greedy decoding — deterministic output
                                        # I want reproducible signals, not
                                        # sampled ones, for a trading pod
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens, not the prompt
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return raw


# ── RAG ablation experiment ───────────────────────────────────────────────────
# I run the full walk-forward evaluation twice: once without RAG (baseline)
# and once with RAG (retrieved episodes prepended to each prompt).
# The comparison is reported honestly in Section 3 — I report whether
# retrieval helped, hurt, or made no difference, and examine specifically
# whether conviction scores changed under retrieval and whether those
# changes were directionally justified.

def run_rag_ablation(pod: SignalPod, orchestrator, blocks: list,
                     actuals_col: str = "actual_direction"):
    """
    Runs the walk-forward evaluation in two conditions:
      - baseline: no retrieval context
      - rag:      retrieve(market_state, k=3) prepended to each prompt

    Returns (baseline_results, rag_results) for comparison in Section 3.
    """
    from retrieve import retrieve
    from eval_suite import run_walkforward  # reuse the same runner

    # Condition 1: baseline (no RAG)
    logger.info("Running baseline (no RAG)...")
    baseline_results = run_walkforward(orchestrator, blocks, actuals_col)

    # Condition 2: RAG
    logger.info("Running RAG condition...")

    # Wrap the pod so it automatically fetches episodes before each call
    class RagPod:
        def __init__(self, base_pod):
            self.base_pod = base_pod

        def __call__(self, market_state: dict) -> str:
            episodes = retrieve(market_state, k=3)
            return self.base_pod(market_state, use_rag=True, episodes=episodes)

    rag_pod       = RagPod(pod)
    rag_orch      = orchestrator.__class__(pod=rag_pod)
    rag_results   = run_walkforward(rag_orch, blocks, actuals_col)

    return baseline_results, rag_results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Experiment 1: rank 8 (primary) ───────────────────────────────────────
    adapter_r8 = train(rank=8, run_name="tinyllama_lora_r8")

    # ── Experiment 2: rank 4 (comparison) ────────────────────────────────────
    # The only variable between these two runs is the LoRA rank.
    # Everything else — data, base model, training args, seed — is identical.
    # This isolates rank as the single experimental variable so the MLflow
    # comparison in Section 3 is clean and defensible.
    adapter_r4 = train(rank=4, run_name="tinyllama_lora_r4")

    logger.info("Both training runs complete.")
    logger.info(f"Rank 8 adapter: {adapter_r8}")
    logger.info(f"Rank 4 adapter: {adapter_r4}")
    logger.info("Load the preferred adapter into SignalPod for walk-forward evaluation.")