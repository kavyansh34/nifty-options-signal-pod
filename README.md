# nifty-options-signal-pod

**Kavyansh Gupta** — Quant Singularity AI-SLM Internship Screening, Summer 2026

A fine-tuned small language model that reads NIFTY 50 options market state snapshots and outputs structured trading signals, wrapped in a deterministic orchestrator that enforces safety rules before anything reaches downstream systems.

---

## What this is

The system has two layers:

**Signal pod** — TinyLlama-1.1B fine-tuned with LoRA (rank 8) on 255 cleaned instruction examples, with weighted sampling to correct class imbalance. Takes a market state snapshot (spot price, ATM IV, IV skew, PCR, ADX, realised vol, India VIX, DTE, moneyness band) and outputs a structured JSON signal with direction (CE/PE/NEUTRAL), conviction (0.0–1.0), and horizon.

**Orchestrator** — a deterministic Python wrapper that applies three rules in sequence before the signal reaches downstream: ADX suppression (ADX < 20 → NEUTRAL without calling model), parse validation (invalid JSON → NEUTRAL + log), and conviction threshold (conviction < 0.40 → downgrade to NEUTRAL). Every decision is logged with a reason code and the values that triggered it.

---

## Repo structure

```
nifty-options-signal-pod/
├── eval_suite.py                  # Eval metrics, thresholds, data cleaning
│                                  # Committed before first training run
├── Orchestrator.py                # Orchestrator — three suppression rules
├── Finetune.py                    # Training script reference
├── nifty-signal-pod-v2.ipynb     # Kaggle training notebook (final version)
├── retrieve.py                    # Provided retrieval function — not modified
├── Requirements.txt
├── README.md
├── report.pdf                     # Written report (4–6 pages)
├── adapters/tinyllama_lora_r8/   # LoRA adapter weights
└── mlruns/                        # MLflow experiment artifacts

Data files (not committed — provided by Quant Singularity):
├── finetune_instructions.jsonl
├── market_states.parquet
└── rag_corpus.jsonl
```

---

## Kaggle notebook

**URL:** `https://www.kaggle.com/kavyansh34/nifty-signal-pod-v2`

Training runs on Kaggle free-tier T4 GPU. To reproduce:
1. Upload `nifty-signal-pod-v2.ipynb` to Kaggle
2. Attach the dataset containing `finetune_instructions.jsonl`, `market_states.parquet`, `rag_corpus.jsonl`, `retrieve.py`
3. Enable GPU T4 x1 in session settings
4. Run all cells in order — MLflow init cell must run before any training cell

---

## Setup

```bash
git clone https://github.com/kavyansh34/nifty-options-signal-pod
cd nifty-options-signal-pod
pip install -r Requirements.txt
```

---

## Run smoke tests (no GPU needed)

```bash
# Data audit and cleaning — prints 255 clean records + 3 findings
python eval_suite.py

# Orchestrator smoke test — all suppression paths fire correctly
python Orchestrator.py
```

Expected output from `Orchestrator.py`:

```
Orchestrator smoke test
ADX < 20 — rule 1 should suppress      → reason: ADX_BELOW_THRESHOLD
ADX ok, valid signal — should pass      → reason: OK
ADX ok, low conviction — rule 3         → reason: LOW_CONVICTION
ADX ok, bad JSON — rule 2               → reason: PARSE_FAIL
ADX ok, missing field — rule 2          → reason: MISSING_FIELD:signal_id
```

---

## Inference

```python
from Orchestrator import Orchestrator
from Finetune import SignalPod

pod = SignalPod(adapter_path='adapters/tinyllama_lora_r8')
orch = Orchestrator(pod=pod)

market_state = {
    'nifty_spot': 22859.61,
    'atm_iv': 13.41,
    'iv_skew_25d': 3.88,
    'pcr': 1.13,
    'adx_14': 29.35,
    'realized_vol_5d': 13.61,
    'vix_india': 14.08,
    'dte_nearest': 2,
    'moneyness_band': 'ATM'
}

output = orch.process(market_state)
print(output)
```

The orchestrator always returns valid JSON. The downstream pipeline never sees raw pod output.

---

## Evaluation

Walk-forward only. Days 31–60, six 5-day rolling blocks. k-fold not used — it introduces look-ahead bias on time series data.

Thresholds committed before training (see `eval_suite.py`):

| Metric | Pass | Warn | Fail |
|--------|------|------|------|
| Directional accuracy | > 52% | 48–52% | < 48% |
| Schema pass rate | > 95% | 90–95% | < 90% |
| Parse failure rate | < 2% | 2–5% | > 5% |
| Conviction calibration | Monotonic | Mostly monotonic | Inverted / flat |
| VIX regime gap | < 8pp | 8–15pp | > 15pp |

Results reported with Wilson score 95% confidence intervals. See `report.pdf` Section 4.

---

## Data audit findings

Three issues found in `finetune_instructions.jsonl` before training:

1. **Rows 47–91 (45 records)** — conviction field contains strings (`"high"`, `"moderate"`, `"low"`, `"weak"`, `"strong"`, `"high confidence"`, `"moderate confidence"`, `"0.8 (high)"`) instead of floats. Dropped entirely — mapping strings to floats would fabricate training labels. Clean records remaining: 255.

2. **All 300 records** — output field uses `"generated_at"` instead of the spec's `"timestamp"`. Renamed in preprocessing so the model learns the correct field name. Without this fix, schema pass rate would be 0%.

3. **No ADX < 20 examples** — the model has no training exposure to trendless markets. The orchestrator suppresses these correctly but the model behaviour in that regime is untested. Documented in Section 5 of the report.

---

## Training decisions

**Why TinyLlama over Phi-2:** The task is structured JSON output, not open-ended reasoning. TinyLlama fits comfortably on T4 with headroom for multiple experiments and is faster at CPU inference where the pod runs in production.

**Why LoRA rank 8:** Adds ~0.6M trainable parameters — sufficient expressiveness for conviction variation across market regimes without overfitting on 255 examples.

**Why weighted sampling instead of oversampling:** The initial training run collapsed to NEUTRAL on all eval predictions due to 44% NEUTRAL class imbalance. Oversampling CE and PE with duplicates caused memorisation. Weighted sampling (CE=2x, PE=2x, NEUTRAL=1x) corrects the imbalance without creating duplicate records.

**Why conviction is not a softmax probability:** The conviction score is generated as text. Softmax over the next token measures how likely that token string is to follow the context — not the model's calibrated uncertainty about direction. Conviction is made meaningful through training on float-valued labels and validated via calibration analysis in the eval suite.

---

## MLflow experiments

Compared LoRa rank8 v/s rank 4 --- found rank 8 more better.

| Run | Rank | Alpha | LR | Epochs | Sampling | Train Loss |
|-----|------|-------|----|--------|----------|------------|
| tinyllama_lora_r8 (v1) | 8 | 16 | 2e-4 | 3 | unweighted | ~1.35 |
| tinyllama_lora_r8 (v2) | 8 | 32 | 5e-4 | 8 | CE/PE 2x weighted | ~0.9x |

Full MLflow artifacts in `mlruns/`.

---

