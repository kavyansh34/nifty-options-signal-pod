# nifty-options-signal-pod

A fine-tuned small language model that reads NIFTY 50 options market state snapshots and outputs structured trading signals, wrapped in a deterministic orchestrator that enforces safety rules before anything reaches downstream systems.

Built as part of the Quant Singularity AI-SLM internship screening project, Summer 2026.

---

## What this is

The system has two layers:

**Signal pod** — TinyLlama-1.1B fine-tuned with LoRA on 255 cleaned instruction examples. Takes a market state snapshot (spot price, ATM IV, IV skew, PCR, ADX, realised vol, India VIX, DTE, moneyness band) and outputs a structured JSON signal with direction (CE/PE/NEUTRAL), conviction (0.0–1.0), and horizon.

**Orchestrator** — a deterministic Python wrapper that applies three rules in sequence before the signal reaches downstream: ADX suppression (< 20 → NEUTRAL without calling model), parse validation (invalid JSON → NEUTRAL + log), and conviction threshold (< 0.40 → downgrade to NEUTRAL). Every decision is logged with a reason code and the values that triggered it.

---

## Repo structure

```
nifty-options-signal-pod/
├── eval_suite.py               # Eval metrics, thresholds, data cleaning
│                               # Committed before first training run
├── orchestrator.py             # Orchestrator — three suppression rules
├── finetune.py                 # Training script (reference)
├── nifty_signal_pod.ipynb      # Kaggle training notebook
├── retrieve.py                 # Provided retrieval function — not modified
├── requirements.txt
├── README.md
└── report.pdf                  # Written report (4–6 pages)

Data files (not committed — too large):
├── finetune_instructions.jsonl
├── market_states.parquet
└── rag_corpus.jsonl
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/nifty-options-signal-pod
cd nifty-options-signal-pod
pip install -r requirements.txt
```

---

## Run the smoke tests (no GPU needed)

```bash
# Verify data audit and cleaning — should print 255 clean records
python eval_suite.py

# Verify orchestrator — all five suppression paths should fire correctly
python orchestrator.py
```

Expected output from `orchestrator.py`:

```
Orchestrator smoke test
ADX < 20 — rule 1 should suppress      → reason: ADX_BELOW_THRESHOLD
ADX ok, valid signal — should pass      → reason: OK
ADX ok, low conviction — rule 3         → reason: LOW_CONVICTION
ADX ok, bad JSON — rule 2               → reason: PARSE_FAIL
ADX ok, missing field — rule 2          → reason: MISSING_FIELD:signal_id
```

---

## Training (Kaggle GPU)

Training runs on Kaggle free-tier T4 GPU. The notebook is at:

**Kaggle notebook URL:** `[Access the Notebook](https://www.kaggle.com/code/kavyanshgupta23/nifty-signal-pod-v4)`

To reproduce:
1. Upload `nifty_signal_pod.ipynb` to Kaggle
2. Attach the "market-data" dataset (contains `finetune_instructions.jsonl`, `market_states.parquet`, `rag_corpus.jsonl`, `retrieve.py`)
3. Enable GPU T4 x1 in session settings
4. Run all cells in order — Cell 3 (MLflow init) must run before any training cell

Two experiments are run:
- `tinyllama_lora_r8` — rank 8, primary configuration
- `tinyllama_lora_r4` — rank 4, comparison

LoRA adapter weights are saved to `/kaggle/working/adapters/` and logged as MLflow artifacts.

---

## Inference (CPU)

After training, load the adapter and run inference through the orchestrator:

```python
from orchestrator import Orchestrator
from finetune import SignalPod

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

Walk-forward only. Days 31–60, evaluated in 5-day rolling blocks. k-fold is not used — it would introduce look-ahead bias on a time series.

Thresholds committed before training (see `eval_suite.py`):

| Metric | Pass | Warn | Fail |
|--------|------|------|------|
| Directional accuracy | > 52% | 48–52% | < 48% |
| Schema pass rate | > 95% | 90–95% | < 90% |
| Parse failure rate | < 2% | 2–5% | > 5% |
| Conviction calibration | Monotonic | Mostly monotonic | Inverted / flat |
| VIX regime gap | < 8pp | 8–15pp | > 15pp |

Results reported with Wilson score 95% confidence intervals. See `report.pdf` Section 4 for full results.

---

## Data audit findings

Three issues were found in `finetune_instructions.jsonl` before training:

1. **Rows 47–91 (45 records)** — conviction field contains strings (`"high"`, `"moderate"`, `"low"`, etc.) instead of floats. Dropped entirely — mapping strings to floats would fabricate training labels.

2. **All 300 records** — output field uses `"generated_at"` instead of the spec's `"timestamp"`. Renamed to `"timestamp"` in preprocessing so the model learns the correct field name.

3. **No ADX < 20 examples** — the model has never seen a trendless market state during training. Documented as a coverage gap in Section 5 of the report.

Clean training set after audit: **255 records**.

---

## Key design decisions

**Why TinyLlama over Phi-2:** The task is structured JSON output, not open-ended reasoning. TinyLlama fits comfortably on T4 with headroom for multiple experiments, and is faster on CPU inference where the pod runs in production.

**Why LoRA rank 8:** Adds ~0.6M trainable parameters — sufficient expressiveness to learn conviction variation across market regimes without overfitting on 255 examples. Rank 4 run as comparison baseline.

**Why conviction is not a softmax probability:** The conviction score is generated as text by the model. Softmax over the next token measures how likely that token string is to follow the context — not the model's calibrated uncertainty about the direction. Conviction is made meaningful through training on float-valued labels and validated via calibration analysis in the eval suite.

---

## Contact

Questions about this submission: refer to the brief contact at surya@quantsingularity.in
