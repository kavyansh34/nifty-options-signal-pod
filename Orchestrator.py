# orchestrator.py
# The orchestrator wraps the signal pod and applies three deterministic
# suppression rules in sequence before anything reaches the downstream pipeline.
# Downstream systems only ever read the orchestrator output — never the raw
# pod signal. This file is the single point of control for that guarantee.
#
# I import validate_schema from eval_suite.py deliberately: the same schema
# rules must apply during both inference and evaluation. Having two copies
# would risk them silently diverging, which would mean passing eval but
# failing in production. Keeping one definition enforces consistency by design.

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from eval_suite import validate_schema, THRESHOLDS

# ── Logging setup ─────────────────────────────────────────────────────────────
# Every orchestrator decision is logged with a reason code and the values
# that triggered it. This is not optional — without it I have no way to
# audit suppression rates per window or diagnose unexpected behaviour
# during the live walkthrough.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("orchestrator")

# ── Reason codes ──────────────────────────────────────────────────────────────
# Fixed string constants so downstream log parsers never break due to
# a typo or phrasing change. compute_orchestrator_stats() in eval_suite.py
# matches against these exact strings when computing suppression rates.

RC_ADX_SUPPRESSED   = "ADX_BELOW_THRESHOLD"
RC_PARSE_FAIL       = "PARSE_FAIL"
RC_LOW_CONVICTION   = "LOW_CONVICTION"
RC_OK               = "OK"

# ── Neutral fallback factory ──────────────────────────────────────────────────
# Every suppression path returns a NEUTRAL signal built by this function.
# I centralised it here so there is exactly one place where the fallback
# structure is defined — if the schema ever changes, one edit covers all paths.

def _neutral_output(reason: str, raw_pod_output: str | None = None) -> dict:
    """
    Builds the standard NEUTRAL fallback the orchestrator emits whenever
    it suppresses or downgrades a pod signal. The raw pod output is included
    in the log entry so I can inspect what the model actually produced,
    even when the orchestrator did not pass it downstream.
    """
    return {
        "direction":  "NEUTRAL",
        "conviction": 0.0,
        "horizon":    "intraday",
        "signal_id":  str(uuid.uuid4()),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "orchestrator_reason": reason,
        "raw_pod_output": raw_pod_output,
    }


# ── Log entry factory ─────────────────────────────────────────────────────────
# Every decision — suppression, downgrade, or pass-through — produces one
# log entry with the same structure. I keep this consistent so
# compute_orchestrator_stats() can aggregate entries without special casing.

def _log_entry(
    reason: str,
    market_state: dict,
    pod_output: dict | None,
    final_output: dict,
    trigger_values: dict,
) -> dict:
    entry = {
        "reason":        reason,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "trigger_values": trigger_values,
        "market_state":  market_state,
        "pod_output":    pod_output,
        "final_output":  final_output,
    }
    logger.info(json.dumps(entry))
    return entry


# ── Core orchestrator ─────────────────────────────────────────────────────────

class Orchestrator:
    """
    Wraps the signal pod and enforces three suppression rules in sequence.
    The rules are applied in the order defined in the brief — ADX check
    first, parse check second, conviction threshold third. Short-circuiting
    on the ADX check means the model is never called in choppy markets,
    which is the most important safety property of the whole system.

    I store every log entry in self.log so eval_suite.compute_orchestrator_stats()
    can aggregate them after a full walk-forward run without needing to
    parse a log file.
    """

    ADX_THRESHOLD        = THRESHOLDS["adx_suppress_threshold"]        # 20.0
    CONVICTION_THRESHOLD = THRESHOLDS["conviction_suppress_threshold"]  # 0.40

    def __init__(self, pod):
        """
        Args:
            pod: any callable that accepts a market_state dict and returns
                 a raw string. Expected to be the fine-tuned LLM pod, but
                 the orchestrator does not depend on its internals — it only
                 cares about what the pod returns as text.
        """
        self.pod = pod
        self.log: list[dict] = []

    def process(self, market_state: dict) -> dict:
        """
        Main entry point. Applies three rules in sequence and returns
        the final orchestrator output. The downstream pipeline calls this —
        it never calls the pod directly.

        Rule 1 — ADX check:
            If ADX is below 20, the market lacks trend structure. I suppress
            the signal entirely and return NEUTRAL without calling the model.
            This is the most important rule: it prevents the pod from issuing
            directional signals in conditions where direction is meaningless.

        Rule 2 — Parse check:
            If the pod output is not valid JSON matching the required schema,
            I return NEUTRAL and log the raw string. A pod that cannot produce
            valid JSON on a given call is not safe to pass downstream regardless
            of what it was trying to say.

        Rule 3 — Conviction threshold:
            If conviction is below 0.40, I downgrade direction to NEUTRAL.
            The pod still ran and produced a parseable signal — I just do not
            trust a low-conviction directional call enough to pass it on.
        """

        # ── Rule 1: ADX check ─────────────────────────────────────────────────
        adx = market_state.get("adx_14", 0.0)
        if adx < self.ADX_THRESHOLD:
            final = _neutral_output(RC_ADX_SUPPRESSED)
            entry = _log_entry(
                reason=RC_ADX_SUPPRESSED,
                market_state=market_state,
                pod_output=None,      # model was never called
                final_output=final,
                trigger_values={"adx_14": adx, "threshold": self.ADX_THRESHOLD},
            )
            self.log.append(entry)
            return final

        # ── Call the pod ──────────────────────────────────────────────────────
        # The pod is called only after the ADX check passes. Whatever it
        # returns is treated as untrusted text until rule 2 validates it.
        raw_output: str = self.pod(market_state)

        # ── Rule 2: Parse check ───────────────────────────────────────────────
        is_valid, parsed, reason_code = validate_schema(raw_output)
        if not is_valid:
            final = _neutral_output(reason_code, raw_pod_output=raw_output)
            entry = _log_entry(
                reason=reason_code,
                market_state=market_state,
                pod_output=None,
                final_output=final,
                trigger_values={"parse_error": reason_code, "raw_length": len(raw_output)},
            )
            self.log.append(entry)
            return final

        # ── Rule 3: Conviction threshold ──────────────────────────────────────
        conviction = float(parsed["conviction"])
        if conviction < self.CONVICTION_THRESHOLD:
            # I downgrade direction to NEUTRAL but preserve the rest of the
            # parsed signal so the log shows what the model was trying to say.
            final = {**parsed, "direction": "NEUTRAL",
                     "orchestrator_reason": RC_LOW_CONVICTION}
            entry = _log_entry(
                reason=RC_LOW_CONVICTION,
                market_state=market_state,
                pod_output=parsed,
                final_output=final,
                trigger_values={
                    "conviction": conviction,
                    "threshold":  self.CONVICTION_THRESHOLD,
                    "original_direction": parsed["direction"],
                },
            )
            self.log.append(entry)
            return final

        # ── All rules passed — signal goes downstream ─────────────────────────
        final = {**parsed, "orchestrator_reason": RC_OK}
        entry = _log_entry(
            reason=RC_OK,
            market_state=market_state,
            pod_output=parsed,
            final_output=final,
            trigger_values={
                "adx_14":     adx,
                "conviction": conviction,
            },
        )
        self.log.append(entry)
        return final

    def reset_log(self):
        """
        Clears the log between evaluation windows so per-block stats
        are computed on that block only, not accumulated across the run.
        """
        self.log = []

    def stats(self):
        """
        Convenience wrapper — returns an OrchestratorStats object for the
        current log. I call this at the end of each 5-day walk-forward block
        and report suppression rates per window in Section 4.
        """
        from eval_suite import compute_orchestrator_stats
        return compute_orchestrator_stats(self.log)


# ── Walk-forward runner ───────────────────────────────────────────────────────
# This function wires the orchestrator to the eval suite's walk-forward
# blocks and returns per-block results in the structure Section 4 expects.

def run_walkforward(orchestrator: Orchestrator, blocks: list, actuals_col: str = "actual_direction"):
    """
    Runs the full walk-forward evaluation.

    Args:
        orchestrator:  an Orchestrator instance with a pod attached
        blocks:        output of eval_suite.make_walkforward_blocks()
        actuals_col:   column name in each block df containing ground truth direction

    Returns:
        List of per-block result dicts, one per 5-day window.
    """
    from eval_suite import (
        directional_accuracy, conviction_calibration,
        is_calibration_monotonic, wilson_ci
    )

    all_results = []

    for start_day, end_day, block_df in blocks:
        orchestrator.reset_log()

        predictions, actuals, convictions = [], [], []

        for _, row in block_df.iterrows():
            market_state = row.to_dict()
            output = orchestrator.process(market_state)

            predictions.append(output["direction"])
            actuals.append(row[actuals_col])

            # Conviction is 0.0 for suppressed/downgraded signals —
            # these are excluded from calibration analysis by the
            # non-NEUTRAL filter inside conviction_calibration().
            convictions.append(float(output.get("conviction", 0.0)))

        # Per-block directional accuracy with Wilson CI
        n_directional = sum(1 for p in predictions if p != "NEUTRAL")
        n_correct     = sum(p == a for p, a in zip(predictions, actuals) if p != "NEUTRAL")
        acc           = directional_accuracy(predictions, actuals)
        ci_lo, ci_hi  = wilson_ci(n_correct, n_directional) if n_directional > 0 else (0.0, 1.0)

        # Per-block orchestrator stats
        stats = orchestrator.stats()

        # Conviction calibration for this block
        calibration = conviction_calibration(convictions, predictions, actuals)

        block_result = {
            "window":               f"days_{start_day}_{end_day}",
            "directional_accuracy": acc,
            "ci_95":                (round(ci_lo, 3), round(ci_hi, 3)),
            "n_total":              len(predictions),
            "n_directional":        n_directional,
            "adx_suppression_rate": stats.suppression_rate,
            "parse_failure_rate":   stats.parse_failure_rate,
            "downgrade_rate":       stats.downgrade_rate,
            "passed_through_rate":  stats.passed_through / stats.total,
            "calibration":          calibration,
            "calibration_monotonic": is_calibration_monotonic(calibration),
        }

        all_results.append(block_result)

        # Print a quick summary per block so I can monitor progress on Kaggle
        print(
            f"Window days {start_day}–{end_day} | "
            f"acc={acc:.3f} [{ci_lo:.3f},{ci_hi:.3f}] | "
            f"suppressed={stats.suppression_rate:.2%} | "
            f"downgraded={stats.downgrade_rate:.2%} | "
            f"parse_fail={stats.parse_failure_rate:.2%}"
        )

    return all_results


# ── Quick smoke test ──────────────────────────────────────────────────────────
# I ran this before connecting the real pod to verify all three rules
# fire correctly and log the right reason codes. It uses a mock pod
# so no GPU or model weights are needed to validate orchestrator logic.

if __name__ == "__main__":
    import random

    class MockPod:
        """
        Simulates four pod behaviours for smoke testing:
        valid signal, low conviction, bad JSON, and missing field.
        Cycles through them in order so all three orchestrator rules
        are exercised in a single test run.
        """
        def __init__(self):
            self._call_count = 0

        def __call__(self, market_state: dict) -> str:
            self._call_count += 1
            mode = self._call_count % 4

            if mode == 1:   # normal valid signal
                return json.dumps({
                    "direction": "CE", "conviction": 0.72,
                    "horizon": "intraday",
                    "signal_id": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            elif mode == 2:  # low conviction — triggers rule 3
                return json.dumps({
                    "direction": "PE", "conviction": 0.31,
                    "horizon": "intraday",
                    "signal_id": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            elif mode == 3:  # malformed JSON — triggers rule 2
                return "not json at all"
            else:            # missing field — triggers rule 2
                return json.dumps({
                    "direction": "CE", "conviction": 0.65,
                    "horizon": "intraday",
                    # signal_id deliberately omitted
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

    orch = Orchestrator(pod=MockPod())

    test_cases = [
        # (description, market_state)
        ("ADX < 20 — rule 1 should suppress",
         {"adx_14": 14.0, "vix_india": 22.0}),

        ("ADX ok, valid signal — should pass through",
         {"adx_14": 28.0, "vix_india": 14.0}),

        ("ADX ok, low conviction — rule 3 should downgrade",
         {"adx_14": 25.0, "vix_india": 13.0}),

        ("ADX ok, bad JSON — rule 2 should catch",
         {"adx_14": 30.0, "vix_india": 15.0}),

        ("ADX ok, missing field — rule 2 should catch",
         {"adx_14": 27.0, "vix_india": 14.5}),
    ]

    print("=" * 60)
    print("Orchestrator smoke test")
    print("=" * 60)

    for desc, ms in test_cases:
        result = orch.process(ms)
        print(f"\n{desc}")
        print(f"  reason : {result['orchestrator_reason']}")
        print(f"  direction : {result['direction']}")
        print(f"  conviction : {result['conviction']}")

    print("\n--- Aggregated stats ---")
    s = orch.stats()
    print(f"  total          : {s.total}")
    print(f"  adx_suppressed : {s.adx_suppressed}")
    print(f"  parse_failed   : {s.parse_failed}")
    print(f"  low_conviction : {s.low_conviction}")
    print(f"  passed_through : {s.passed_through}")
    print("\nSmoke test complete. All paths exercised.")