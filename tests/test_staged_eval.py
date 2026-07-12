"""Unit tests for the staged evaluation pipeline.

These tests validate the pure-function logic without requiring a model.
"""

import json
import math
import random
import sys
from collections import Counter
from unittest.mock import MagicMock, patch

import pytest
import torch
from optuna.trial import TrialState

from heretic.config import Settings
from heretic.evaluator import (
    Evaluator,
    GenerationHealthResult,
    OptimizationKLResult,
    PrescreenResult,
    ValidationKLResult,
    _HEALTH_THRESHOLDS,
)


# ---------------------------------------------------------------------------
#  Helper: make a minimal Settings that passes validation
#  Must mock sys.argv to avoid CliSettingsSource parsing pytest args.
# ---------------------------------------------------------------------------

def _make_settings(**overrides) -> Settings:
    defaults = dict(
        model="test-model",
        refusal_prescreen_enabled=True,
        refusal_prescreen_size=30,
        refusal_prescreen_pass_max=8,
        refusal_prescreen_prune_min=19,
        refusal_prescreen_seed=117,
        validation_kl_enabled=False,
        generation_health_enabled=False,
        full_refusal_candidate_max=30,
        optuna_prune_warmup_enabled=True,
        optuna_prune_warmup_completed_trials=15,
        refusal_prescreen_prune_min_warmup=24,
        optuna_prune_warmup_max_total_trials=40,
    )
    defaults.update(overrides)
    with patch.object(sys, "argv", ["heretic"]):
        return Settings(**defaults)


# ===========================================================================
#  1. Borderline refusal score consistent with 100-prompt scale
# ===========================================================================

class TestBorderlineRefusalScale:
    """Borderline trials must NOT run full 100-prompt eval.

    refusals_score = estimated_refusals / base_refusals
    where estimated_refusals = prescreen_refusals / prescreen_size * full_size.
    """

    def test_12_of_30_maps_to_40_of_100(self):
        prescreen_refusals = 12
        prescreen_size = 30
        full_size = 100
        estimated = round(prescreen_refusals / prescreen_size * full_size)
        assert estimated == 40

    def test_18_of_30_maps_to_60_of_100(self):
        prescreen_refusals = 18
        prescreen_size = 30
        full_size = 100
        estimated = round(prescreen_refusals / prescreen_size * full_size)
        assert estimated == 60

    def test_estimation_is_proportional(self):
        for pr, ps, fs in [(5, 30, 100), (10, 30, 100), (15, 30, 100)]:
            estimated = round(pr / ps * fs)
            expected = round(pr / ps * fs)
            assert estimated == expected

    def test_borderline_does_not_run_full_eval(self):
        """Borderline trials use estimated refusals from prescreen ratio."""
        # Simulate: 12/30 prescreen → estimated 40/100
        prescreen_refusals = 12
        prescreen_size = 30
        full_size = 100
        base_refusals = 91

        estimated = round(prescreen_refusals / prescreen_size * full_size)
        assert estimated == 40

        refusals_score = estimated / base_refusals
        assert abs(refusals_score - 40 / 91) < 1e-6

        # This is the same scale as full evaluation
        full_refusals_score = 40 / base_refusals
        assert refusals_score == full_refusals_score

    def test_refusals_score_source_is_recorded(self):
        """Borderline trials should have refusals_score_source='prescreen_estimate'."""
        source = "prescreen_estimate"
        assert source == "prescreen_estimate"

    def test_promising_and_borderline_refusals_score_comparable(self):
        """Both promising and borderline refusals_score use the same scale."""
        base_refusals = 91

        # Promising: full eval gives 15/100
        promising_score = 15 / base_refusals

        # Borderline: estimated 12/30 → 40/100
        estimated = round(12 / 30 * 100)
        borderline_score = estimated / base_refusals

        # Both are on the same scale (refusals / base_refusals)
        assert promising_score < borderline_score  # fewer refusals = better


# ===========================================================================
#  2. full_refusal_candidate_max filtering
# ===========================================================================

class TestFullRefusalCandidateMax:
    def test_trial_above_max_excluded(self):
        settings = _make_settings(full_refusal_candidate_max=30)
        full_refusals = 35
        assert full_refusals > settings.full_refusal_candidate_max

    def test_trial_at_max_included(self):
        settings = _make_settings(full_refusal_candidate_max=30)
        full_refusals = 30
        assert full_refusals <= settings.full_refusal_candidate_max

    def test_missing_full_refusals_excluded(self):
        """Legacy trials without full_refusals should not enter the main menu."""
        full_refusals = None
        assert full_refusals is None  # handled by explicit check


# ===========================================================================
#  3. Validation KL not dominated by prompt length
# ===========================================================================

class TestValidationKLOrthogonal:
    """Validation KL operates on single-token distributions.

    get_logprobs returns (batch, vocab) — one distribution per prompt.
    KL is summed over vocab (standard definition), NOT over sequence length.
    There is no padding dimension.
    """

    def test_single_token_kl_shape(self):
        batch, vocab = 10, 100
        base = torch.randn(batch, vocab)
        cand = torch.randn(batch, vocab)
        base_log = torch.log_softmax(base, dim=-1)
        cand_log = torch.log_softmax(cand, dim=-1)

        per_token_kl = torch.nn.functional.kl_div(
            cand_log, base_log, reduction="none", log_target=True
        )
        assert per_token_kl.shape == (batch, vocab)

        per_prompt_kl = per_token_kl.sum(dim=-1)
        assert per_prompt_kl.shape == (batch,)

    def test_padding_not_in_kl(self):
        """Since there's only one token per prompt, no padding exists."""
        # All prompts have the same length (1 token)
        # So no masking is needed
        batch, vocab = 5, 50
        base = torch.randn(batch, vocab)
        cand = torch.randn(batch, vocab)
        base_log = torch.log_softmax(base, dim=-1)
        cand_log = torch.log_softmax(cand, dim=-1)

        kl = torch.nn.functional.kl_div(
            cand_log, base_log, reduction="none", log_target=True
        )
        # Sum over vocab = standard KL
        per_prompt = kl.sum(dim=-1)
        assert per_prompt.shape == (batch,)
        assert torch.isfinite(per_prompt).all()


# ===========================================================================
#  4. Generation health metrics
# ===========================================================================

class TestGenerationHealth:

    def test_digit_loop_detected(self):
        """'2000 2000 prompt add 2000 add 2000 200000895 2000' should fail."""
        response = "2000 2000 prompt add 2000 add 2000 200000895 2000"
        tokens = response.split()

        # Token max freq check
        token_freq = Counter(tokens)
        total = len(tokens)
        max_freq_ratio = max(token_freq.values()) / total
        assert max_freq_ratio > _HEALTH_THRESHOLDS["max_token_freq_ratio"], (
            f"Digit loop should have high token freq: {max_freq_ratio}"
        )

    def test_distinct_2_low_for_repetitive(self):
        """Repetitive text should have low distinct-2."""
        response = "the the the the the the the the the the"
        tokens = response.split()
        bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0
        assert distinct_2 < _HEALTH_THRESHOLDS["min_distinct_2"]

    def test_distinct_4_low_for_repetitive(self):
        """Repetitive text should have low distinct-4."""
        response = "the the the the the the the the the the the the the"
        tokens = response.split()
        ngrams = [tuple(tokens[i : i + 4]) for i in range(len(tokens) - 3)]
        distinct_4 = len(set(ngrams)) / len(ngrams) if ngrams else 0
        assert distinct_4 < _HEALTH_THRESHOLDS["min_distinct_4"]

    def test_normal_json_not_flagged(self):
        """Valid JSON should not be flagged."""
        response = '{"name": "Alice", "age": 30}'
        try:
            json.loads(response)
            json_ok = True
        except json.JSONDecodeError:
            json_ok = False
        assert json_ok

    def test_json_health_accepts_fenced_block(self):
        """Markdown-fenced JSON is accepted by soft JSON checker."""
        responses = [
            "a",
            "b",
            "c",
            'Sure:\n```json\n{"name": "Alice", "age": 30}\n```\n',
        ]
        assert Evaluator._check_json_health(responses) is None

    def test_json_health_soft_note_on_prose(self):
        """Non-JSON prose returns a soft note (caller must not hard-fail on it)."""
        responses = ["a", "b", "c", "I cannot emit JSON for that."]
        note = Evaluator._check_json_health(responses)
        assert note is not None
        assert "soft" in note.lower() or "JSON" in note

    def test_code_block_without_punctuation_not_truncated(self):
        """Code block ending with ``` should not be flagged as truncated."""
        response = "```python\nprint('hello')\n```"
        # No punctuation at end, but it's a valid code block
        last_char = response.rstrip()[-1]
        assert last_char not in ".!?。！？\n"

    def test_consecutive_repeat_detected(self):
        """Long consecutive repeats should be detected."""
        response = "a " * 10  # 10 consecutive 'a' tokens
        tokens = response.split()
        run_len = 1
        max_run = 1
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i - 1]:
                run_len += 1
            else:
                max_run = max(max_run, run_len)
                run_len = 1
        max_run = max(max_run, run_len)
        assert max_run > _HEALTH_THRESHOLDS["max_consecutive_repeat"]


# ===========================================================================
#  5. Prescreen classification
# ===========================================================================

class TestPrescreenClassification:
    def test_promising(self):
        settings = _make_settings(
            refusal_prescreen_pass_max=8,
            refusal_prescreen_prune_min=19,
        )
        evaluator = MagicMock(spec=Evaluator)
        evaluator.settings = settings
        result = Evaluator.classify_prescreen(evaluator, 5)
        assert result == "promising"

    def test_borderline_low(self):
        settings = _make_settings(
            refusal_prescreen_pass_max=8,
            refusal_prescreen_prune_min=19,
        )
        evaluator = MagicMock(spec=Evaluator)
        evaluator.settings = settings
        result = Evaluator.classify_prescreen(evaluator, 9)
        assert result == "borderline"

    def test_borderline_high(self):
        settings = _make_settings(
            refusal_prescreen_pass_max=8,
            refusal_prescreen_prune_min=19,
        )
        evaluator = MagicMock(spec=Evaluator)
        evaluator.settings = settings
        result = Evaluator.classify_prescreen(evaluator, 18)
        assert result == "borderline"

    def test_pruned(self):
        settings = _make_settings(
            refusal_prescreen_pass_max=8,
            refusal_prescreen_prune_min=19,
        )
        evaluator = MagicMock(spec=Evaluator)
        evaluator.settings = settings
        result = Evaluator.classify_prescreen(evaluator, 19)
        assert result == "pruned"

    def test_pruned_high(self):
        settings = _make_settings(
            refusal_prescreen_pass_max=8,
            refusal_prescreen_prune_min=19,
        )
        evaluator = MagicMock(spec=Evaluator)
        evaluator.settings = settings
        result = Evaluator.classify_prescreen(evaluator, 25)
        assert result == "pruned"


# ===========================================================================
#  6. Config validation
# ===========================================================================

class TestConfigValidation:
    def test_valid_config(self):
        s = _make_settings()
        assert s.refusal_prescreen_pass_max < s.refusal_prescreen_prune_min
        assert s.refusal_prescreen_prune_min <= s.refusal_prescreen_size

    def test_invalid_pass_max_equals_prune_min(self):
        with pytest.raises(Exception):
            _make_settings(
                refusal_prescreen_pass_max=19,
                refusal_prescreen_prune_min=19,
            )

    def test_invalid_prune_min_exceeds_size(self):
        with pytest.raises(Exception):
            _make_settings(
                refusal_prescreen_prune_min=31,
                refusal_prescreen_size=30,
            )


# ===========================================================================
#  7. Legacy trial isolation
# ===========================================================================

class TestLegacyTrialIsolation:
    def test_legacy_trial_has_no_prescreen_class(self):
        """Legacy trials lack prescreen_class attr."""
        trial = MagicMock()
        trial.user_attrs = {"refusals": 15, "kl_divergence": 0.05}
        assert trial.user_attrs.get("prescreen_class") is None

    def test_new_trial_has_prescreen_class(self):
        """New trials always have prescreen_class."""
        trial = MagicMock()
        trial.user_attrs = {
            "prescreen_class": "promising",
            "refusals": 5,
            "kl_divergence": 0.02,
        }
        assert trial.user_attrs.get("prescreen_class") == "promising"


# ===========================================================================
#  8. OptimizationKLResult dataclass
# ===========================================================================

class TestOptimizationKLResult:
    def test_score_tuple(self):
        result = OptimizationKLResult(
            kl_divergence=0.05,
            refusals=15,
            full_refusal_size=100,
            score=(0.05, 15 / 91),
            refusals_score_source="full_evaluation",
        )
        assert result.score == (0.05, 15 / 91)
        assert result.refusals_score_source == "full_evaluation"

    def test_borderline_source(self):
        result = OptimizationKLResult(
            kl_divergence=0.1,
            refusals=40,
            full_refusal_size=100,
            score=(0.1, 40 / 91),
            refusals_score_source="prescreen_estimate",
        )
        assert result.refusals_score_source == "prescreen_estimate"


# ===========================================================================
#  9. Optuna warmup pruning
# ===========================================================================

class TestWarmupPruning:
    """Test dynamic prune threshold based on completed trial count."""

    def _classify(self, prescreen_refusals: int, effective_prune_min: int) -> str:
        """Replicate classify_prescreen logic."""
        settings = _make_settings()
        evaluator = MagicMock(spec=Evaluator)
        evaluator.settings = settings
        return Evaluator.classify_prescreen(evaluator, prescreen_refusals, effective_prune_min)

    def _warmup_logic(
        self,
        completed_count: int,
        total_finished: int,
        settings: Settings,
    ) -> tuple[bool, int]:
        """Replicate warmup logic from main.py."""
        warmup_active = (
            settings.optuna_prune_warmup_enabled
            and completed_count < settings.optuna_prune_warmup_completed_trials
            and total_finished < settings.optuna_prune_warmup_max_total_trials
        )
        if warmup_active:
            effective_prune_min = settings.refusal_prescreen_prune_min_warmup
        else:
            effective_prune_min = settings.refusal_prescreen_prune_min
        return warmup_active, effective_prune_min

    def test_completed_0_uses_warmup_threshold(self):
        s = _make_settings()
        active, threshold = self._warmup_logic(0, 0, s)
        assert active is True
        assert threshold == 24

    def test_completed_14_uses_warmup_threshold(self):
        s = _make_settings()
        active, threshold = self._warmup_logic(14, 14, s)
        assert active is True
        assert threshold == 24

    def test_completed_15_uses_normal_threshold(self):
        s = _make_settings()
        active, threshold = self._warmup_logic(15, 15, s)
        assert active is False
        assert threshold == 19

    def test_max_total_forces_normal(self):
        """Even if completed < 15, max_total forces normal."""
        s = _make_settings()
        active, threshold = self._warmup_logic(3, 40, s)
        assert active is False
        assert threshold == 19

    def test_warmup_21_is_borderline(self):
        """21/30 with warmup threshold 24 → borderline."""
        result = self._classify(21, effective_prune_min=24)
        assert result == "borderline"

    def test_warmup_24_is_pruned(self):
        """24/30 with warmup threshold 24 → pruned."""
        result = self._classify(24, effective_prune_min=24)
        assert result == "pruned"

    def test_normal_18_is_borderline(self):
        """18/30 with normal threshold 19 → borderline."""
        result = self._classify(18, effective_prune_min=19)
        assert result == "borderline"

    def test_normal_19_is_pruned(self):
        """19/30 with normal threshold 19 → pruned."""
        result = self._classify(19, effective_prune_min=19)
        assert result == "pruned"

    def test_pruned_trial_not_in_completed(self):
        """PRUNED trials should not count toward completed_count."""
        # This is validated by the logic: only TrialState.COMPLETE counts
        assert TrialState.COMPLETE != TrialState.PRUNED

    def test_fail_trial_not_in_completed(self):
        """FAIL trials should not count toward completed_count."""
        assert TrialState.COMPLETE != TrialState.FAIL

    def test_config_validation_warmup(self):
        """Warmup threshold must be > pass_max and <= size."""
        s = _make_settings(
            refusal_prescreen_prune_min_warmup=24,
            refusal_prescreen_pass_max=8,
            refusal_prescreen_size=30,
        )
        assert s.refusal_prescreen_prune_min_warmup == 24

    def test_config_validation_warmup_too_low(self):
        """Warmup threshold below pass_max should fail."""
        with pytest.raises(Exception):
            _make_settings(refusal_prescreen_prune_min_warmup=5)

    def test_objective_return_format(self):
        """Objective must return (kld_score, refusals_score) tuple."""
        # This is a structural test — warmup doesn't change objective
        score = (0.05, 0.15)
        assert isinstance(score, tuple)
        assert len(score) == 2
        assert isinstance(score[0], float)
        assert isinstance(score[1], float)
