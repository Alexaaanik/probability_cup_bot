"""Probabilities: fair odds, LLM adjustment, shrinkage."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from openai import OpenAI

from config import (
    API_PROB_MAX,
    API_PROB_MIN,
    LLM_MAX_ADJUSTMENT,
    MARKET_STATUS_OPEN,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
    NO_ODDS_BASE_PROBABILITY,
    PROBABILITY_CENTER,
    PROP_BASE_PROBABILITY,
    PROP_LLM_MAX_ADJUSTMENT,
    PROP_SHRINKAGE_FACTOR,
    SHRINKAGE_FACTOR,
)
from llm_client import LlmCallBudget, get_adjustment
from matcher import normalize_team_name
from odds_client import OddsMatch

logger = logging.getLogger(__name__)

_WIN_MARKET_PATTERN = re.compile(
    r"^Will (.+) win the match\?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FairProbabilities:
    """Normalized h2h outcome probabilities (sum = 1.0)."""

    home_win: float
    draw: float
    away_win: float
    home_team: str
    away_team: str


@dataclass(frozen=True)
class LlmAdjustment:
    adjustment: float
    reason: str
    llm_called: bool
    llm_raw_response: str


@dataclass(frozen=True)
class ModelDecision:
    """Full trace of one decision — written to the JSONL log."""

    market_id: str
    match_name: str
    question: str
    market_kind: str
    base_probability: float | None
    adjustment: float
    adjustment_reason: str
    adjusted_probability: float | None
    shrunk_probability: float | None
    api_probability: int | None
    skipped: bool
    pending: bool
    skip_reason: str
    llm_called: bool
    llm_reason: str
    llm_raw_response: str


def step1_fair_probabilities_from_odds(odds_match: OddsMatch) -> FairProbabilities:
    """fair_prob = (1/odds) / sum(1/odds)"""
    raw: dict[str, float] = {}
    for outcome in odds_match.outcomes:
        if outcome.average_odds <= 0:
            continue
        raw[outcome.name] = 1.0 / outcome.average_odds

    if not raw:
        raise ValueError(f"No valid odds for {odds_match.home_team}")

    total = sum(raw.values())
    fair = {name: value / total for name, value in raw.items()}

    home_key = odds_match.home_team
    away_key = odds_match.away_team
    draw_prob = fair.get("Draw", fair.get("draw", 0.0))

    home_prob = fair.get(home_key, 0.0)
    away_prob = fair.get(away_key, 0.0)

    if home_prob == 0.0 or away_prob == 0.0:
        norm_fair = {normalize_team_name(k): v for k, v in fair.items()}
        home_prob = norm_fair.get(normalize_team_name(home_key), home_prob)
        away_prob = norm_fair.get(normalize_team_name(away_key), away_prob)

    return FairProbabilities(
        home_win=home_prob,
        draw=draw_prob,
        away_win=away_prob,
        home_team=odds_match.home_team,
        away_team=odds_match.away_team,
    )


def parse_win_market_team(question: str) -> str | None:
    """Extract the team from 'Will X win the match?'."""
    match = _WIN_MARKET_PATTERN.match(question.strip())
    if not match:
        return None
    return match.group(1).strip()


def is_win_market(question: str) -> bool:
    return parse_win_market_team(question) is not None


def base_probability_for_win_market(
    question: str,
    fair_probs: FairProbabilities,
) -> float | None:
    """Base probability for a match-winner market. None for other market types."""
    team_in_question = parse_win_market_team(question)
    if team_in_question is None:
        return None

    norm_q = normalize_team_name(team_in_question)
    norm_home = normalize_team_name(fair_probs.home_team)
    norm_away = normalize_team_name(fair_probs.away_team)

    if norm_q == norm_home:
        return fair_probs.home_win
    if norm_q == norm_away:
        return fair_probs.away_win

    logger.warning(
        "Team '%s' from question did not match %s / %s",
        team_in_question,
        fair_probs.home_team,
        fair_probs.away_team,
    )
    return None


def step3_llm_adjustment(
    *,
    match_name: str,
    market_question: str,
    base_probability: float,
    skip_llm: bool,
    llm_budget: LlmCallBudget,
    openrouter_client: OpenAI | None,
    max_adjustment: float = LLM_MAX_ADJUSTMENT,
) -> LlmAdjustment:
    """LLM adjustment. Without key, --skip-llm, or on limit — adjustment = 0."""
    if skip_llm:
        return LlmAdjustment(
            adjustment=0.0,
            reason="--skip-llm",
            llm_called=False,
            llm_raw_response="",
        )

    if openrouter_client is None:
        return LlmAdjustment(
            adjustment=0.0,
            reason="OPENROUTER_API_KEY not set",
            llm_called=False,
            llm_raw_response="",
        )

    if not llm_budget.can_call():
        if not llm_budget.limit_logged:
            logger.warning(
                "LLM call limit reached (%d/%d) — using base_probability only",
                llm_budget.used,
                llm_budget.max_calls,
            )
            llm_budget.limit_logged = True
        return LlmAdjustment(
            adjustment=0.0,
            reason="llm_limit_reached",
            llm_called=False,
            llm_raw_response="",
        )

    result = get_adjustment(
        match_name,
        base_probability,
        market_question,
        client=openrouter_client,
    )
    llm_budget.record_call()

    raw_adj = float(result["adjustment"])
    clamped = max(-max_adjustment, min(max_adjustment, raw_adj))
    reason = str(result["reason"])
    if clamped != raw_adj:
        reason = f"{reason} [clamped to ±{max_adjustment}]"

    return LlmAdjustment(
        adjustment=clamped,
        reason=reason,
        llm_called=True,
        llm_raw_response=str(result.get("raw_response", "")),
    )


def apply_shrinkage(adjusted_probability: float, *, factor: float) -> float:
    """Shrink toward 0.5 before submission."""
    return PROBABILITY_CENTER + (
        (adjusted_probability - PROBABILITY_CENTER) * factor
    )


def step4_apply_shrinkage(adjusted_probability: float) -> float:
    return apply_shrinkage(adjusted_probability, factor=SHRINKAGE_FACTOR)


def clamp_probability(probability: float) -> float:
    return max(MIN_PROBABILITY, min(MAX_PROBABILITY, probability))


def to_api_probability(probability: float) -> int:
    """Round to integer 1–99 for the SportsPredict API."""
    clamped = clamp_probability(probability)
    api_value = round(clamped * 100)
    return max(API_PROB_MIN, min(API_PROB_MAX, api_value))


def _skipped_decision(
    *,
    market_id: str,
    match_name: str,
    question: str,
    market_kind: str,
    skip_reason: str,
) -> ModelDecision:
    return ModelDecision(
        market_id=market_id,
        match_name=match_name,
        question=question,
        market_kind=market_kind,
        base_probability=None,
        adjustment=0.0,
        adjustment_reason="",
        adjusted_probability=None,
        shrunk_probability=None,
        api_probability=None,
        skipped=True,
        pending=False,
        skip_reason=skip_reason,
        llm_called=False,
        llm_reason="",
        llm_raw_response="",
    )


def _finalize_decision(
    *,
    market_id: str,
    match_name: str,
    question: str,
    market_kind: str,
    base: float,
    llm: LlmAdjustment,
    shrinkage_factor: float,
    pending: bool = False,
) -> ModelDecision:
    adjusted = clamp_probability(base + llm.adjustment)
    shrunk = apply_shrinkage(adjusted, factor=shrinkage_factor)
    api_prob = to_api_probability(shrunk)

    return ModelDecision(
        market_id=market_id,
        match_name=match_name,
        question=question,
        market_kind=market_kind,
        base_probability=base,
        adjustment=llm.adjustment,
        adjustment_reason=llm.reason,
        adjusted_probability=adjusted,
        shrunk_probability=shrunk,
        api_probability=api_prob,
        skipped=False,
        pending=pending,
        skip_reason="pending_no_odds" if pending else "",
        llm_called=llm.llm_called,
        llm_reason=llm.reason,
        llm_raw_response=llm.llm_raw_response,
    )


def build_decision_for_market(
    *,
    market_id: str,
    match_name: str,
    question: str,
    market_status: str,
    fair_probs: FairProbabilities | None,
    skip_llm: bool,
    llm_budget: LlmCallBudget,
    openrouter_client: OpenAI | None,
) -> ModelDecision:
    """Build a full decision for one market."""
    market_kind = "win" if is_win_market(question) else "prop"

    if market_status != MARKET_STATUS_OPEN:
        return _skipped_decision(
            market_id=market_id,
            match_name=match_name,
            question=question,
            market_kind=market_kind,
            skip_reason=f"market not open (status={market_status})",
        )

    if market_kind == "win":
        return _build_win_decision(
            market_id=market_id,
            match_name=match_name,
            question=question,
            fair_probs=fair_probs,
            skip_llm=skip_llm,
            llm_budget=llm_budget,
            openrouter_client=openrouter_client,
        )

    return _build_prop_decision(
        market_id=market_id,
        match_name=match_name,
        question=question,
        fair_probs=fair_probs,
        skip_llm=skip_llm,
        llm_budget=llm_budget,
        openrouter_client=openrouter_client,
    )


def _build_win_decision(
    *,
    market_id: str,
    match_name: str,
    question: str,
    fair_probs: FairProbabilities | None,
    skip_llm: bool,
    llm_budget: LlmCallBudget,
    openrouter_client: OpenAI | None,
) -> ModelDecision:
    if fair_probs is None:
        llm = step3_llm_adjustment(
            match_name=match_name,
            market_question=question,
            base_probability=NO_ODDS_BASE_PROBABILITY,
            skip_llm=skip_llm,
            llm_budget=llm_budget,
            openrouter_client=openrouter_client,
            max_adjustment=PROP_LLM_MAX_ADJUSTMENT,
        )
        return _finalize_decision(
            market_id=market_id,
            match_name=match_name,
            question=question,
            market_kind="win",
            base=NO_ODDS_BASE_PROBABILITY,
            llm=llm,
            shrinkage_factor=PROP_SHRINKAGE_FACTOR,
            pending=True,
        )

    base = base_probability_for_win_market(question, fair_probs)
    if base is None:
        return _skipped_decision(
            market_id=market_id,
            match_name=match_name,
            question=question,
            market_kind="win",
            skip_reason="team from question did not match odds line",
        )

    llm = step3_llm_adjustment(
        match_name=match_name,
        market_question=question,
        base_probability=base,
        skip_llm=skip_llm,
        llm_budget=llm_budget,
        openrouter_client=openrouter_client,
    )
    return _finalize_decision(
        market_id=market_id,
        match_name=match_name,
        question=question,
        market_kind="win",
        base=base,
        llm=llm,
        shrinkage_factor=SHRINKAGE_FACTOR,
    )


def _build_prop_decision(
    *,
    market_id: str,
    match_name: str,
    question: str,
    fair_probs: FairProbabilities | None,
    skip_llm: bool,
    llm_budget: LlmCallBudget,
    openrouter_client: OpenAI | None,
) -> ModelDecision:
    """Prop markets: neutral base, smaller LLM band, stronger shrinkage."""
    base = PROP_BASE_PROBABILITY
    llm = step3_llm_adjustment(
        match_name=match_name,
        market_question=question,
        base_probability=base,
        skip_llm=skip_llm,
        llm_budget=llm_budget,
        openrouter_client=openrouter_client,
        max_adjustment=PROP_LLM_MAX_ADJUSTMENT,
    )
    return _finalize_decision(
        market_id=market_id,
        match_name=match_name,
        question=question,
        market_kind="prop",
        base=base,
        llm=llm,
        shrinkage_factor=PROP_SHRINKAGE_FACTOR,
    )
