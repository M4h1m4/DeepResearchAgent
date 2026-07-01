"""
PII Detection and Hallucination Prevention guardrails.

PII: Uses Microsoft Presidio if installed (pip install presidio-analyzer presidio-anonymizer
     plus: python -m spacy download en_core_web_lg). Falls back to regex patterns for
     common entities (EMAIL, PHONE, SSN, CREDIT_CARD, IP_ADDRESS) when Presidio is absent.

Hallucination: LLM-as-judge — sends the answer + retrieved context to the LLM and asks
     it to score faithfulness (0–1) and list unsupported claims. Answers below
     FAITHFULNESS_THRESHOLD get a warning injected and flagged in response metadata.
"""

import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from config import settings
from config.logging_config import get_logger

logger = get_logger(__name__)

FAITHFULNESS_THRESHOLD = 0.7

# Regex fallback — covers the most common PII types without external deps
_REGEX_PATTERNS: Dict[str, re.Pattern] = {
    "EMAIL":       re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE":       re.compile(r"\b(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}\b"),
    "SSN":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"),
    "IP_ADDRESS":  re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PIIResult:
    has_pii: bool
    entities: List[Dict]          # [{"type": "EMAIL", "start": 5, "end": 16}]
    redacted_text: str


@dataclass
class HallucinationResult:
    is_grounded: bool
    faithfulness_score: float     # 0.0 – 1.0
    unsupported_claims: List[str] = field(default_factory=list)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# PII Detector
# ---------------------------------------------------------------------------

class PIIDetector:
    """
    Detects and redacts PII in text.
    Uses Presidio when available, regex patterns otherwise.
    Call detect_and_redact() on any text before returning it to the user.
    """

    def __init__(self):
        self._analyzer = None
        self._anonymizer = None
        self._load_presidio()

    def _load_presidio(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            logger.info("Presidio PII engine loaded")
        except ImportError:
            logger.info(
                "Presidio not installed — using regex PII detection. "
                "Install with: pip install presidio-analyzer presidio-anonymizer "
                "&& python -m spacy download en_core_web_lg"
            )

    def detect_and_redact(self, text: str) -> PIIResult:
        if not text or not text.strip():
            return PIIResult(has_pii=False, entities=[], redacted_text=text)
        if self._analyzer:
            return self._presidio_scan(text)
        return self._regex_scan(text)

    def _presidio_scan(self, text: str) -> PIIResult:
        from presidio_anonymizer.entities import OperatorConfig
        try:
            results = self._analyzer.analyze(text=text, language="en")
            entities = [
                {"type": r.entity_type, "start": r.start, "end": r.end, "score": round(r.score, 3)}
                for r in results
            ]
            if results:
                anon = self._anonymizer.anonymize(
                    text=text,
                    analyzer_results=results,
                    operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"})},
                )
                redacted = anon.text
            else:
                redacted = text
            if entities:
                logger.warning("PII detected (Presidio)", extra={"types": [e["type"] for e in entities]})
            return PIIResult(has_pii=bool(entities), entities=entities, redacted_text=redacted)
        except Exception as e:
            logger.warning("Presidio scan failed, falling back to regex", extra={"error": str(e)})
            return self._regex_scan(text)

    def _regex_scan(self, text: str) -> PIIResult:
        all_matches = []
        for pii_type, pattern in _REGEX_PATTERNS.items():
            for m in pattern.finditer(text):
                all_matches.append((m.start(), m.end(), pii_type))

        if not all_matches:
            return PIIResult(has_pii=False, entities=[], redacted_text=text)

        entities = [{"type": t, "start": s, "end": e} for s, e, t in all_matches]
        logger.warning("PII detected (regex)", extra={"types": [e["type"] for e in entities]})

        # Replace from right to left to keep offsets valid
        chars = list(text)
        for start, end, _ in sorted(all_matches, key=lambda x: x[0], reverse=True):
            chars[start:end] = list("<REDACTED>")

        return PIIResult(has_pii=True, entities=entities, redacted_text="".join(chars))


# ---------------------------------------------------------------------------
# Hallucination Guard
# ---------------------------------------------------------------------------

class HallucinationGuard:
    """
    LLM-as-judge faithfulness checker.

    Sends (question, answer, context) to the LLM and asks it to:
      1. Score faithfulness 0–1
      2. List every claim in the answer not supported by context

    If faithfulness_score < FAITHFULNESS_THRESHOLD the answer is considered
    ungrounded. Callers should add a caveat or surface unsupported_claims.
    """

    _PROMPT = """\
You are a faithfulness judge for a RAG system. Your only job is to check \
whether the ANSWER is supported by the PROVIDED CONTEXT.

QUESTION: {question}

PROVIDED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Instructions:
1. Break the answer into individual factual claims.
2. For each claim decide: is it directly supported by the context above?
3. List every claim that is NOT supported (these are potential hallucinations).
4. Give a faithfulness_score from 0.0 (fully hallucinated) to 1.0 (fully grounded).

Respond with JSON only — no prose outside the JSON block:
{{
  "faithfulness_score": <float>,
  "unsupported_claims": ["<claim>", ...],
  "reasoning": "<one sentence summary>"
}}"""

    def check(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> HallucinationResult:
        if not answer or not answer.strip():
            return HallucinationResult(is_grounded=True, faithfulness_score=1.0, reasoning="Empty answer")

        if not context or not context.strip():
            return HallucinationResult(
                is_grounded=False,
                faithfulness_score=0.0,
                unsupported_claims=["Answer generated without any retrieved context"],
                reasoning="No context was retrieved to ground this answer",
            )

        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY not set — hallucination check skipped")
            return HallucinationResult(is_grounded=True, faithfulness_score=1.0, reasoning="Skipped — no API key")

        try:
            from langchain_openai import ChatOpenAI
            from langchain.schema import HumanMessage

            llm = ChatOpenAI(model=settings.openai_model, temperature=0.0, openai_api_key=settings.openai_api_key, request_timeout=15)
            prompt = self._PROMPT.format(
                question=question,
                context=context[:12000],
                answer=answer[:3000],
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()

            # Strip markdown fences if present
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            data = json.loads(raw)
            score = float(data.get("faithfulness_score", 1.0))
            unsupported = data.get("unsupported_claims", [])
            reasoning = data.get("reasoning", "")

            logger.info(
                "Hallucination check complete",
                extra={"score": score, "unsupported_count": len(unsupported), "is_grounded": score >= FAITHFULNESS_THRESHOLD},
            )
            return HallucinationResult(
                is_grounded=score >= FAITHFULNESS_THRESHOLD,
                faithfulness_score=score,
                unsupported_claims=unsupported,
                reasoning=reasoning,
            )

        except Exception as e:
            logger.warning("Hallucination check failed", extra={"error": str(e)})
            # Fail open — don't block the response if the judge itself errors
            return HallucinationResult(
                is_grounded=True,
                faithfulness_score=1.0,
                reasoning=f"Check skipped: {str(e)}",
            )
