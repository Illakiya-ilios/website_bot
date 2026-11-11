"""
LLM Guard integration for the Ilios Sales Agent API.

Input scanners run on the user's message before it reaches the RAG chain.
Output scanners run on the model's reply before it is returned to the user.

Environment variables (all optional):
  LLM_GUARD_ENABLED            "true" (default) / "false" to disable all scanning
  LLM_GUARD_FAIL_OPEN          "false" (default) blocks the request if a scanner
                               crashes; "true" lets the request through
  PROMPT_INJECTION_THRESHOLD   default 0.92
  TOXICITY_THRESHOLD           default 0.5 (input) / 0.7 (output)
"""
import logging
import os
from dataclasses import dataclass

import structlog
from llm_guard import scan_output, scan_prompt
from llm_guard.input_scanners import PromptInjection, Secrets
from llm_guard.input_scanners import Toxicity as InputToxicity
from llm_guard.output_scanners import Toxicity as OutputToxicity

logger = logging.getLogger("ilios-api.guard")

# llm-guard logs every scanner score at debug level via structlog; keep only
# warnings (detections) and errors in our logs.
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


GUARD_ENABLED = _env_bool("LLM_GUARD_ENABLED", True)
FAIL_OPEN = _env_bool("LLM_GUARD_FAIL_OPEN", False)


@dataclass
class GuardResult:
    text: str          # sanitized text (e.g. secrets redacted)
    allowed: bool
    reason: str = ""   # names of the scanners that flagged the text


_input_scanners = []
_output_scanners = []

if GUARD_ENABLED:
    logger.info("Loading LLM Guard scanners (first run downloads models)...")
    # Secrets only redacts; it is run separately so the redaction placeholder
    # doesn't confuse the injection/toxicity classifiers.
    _secrets_scanner = Secrets(redact_mode="all")
    _input_scanners = [
        PromptInjection(
            threshold=float(os.getenv("PROMPT_INJECTION_THRESHOLD", "0.92"))
        ),
        InputToxicity(threshold=float(os.getenv("TOXICITY_THRESHOLD", "0.5"))),
    ]
    # Output PII scanning is intentionally omitted: the bot is expected to share
    # Ilios Digital's public contact details (emails, phone numbers).
    _output_scanners = [OutputToxicity(threshold=0.7)]
    logger.info("LLM Guard ready.")
else:
    logger.warning("LLM Guard is disabled (LLM_GUARD_ENABLED=false).")


def _failed(results_valid: dict) -> str:
    return ", ".join(name for name, ok in results_valid.items() if not ok)


def check_input(message: str) -> GuardResult:
    if not _input_scanners:
        return GuardResult(message, True)
    try:
        _, valid, scores = scan_prompt(_input_scanners, message)
        sanitized, secrets_ok, _ = _secrets_scanner.scan(message)
    except Exception:
        logger.error("LLM Guard input scan failed.", exc_info=True)
        return GuardResult(message, FAIL_OPEN, "scanner error")

    if not secrets_ok:
        logger.warning("Secrets redacted from user input.")

    failed = _failed(valid)
    if failed:
        logger.warning("Input blocked by: %s (scores=%s)", failed, scores)
        return GuardResult(sanitized, False, failed)
    return GuardResult(sanitized, True)


def check_output(message: str, answer: str) -> GuardResult:
    if not _output_scanners:
        return GuardResult(answer, True)
    try:
        sanitized, valid, scores = scan_output(_output_scanners, message, answer)
    except Exception:
        logger.error("LLM Guard output scan failed.", exc_info=True)
        return GuardResult(answer, FAIL_OPEN, "scanner error")

    failed = _failed(valid)
    if failed:
        logger.warning("Output blocked by: %s (scores=%s)", failed, scores)
        return GuardResult(sanitized, False, failed)
    return GuardResult(sanitized, True)
