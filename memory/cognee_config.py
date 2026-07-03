"""
Explicit Cognee provider configuration. Import and call configure() ONCE, before
any cognee.remember()/recall() call, anywhere in the process — including in
scripts/run_session.py and server/main.py.

IMPORTANT: Cognee defaults to an OpenAI provider if left unconfigured. Skipping
this file means an accidental OpenAI bill even though nothing in this codebase
calls OpenAI directly — remember() uses an LLM internally to extract entities
and relations for the graph.

Verify exact parameter names/casing against docs.cognee.ai/setup-configuration/overview
before relying on this in a live demo — Cognee's config surface can shift between
versions, and this file uses the env-var pattern documented there, which is the
most stable interface across versions. If cognee.config.set(...) calls are needed
instead of env vars in your installed version, add them here, not scattered
elsewhere in the codebase, so there's exactly one place to fix.
"""
import os
from dotenv import load_dotenv

_configured = False


def configure():
    global _configured
    if _configured:
        return
    load_dotenv()

    required = ["LLM_PROVIDER", "LLM_MODEL", "EMBEDDING_PROVIDER"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        raise RuntimeError(
            f"Missing required env vars for Cognee: {missing}. "
            f"Copy .env.example to .env and fill it in BEFORE running anything — "
            f"otherwise Cognee may silently fall back to a paid OpenAI default."
        )

    if os.environ["LLM_PROVIDER"] == "groq" and not os.environ.get("LLM_API_KEY"):
        raise RuntimeError("LLM_PROVIDER=groq but LLM_API_KEY is not set in .env")

    # These env vars are read by Cognee itself at call time — no direct cognee
    # import needed here, which keeps this module import-safe even before
    # `pip install cognee` has fully resolved in a fresh environment.
    _configured = True
    print(
        f"[cognee_config] configured: LLM_PROVIDER={os.environ['LLM_PROVIDER']} "
        f"EMBEDDING_PROVIDER={os.environ['EMBEDDING_PROVIDER']} "
        f"(vector/graph store: local default unless overridden)"
    )
