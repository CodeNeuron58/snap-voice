"""Boot warm-up — pay the first-turn cost before the first user arrives.

Deepgram keeps models hot; we do the same on-device: a silent Whisper pass and a
one-token LLM completion at startup, so the first real interaction skips mel
filterbank setup, chat-template parsing, and compute-graph allocation.
"""

import asyncio
import logging
import time

from snap import config

log = logging.getLogger(__name__)


async def warm_up(stt=None, llm=None) -> None:
    """Warm whatever services were passed in (duck-typed `warm_up()` methods)."""
    if not config.WARM_UP_ON_BOOT:
        return
    t0 = time.perf_counter()
    for service in (stt, llm):
        if service is not None and hasattr(service, "warm_up"):
            try:
                result = service.warm_up()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:  # noqa: BLE001 — warm-up must never block boot
                log.warning("warm-up skipped (%s)", exc)
    log.info("warm-up: models hot in %.2fs", time.perf_counter() - t0)
