"""A helper that runs the concurrent loops of a pipeline."""

import asyncio
import logging
from collections.abc import Coroutine

logger = logging.getLogger(__name__)


async def run_until_first_exit(name: str, loops: dict[str, Coroutine]) -> None:
    """Run ``loops`` concurrently until one ends or fails, then cancel the others.

    The function logs a failure and does not raise it, so the shutdown code of the caller still runs.
    """
    tasks = [asyncio.create_task(coroutine, name=label) for label, coroutine in loops.items()]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if not task.cancelled() and task.exception():
                logger.error(f"{name}: {task.get_name()} failed", exc_info=task.exception())
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
