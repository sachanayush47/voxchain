"""run_until_first_exit: the first loop to end stops the others, and a failure is logged, not raised."""

import asyncio
import logging

from voxkit.tasks import run_until_first_exit


async def test_the_first_loop_to_finish_cancels_the_rest():
    cancelled = []

    async def forever():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    async def quick():
        return None

    await asyncio.wait_for(run_until_first_exit("test", {"forever": forever(), "quick": quick()}), 1)
    assert cancelled == [True]


async def test_a_failing_loop_is_logged_and_not_raised(caplog):
    async def broken():
        raise RuntimeError("Boom")

    async def forever():
        await asyncio.Event().wait()

    with caplog.at_level(logging.ERROR, logger="voxkit.tasks"):
        await asyncio.wait_for(run_until_first_exit("Pipeline", {"broken": broken(), "forever": forever()}), 1)
    assert "Pipeline: broken failed" in caplog.text
