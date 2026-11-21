"""EndOfTurnGate: it holds transcripts that sound incomplete, and a detector error counts as a complete turn."""

import asyncio

from voxchain.core.turn_gate import EndOfTurnGate


class BrokenDetector:
    def push_audio(self, chunk):
        pass

    def start_turn(self):
        pass

    async def is_end_of_turn(self, transcript):
        raise RuntimeError("Model missing")


async def test_a_detector_error_commits_the_turn_and_goes_to_on_error():
    committed, errors = [], []
    gate = EndOfTurnGate(BrokenDetector(), timeout=5.0, commit=committed.append, on_error=errors.append)
    gate.on_transcript("book a table")
    for _ in range(5):
        await asyncio.sleep(0)
    assert committed == ["book a table"]
    assert [str(error) for error in errors] == ["Model missing"]
