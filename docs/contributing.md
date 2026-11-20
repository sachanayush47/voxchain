# Contributing

## Development

```bash
uv sync                     # Install the dependencies and dev tools
uv run pytest               # Run the tests (no network or credentials necessary)
uv run ruff check --fix .   # Lint
uv run ruff format .        # Format
uv run pre-commit install   # Do this once for each clone: lint and format at each commit
```

The `examples/` folder has scripts that use your microphone and speakers.

To build these docs locally, run `uv sync --extra docs && uv run mkdocs serve`. The command `uv run mkdocs build --strict` must pass with no warnings.

The contributor conventions (coding standards, design principles, tests and language) are in [`AGENTS.md`](https://github.com/sachanayush47/voxchain/blob/master/AGENTS.md). All prose follows ASD-STE100 Simplified Technical English.

## How to add a provider

Implement the interface in `voxchain/stt/base.py`, `voxchain/tts/base.py` or `voxchain/realtime/base.py`. Compose the shared pieces. Do not inherit behaviour from them:

- [`ProviderSession`][voxchain.session.ProviderSession] gives you the socket, the reconnects and the event stream. You supply `open` (connect and configure) and `handle` (change one message into events).
- [`VoiceGate`][voxchain.vad.gate.VoiceGate], if the vendor does not report voice activity.
- [`ResponseGate`][voxchain.realtime.response_gate.ResponseGate] and [`ConversationHistory`][voxchain.realtime.history.ConversationHistory], for speech-to-speech.

Put the wire vocabulary of the vendor (URLs, message types as `StrEnum`s, handshake) in `voxchain/protocols/<vendor>.py`. The current providers are compact references. The pipelines depend only on the interfaces.

Each new provider must have tests that use no network, and a `::: voxchain.<module>` block on the correct page of the API reference.

## Release

1. Increase `version` in `pyproject.toml` and `__version__` in `voxchain/__init__.py`.
2. Push the change to `master`.
3. Publish a GitHub release with the tag `v<version>`, for example `gh release create v0.1.1 --target master`.

The release workflow makes sure that the tag agrees with the version. Then it lints, tests and builds the package, and uploads it to PyPI through trusted publishing. No API token is necessary.
