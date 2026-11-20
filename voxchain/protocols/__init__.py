"""The wire vocabulary of each vendor, which its providers share: endpoints, message types and handshakes."""


def compact(values: dict) -> dict:
    """``values`` without its ``None`` entries, so that the vendor's defaults apply to options with no value."""
    return {key: value for key, value in values.items() if value is not None}
