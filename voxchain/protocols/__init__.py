"""Vendor wire vocabularies shared by that vendor's providers: endpoints, message types, handshakes."""


def compact(values: dict) -> dict:
    """``values`` without its ``None`` entries, so unset options keep the vendor's defaults."""
    return {key: value for key, value in values.items() if value is not None}
