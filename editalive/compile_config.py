"""Process-wide torch.compile configuration set before model imports."""

_ENABLED = False


def configure_compile(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = bool(enabled)


def is_compile_enabled() -> bool:
    return _ENABLED
