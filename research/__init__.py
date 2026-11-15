"""Research visit-detail pseudonymisation service."""

__all__ = ["create_app"]


def create_app(settings=None):
    from .web import create_app as _create_app

    return _create_app(settings)
