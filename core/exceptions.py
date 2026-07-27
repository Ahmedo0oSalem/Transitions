"""Custom exceptions for TRANSITIONS."""


class TransitionsError(Exception):
    """Base exception for package-specific failures."""


class DataLoadError(TransitionsError):
    """Raised when required match data cannot be loaded."""


class ConfigurationError(TransitionsError):
    """Raised when an invalid configuration is supplied."""
