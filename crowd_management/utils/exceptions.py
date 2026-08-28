"""Project-specific exception types."""


class CrowdManagementError(RuntimeError):
    """Base exception for the crowd management project."""


class ConfigurationError(CrowdManagementError):
    """Raised when configuration is invalid or incomplete."""


class DatasetValidationError(CrowdManagementError):
    """Raised when a dataset or source cannot be validated."""


class VideoSourceError(CrowdManagementError):
    """Raised when a video source cannot be opened or read."""


class FrameBufferError(CrowdManagementError):
    """Raised when a frame buffer fails to perform a required operation."""
