class ApplicationError(Exception):
    """Base exception for application-level errors."""


class DatasetError(ApplicationError):
    """Raised when a dataset cannot be processed."""


class DatasetValidationError(DatasetError):
    """Raised when a dataset fails validation."""


class PreprocessingError(ApplicationError):
    """Raised when preprocessing fails."""


class AgentError(ApplicationError):
    """Raised when agent execution fails."""