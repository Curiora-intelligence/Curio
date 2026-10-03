class RequestConflictError(ValueError):
    """A durable request exists but cannot safely be replayed."""
