from collections.abc import Iterator


def exception_leaves(error: BaseException) -> Iterator[BaseException]:
    """Inspect underlying validation failures without depending on traceback text."""
    if isinstance(error, BaseExceptionGroup):
        for nested in error.exceptions:
            yield from exception_leaves(nested)
    else:
        yield error
