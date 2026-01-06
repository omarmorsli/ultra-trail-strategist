"""
Async utilities for wrapping synchronous operations.

Provides helpers to run blocking I/O operations in a thread pool
without blocking the async event loop.
"""

import asyncio
from functools import partial, wraps
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def run_in_thread(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to run a synchronous function in a thread pool.

    Use this to wrap blocking I/O operations (API calls, file I/O)
    so they don't block the async event loop.

    Parameters
    ----------
    func : Callable
        Synchronous function to wrap.

    Returns
    -------
    Callable
        Async function that runs the original in a thread.

    Example
    -------
    >>> @run_in_thread
    ... def blocking_api_call(url):
    ...     return requests.get(url).json()
    ...
    >>> result = await blocking_api_call("https://api.example.com")
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        # Use partial to bind args/kwargs
        bound_func = partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, bound_func)

    return wrapper  # type: ignore[return-value]


async def to_thread(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Run a synchronous function in a thread pool.

    Similar to asyncio.to_thread() (Python 3.9+) but with kwargs support.

    Parameters
    ----------
    func : Callable
        Synchronous function to run.
    *args
        Positional arguments for the function.
    **kwargs
        Keyword arguments for the function.

    Returns
    -------
    T
        Result of the function.

    Example
    -------
    >>> result = await to_thread(requests.get, "https://api.example.com", timeout=5)
    """
    loop = asyncio.get_running_loop()
    bound_func = partial(func, *args, **kwargs)
    return await loop.run_in_executor(None, bound_func)
