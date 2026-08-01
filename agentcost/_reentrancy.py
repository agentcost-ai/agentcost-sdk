"""
AgentCost re-entrancy guard.

Shared by every provider interceptor so that a higher-level integration
(LangChain) calling a provider SDK under the hood is recorded once, not twice.

A ContextVar, not threading.local(): under asyncio every concurrent request
runs on the same thread, so a thread-local depth raised by one in-flight call
would make its siblings look nested and drop them. Each asyncio task gets its
own copy of a ContextVar, while everything a coroutine awaits still sees its
depth — so genuine nesting (LangChain -> OpenAI) is detected and sibling
requests are not.
"""

import contextvars

_tracking_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "agentcost_tracking_depth", default=0
)


def in_tracking() -> bool:
    """True if we are already inside a tracked call in this context."""
    return _tracking_depth.get() > 0


def enter_tracking() -> contextvars.Token:
    """Mark the start of a tracked call. Pass the token to exit_tracking()."""
    return _tracking_depth.set(_tracking_depth.get() + 1)


def exit_tracking(token: contextvars.Token) -> None:
    """Restore the depth recorded before the matching enter_tracking()."""
    _tracking_depth.reset(token)
