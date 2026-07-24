"""Upstream concurrency limiting for homelab GPU backends."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class UpstreamConcurrencyLimiter:
    """Non-blocking slot limiter for expensive upstream generations."""

    limit: int
    _active: int = field(default=0, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.limit < 1:
            msg = "concurrency limit must be >= 1"
            raise ValueError(msg)

    @property
    def active(self) -> int:
        return self._active

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._active >= self.limit:
                return False
            self._active += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            if self._active > 0:
                self._active -= 1
