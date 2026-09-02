"""Cluster clock and SLURM time-string helpers (design sections 2 "clock.py", 5.2 and 6.0).

Rule 5 of the design: no laptop-clock comparisons. ``ClusterClock`` extrapolates the cluster's own
``date +%s`` with the local *monotonic* clock, converts SLURM timestamps (epoch or ISO with the
discovered tz offset) to cluster epoch seconds, and detects clock jumps (laptop sleep).
This module imports nothing from the package.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone

# Sentinels SLURM prints instead of a timestamp/duration (design section 6.0).
SENTINELS: frozenset[str] = frozenset({
    "N/A", "Unknown", "None", "None assigned", "(null)", "UNLIMITED", "Partition_Limit", "INVALID",
    "infinite", "NONE", "",
})

_ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})$")
_EPOCH_RE = re.compile(r"^-?\d+$")
_DURATION_RE = re.compile(r"^(?:(\d+)-)?(?:(\d+):)?(?:(\d+):)?(\d+)$")


class ClusterClock:
    """Tracks the cluster's epoch time via ``::NOW`` and extrapolates it with ``time.monotonic()``.

    ``epoch_format`` is the discovered ``caps.epoch_format`` (SLURM honours ``SLURM_TIME_FORMAT=%s``);
    ``tz_offset_s`` is the discovered ``date +%z`` offset in seconds (east of UTC positive).
    """

    def __init__(self, epoch_format: bool = True, tz_offset_s: int = 0) -> None:
        self.epoch_format = bool(epoch_format)
        self.tz_offset_s = int(tz_offset_s)
        self._remote_epoch: int | None = None
        self._mono_at_update: float | None = None
        self._wall_at_update: float | None = None
        self._prev_offset: float | None = None   # remote_epoch - monotonic at the previous update
        self._last_jump_s: float = 0.0

    # -- synchronisation -------------------------------------------------------------------------

    def update_from_remote(self, remote_now_epoch: int, *, monotonic: float | None = None,
                           wall: float | None = None) -> None:
        """Record the cluster's ``date +%s`` together with the local monotonic (and wall) reading."""
        mono = time.monotonic() if monotonic is None else monotonic
        self._wall_at_update = time.time() if wall is None else wall
        offset = float(remote_now_epoch) - mono
        if self._prev_offset is not None:
            self._last_jump_s = abs(offset - self._prev_offset)
        else:
            self._last_jump_s = 0.0
        self._prev_offset = offset
        self._remote_epoch = int(remote_now_epoch)
        self._mono_at_update = mono

    @property
    def synced(self) -> bool:
        return self._remote_epoch is not None

    def remote_now(self, *, monotonic: float | None = None) -> int:
        """Extrapolated cluster epoch seconds. Falls back to the local wall clock before the first sync."""
        if self._remote_epoch is None or self._mono_at_update is None:
            return int(time.time())
        mono = time.monotonic() if monotonic is None else monotonic
        return int(self._remote_epoch + (mono - self._mono_at_update))

    def jump_detected(self, threshold_s: float = 120, *, monotonic: float | None = None,
                      wall: float | None = None) -> bool:
        """True when a clock jump > ``threshold_s`` is visible (laptop sleep, design section 5.2).

        Two signals: (a) the offset between cluster epoch and local monotonic changed by more than the
        threshold between the last two syncs; (b) since the last sync the local wall clock advanced by
        more than the monotonic clock did plus the threshold (a suspend the monotonic clock slept through).
        ``monotonic``/``wall`` override the current readings (tests).
        """
        if self._last_jump_s > threshold_s:
            return True
        if self._mono_at_update is None or self._wall_at_update is None:
            return False
        mono = time.monotonic() if monotonic is None else monotonic
        now_wall = time.time() if wall is None else wall
        mono_elapsed = mono - self._mono_at_update
        wall_elapsed = now_wall - self._wall_at_update
        return abs(wall_elapsed - mono_elapsed) > threshold_s

    # -- conversion ------------------------------------------------------------------------------

    def to_epoch(self, value: object) -> int | None:
        """Convert a SLURM timestamp field to cluster epoch seconds.

        Accepts ints/floats (already epoch), digit strings (epoch), ``YYYY-MM-DDTHH:MM:SS`` (cluster-local
        time, shifted by ``tz_offset_s``) and returns None for every sentinel or unparsable value.
        """
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return int(value)
        s = str(value).strip()
        if s in SENTINELS:
            return None
        if _EPOCH_RE.match(s):
            return int(s)
        m = _ISO_RE.match(s)
        if m:
            y, mo, d, hh, mm, ss = (int(g) for g in m.groups())
            try:
                local = datetime(y, mo, d, hh, mm, ss, tzinfo=timezone(timedelta(seconds=self.tz_offset_s)))
            except ValueError:
                return None
            return int(local.timestamp())
        return None

    def age_s(self, ts: object, *, monotonic: float | None = None) -> float:
        """Seconds elapsed on the cluster since ``ts`` (epoch or SLURM string); ``inf`` for sentinels."""
        epoch = self.to_epoch(ts)
        if epoch is None:
            return float("inf")
        return float(self.remote_now(monotonic=monotonic) - epoch)


# -- durations ------------------------------------------------------------------------------------

def parse_duration(s: object) -> int | None:
    """Parse a SLURM time string into seconds (design section 3.2 ``Resources.time``, section 6.0).

    Accepted: ``MM``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM``, ``D-HH:MM:SS``; sentinels
    (``UNLIMITED``, ``infinite``, ``NONE``, ``Partition_Limit``, ``N/A`` ...) and garbage return None.
    Ints are minutes (``sbatch -t 30``).
    """
    if s is None or isinstance(s, bool):
        return None
    if isinstance(s, int):
        return s * 60 if s >= 0 else None
    text = str(s).strip()
    if text in SENTINELS or text.lower() in {"unlimited", "infinite", "none"}:
        return None
    m = _DURATION_RE.match(text)
    if not m:
        return None
    days, a, b, last = m.groups()
    parts = [p for p in (a, b) if p is not None]
    if days is not None:
        # D-HH | D-HH:MM | D-HH:MM:SS
        fields = [int(p) for p in parts] + [int(last)]
        hh, mm, ss = (fields + [0, 0])[:3] if len(fields) < 3 else fields
        return int(days) * 86400 + hh * 3600 + mm * 60 + ss
    fields = [int(p) for p in parts] + [int(last)]
    if len(fields) == 1:            # MM
        return fields[0] * 60
    if len(fields) == 2:            # MM:SS
        return fields[0] * 60 + fields[1]
    hh, mm, ss = fields             # HH:MM:SS
    return hh * 3600 + mm * 60 + ss


def format_duration(seconds: object) -> str | None:
    """Render seconds as ``HH:MM:SS`` or ``D-HH:MM:SS`` (sbatch ``-t`` syntax). None/negative -> None."""
    if seconds is None or isinstance(seconds, bool):
        return None
    total = int(seconds)
    if total < 0:
        return None
    days, rem = divmod(total, 86400)
    hh, rem = divmod(rem, 3600)
    mm, ss = divmod(rem, 60)
    if days:
        return f"{days}-{hh:02d}:{mm:02d}:{ss:02d}"
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


__all__ = ["SENTINELS", "ClusterClock", "parse_duration", "format_duration"]
