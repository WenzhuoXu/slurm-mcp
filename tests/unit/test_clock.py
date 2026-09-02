"""Unit tests for slurm_mcp.clock (design sections 5.2, 6.0)."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, strategies as st

from slurm_mcp.clock import SENTINELS, ClusterClock, format_duration, parse_duration

SENTINEL_VALUES = ["N/A", "Unknown", "None", "None assigned", "(null)", "UNLIMITED", "Partition_Limit", "INVALID"]


# --- ClusterClock ----------------------------------------------------------------------------------

def test_remote_now_extrapolates_with_monotonic():
    c = ClusterClock(epoch_format=True, tz_offset_s=0)
    c.update_from_remote(1_000_000, monotonic=100.0, wall=5000.0)
    assert c.synced
    assert c.remote_now(monotonic=100.0) == 1_000_000
    assert c.remote_now(monotonic=160.5) == 1_000_060


def test_remote_now_before_sync_falls_back_to_wall_clock():
    c = ClusterClock(True, 0)
    assert not c.synced
    assert abs(c.remote_now() - time.time()) < 5


def test_update_uses_real_monotonic_by_default():
    c = ClusterClock(True, 0)
    c.update_from_remote(123456)
    assert 123456 <= c.remote_now() <= 123456 + 5


@pytest.mark.parametrize("value", SENTINEL_VALUES + ["", None, "garbage", "2026-13-40T99:99:99", True])
def test_to_epoch_sentinels_and_garbage(value):
    assert ClusterClock(False, -4 * 3600).to_epoch(value) is None


def test_sentinel_set_contains_design_list():
    assert set(SENTINEL_VALUES) <= SENTINELS


@pytest.mark.parametrize("value,expected", [
    (1756768683, 1756768683), ("1756768683", 1756768683), (1756768683.9, 1756768683), (" 42 ", 42), ("0", 0),
])
def test_to_epoch_epoch_values(value, expected):
    assert ClusterClock(True, 0).to_epoch(value) == expected


@pytest.mark.parametrize("tz_offset_s,iso", [
    (0, "2026-09-01T20:58:03"),
    (-4 * 3600, "2026-09-01T20:58:03"),     # EDT (TRACE fixture squeue_me_start)
    (-5 * 3600, "2026-01-01T00:00:00"),     # EST
    (3600, "2026-06-15T12:00:00"),
    (5 * 3600 + 1800, "2026-03-10T08:15:00"),
])
def test_to_epoch_iso_uses_tz_offset(tz_offset_s, iso):
    naive = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S")
    expected = int(naive.replace(tzinfo=timezone(timedelta(seconds=tz_offset_s))).timestamp())
    assert ClusterClock(False, tz_offset_s).to_epoch(iso) == expected
    # the same wall-clock string one hour further east is one hour earlier in epoch terms
    assert ClusterClock(False, tz_offset_s + 3600).to_epoch(iso) == expected - 3600


def test_to_epoch_iso_known_value():
    assert ClusterClock(False, 0).to_epoch("2026-09-01T20:58:03") == 1788296283
    assert ClusterClock(False, -4 * 3600).to_epoch("2026-09-01T20:58:03") == 1788296283 + 4 * 3600


def test_to_epoch_iso_fixture_value_from_trace():
    # tests/fixtures/trace/squeue_me_start.out row 1 was captured in America/New_York (EDT, -0400).
    c = ClusterClock(False, -4 * 3600)
    e = c.to_epoch("2026-09-01T20:58:03")
    assert e is not None and e == c.to_epoch(str(e))


def test_age_s():
    c = ClusterClock(True, 0)
    c.update_from_remote(2000, monotonic=0.0, wall=0.0)
    assert c.age_s(1400, monotonic=10.0) == 610.0
    assert c.age_s("1400", monotonic=10.0) == 610.0
    assert c.age_s("N/A") == float("inf")


def test_jump_detected_between_syncs():
    c = ClusterClock(True, 0)
    c.update_from_remote(1000, monotonic=0.0, wall=0.0)
    assert not c.jump_detected(monotonic=0.0, wall=0.0)
    c.update_from_remote(1060, monotonic=60.0, wall=60.0)        # consistent
    assert not c.jump_detected(monotonic=60.0, wall=60.0)
    c.update_from_remote(5000, monotonic=120.0, wall=120.0)      # cluster time leapt by 3880 s (sleep)
    assert c.jump_detected(monotonic=120.0, wall=120.0)
    assert not c.jump_detected(threshold_s=10_000, monotonic=120.0, wall=120.0)


def test_jump_detected_wall_vs_monotonic():
    c = ClusterClock(True, 0)
    c.update_from_remote(1000, monotonic=0.0, wall=0.0)
    assert not c.jump_detected(monotonic=100.0, wall=100.0)
    assert not c.jump_detected(monotonic=100.0, wall=190.0)      # 90 s drift < 120
    assert c.jump_detected(monotonic=100.0, wall=400.0)          # laptop slept 300 s
    assert c.jump_detected(threshold_s=60, monotonic=100.0, wall=190.0)


def test_jump_detected_unsynced_false():
    assert not ClusterClock(True, 0).jump_detected()


# --- durations -------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,seconds", [
    ("30", 1800), ("0", 0), ("05:30", 330), ("00:30", 30), ("01:00:00", 3600), ("1-00:00:00", 86400),
    ("1-12", 129600), ("1-12:30", 131400), ("2-03:04:05", 2 * 86400 + 3 * 3600 + 4 * 60 + 5),
    ("1-00:00", 86400), ("100:00:00", 360000), (" 15 ", 900), (30, 1800), (0, 0),
])
def test_parse_duration_ok(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["UNLIMITED", "infinite", "NONE", "None", "N/A", "Partition_Limit", "INVALID",
                                  "Unknown", "", None, "abc", "1:2:3:4", "-5", "1-2-3", "1:", ":5", True, -3])
def test_parse_duration_none(text):
    assert parse_duration(text) is None


@pytest.mark.parametrize("seconds,text", [
    (0, "00:00:00"), (30, "00:00:30"), (3600, "01:00:00"), (86400, "1-00:00:00"), (90061, "1-01:01:01"),
    (359999, "4-03:59:59"), (3600.9, "01:00:00"),
])
def test_format_duration(seconds, text):
    assert format_duration(seconds) == text


@pytest.mark.parametrize("value", [None, -1, True])
def test_format_duration_none(value):
    assert format_duration(value) is None


@given(st.integers(min_value=0, max_value=400 * 86400))
def test_format_parse_roundtrip(seconds):
    assert parse_duration(format_duration(seconds)) == seconds


@given(st.integers(min_value=0, max_value=2**31), st.integers(min_value=-14 * 3600, max_value=14 * 3600))
def test_to_epoch_epoch_string_roundtrip(epoch, tz):
    assert ClusterClock(True, tz).to_epoch(str(epoch)) == epoch
