"""
Hype detection for Streamer and Gambling modes: loud reactions, from audio only.

One FFmpeg pass decodes the audio track (mono, 16 kHz; no video decode) and
reports the RMS level of each 1-second window with `astats`. A window is a
spike when it is well above *this video's own* typical level: the threshold
is the median window level plus the larger of a minimum jump and a multiple
of the level spread (median absolute deviation), so a quiet podcast-like
stream and a loud gameplay stream are judged against themselves, not a fixed
dB value. Spikes close together are merged into one moment.

The moments are hints for the clip planner, never clips by themselves.
"""

import asyncio
import logging
import math
import re
from dataclasses import asdict, dataclass
from statistics import median
from typing import Optional

from clip_engine.services.media_process import MEDIA_INPUT_OPTIONS, MEDIA_TIMEOUT_SECONDS, media_process

logger = logging.getLogger(__name__)

CONTENT_MODES = ("podcast", "streamer", "gambling")
# Modes that look for loudness spikes; podcast plans from the transcript only.
HYPE_MODES = ("streamer", "gambling")

WINDOW_SECONDS = 1.0
SAMPLE_RATE = 16000
# Windows quieter than this are digital silence (muted mic, intro cards) and
# would drag the typical level down; they are left out of the baseline only.
SILENCE_FLOOR_DB = -70.0
# A spike must be at least this far above the video's median window (6 dB is
# about twice the RMS amplitude) ...
MIN_JUMP_DB = 6.0
# ... and at least this many robust standard deviations above it.
SPREAD_FACTOR = 2.5
# Spikes this close together are one moment (a scream, a breath, a scream).
MERGE_GAP_SECONDS = 4.0
# The planner gets at most this many moments, the strongest ones.
MAX_MOMENTS = 25
# Too little audio to know what "normal" sounds like.
MIN_WINDOWS = 20

_PTS = re.compile(rb"pts_time:(-?[0-9.]+)")
_RMS = re.compile(rb"lavfi\.astats\.Overall\.RMS_level=(-?[0-9.]+|-?inf|nan)")


@dataclass
class HypeMoment:
    """A loud stretch of the source, in source seconds."""

    start_seconds: float
    end_seconds: float
    peak_seconds: float
    # RMS level of the loudest 1-second window (dBFS).
    peak_db: float
    # How far that window is above the video's median window.
    above_average_db: float

    def to_dict(self) -> dict:
        return {key: round(value, 2) for key, value in asdict(self).items()}


@dataclass
class HypeAnalysis:
    moments: list[HypeMoment]
    # The video's own typical level and the level a window had to reach.
    baseline_db: Optional[float] = None
    threshold_db: Optional[float] = None
    windows: int = 0


def loudness_command(
    video_path: str, start_seconds: Optional[float] = None, end_seconds: Optional[float] = None,
) -> list[str]:
    """FFmpeg command printing one RMS level per 1-second window of audio."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-v", "error"]
    if start_seconds:
        cmd += ["-ss", f"{start_seconds:.3f}"]
    if end_seconds is not None:
        cmd += ["-t", f"{max(0.0, end_seconds - (start_seconds or 0)):.3f}"]
    samples = int(SAMPLE_RATE * WINDOW_SECONDS)
    cmd += [
        *MEDIA_INPUT_OPTIONS, "-i", video_path,
        "-map", "0:a:0", "-vn",
        # Resample and downmix inside the graph so each window is exactly
        # WINDOW_SECONDS long (-ar would resample after asetnsamples).
        "-af", (
            f"aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=mono,"
            f"asetnsamples=n={samples}:p=0,astats=metadata=1:reset=1,"
            "ametadata=mode=print:key=lavfi.astats.Overall.RMS_level:file=-"
        ),
        "-f", "null", "-",
    ]
    return cmd


def parse_astats_lines(lines, offset_seconds: float = 0.0) -> list[tuple[float, float]]:
    """(window start in source seconds, RMS dBFS) from `ametadata` print output.

    Silent windows report -inf; they are kept as -inf.
    """
    windows: list[tuple[float, float]] = []
    pts: Optional[float] = None
    for line in lines:
        if match := _PTS.search(line):
            pts = float(match.group(1))
        elif (match := _RMS.search(line)) and pts is not None:
            raw = match.group(1).decode()
            value = float(raw) if raw not in ("nan", "-nan") else -math.inf
            windows.append((round(offset_seconds + pts, 3), value))
            pts = None
    return windows


def measure_loudness(
    video_path: str, start_seconds: Optional[float] = None, end_seconds: Optional[float] = None,
    timeout: float = MEDIA_TIMEOUT_SECONDS,
) -> list[tuple[float, float]]:
    """Per-window loudness of the audio track. Streams FFmpeg's output line by line."""
    cmd = loudness_command(video_path, start_seconds, end_seconds)
    with media_process(cmd, timeout=timeout) as (process, stderr):
        windows = parse_astats_lines(iter(lambda: process.stdout.readline(4096), b""), start_seconds or 0.0)
    if process.returncode != 0:
        raise RuntimeError(f"Loudness analysis failed: {bytes(stderr).decode(errors='replace')[-300:]}")
    return windows


def detect_hype_moments(
    windows: list[tuple[float, float]],
    *,
    min_jump_db: float = MIN_JUMP_DB,
    spread_factor: float = SPREAD_FACTOR,
    merge_gap_seconds: float = MERGE_GAP_SECONDS,
    max_moments: int = MAX_MOMENTS,
    window_seconds: float = WINDOW_SECONDS,
) -> HypeAnalysis:
    """Loud moments relative to this video's own level. Pure; see the module docstring."""
    levels = [db for _, db in windows if math.isfinite(db) and db > SILENCE_FLOOR_DB]
    if len(levels) < MIN_WINDOWS:
        return HypeAnalysis(moments=[], windows=len(windows))
    baseline = median(levels)
    spread = 1.4826 * median(abs(db - baseline) for db in levels)
    threshold = baseline + max(min_jump_db, spread_factor * spread)

    moments: list[HypeMoment] = []
    for start, db in sorted(windows):
        if not (math.isfinite(db) and db >= threshold):
            continue
        end = start + window_seconds
        last = moments[-1] if moments else None
        if last and start - last.end_seconds <= merge_gap_seconds:
            last.end_seconds = end
            if db > last.peak_db:
                last.peak_seconds, last.peak_db = start, db
        else:
            moments.append(HypeMoment(start, end, start, db, 0.0))
    for moment in moments:
        moment.above_average_db = moment.peak_db - baseline
    if len(moments) > max_moments:
        strongest = sorted(moments, key=lambda m: m.peak_db, reverse=True)[:max_moments]
        moments = sorted(strongest, key=lambda m: m.start_seconds)
    return HypeAnalysis(moments=moments, baseline_db=baseline, threshold_db=threshold, windows=len(windows))


async def find_hype_moments(
    video_path: str, start_seconds: Optional[float] = None, end_seconds: Optional[float] = None,
) -> HypeAnalysis:
    """Measure and detect off the event loop. Never raises: no hints is a valid outcome."""
    try:
        windows = await asyncio.to_thread(measure_loudness, video_path, start_seconds, end_seconds)
    except Exception as e:
        logger.warning(f"Hype detection skipped; planning from the transcript only: {e}")
        return HypeAnalysis(moments=[])
    analysis = detect_hype_moments(windows)
    logger.info(
        "Hype detection: %s moments in %s windows (baseline %s dB, threshold %s dB)",
        len(analysis.moments), analysis.windows,
        f"{analysis.baseline_db:.1f}" if analysis.baseline_db is not None else "n/a",
        f"{analysis.threshold_db:.1f}" if analysis.threshold_db is not None else "n/a",
    )
    return analysis


def format_hype_hints(moments: list[HypeMoment], content_mode: str) -> str:
    """The planner hint block. Empty when there are no moments."""
    if not moments:
        return ""
    lines = "\n".join(
        f"- {m.start_seconds:.1f}s - {m.end_seconds:.1f}s (peak at {m.peak_seconds:.1f}s, "
        f"+{m.above_average_db:.1f} dB above this video's average)"
        for m in moments
    )
    subject = "big wins, bonus hits and reactions" if content_mode == "gambling" else "reactions, screaming and hype"
    return (
        "\n\n## HYPE MOMENTS (audio loudness spikes)\n\n"
        f"These are moments where the audio gets much louder than this video's own average, "
        f"usually {subject}:\n{lines}\n\n"
        "Treat them as strong candidates. When you pick one, START the clip a few seconds "
        "(about 3-8 s) BEFORE the spike so the build-up is included, and let the reaction "
        "play out before ending. A spike is evidence, not a requirement: every clip must "
        "still follow the duration and quality rules, and you may pick moments without a spike."
    )
