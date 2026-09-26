"""Hype detection (Streamer/Gambling modes): loudness windows, relative
thresholds, merging, planner hints, and a real FFmpeg measurement."""

import asyncio
import json
import math
import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from clip_engine.services.ai_clipping_pipeline import ClippingJobRequest
from clip_engine.services.hype_detector import (
    HypeMoment,
    detect_hype_moments,
    format_hype_hints,
    loudness_command,
    measure_loudness,
    parse_astats_lines,
)
from tests.test_planner import FakeClient, clip, completion, make_planner, make_transcript

FFMPEG = os.environ.get("TEST_FFMPEG") or shutil.which("ffmpeg")


def stream(levels_db, spikes=()):
    """1-second windows at the given base levels, with (second, dB) overrides."""
    windows = [(float(t), level) for t, level in enumerate(levels_db)]
    for second, db in spikes:
        windows[second] = (float(second), db)
    return windows


def calm(seconds=120, level=-35.0):
    # A little natural variation (+/- 1.5 dB).
    return [level + 1.5 * math.sin(t / 3) for t in range(seconds)]


class TestDetection:
    def test_finds_spikes_well_above_the_videos_own_average(self):
        analysis = detect_hype_moments(stream(calm(), [(30, -22), (80, -20)]))
        assert [(m.start_seconds, m.end_seconds, m.peak_seconds) for m in analysis.moments] == [
            (30, 31, 30), (80, 81, 80),
        ]
        assert analysis.moments[1].above_average_db == pytest.approx(15, abs=0.5)
        assert analysis.baseline_db == pytest.approx(-35, abs=0.5)

    def test_threshold_is_relative_not_a_fixed_db_value(self):
        quiet = detect_hype_moments(stream(calm(level=-45), [(40, -32)]))
        loud = detect_hype_moments(stream(calm(level=-15), [(40, -2)]))
        assert [m.start_seconds for m in quiet.moments] == [m.start_seconds for m in loud.moments] == [40]
        assert loud.threshold_db - quiet.threshold_db == pytest.approx(30, abs=0.5)
        # A constantly loud video has no hype moments of its own.
        assert detect_hype_moments(stream([-12.0] * 120)).moments == []

    def test_small_bumps_are_not_hype(self):
        assert detect_hype_moments(stream(calm(), [(30, -31), (60, -30.5)])).moments == []

    def test_noisy_videos_need_a_bigger_jump(self):
        noisy = [-45.0 + (8 if t % 2 else -8) for t in range(120)]
        # +10 dB over the typical level is inside this video's normal swing ...
        assert detect_hype_moments(stream(noisy, [(51, -35)])).moments == []
        # ... a much bigger jump is not.
        assert [m.start_seconds for m in detect_hype_moments(stream(noisy, [(51, -2)])).moments] == [51]

    def test_nearby_spikes_merge_into_one_moment(self):
        analysis = detect_hype_moments(stream(calm(), [(30, -22), (31, -20), (34, -21), (45, -22)]))
        moments = [(m.start_seconds, m.end_seconds, m.peak_seconds, m.peak_db) for m in analysis.moments]
        assert moments == [(30, 35, 31, -20), (45, 46, 45, -22)]

    def test_silence_does_not_lower_the_baseline(self):
        levels = [-math.inf] * 60 + [-90.0] * 20 + calm(60)
        analysis = detect_hype_moments(stream(levels, [(100, -22)]))
        assert analysis.baseline_db == pytest.approx(-35, abs=1)
        assert [m.start_seconds for m in analysis.moments] == [100]

    def test_too_little_audio_gives_no_moments(self):
        assert detect_hype_moments(stream(calm(10), [(5, 0)])).moments == []
        assert detect_hype_moments([]).moments == []

    def test_keeps_the_strongest_moments_in_time_order(self):
        spikes = [(10 + 10 * i, -25 + i) for i in range(8)]
        analysis = detect_hype_moments(stream(calm(), spikes), max_moments=3)
        assert [m.start_seconds for m in analysis.moments] == [60, 70, 80]


class TestFfmpegOutput:
    def test_parses_ametadata_print_output(self):
        lines = [
            b"frame:0    pts:0       pts_time:0\n",
            b"lavfi.astats.Overall.RMS_level=-35.5\n",
            b"frame:1    pts:16000   pts_time:1\n",
            b"lavfi.astats.Overall.RMS_level=-inf\n",
            b"frame:2    pts:32000   pts_time:2\n",
            b"lavfi.astats.Overall.RMS_level=-12.25\n",
        ]
        assert parse_astats_lines(lines, offset_seconds=60) == [(60, -35.5), (61, -math.inf), (62, -12.25)]

    def test_command_resamples_before_windowing_and_seeks_the_range(self):
        cmd = loudness_command("/v.mp4", 60, 90)
        assert cmd[cmd.index("-ss") + 1] == "60.000" and cmd[cmd.index("-t") + 1] == "30.000"
        graph = cmd[cmd.index("-af") + 1]
        assert graph.index("aresample=16000") < graph.index("asetnsamples=n=16000")
        assert "-vn" in cmd and "0:a:0" in cmd
        assert "-ss" not in loudness_command("/v.mp4")


class TestPlannerHints:
    MOMENTS = [HypeMoment(95, 99, 96, -18, 14.5), HypeMoment(200, 202, 200, -20, 12)]

    def test_hint_text_asks_for_the_build_up(self):
        text = format_hype_hints(self.MOMENTS, "streamer")
        assert "- 95.0s - 99.0s (peak at 96.0s, +14.5 dB above this video's average)" in text
        assert "BEFORE the spike" in text
        assert format_hype_hints([], "streamer") == ""
        assert "big wins" in format_hype_hints(self.MOMENTS, "gambling")

    def _messages(self, monkeypatch, **kwargs):
        planner = make_planner()
        planner._http_client = FakeClient([(200, completion(json.dumps({"clips": [clip(90, 120)]})))])
        captured = []
        original = planner._build_vision_messages
        monkeypatch.setattr(planner, "_build_vision_messages", lambda *a, **k: captured.append(original(*a, **k)) or captured[-1])
        asyncio.run(planner.plan_clips(
            transcript_result=make_transcript(300), video_metadata=SimpleNamespace(duration_seconds=300),
            max_clips=1, auto_clip_count=False, min_duration_seconds=15, max_duration_seconds=60, **kwargs,
        ))
        return captured[0]

    def test_podcast_prompt_is_unchanged(self, monkeypatch):
        before = self._messages(monkeypatch)
        podcast = self._messages(monkeypatch, content_mode="podcast", hype_moments=self.MOMENTS)
        assert json.dumps(podcast) == json.dumps(before)
        assert "HYPE" not in json.dumps(podcast) and "CONTENT MODE" not in json.dumps(podcast)

    @pytest.mark.parametrize("mode,heading", [("streamer", "STREAMER"), ("gambling", "GAMBLING STREAM")])
    def test_streamer_and_gambling_get_guidance_and_hints(self, monkeypatch, mode, heading):
        system, user = self._messages(monkeypatch, content_mode=mode, hype_moments=self.MOMENTS)
        assert f"## CONTENT MODE: {heading}" in system["content"]
        text = user["content"][0]["text"]
        assert "## HYPE MOMENTS" in text and "peak at 96.0s" in text and "peak at 200.0s" in text

    def test_hints_outside_the_selected_range_are_dropped(self, monkeypatch):
        _, user = self._messages(monkeypatch, content_mode="streamer", hype_moments=self.MOMENTS,
                                 start_time_seconds=150, end_time_seconds=300)
        text = user["content"][0]["text"]
        assert "peak at 200.0s" in text and "peak at 96.0s" not in text

    def test_no_moments_means_no_hint_block(self, monkeypatch):
        _, user = self._messages(monkeypatch, content_mode="streamer", hype_moments=[])
        assert "HYPE" not in user["content"][0]["text"]


def test_job_request_validates_the_mode():
    assert ClippingJobRequest(video_url="https://example.com/v", job_id="j").content_mode == "podcast"
    assert ClippingJobRequest(video_url="https://example.com/v", job_id="j", content_mode="gambling").content_mode == "gambling"
    with pytest.raises(ValueError):
        ClippingJobRequest(video_url="https://example.com/v", job_id="j", content_mode="vlog")


@pytest.mark.skipif(not FFMPEG, reason="FFmpeg required")
def test_measures_real_audio_in_one_second_windows(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", os.path.dirname(FFMPEG) + os.pathsep + os.environ.get("PATH", ""))
    path = tmp_path / "stream.mkv"
    # Pink noise with bursts 5x louder (+14 dB) at 30-32 s and 50-51 s.
    subprocess.run([
        FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "anoisesrc=color=pink:amplitude=0.05:d=70:r=48000",
        "-af", "volume='if(between(t,30,32)+between(t,50,51),5,1)':eval=frame", "-c:a", "pcm_s16le", str(path),
    ], check=True, timeout=60)
    windows = measure_loudness(str(path))
    assert len(windows) == 70 and windows[1][0] == 1.0
    moments = detect_hype_moments(windows).moments
    assert [(m.start_seconds, m.end_seconds) for m in moments] == [(30, 32), (50, 51)]
    ranged = measure_loudness(str(path), 20, 40)
    assert len(ranged) == 20 and ranged[0][0] == 20.0
