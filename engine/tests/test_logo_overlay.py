"""Logo watermark: positioning math, the overlay filter string, and a real render.

The unit tests stub nothing but FFmpeg's presence; the integration test at the
bottom renders short 9:16 and 16:9 clips with a tiny transparent PNG and reads
the pixels back. It is skipped when FFmpeg is missing, like test_av_sync.
"""

import asyncio
import os
import shutil
import subprocess

import pytest
from PIL import Image

from clip_engine.config import LayoutStyle
from clip_engine.services.ai_clipping_pipeline import ClippingJobRequest
from clip_engine.services.layout_analyzer import ClipLayoutPlan, LayoutType, ShotLayout
from clip_engine.services.rendering_service import (
    LogoOverlay,
    RenderingService,
    RenderRequest,
    logo_geometry,
)

FFMPEG = os.environ.get("TEST_FFMPEG") or shutil.which("ffmpeg")
FFPROBE = os.environ.get("TEST_FFPROBE") or shutil.which("ffprobe")


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(RenderingService, "_verify_ffmpeg", lambda self: None)
    return RenderingService()


def logo_png(path, size=(20, 10), opaque_left=True):
    """Red on the left half, fully transparent on the right half."""
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    if opaque_left:
        for x in range(size[0] // 2):
            for y in range(size[1]):
                image.putpixel((x, y), (255, 0, 0, 255))
    image.save(path)
    return str(path)


def request(tmp_path, **overrides):
    values = dict(
        video_path="in.mp4", output_path=str(tmp_path / "clips" / "clip_00.mp4"),
        start_time_ms=0, end_time_ms=10_000, source_width=1920, source_height=1080,
        apply_padding=False,
    )
    values.update(overrides)
    os.makedirs(os.path.dirname(values["output_path"]), exist_ok=True)
    return RenderRequest(**values)


def plan(src_w=1920, src_h=1080):
    return ClipLayoutPlan([ShotLayout(0, 10_000, LayoutType.SCREEN)], src_w, src_h)


class TestGeometry:
    def test_width_is_a_fraction_of_the_output_width_and_height_keeps_the_aspect(self):
        logo = LogoOverlay(path="logo.png", x=0.81, y=0.0225, width=0.15)
        assert logo_geometry(logo, 200, 100, 1080, 1920) == (875, 43, 162, 81)

    def test_position_is_a_fraction_of_the_actual_output_size(self):
        logo = LogoOverlay(path="logo.png", x=0.25, y=0.5, width=0.1)
        assert logo_geometry(logo, 100, 100, 1920, 1080) == (480, 540, 192, 192)
        assert logo_geometry(logo, 100, 100, 3840, 2160) == (960, 1080, 384, 384)

    def test_clamped_inside_the_frame_on_the_right_and_bottom(self):
        logo = LogoOverlay(path="logo.png", x=0.95, y=0.99, width=0.3)
        x, y, w, h = logo_geometry(logo, 200, 100, 1080, 1920)
        assert (w, h) == (324, 162)
        assert (x, y) == (1080 - 324, 1920 - 162)

    def test_edges_are_reachable(self):
        logo = LogoOverlay(path="logo.png", x=0, y=0, width=1)
        assert logo_geometry(logo, 200, 100, 1080, 1920) == (0, 0, 1080, 540)

    def test_tall_logo_on_a_landscape_frame_shrinks_to_fit(self):
        logo = LogoOverlay(path="logo.png", x=0.9, y=0.5, width=0.5)
        x, y, w, h = logo_geometry(logo, 100, 1000, 1920, 1080)
        assert h == 1080 and w == 108
        assert (x, y) == (1728, 0)

    def test_tiny_logo_is_at_least_one_pixel(self):
        logo = LogoOverlay(path="logo.png", x=0, y=0, width=0.0001)
        assert logo_geometry(logo, 1000, 10, 1080, 1920)[2:] == (1, 1)

    @pytest.mark.parametrize("field,value", [
        ("x", -0.01), ("y", 1.5), ("width", 0), ("width", 1.2), ("opacity", 2),
        ("x", float("nan")), ("y", float("inf")), ("x", True), ("width", "0.2"),
    ])
    def test_rejects_out_of_range_values(self, field, value):
        values = dict(path="logo.png", x=0.1, y=0.1, width=0.2, opacity=1.0)
        values[field] = value
        with pytest.raises(ValueError):
            LogoOverlay(**values)

    def test_job_request_converts_the_bridge_dict(self):
        job = ClippingJobRequest(video_url="https://example.com/v", job_id="job",
                                 logo={"path": "/x/logo.png", "x": 0.1, "y": 0.2, "width": 0.3})
        assert job.logo == LogoOverlay(path="/x/logo.png", x=0.1, y=0.2, width=0.3, opacity=1.0)
        assert ClippingJobRequest(video_url="https://example.com/v", job_id="job").logo is None


class TestOverlayGraph:
    def test_no_logo_leaves_the_graph_unchanged(self, service, tmp_path):
        base = request(tmp_path, title_text="A title for this clip")
        explicit = request(tmp_path, title_text="A title for this clip", logo=None)
        overlays = service._overlays(base, plan(), 1080, 1920, False)
        assert overlays == service._overlays(explicit, plan(), 1080, 1920, False)
        assert len(overlays) == 1
        graph, inputs = service._compose_overlays("[0:v]null[base];[base]null[captioned]", overlays)
        # Exactly the graph the title card produced before logos existed.
        assert graph == (
            "[0:v]null[base];[base]null[captioned];"
            f"[captioned][1:v]overlay=x='(W-w)/2':y='{overlays[0][2]}':shortest=1[composited];"
            "[composited]null[out]"
        )
        assert inputs == [overlays[0][0]]
        assert not any("logo-" in path for path in os.listdir(tmp_path / "clips"))
        # And without any overlay at all.
        assert service._overlays(request(tmp_path), plan(), 1080, 1920, False) == []
        assert service._compose_overlays("G", []) == ("G;[captioned]null[out]", [])

    def test_logo_is_drawn_last_with_alpha_and_no_enable(self, service, tmp_path):
        path = logo_png(tmp_path / "logo.png", size=(200, 100))
        req = request(
            tmp_path, title_text="A title for this clip",
            banner_platform="youtube", banner_channel_url="youtube.com/@example",
            logo=LogoOverlay(path=path, x=0.81, y=0.0225, width=0.15),
        )
        overlays = service._overlays(req, plan(), 1080, 1920, False)
        assert len(overlays) == 3
        logo_path, x, y, enable, image_filter = overlays[-1]
        assert os.path.basename(logo_path) == "logo-0-10000.png"
        assert (x, y, enable, image_filter) == ("875", "43", "", "format=rgba")
        graph, inputs = service._compose_overlays("G", overlays, "setpts=PTS/1.5")
        assert inputs[-1] == logo_path
        assert graph.endswith(
            ";[3:v]format=rgba[img3]"
            ";[ov2][img3]overlay=x='875':y='43':shortest=1[composited]"
            ";[composited]setpts=PTS/1.5[out]"
        )
        assert "enable" not in graph.split("[img3];")[-1]

    def test_landscape_uses_the_landscape_output_size(self, service, tmp_path):
        path = logo_png(tmp_path / "logo.png", size=(100, 50))
        req = request(tmp_path, aspect_ratio="16:9", logo=LogoOverlay(path=path, x=0.5, y=0.9, width=0.1))
        (overlay,) = service._overlays(req, plan(), 2560, 1440, True)
        # 10% of 2560 wide at (50%, 90%) of the 2560x1440 frame.
        assert overlay[1:3] == ("1280", "1296")
        with Image.open(overlay[0]) as scaled:
            assert scaled.size == (256, 128)

    def test_scaled_copy_keeps_alpha_and_applies_opacity(self, service, tmp_path):
        path = logo_png(tmp_path / "logo.png", size=(40, 20))
        req = request(tmp_path, logo=LogoOverlay(path=path, x=0, y=0, width=0.1, opacity=0.5))
        out_path, x, y = service._logo_overlay_image(req, 1080, 1920)
        assert (x, y) == (0, 0)
        with Image.open(out_path) as scaled:
            assert scaled.mode == "RGBA" and scaled.size == (108, 54)
            assert scaled.getpixel((10, 27))[3] == 128  # opaque half at 50%
            assert scaled.getpixel((100, 27))[3] == 0   # transparent half stays clear
        # The user's own file is never the one passed to FFmpeg (inputs are deleted after rendering).
        assert out_path != path
        with Image.open(path) as original:
            assert original.size == (40, 20)

    def test_a_bad_logo_does_not_leave_other_overlay_images_behind(self, service, tmp_path):
        missing = str(tmp_path / "missing.png")
        req = request(tmp_path, title_text="A title for this clip", logo=LogoOverlay(path=missing, x=0, y=0, width=0.1))
        with pytest.raises(OSError):
            service._overlays(req, plan(), 1080, 1920, False)
        assert os.listdir(tmp_path / "clips") == []

    def test_non_png_logo_is_rejected(self, service, tmp_path):
        jpeg = tmp_path / "logo.png"
        Image.new("RGB", (10, 10)).save(jpeg, "JPEG")
        req = request(tmp_path, logo=LogoOverlay(path=str(jpeg), x=0, y=0, width=0.1))
        with pytest.raises(Exception, match="PNG"):
            service._logo_overlay_image(req, 1080, 1920)


def run(*args):
    result = subprocess.run([str(a) for a in args], capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr.decode(errors="replace")[-4000:]
    return result.stdout


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="FFmpeg and FFprobe required")
class TestLogoRender:
    @pytest.fixture(autouse=True)
    def media_path(self, monkeypatch):
        monkeypatch.setenv("PATH", os.path.dirname(FFMPEG) + os.pathsep + os.environ.get("PATH", ""))

    @pytest.mark.parametrize("aspect_ratio,source_size", [("9:16", "320x180"), ("16:9", "320x180")])
    def test_logo_is_burned_in_with_its_transparency(self, service, tmp_path, aspect_ratio, source_size):
        source = tmp_path / "source.mkv"
        run(FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", f"color=black:s={source_size}:r=30:d=2",
            "-c:v", "ffv1", source)
        # FFV1 in, MPEG-4 out: available in both the LGPL bundle and distro builds.
        service._video_codec_args = lambda *args: ["-c:v", "mpeg4", "-q:v", "2"]
        path = logo_png(tmp_path / "logo.png", size=(20, 10))
        logo = LogoOverlay(path=path, x=0.3, y=0.4, width=0.2)
        req = request(
            tmp_path, video_path=str(source), end_time_ms=1500, source_width=320, source_height=180,
            include_audio=False, include_captions=False, aspect_ratio=aspect_ratio,
            layout_style=LayoutStyle.FIT, logo=logo,
        )
        result = asyncio.run(service.render_clip(req))
        out_w, out_h = result.output_width, result.output_height
        assert (out_w > out_h) == (aspect_ratio == "16:9")

        frame = tmp_path / "frame.png"
        run(FFMPEG, "-v", "error", "-y", "-ss", "0.5", "-i", result.output_path, "-frames:v", "1", frame)
        x, y, w, h = logo_geometry(logo, 20, 10, out_w, out_h)
        with Image.open(frame) as image:
            rgb = image.convert("RGB")
            assert rgb.size == (out_w, out_h)
            red = rgb.getpixel((x + w // 4, y + h // 2))
            clear = rgb.getpixel((x + 3 * w // 4, y + h // 2))
            outside = rgb.getpixel((max(0, x - 20), y + h // 2))
        assert red[0] > 180 and red[1] < 80 and red[2] < 80, red
        assert max(clear) < 60, clear      # transparent half shows the black video
        assert max(outside) < 60, outside
        # The temporary scaled copy is cleaned up; the user's logo is untouched.
        assert not os.path.exists(os.path.join(os.path.dirname(result.output_path), "logo-0-1500.png"))
        assert os.path.isfile(path)
