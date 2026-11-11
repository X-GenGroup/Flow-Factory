# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Background log writer ordering/lifecycle and budget-fitted video encoding."""

import math
import threading
import time
from types import SimpleNamespace

import imageio
import numpy as np
import pytest
from PIL import Image

import flow_factory.logger.abc as logger_abc
from flow_factory.hparams.log_args import LogArguments
from flow_factory.logger.abc import Logger
from flow_factory.logger.formatting import LogVideo


class RecordingLogger(Logger):
    """Record writes; a video converts only after ``release`` is set."""

    def _init_platform(self):
        self.platform = None
        self.records = []
        self.release = threading.Event()
        self.release.set()
        self.seen_frames = []
        self.fail_video = False
        self.finished = 0

    def _convert_to_platform(self, value, height=None, width=None):
        if isinstance(value, LogVideo):
            assert self.release.wait(timeout=30), "test never released the writer"
            if self.fail_video:
                raise ValueError("encoder exploded")
            self.seen_frames.append(value.get_numpy().copy())
            return "video"
        return value

    def _log_impl(self, data, step):
        self.records.append((step, data))

    def _finish_platform(self):
        self.finished += 1


def make_logger(tmp_path, **log):
    config = SimpleNamespace(log_args=LogArguments(save_dir=str(tmp_path), **log))
    return RecordingLogger(config)


def video(t=4, h=8, w=8):
    return LogVideo(np.zeros((t, h, w, 3), dtype=np.uint8), fps=24)


def wait_for(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


def test_writer_keeps_call_order_without_blocking_the_caller(tmp_path):
    run = make_logger(tmp_path)
    run.release.clear()
    start = time.monotonic()
    run.log_data({"samples": [video()], "loss": 1.0}, step=0)
    run.log_data({"loss": 2.0}, step=1)
    assert time.monotonic() - start < 5
    assert run.records == []
    run.release.set()
    run.close()
    assert run.records == [(0, {"samples": ["video"], "loss": 1.0}), (1, {"loss": 2.0})]
    assert run.finished == 1


def test_media_is_copied_before_the_caller_returns(tmp_path):
    run = make_logger(tmp_path)
    run.release.clear()
    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    run.log_data({"samples": [LogVideo(frames)]}, step=0)
    frames[:] = 255
    run.release.set()
    run.close()
    assert len(run.seen_frames) == 1 and not run.seen_frames[0].any()


def test_media_beyond_the_pending_limit_is_dropped_but_scalars_are_kept(tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(logger_abc.logger, "warning", warnings.append)
    run = make_logger(tmp_path, media_max_pending=1)
    run.release.clear()
    run.log_data({"samples": [video()]}, step=0)
    run.log_data({"samples": [video()], "loss": 3.0}, step=1)
    run.release.set()
    run.close()
    assert run.records == [(0, {"samples": ["video"]}), (1, {"loss": 3.0})]
    assert len(warnings) == 1 and "step 1" in warnings[0] and "media_max_pending=1" in warnings[0]


def test_writer_errors_are_raised_on_the_caller_thread(tmp_path):
    run = make_logger(tmp_path)
    run.fail_video = True
    run.log_data({"samples": [video()]}, step=7)
    run._queue.join()
    with pytest.raises(RuntimeError, match="step 7") as caught:
        run.log_data({"loss": 1.0}, step=8)
    assert isinstance(caught.value.__cause__, ValueError)
    run.close()


def test_close_without_wait_discards_queued_calls_and_can_be_joined_later(tmp_path):
    run = make_logger(tmp_path)
    run.release.clear()
    run.log_data({"samples": [video()]}, step=0)
    wait_for(lambda: run._queue.unfinished_tasks == 1 and run._queue.qsize() == 0)
    run.log_data({"loss": 1.0}, step=1)
    run.close(wait=False)
    assert run.finished == 0
    run.release.set()
    run.close(wait=True)
    assert run.records == [(0, {"samples": ["video"]})]
    assert run._pending_media == 0 and run.finished == 1
    with pytest.raises(RuntimeError, match="after the logger was closed"):
        run.log_data({"loss": 2.0}, step=2)


def test_synchronous_mode_writes_inline(tmp_path):
    run = make_logger(tmp_path, async_logging=False)
    run.log_data({"loss": 1.0}, step=0)
    assert run.records == [(0, {"loss": 1.0})]
    run.close()
    assert run.finished == 1


def _moving_video(frames=24, height=96, width=160):
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack([x * 255 / width, y * 255 / height, (x + y) * 127 / (height + width)], -1)
    texture = rng.integers(0, 60, (height, width, 3))
    clip = [np.roll(base + texture, 5 * t, axis=1) % 256 for t in range(frames)]
    return np.stack(clip).astype(np.uint8)


def _gif(path):
    frames = imageio.mimread(path)
    with Image.open(path) as image:
        duration = image.info["duration"]
    return np.stack(frames), duration


def test_fitting_lowers_frame_rate_before_resolution_and_keeps_duration():
    clip = LogVideo(_moving_video(), fps=24)
    full = clip.get_value("gif")
    budget = int(math.ceil(len(open(full, "rb").read()) / 2.5))
    fitted = clip.get_fitted_value("gif", budget, min_fps=8, min_side=16)
    assert fitted is not None and len(open(fitted, "rb").read()) <= budget
    frames, duration = _gif(fitted)
    stride = math.ceil(24 / len(frames))
    assert stride > 1 and frames.shape[1:3] == (96, 160)
    assert duration == pytest.approx(1000 * stride / 24, abs=10)
    clip.cleanup()


def test_fitting_resizes_only_after_reaching_the_frame_rate_floor():
    clip = LogVideo(_moving_video(), fps=24)
    budget = int(len(open(clip.get_value("gif"), "rb").read()) / 12)
    fitted = clip.get_fitted_value("gif", budget, min_fps=8, min_side=16)
    assert fitted is not None and len(open(fitted, "rb").read()) <= budget
    frames, _ = _gif(fitted)
    height, width = frames.shape[1:3]
    assert len(frames) == 8  # stride 3 = 24 fps / 8 fps floor
    assert height < 96 and height % 2 == 0 and width % 2 == 0
    assert width / height == pytest.approx(160 / 96, rel=0.1)
    clip.cleanup()


def test_fitting_converges_when_bytes_per_pixel_grow_as_the_clip_shrinks(monkeypatch, tmp_path):
    """Model GIF frame differencing: subsampled frames cost more and area is sublinear.

    The probe runs at stride 1, so a stride-3 attempt overshoots; recalibrating towards the
    limit itself then approaches it from above (the 124x480x864 H3 clip that was skipped).
    """
    clip = LogVideo(np.zeros((124, 480, 864, 3), dtype=np.uint8), fps=24)
    writes = []

    def fake_write(arr, fps, format, *, with_audio):
        frames, height, width = arr.shape[:3]
        stride = 1 if frames <= 8 else round(124 / frames)
        size = int(0.66 * frames * (height * width) ** 0.9 * (480 * 864) ** 0.1 * stride**0.3)
        path = tmp_path / f"{len(writes)}.{format}"
        path.write_bytes(b"\0" * size)
        writes.append(size)
        return str(path)

    monkeypatch.setattr(clip, "_write", fake_write)
    budget = 10 * 1024**2
    fitted = clip.get_fitted_value("gif", budget, min_fps=8, min_side=128)
    assert fitted is not None and writes[-1] <= budget
    assert len(writes) <= 1 + 3  # probe plus at most _FIT_MAX_ATTEMPTS encodings


def test_fitting_skips_a_video_that_cannot_meet_the_floors(monkeypatch):
    warnings = []
    monkeypatch.setattr("flow_factory.logger.formatting.logger.warning", warnings.append)
    clip = LogVideo(_moving_video(), fps=24, caption="tiny")
    assert clip.get_fitted_value("gif", 500, min_fps=8, min_side=64) is None
    assert len(warnings) == 1 and "'tiny'" in warnings[0] and "500 bytes" in warnings[0]


def test_backend_budget_defaults_and_overrides(tmp_path):
    class Budgeted(RecordingLogger):
        platform_media_max_bytes = 1234

    default = Budgeted(SimpleNamespace(log_args=LogArguments(save_dir=str(tmp_path))))
    override = Budgeted(
        SimpleNamespace(log_args=LogArguments(save_dir=str(tmp_path), media_max_bytes=99))
    )
    assert (default.media_max_bytes, override.media_max_bytes) == (1234, 99)
    default.close()
    override.close()


def test_platform_item_limit_truncates_with_a_warning(tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(logger_abc.logger, "warning", warnings.append)

    class Limited(RecordingLogger):
        platform_media_max_items = 2

    run = Limited(SimpleNamespace(log_args=LogArguments(save_dir=str(tmp_path))))
    run.log_data({"samples": [video(), video(), video()]}, step=4)
    run.close()
    assert run.records == [(4, {"samples": ["video", "video"]})]
    assert len(warnings) == 1 and "2 of 3" in warnings[0]


def _backend(monkeypatch, name, tmp_path, platform):
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, name, platform)
    monkeypatch.delitem(sys.modules, f"flow_factory.logger.{name}", raising=False)
    module = importlib.import_module(f"flow_factory.logger.{name}")
    cls = getattr(module, "SwanlabLogger" if name == "swanlab" else "WandbLogger")
    config = SimpleNamespace(
        log_args=LogArguments(save_dir=str(tmp_path), project="p", run_name="r"),
        to_dict=lambda: {},
    )
    return cls(config)


def test_swanlab_fits_gifs_to_its_platform_limit_and_wandb_writes_mp4(tmp_path, monkeypatch):
    made = []
    platform = SimpleNamespace(
        init=lambda **kwargs: None,
        log=lambda data, step: made.append((step, data)),
        finish=lambda: made.append("finish"),
        Video=lambda path, caption=None, format=None: (path, open(path, "rb").read()[:6], format),
        Image=lambda *args, **kwargs: None,
    )
    swan = _backend(monkeypatch, "swanlab", tmp_path, platform)
    assert (swan.video_format, swan.media_max_bytes) == ("gif", 10 * 1024**2)
    swan.log_data({"samples": [LogVideo(_moving_video(frames=4), fps=24)]}, step=1)
    swan.close()
    (step, data), finish = made
    ((path, magic, _),) = data["samples"]
    assert step == 1 and magic in (b"GIF87a", b"GIF89a") and finish == "finish"

    made.clear()
    wb = _backend(monkeypatch, "wandb", tmp_path, platform)
    assert (wb.video_format, wb.media_max_bytes) == ("mp4", None)
    wb.log_data({"samples": [LogVideo(_moving_video(frames=4), fps=24)]}, step=2)
    wb.close()
    (step, data), _ = made
    ((path, _, fmt),) = data["samples"]
    assert step == 2 and path.endswith(".mp4") and fmt == "mp4"


@pytest.mark.parametrize("wait", [True, False])
def test_trainer_cleanup_closes_the_logger(wait):
    from flow_factory.trainers.abc import BaseTrainer

    calls = []
    trainer = object.__new__(BaseTrainer)
    trainer.logger = SimpleNamespace(close=lambda *, wait: calls.append(wait))
    trainer.cleanup(wait=wait)
    assert calls == [wait]


def test_logged_train_sample_count_is_validated():
    from flow_factory.advantage.advantage_processor import AdvantageProcessor

    with pytest.raises(ValueError, match="max_logged_samples"):
        AdvantageProcessor(accelerator=None, reward_weights={}, group_size=2, max_logged_samples=-1)


@pytest.mark.parametrize(
    "values",
    [
        {"async_logging": "yes"},
        {"media_max_pending": 0},
        {"media_max_bytes": 0},
        {"media_min_fps": 0},
        {"media_min_side": 0},
        {"max_logged_train_samples": -1},
    ],
)
def test_invalid_log_arguments(tmp_path, values):
    with pytest.raises((TypeError, ValueError)):
        LogArguments(save_dir=str(tmp_path), **values)
