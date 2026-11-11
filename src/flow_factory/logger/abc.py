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

# src/flow_factory/logger/abc.py
import queue
import threading
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional

from ..hparams import *
from ..utils.logger_utils import setup_logger
from .formatting import LogFormatter, LogImage, LogTable, LogVideo

logger = setup_logger(__name__)

_MEDIA_TYPES = (LogImage, LogVideo, LogTable)
_STOP = object()


def _has_media(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_has_media(item) for item in value)
    return isinstance(value, _MEDIA_TYPES)


def _materialize(value: Any) -> None:
    if isinstance(value, (list, tuple)):
        for item in value:
            _materialize(item)
    elif isinstance(value, _MEDIA_TYPES):
        value.materialize()


class Logger(ABC):
    """Backend-neutral logging with an optional single background writer.

    With ``log.async_logging`` the caller formats each call, copies its media to owned CPU
    data and enqueues it. One writer thread converts and sends calls in FIFO order, so steps
    reach the backend in call order and backend APIs run on that thread only. Writer errors
    are raised on the caller's next ``log_data`` or ``close``.
    """

    platform: Any
    # Container used for LogVideo on this backend.
    video_format: ClassVar[str] = "mp4"
    # Platform limit per media item and per key, None when unlimited.
    platform_media_max_bytes: ClassVar[Optional[int]] = None
    platform_media_max_items: ClassVar[Optional[int]] = None

    def __init__(self, config: Arguments):
        self.config = config
        log_args = config.log_args
        self.clean_up_freq = 10
        self._pending_cleanup: List[Dict] = []
        self.media_max_bytes = (
            log_args.media_max_bytes
            if log_args.media_max_bytes is not None
            else type(self).platform_media_max_bytes
        )
        self.media_min_fps = log_args.media_min_fps
        self.media_min_side = log_args.media_min_side
        self.media_max_pending = log_args.media_max_pending
        self.async_logging = log_args.async_logging
        self._closed = False
        self._init_platform()
        if self.async_logging:
            self._queue: "queue.Queue[Any]" = queue.Queue()
            self._lock = threading.Lock()
            self._pending_media = 0
            self._writer_error: Optional[BaseException] = None
            self._writer_error_step: Optional[int] = None
            self._stop_sent = False
            self._writer = threading.Thread(
                target=self._run_writer, name="flow-factory-logger", daemon=True
            )
            self._writer.start()

    @abstractmethod
    def _init_platform(self):
        pass

    def log_data(
        self,
        data: Dict[str, Any],
        step: int,
        keys: Optional[str] = None,
    ):
        if self._closed:
            raise RuntimeError(f"log_data(step={step}) called after the logger was closed")
        if self.async_logging:
            self._raise_writer_error()

        # 1. Process rules (Mean, Paths, wrappers) into IR
        formatted_dict = LogFormatter.format_dict(data)

        # 2. Filter keys if requested
        if keys:
            valid_keys = keys.split(",")
            formatted_dict = {k: v for k, v in formatted_dict.items() if k in valid_keys}

        if not self.async_logging:
            self._write(formatted_dict, step)
            return

        media_keys = [k for k, v in formatted_dict.items() if _has_media(v)]
        counted = False
        if media_keys:
            with self._lock:
                counted = self._pending_media < self.media_max_pending
                if counted:
                    self._pending_media += 1
                pending = self._pending_media
            if counted:
                for key in media_keys:
                    _materialize(formatted_dict[key])
            else:
                logger.warning(
                    f"Dropping media {media_keys} at step {step}: {pending} media log call(s) "
                    f"are still queued or encoding (log.media_max_pending="
                    f"{self.media_max_pending}); scalars are still logged"
                )
                for key in media_keys:
                    formatted_dict.pop(key)
        self._queue.put((formatted_dict, step, counted))

    def close(self, *, wait: bool = True) -> None:
        """Flush and finish the backend, or discard queued calls when ``wait`` is False.

        Safe to call more than once; a later ``close(wait=True)`` joins a previous
        nonblocking close.
        """
        self._closed = True
        if self.async_logging:
            if not wait:
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is not _STOP and item[2]:
                        with self._lock:
                            self._pending_media -= 1
                    self._queue.task_done()
            if not self._stop_sent:
                self._stop_sent = True
                self._queue.put(_STOP)
            if not wait:
                return
            self._writer.join()
            self._raise_writer_error()
        elif not wait:
            return
        for data in self._pending_cleanup:
            self._cleanup_temp_files(data)
        self._pending_cleanup.clear()
        self._finish_platform()

    def _run_writer(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                self._queue.task_done()
                return
            formatted_dict, step, counted = item
            try:
                if self._writer_error is None:
                    self._write(formatted_dict, step)
            except BaseException as error:  # re-raised on the caller thread
                self._writer_error = error
                self._writer_error_step = step
            finally:
                if counted:
                    with self._lock:
                        self._pending_media -= 1
                self._queue.task_done()

    def _raise_writer_error(self) -> None:
        if self._writer_error is not None:
            error, self._writer_error = self._writer_error, None
            raise RuntimeError(
                f"{type(self).__name__} background writer failed while logging step "
                f"{self._writer_error_step}"
            ) from error

    def _write(self, formatted_dict: Dict[str, Any], step: int) -> None:
        # 3. Convert IR to Platform Objects
        final_dict = {}
        for k, v in formatted_dict.items():
            v = self._limit_items(k, v, step)
            converted = self._recursive_convert(v)
            if converted is None:
                continue
            if isinstance(
                converted, dict
            ):  # for LogTable conversion returning dict, e.g., SwanlabLogger
                final_dict.update(converted)
            else:
                final_dict[k] = converted

        # 4. Actual Logging
        if final_dict:
            self._log_impl(final_dict, step)

        # 5. Cleanup temporary files periodically
        if len(self._pending_cleanup) >= self.clean_up_freq:
            first_data = self._pending_cleanup.pop(0)
            self._cleanup_temp_files(first_data)
        self._pending_cleanup.append(formatted_dict)

    def _limit_items(self, key: str, value: Any, step: int) -> Any:
        limit = type(self).platform_media_max_items
        if limit is None or not isinstance(value, (list, tuple)) or len(value) <= limit:
            return value
        logger.warning(
            f"Logging {limit} of {len(value)} media items for {key!r} at step {step}: "
            f"{type(self).__name__} keeps at most {limit} per key and step"
        )
        return value[:limit]

    def _recursive_convert(
        self, value: Any, height: Optional[int] = None, width: Optional[int] = None
    ) -> Any:
        """Recursively convert IR objects to platform objects, dropping skipped media."""
        if isinstance(value, (list, tuple)):
            converted = [self._recursive_convert(v, height, width) for v in value if v is not None]
            return [v for v in converted if v is not None]
        return self._convert_to_platform(value, height, width)

    def _video_path(
        self, value: LogVideo, height: Optional[int] = None, width: Optional[int] = None
    ) -> Optional[str]:
        """Encode a video in this backend's format, within its size budget when one applies."""
        if self.media_max_bytes is None:
            return value.get_value(type(self).video_format, height, width)
        return value.get_fitted_value(
            type(self).video_format,
            self.media_max_bytes,
            min_fps=self.media_min_fps,
            min_side=self.media_min_side,
            height=height,
            width=width,
        )

    def _cleanup_temp_files(self, data: Dict):
        for value in data.values():
            if isinstance(value, (LogImage, LogVideo, LogTable)):
                value.cleanup()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, (LogImage, LogVideo, LogTable)):
                        item.cleanup()

    def _finish_platform(self) -> None:
        """Flush and close the backend run after the writer has drained."""

    @abstractmethod
    def _convert_to_platform(
        self, value: Any, height: Optional[int] = None, width: Optional[int] = None
    ) -> Any:
        """Convert a single IR object to platform-specific object.

        Args:
            value: IR object (LogImage, LogVideo, LogTable) or pass-through value.
            height: Optional target height for resize (aspect-ratio preserved if width is None).
            width: Optional target width for resize (aspect-ratio preserved if height is None).

        Returns:
            Platform-specific object (e.g., wandb.Image, swanlab.Video), or None to skip a
            video that cannot meet the backend's size budget.
        """
        pass

    @abstractmethod
    def _log_impl(self, data: Dict, step: int):
        pass
