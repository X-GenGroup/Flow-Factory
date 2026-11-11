# Logging

Set `log.logging_backend` to `wandb`, `swanlab`, `tensorboard`, `none`, or a fully qualified
`Logger` subclass. Only the main process logs.

## Background writer

With `log.async_logging: true` (default), each `log_data` call is formatted on the training
thread, its media are copied to owned CPU data, and the call is queued. One writer thread
converts and sends calls in FIFO order:

- Steps reach the backend in call order, so W&B's increasing-step rule holds.
- All backend API calls run on the writer thread.
- Training never waits for video encoding or upload. Scalars queued behind a media call
  appear once that call has been written.
- A writer error is raised on the training thread at the next `log_data` or at shutdown.
- Normal shutdown drains the queue and finishes the run. An interrupt discards queued calls.

`log.media_max_pending` (default 2) bounds the media-bearing calls queued or encoding at once.
The media of a further call are dropped with a warning; its scalars are still logged.
`log.async_logging: false` writes inline on the training thread.

## Video encoding and size budgets

| Backend | Container | Default per-video budget |
|---|---|---|
| W&B | MP4 (with audio when the sample has it) | none |
| SwanLab | GIF | 10 MiB, the size SwanLab accepts; at most 108 items per key and step |
| TensorBoard | GIF image summary (no `moviepy` needed) | none |

`log.media_max_bytes` overrides the budget for any backend. A video that exceeds its budget is
fitted to this video's measured encoded size, not a fixed ratio. A short probe measures bytes
per frame-pixel; each attempt then recalibrates on the size it produced.

1. Lower the frame rate by an integer stride, down to `log.media_min_fps` (default 8). The
   output frame rate is divided by the same stride, so duration and A/V timing are unchanged.
2. If that is not enough, resize, preserving the aspect ratio and even dimensions, down to a
   short side of `log.media_min_side` (default 128).
3. A video that still does not fit is skipped with a warning naming its size and limits.

`log.max_logged_train_samples` (default 30) sets the training samples logged as media per
acquisition; `0` logs none.
