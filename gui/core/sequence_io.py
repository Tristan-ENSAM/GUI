# -*- coding: utf-8 -*-
"""
Lightweight readers for the raw experimental streams.

`ImageSequence` gives uniform random access (`n_frames`, `frame(i)`) to an
image stream, whatever the backing store:
  * an in-memory numpy array (n_frames, H, W[, C])  -- used by tests,
  * a directory of image files (sorted),
  * a multi-page TIFF or a video file (read lazily via imageio).

`load_forces` reads a text/CSV force file into (t, Fc, Ff) arrays.

File backends are best-effort: if a format can't be read, a clear
RuntimeError is raised so the UI can report it. The array backend has no
external dependency and is fully exercised by the tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np

_IMAGE_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".npy")


class ImageSequence:
    """Random-access view over an image stream.

    Construct via `from_array` (in-memory) or `from_path` (files). Frames are
    returned as 2-D (grayscale) or 3-D (H, W, C) numpy arrays. `fps` and `t0`
    let callers turn a frame index into a time."""

    def __init__(self, fps: float = 1.0, t0: float = 0.0, source: str = ""):
        self.fps = float(fps) if fps else 1.0
        self.t0 = float(t0)
        self.source = source
        self._array: Optional[np.ndarray] = None   # (n, H, W[, C])
        self._files: list[Path] = []               # one image per frame
        self._reader = None                         # imageio reader (tiff/video)
        self._reader_len = 0

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_array(cls, arr, fps: float = 1.0, t0: float = 0.0,
                   source: str = "<array>") -> "ImageSequence":
        seq = cls(fps=fps, t0=t0, source=source)
        a = np.asarray(arr)
        if a.ndim == 2:
            a = a[None, ...]            # single frame -> (1, H, W)
        if a.ndim not in (3, 4):
            raise ValueError("image array must be (n,H,W) or (n,H,W,C)")
        seq._array = a
        return seq

    @classmethod
    def from_path(cls, path, fps: float = 1.0, t0: float = 0.0) -> "ImageSequence":
        p = Path(path)
        seq = cls(fps=fps, t0=t0, source=str(p))
        if not p.exists():
            raise RuntimeError("Path does not exist: %s" % p)
        if p.is_dir():
            files = sorted(
                q for q in p.iterdir()
                if q.suffix.lower() in _IMAGE_EXTS)
            if not files:
                raise RuntimeError("No image files in directory: %s" % p)
            seq._files = files
            return seq
        if p.suffix.lower() == ".npy":
            return cls.from_array(np.load(p), fps=fps, t0=t0, source=str(p))
        if p.suffix.lower() == ".npz":
            with np.load(p) as z:
                key = "frames" if "frames" in z else list(z.keys())[0]
                return cls.from_array(z[key], fps=fps, t0=t0, source=str(p))
        # multi-page TIFF or video -> lazy reader via imageio
        try:
            import imageio.v3 as iio
            # number of frames
            try:
                props = iio.improps(p, plugin=None)
                n = int(props.n_images) if props.n_images else 0
            except Exception:
                n = 0
            seq._reader = (iio, p)
            seq._reader_len = n
            if n == 0:
                # Fall back: read all into memory (small files only).
                stack = iio.imread(p)
                return cls.from_array(stack, fps=fps, t0=t0, source=str(p))
            return seq
        except Exception as e:
            raise RuntimeError("Unsupported image stream %s (%s)" % (p, e))

    # -- access -----------------------------------------------------------
    @property
    def n_frames(self) -> int:
        if self._array is not None:
            return int(self._array.shape[0])
        if self._files:
            return len(self._files)
        if self._reader is not None:
            return int(self._reader_len)
        return 0

    def frame(self, i: int) -> np.ndarray:
        n = self.n_frames
        if n == 0:
            raise IndexError("empty sequence")
        i = max(0, min(int(i), n - 1))
        if self._array is not None:
            return np.asarray(self._array[i])
        if self._files:
            import imageio.v3 as iio
            return np.asarray(iio.imread(self._files[i]))
        if self._reader is not None:
            iio, p = self._reader
            return np.asarray(iio.imread(p, index=i))
        raise IndexError("no backend")

    def time(self, i: int) -> float:
        return self.t0 + int(i) / self.fps


def load_forces(path, fps: float = 1.0, col_t: int = -1,
                col_fc: int = 0, col_ff: int = 1):
    """Read a force file into (t, Fc, Ff) 1-D arrays.

    `col_t = -1` means there is no time column: time is derived as
    index / fps. Accepts whitespace- or comma-separated text/CSV. Lines that
    can't be parsed as numbers (headers) are skipped."""
    p = Path(path)
    if not p.exists():
        raise RuntimeError("Force file does not exist: %s" % p)
    # Try comma first, then whitespace.
    data = None
    for delim in (",", None):
        try:
            data = np.genfromtxt(p, delimiter=delim, comments="#")
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if np.isfinite(data).any():
                break
        except Exception:
            data = None
    if data is None or data.size == 0 or not np.isfinite(data).any():
        raise RuntimeError("Could not parse force file: %s" % p)
    # Drop fully-NaN rows (e.g. a header line).
    data = data[np.isfinite(data).any(axis=1)]
    fc = data[:, col_fc]
    ff = data[:, col_ff]
    if col_t is not None and col_t >= 0 and col_t < data.shape[1]:
        t = data[:, col_t]
    else:
        t = np.arange(data.shape[0], dtype=float) / (float(fps) if fps else 1.0)
    return t, fc, ff
