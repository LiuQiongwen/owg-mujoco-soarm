"""
owg_robot/viewer_utils.py

Interactive MuJoCo viewer for the OWG / SO-ARM101 pipeline.

Linux/X11/NVIDIA compatibility
-------------------------------
``mujoco.viewer.launch_passive()`` segfaults on Linux/X11/NVIDIA because it
initialises GLFW from a daemon thread.  **Importing this module** applies three
patches that make both ``launch_passive`` and ``MujocoViewer`` work correctly:

  1. ``XInitThreads()``   — enable multi-threaded X11 before GLFW opens a
                            display.  Must precede the first Xlib call.
  2. Non-daemon viewer thread — ``launch_passive`` spawns a daemon thread for
                            the render loop; Python kills it before GL cleanup
                            runs, corrupting NVIDIA's context state.  Patching
                            it to non-daemon lets Python wait for proper cleanup.
  3. ``glfw.terminate`` no-op — prevents a double-free when the render loop
                            already cleaned up GLFW before atexit fires.

After importing this module the standard pattern works unchanged:

    import owg_robot.viewer_utils          # apply patches
    import mujoco.viewer

    with mujoco.viewer.launch_passive(m, d) as v:
        for _ in range(300):
            mujoco.mj_step(m, d)
            v.sync()
            time.sleep(1/60)

``MujocoViewer`` (below) provides additional features: slowdown, pause/resume,
overlay labels and markers, and a clean callback-based API.

Quick start (MujocoViewer)
--------------------------
    from owg_robot.viewer_utils import MujocoViewer

    v = MujocoViewer(env.model, env.data, slowdown=2.0)
    v.overlay.set_phase("approach")
    v.overlay.add_marker(obj_pos, label="target")

    def loop(v):
        while v.is_alive():
            mujoco.mj_step(env.model, env.data)
            v.sync()
            v.sleep()           # honours slowdown + pause

    v.start(loop)               # blocks until viewer window is closed

Usable from
-----------
- demo.py (pass a viewer to the step hook)
- calibration scripts
- benchmark replay
- IK debugging
"""

from __future__ import annotations

import ctypes
import queue
import sys
import threading
import time
from typing import Callable, Optional

import mujoco
import mujoco.viewer as _mjv
import numpy as np

__all__ = ["MujocoViewer", "Overlay", "passive_viewer"]

# ── Linux/X11/NVIDIA compatibility patches ───────────────────────────────────
# Applied once at import time; safe to import early in any script.

def _apply_glfw_patches() -> None:
    """Patch GLFW and threading so launch_passive works on Linux/X11/NVIDIA."""
    if sys.platform != "linux":
        return

    # 1. XInitThreads — must be called before the first Xlib function.
    #    Without it, concurrent X11 calls from GLFW's daemon thread cause a
    #    segfault on NVIDIA's proprietary GLX driver.
    try:
        ctypes.cdll.LoadLibrary("libX11.so.6").XInitThreads()
    except Exception:
        pass   # not on X11 (Wayland, headless) — skip silently

    # 2. Non-daemon viewer thread — launch_passive spawns a daemon thread for
    #    the GLFW render loop.  Python kills daemon threads before Py_Finalize,
    #    leaving the NVIDIA GL context in a partially-destroyed state, which
    #    causes a segfault in the subsequent GC / C-extension teardown.
    #    Making the thread non-daemon lets Python wait for simulate.destroy()
    #    to complete before starting the shutdown sequence.
    #
    #    We only patch threads spawned by mujoco.viewer (identified by the
    #    _launch_internal target), not all threads in the process.
    import mujoco.viewer as _mjv_mod
    _orig_Thread = threading.Thread

    class _MujocoAwareThread(_orig_Thread):
        def start(self) -> None:
            target = getattr(self, "_target", None)
            if target is not None:
                qn = getattr(target, "__qualname__", "") or ""
                if "_launch_internal" in qn or "launch_passive" in qn:
                    self.daemon = False
            super().start()

    threading.Thread = _MujocoAwareThread

    # 3. glfw.terminate no-op — mujoco.viewer registers glfw.terminate as an
    #    atexit handler.  By the time atexit fires, simulate.destroy() (in the
    #    now-non-daemon thread) has already terminated GLFW.  Calling terminate
    #    again from the main thread causes an NVIDIA driver crash.
    try:
        import glfw as _glfw
        _glfw.terminate = lambda: None
    except Exception:
        pass


_apply_glfw_patches()

# seconds to wait for GLFW window to open
_HANDLE_TIMEOUT = 15.0
_WORKER_JOIN    = 3.0


# ── Overlay ───────────────────────────────────────────────────────────────────

class Overlay:
    """Thread-safe overlay: phase labels, key-value info, and 3-D marker spheres.

    Geoms are written into the viewer's ``user_scn`` (MjvScene) on every
    ``sync()`` call, so the viewer sees fresh overlay state each frame.

    Label positions are in world space.  ``phase_pos`` anchors the top-left
    text column; info lines stack downward by ``line_spacing`` (metres).

    Parameters
    ----------
    phase_pos : (3,) array-like
        World-space anchor for the phase label.  Default is above and to the
        right of the SO-ARM101 workspace.
    line_spacing : float
        Vertical gap between consecutive info lines (metres).
    """

    def __init__(
        self,
        phase_pos: tuple = (0.55, 0.0, 1.45),
        line_spacing: float = 0.07,
    ):
        self._lock        = threading.Lock()
        self._phase: str  = ""
        self._info: dict  = {}
        self._order: list = []
        self._markers: list = []

        self.phase_pos    = np.array(phase_pos, dtype=float)
        self.line_spacing = float(line_spacing)

    # ── Public setters (all thread-safe) ─────────────────────────────────────

    def set_phase(self, label: str) -> None:
        """Update the phase label shown at the top of the overlay column."""
        with self._lock:
            self._phase = str(label)

    def add_info(self, key: str, value) -> None:
        """Add or update a key-value line below the phase label."""
        with self._lock:
            if key not in self._info:
                self._order.append(key)
            self._info[key] = str(value)

    def clear_info(self) -> None:
        """Remove all info lines (phase label is preserved)."""
        with self._lock:
            self._info.clear()
            self._order.clear()

    def add_marker(
        self,
        pos,
        rgba: tuple = (1.0, 0.2, 0.0, 0.7),
        size: float = 0.02,
        label: str = "",
    ) -> None:
        """Add a 3-D sphere marker at *pos*."""
        with self._lock:
            self._markers.append(
                (np.array(pos, dtype=float), np.array(rgba, dtype=float),
                 float(size), str(label))
            )

    def clear_markers(self) -> None:
        """Remove all marker spheres."""
        with self._lock:
            self._markers.clear()

    # ── Internal: write into MjvScene ────────────────────────────────────────

    def apply(self, scn: mujoco.MjvScene) -> None:
        """Rebuild overlay geoms in *scn*.  Resets scn.ngeom to 0 first.

        Called automatically by ``MujocoViewer.sync()``.
        """
        with self._lock:
            phase   = self._phase
            info    = [(k, self._info[k]) for k in self._order]
            markers = list(self._markers)

        scn.ngeom = 0
        n = 0
        lim = scn.maxgeom

        def _label(pos: np.ndarray, text: str) -> None:
            nonlocal n
            if n >= lim or not text:
                return
            try:
                g = scn.geoms[n]
                mujoco.mjv_initGeom(
                    g,
                    mujoco.mjtGeom.mjGEOM_LABEL,
                    np.zeros(3),
                    pos,
                    np.eye(3).ravel(),
                    np.ones(4, dtype=float),
                )
                g.label = text[:99]
                n += 1
            except Exception:
                pass

        def _sphere(pos: np.ndarray, rgba: np.ndarray, sz: float, lbl: str) -> None:
            nonlocal n
            if n >= lim:
                return
            try:
                g = scn.geoms[n]
                mujoco.mjv_initGeom(
                    g,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.full(3, sz, dtype=float),
                    pos,
                    np.eye(3).ravel(),
                    rgba,
                )
                if lbl:
                    g.label = lbl[:99]
                n += 1
            except Exception:
                pass

        if phase:
            _label(self.phase_pos.copy(), f"[{phase}]")
        for i, (k, v) in enumerate(info):
            p = self.phase_pos - [0.0, 0.0, (i + 1) * self.line_spacing]
            _label(p, f"{k}: {v}")
        for pos, rgba, sz, lbl in markers:
            _sphere(pos, rgba, sz, lbl)

        scn.ngeom = n


# ── MujocoViewer ──────────────────────────────────────────────────────────────

class MujocoViewer:
    """Interactive MuJoCo viewer that runs GLFW on the calling thread.

    The GLFW render loop blocks ``start()`` on the calling (main) thread.
    Physics and control code run in a worker daemon thread that is started
    automatically when ``physics_fn`` is passed to ``start()``.

    Parameters
    ----------
    model, data :
        The MuJoCo model/data pair shared with the worker thread.
    slowdown : float
        Real-time multiplier for ``sleep()``.  1.0 = real time; 3.0 = 3× slower.
    overlay : Overlay, optional
        Shared overlay instance.  One is created automatically if not provided.
    show_left_ui, show_right_ui : bool
        Toggle the MuJoCo viewer side panels.

    Thread safety
    -------------
    ``sync()`` and all overlay setters are thread-safe and intended to be called
    from the worker thread.  ``stop()``, ``pause()``, and ``resume()`` are also
    thread-safe.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data:  mujoco.MjData,
        slowdown:      float = 1.0,
        overlay:       Optional[Overlay] = None,
        show_left_ui:  bool = True,
        show_right_ui: bool = True,
    ):
        self.model    = model
        self.data     = data
        self.slowdown = max(1e-3, float(slowdown))
        self.overlay  = overlay if overlay is not None else Overlay()

        self._lui = show_left_ui
        self._rui = show_right_ui

        self._handle: Optional[object] = None   # mujoco.viewer.Handle
        self._stopped = threading.Event()
        self._paused  = threading.Event()
        self._paused.set()   # set = not paused → sleep() returns immediately
        self._worker: Optional[threading.Thread] = None

    # ── Status ────────────────────────────────────────────────────────────────

    def is_alive(self) -> bool:
        """True while the viewer window is open and stop() has not been called."""
        if self._stopped.is_set():
            return False
        h = self._handle
        return True if h is None else h.is_running()

    # ── Control ───────────────────────────────────────────────────────────────

    def pause(self) -> None:
        """Pause: subsequent ``sleep()`` calls block until ``resume()``."""
        self._paused.clear()

    def resume(self) -> None:
        """Resume after ``pause()``."""
        self._paused.set()

    def stop(self) -> None:
        """Signal the viewer and worker to stop.  Safe to call from any thread."""
        self._stopped.set()
        h = self._handle
        if h is not None:
            try:
                h.close()
            except Exception:
                pass

    # Aliases matching requirements
    def start_viewer(self, physics_fn: Optional[Callable] = None) -> None:
        """Alias for start()."""
        self.start(physics_fn)

    def stop_viewer(self) -> None:
        """Alias for stop()."""
        self.stop()

    # ── Sync and sleep ────────────────────────────────────────────────────────

    def sync(self) -> None:
        """Push model/data state and overlay to the viewer.

        Call this from the worker thread after updating ``data`` (e.g. after
        ``mujoco.mj_step()``).  No-op if the viewer is not yet ready.
        """
        h = self._handle
        if h is None:
            return
        scn = h.user_scn   # MjvScene for user geoms (public property)
        if scn is not None:
            self.overlay.apply(scn)
        if h.is_running():
            h.sync()

    def sleep(self, dt: Optional[float] = None) -> None:
        """Sleep for one timestep × slowdown.  Blocks indefinitely if paused.

        Parameters
        ----------
        dt : float, optional
            Override the sleep duration (seconds, before slowdown scaling).
            Defaults to ``model.opt.timestep``.
        """
        self._paused.wait()   # blocks until resumed
        seconds = (dt if dt is not None else self.model.opt.timestep) * self.slowdown
        if seconds > 0:
            time.sleep(seconds)

    def wait_if_paused(self) -> None:
        """Block until ``resume()`` is called.  No-op if not paused."""
        self._paused.wait()

    # ── Start ─────────────────────────────────────────────────────────────────

    def start(
        self,
        physics_fn: Optional[Callable[["MujocoViewer"], None]] = None,
    ) -> None:
        """Launch the viewer.  **Blocks the calling thread** until closed.

        Parameters
        ----------
        physics_fn : callable, optional
            ``fn(viewer)`` — executed in a daemon thread.  Should loop while
            ``viewer.is_alive()``, update ``data``, and call ``viewer.sync()``.
            The viewer closes automatically when *physics_fn* returns.

        Raises
        ------
        RuntimeError
            If the viewer handle is not delivered within ``_HANDLE_TIMEOUT``
            seconds (GLFW failed to open).
        """
        hq = queue.Queue(1)
        self._stopped.clear()
        self._handle = None

        def _worker() -> None:
            try:
                h = hq.get(timeout=_HANDLE_TIMEOUT)
            except queue.Empty:
                print("[MujocoViewer] ERROR: timed out waiting for GLFW handle",
                      file=sys.stderr)
                self._stopped.set()
                return

            self._handle = h

            if physics_fn is not None:
                try:
                    physics_fn(self)
                except Exception as exc:
                    import traceback
                    print(f"[MujocoViewer] physics_fn raised {type(exc).__name__}: {exc}",
                          file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                finally:
                    self.stop()

        self._worker = threading.Thread(
            target=_worker, daemon=True, name="mujoco-viewer-physics"
        )
        self._worker.start()

        # GLFW render loop runs on this (calling) thread — blocks here.
        _mjv._launch_internal(
            self.model, self.data,
            run_physics_thread=False,
            handle_return=hq,
            show_left_ui=self._lui,
            show_right_ui=self._rui,
        )

        # Viewer closed (window destroyed or stop() called).
        self._stopped.set()
        if self._worker is not None:
            self._worker.join(timeout=_WORKER_JOIN)

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "MujocoViewer":
        return self

    def __exit__(self, *_) -> None:
        self.stop()


# ── Convenience function ──────────────────────────────────────────────────────

def passive_viewer(
    model: mujoco.MjModel,
    data:  mujoco.MjData,
    physics_fn: Optional[Callable[["MujocoViewer"], None]] = None,
    slowdown: float = 1.0,
    overlay:  Optional[Overlay] = None,
    **kw,
) -> MujocoViewer:
    """Launch a passive viewer and block until the window is closed.

    Equivalent to creating a ``MujocoViewer`` and calling ``start()``.

    Parameters
    ----------
    model, data :
        MuJoCo simulation objects.
    physics_fn : callable, optional
        ``fn(viewer)`` — runs in a worker thread.
    slowdown : float
        Slow-motion multiplier for ``viewer.sleep()``.
    overlay : Overlay, optional
        Shared overlay for phase labels and markers.
    **kw :
        Extra kwargs forwarded to ``MujocoViewer`` (``show_left_ui``, etc.).

    Returns
    -------
    MujocoViewer
        The closed viewer (useful for inspecting final state).

    Example
    -------
    ::
        from owg_robot.viewer_utils import passive_viewer

        def run(v):
            while v.is_alive():
                mujoco.mj_step(model, data)
                v.sync()
                v.sleep()

        passive_viewer(model, data, run, slowdown=3.0)
    """
    v = MujocoViewer(model, data, slowdown=slowdown, overlay=overlay, **kw)
    v.start(physics_fn)
    return v
