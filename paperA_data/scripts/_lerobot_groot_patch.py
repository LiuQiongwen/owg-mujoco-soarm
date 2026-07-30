"""
Works around a real, unrelated bug in the installed lerobot==0.4.4 package
(not our code, not our config): `lerobot.policies.groot.groot_n1`'s
`GR00TN15Config` dataclass defines `backbone_cfg`/`action_head_cfg`/
`action_horizon`/`action_dim` with `field(init=False, ...)` and no default,
which Python's dataclass machinery rejects at class-definition time
("non-default argument ... follows default argument") because
`compute_dtype` (declared after them) has a default. This breaks every
`lerobot-*` CLI (calibrate/teleoperate/record/train/...) since they all
import `lerobot.teleoperators`/`lerobot.robots`, which transitively import
this module even though we never use the GRoot policy.

`GR00TN15Config` defines its own `__init__(self, **kwargs)` that bypasses
the dataclass-generated one entirely (see the class body), so adding
`default=None` to these four fields only satisfies the class-definition-time
validation -- it changes no runtime behaviour.

Rather than edit the installed package file directly (a site-packages edit,
avoided here per project convention -- reinstalling/upgrading lerobot would
silently revert it anyway), this pre-registers a patched copy of the module
in sys.modules before anything else imports it by name, using the same
sys.modules pre-registration technique as this project's existing
datasets/lerobot collision workaround (see real_hw_connect.py).

Usage: `import _lerobot_groot_patch` (for its side effect) before any other
lerobot import, in any script that needs to call lerobot-calibrate/-record/
-teleoperate/-train functionality programmatically.
"""
import importlib.util
import sys

_MODULE_NAME = "lerobot.policies.groot.groot_n1"


def fix_datasets_collision():
    """Ensure the real HuggingFace `datasets` package (not this project's own
    `datasets/` directory) is what `import datasets` resolves to, for the
    rest of the process lifetime.

    When Python is invoked with `-c` (or a script) from within the project
    root, `sys.path[0]` is automatically set to `''` (cwd) -- which resolves
    BEFORE site-packages and shadows the real `datasets` package with this
    project's own `datasets/__init__.py`, even before any explicit
    `sys.path.insert(PROJECT_ROOT)` call. Temporarily strip cwd-equivalent
    entries, import the real package once (caching it in sys.modules), then
    restore sys.path -- subsequent `import datasets` anywhere in the process
    reuse the cached module regardless of what's on sys.path afterward.
    """
    if "datasets" in sys.modules and hasattr(sys.modules["datasets"], "Dataset"):
        return  # already the real package
    saved = sys.path[:]
    sys.path = [p for p in sys.path if p not in ("", ".", "/lena/projects/OWG-main")]
    sys.modules.pop("datasets", None)
    import datasets  # noqa: F401  (real HF package, now cached in sys.modules)
    sys.path = saved


def apply():
    if _MODULE_NAME in sys.modules:
        return  # already imported (patched or not) -- nothing to do

    # A faithful patch (read the real file, fix the dataclass fields, exec
    # it) turned out to hit a SECOND problem: groot_n1.py itself imports
    # lerobot.policies.groot.action_head.* mid-file, which re-enters
    # lerobot.policies's __init__ cascade and tries to pull GR00TN15 back
    # out of this same module while it's still mid-exec (a real circular
    # import, not just the dataclass bug). We never use the GRoot policy at
    # all -- modeling_groot.py only needs the *name* GR00TN15 to be
    # importable (as `from lerobot.policies.groot.groot_n1 import GR00TN15`,
    # used solely as `GR00TN15.from_pretrained(...)` inside GrootPolicy,
    # which we also never instantiate) -- so register a minimal stub instead
    # of executing the real implementation.
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(_MODULE_NAME, loader=None)
    )
    module.__package__ = "lerobot.policies.groot"

    class GR00TN15:  # noqa: N801 -- matches the real class name being stubbed
        """Stub -- the real GR00TN15 is never used by this project; this
        only exists so `from lerobot.policies.groot.groot_n1 import GR00TN15`
        succeeds. See this file's module docstring."""

        @classmethod
        def from_pretrained(cls, *a, **kw):
            raise NotImplementedError(
                "GR00TN15 is stubbed out by _lerobot_groot_patch.py (unrelated "
                "upstream lerobot bug workaround) -- the real GRoot policy is "
                "not available in this environment and is not used by this project."
            )

    module.GR00TN15 = GR00TN15
    sys.modules[_MODULE_NAME] = module


fix_datasets_collision()
apply()
