"""
RoboSuite MujocoXMLObject wrappers around this project's actual YCB meshes
(tango_robot/assets/ycb_objects/), so the Piper+RoboSuite scene uses the same
Pear/TomatoSoupCan geometry as every other GeoEBM/consensus/LGGSN result in
the project, instead of RoboSuite's own generic look-alike objects
(CanObject, etc., which are NOT YCB meshes).

Mass/friction/scale and mesh filenames match
configs/objects/ycb_mujoco_manifest.yaml exactly. bottom/top/horizontal_radius
sites were computed directly from each mesh's own vertex bounding box (see
assets/pear.xml, assets/tomato_soup_can.xml).
"""
from pathlib import Path

from robosuite.models.objects import MujocoXMLObject

_ASSET_DIR = Path(__file__).resolve().parent / "assets"


class YcbPearObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_ASSET_DIR / "pear.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class YcbTomatoSoupCanObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_ASSET_DIR / "tomato_soup_can.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class YcbBananaObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_ASSET_DIR / "banana.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class YcbMustardBottleObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_ASSET_DIR / "mustard_bottle.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class YcbCrackerBoxObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_ASSET_DIR / "cracker_box.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class YcbPowerDrillObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_ASSET_DIR / "power_drill.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class YcbMediumClampObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            str(_ASSET_DIR / "medium_clamp.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
