"""Convert robot_master quadruped-mount URDF and OBJ assets to a loadable MJCF.

Adapted from convert_robot_master_to_mjcf.py for the Unitree-mounted variant:
  - Source URDF uses .obj meshes (no GLTF conversion needed).
  - The table planar-mate locked joints are gone; the base is rigid.
  - Output goes to robot_master_urdf_quadruped/mjcf/.

Usage:
    python convert_robot_master_quadruped_to_mjcf.py
"""

from __future__ import annotations

import math
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import mujoco

from robot_master_configuration import (
    AK40_MOTOR_TORQUE_LIMIT_NM,
    HARDWARE_ZERO_ACTUATOR_CTRL_RAD,
    HARDWARE_ZERO_KEYFRAME_NAME,
    JOINT_RANGE_OVERRIDES_RAD,
    REVOLUTE_2_REVERSED_REF_RAD,
    hardware_zero_joint_positions,
)


ROOT      = Path(__file__).resolve().parent
URDF      = Path("C:/school/sim/robot_master_copy_1/urdf/robot_master_copy_1.urdf")
MESH_DIR  = URDF.parent.parent / "meshes"
OUTPUT_DIR = ROOT / "robot_master_urdf_quadruped" / "mjcf"
ASSET_DIR  = OUTPUT_DIR / "assets"
MJCF       = OUTPUT_DIR / "robot_master.xml"

# Package prefix used by the OnShape URDF exporter for this assembly.
PACKAGE_PREFIX = "package://robot_master_copy_1/meshes/"

# CAD-only helper frames and a loop-closure link have near-zero inertia.
MIN_MASS      = 1e-4
MIN_INERTIA   = 1e-9
HARDWARE_ZERO_SETTLE_TIME_S = 10.0

# CubeMars motor torque limits (N·m at the output shaft).
AK60_6_PEAK_TORQUE_NM  = 9.0
AK70_10_PEAK_TORQUE_NM = 24.8
AK40_10_PEAK_TORQUE_NM = AK40_MOTOR_TORQUE_LIMIT_NM

# AK40 gear ratio (60-tooth output / 48-tooth pinion).
AK40_10_REDUCTION            = 60.0 / 48.0
AK40_10_OUTPUT_PEAK_TORQUE_NM = AK40_10_PEAK_TORQUE_NM * AK40_10_REDUCTION
AK40_10_JOINT_RANGE_RAD      = JOINT_RANGE_OVERRIDES_RAD["revolute_4"]

# Position-control stiffnesses (kp = peak torque / 1 rad error).
AK60_6_POSITION_KP            = AK60_6_PEAK_TORQUE_NM
AK70_10_POSITION_KP           = AK70_10_PEAK_TORQUE_NM
AK40_10_POSITION_KP           = AK40_10_OUTPUT_PEAK_TORQUE_NM
AK40_10_ACTUATOR_POSITION_KP  = AK40_10_POSITION_KP / AK40_10_REDUCTION**2

CONTINUOUS_JOINT_SLIDER_RANGE = "-6.283185307 6.283185307"

# revolute_2 is reversed so its hard stop is at 0 and travel is positive.
REVERSED_JOINT_REFS_RAD = {"revolute_2": REVOLUTE_2_REVERSED_REF_RAD}
REVOLUTE_2_CTRL_RANGE   = f"0 {REVOLUTE_2_REVERSED_REF_RAD + 6.28319:.10g}"

# Visual meshes are kept collision-free; capsules proxy the articulated links.
VISUAL_COLLISION_TYPE    = "0"
COLLISION_PROXY_TYPE     = "2"
COLLISION_PROXY_RADIUS_M = 0.015

# Capsule endpoints measured from the URDF reference pose (same arm geometry).
COLLISION_CAPSULES = (
    ("collision_revolute_3_to_4", "35t_htd_custom_pulley_1", "0 0 0 0.01158696 -0.17936519 0.08813909"),
    ("collision_revolute_4_to_6", "part_8_7",                "0 0 0 0.00245809 -0.0992568 -0.0119186"),
    ("collision_revolute_6_to_8", "part_1_35",               "0 0 0 -0.01149162 0.04866149 0"),
    ("collision_revolute_5_to_7", "part_8_4",                "0 0 0 -0.01826471 -0.10292344 0.01456664"),
    ("collision_revolute_7_to_9", "part_1_45",               "0 0 0 0.0130283 0.0458512 0.0150973"),
)

# Neighbouring links share a hinge pivot — exclude their capsule overlap.
COLLISION_EXCLUDE_PAIRS = (
    ("35t_htd_custom_pulley_1", "part_8_7"),
    ("part_8_7",                "part_1_35"),
    ("part_1_35",               "part_8_2"),
    ("part_8_4",                "part_1_45"),
    ("part_1_45",               "part_8_1"),
)

# Joints to lock (treat as fixed).  The old table planar-mate joints
# (planar_1, revolute_10/11) are absent in this URDF; only the loop-closure
# helpers remain.  Fixed joints in the URDF are already skipped automatically.
LOCKED_JOINTS: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def floats(value: str | None, count: int, default: float = 0.0) -> list[float]:
    if value is None:
        return [default] * count
    values = [float(item) for item in value.split()]
    if len(values) != count:
        raise ValueError(f"Expected {count} values, got {value!r}")
    return values


def format_values(values: list[float]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def rpy_to_quat(rpy: str | None) -> str:
    roll, pitch, yaw = floats(rpy, 3)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return format_values([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def origin_attributes(origin: ET.Element | None) -> dict[str, str]:
    if origin is None:
        return {}
    attributes: dict[str, str] = {}
    if origin.get("xyz") is not None:
        attributes["pos"] = origin.attrib["xyz"]
    if origin.get("rpy") is not None:
        attributes["quat"] = rpy_to_quat(origin.attrib["rpy"])
    return attributes


def inertial_attributes(link: ET.Element) -> dict[str, str]:
    inertial = link.find("inertial")
    if inertial is None:
        return {"pos": "0 0 0", "mass": format_values([MIN_MASS]),
                "diaginertia": format_values([MIN_INERTIA] * 3)}
    mass     = max(float(inertial.find("mass").attrib["value"]), MIN_MASS)
    inertia  = inertial.find("inertia")
    diagonal = [max(abs(float(inertia.attrib[k])), MIN_INERTIA) for k in ("ixx", "iyy", "izz")]
    largest  = max(diagonal)
    if sum(diagonal) - largest <= largest:
        diagonal = [max(v, largest * 0.5001) for v in diagonal]
    attributes = {"pos": "0 0 0", "mass": format_values([mass]),
                  "diaginertia": format_values(diagonal)}
    attributes.update(origin_attributes(inertial.find("origin")))
    return attributes


def joint_element(joint: ET.Element) -> ET.Element:
    joint_type = joint.attrib["type"]
    attributes: dict[str, str] = {"name": joint.attrib["name"]}
    if joint_type == "revolute":
        attributes["type"] = "hinge"
    elif joint_type == "continuous":
        attributes["type"] = "hinge"
        attributes["limited"] = "false"
    elif joint_type == "prismatic":
        attributes["type"] = "slide"
    else:
        raise ValueError(f"Unexpected movable joint type: {joint_type}")

    axis = joint.find("axis")
    if axis is not None:
        attributes["axis"] = axis.attrib["xyz"]
    override_range = JOINT_RANGE_OVERRIDES_RAD.get(joint.attrib["name"])
    limit = joint.find("limit")
    if override_range is not None:
        attributes["range"] = format_values(list(override_range))
    elif limit is not None and joint_type != "continuous":
        attributes["range"] = f"{limit.attrib['lower']} {limit.attrib['upper']}"

    reversed_ref = REVERSED_JOINT_REFS_RAD.get(joint.attrib["name"])
    if reversed_ref is not None:
        attributes["axis"] = format_values([0.0 - float(v) for v in attributes["axis"].split()])
        attributes["ref"]  = f"{reversed_ref:g}"
        lower, upper = (float(v) for v in attributes["range"].split())
        attributes["range"] = format_values([reversed_ref - upper, reversed_ref - lower])
    return ET.Element("joint", attributes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    robot  = ET.parse(URDF).getroot()
    links  = {link.attrib["name"]: link for link in robot.findall("link")}
    joints = robot.findall("joint")
    children:    dict[str, list[ET.Element]] = defaultdict(list)
    child_joints: dict[str, ET.Element]      = {}
    for joint in joints:
        parent = joint.find("parent").attrib["link"]
        child  = joint.find("child").attrib["link"]
        children[parent].append(joint)
        child_joints[child] = joint
    root_links = sorted(set(links) - set(child_joints))
    if len(root_links) != 1:
        raise ValueError(f"Expected exactly one URDF root link, got {root_links}")

    # Collect unique OBJ source filenames and copy them into the asset dir.
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    source_files = sorted({
        Path(mesh.attrib["filename"].removeprefix(PACKAGE_PREFIX)).name
        for link in links.values()
        for mesh in link.findall("visual/geometry/mesh")
    })
    mesh_assets: dict[str, tuple[str, str]] = {}   # source_name -> (asset_name, asset_file)
    for index, source_name in enumerate(source_files, start=1):
        src  = MESH_DIR / source_name
        dest_name = f"{index:03d}_{src.stem}.obj"
        dest = ASSET_DIR / dest_name
        shutil.copy2(src, dest)
        mesh_assets[source_name] = (f"mesh_{index:03d}", f"assets/{dest_name}")

    # Build MJCF tree.
    mujoco_xml = ET.Element("mujoco", model=robot.attrib.get("name", "robot_master"))
    ET.SubElement(mujoco_xml, "compiler", angle="radian", autolimits="true")
    ET.SubElement(mujoco_xml, "option", integrator="implicitfast")
    visual = ET.SubElement(mujoco_xml, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    assets = ET.SubElement(mujoco_xml, "asset")
    for source_name in source_files:
        asset_name, asset_file = mesh_assets[source_name]
        ET.SubElement(assets, "mesh", name=asset_name, file=asset_file)

    worldbody      = ET.SubElement(mujoco_xml, "worldbody")
    body_elements: dict[str, ET.Element] = {}
    model_joint_names: list[str]         = []

    def add_body(link_name: str, parent: ET.Element) -> None:
        link         = links[link_name]
        attributes   = {"name": link_name}
        parent_joint = child_joints.get(link_name)
        if parent_joint is not None:
            attributes.update(origin_attributes(parent_joint.find("origin")))
        body = ET.SubElement(parent, "body", attributes)
        body_elements[link_name] = body
        ET.SubElement(body, "inertial", inertial_attributes(link))
        if (
            parent_joint is not None
            and parent_joint.attrib["type"] != "fixed"
            and parent_joint.attrib["name"] not in LOCKED_JOINTS
        ):
            body.append(joint_element(parent_joint))
            model_joint_names.append(parent_joint.attrib["name"])

        for visual_element in link.findall("visual"):
            mesh = visual_element.find("geometry/mesh")
            if mesh is None:
                continue
            source_name = Path(mesh.attrib["filename"].removeprefix(PACKAGE_PREFIX)).name
            asset_name, _ = mesh_assets[source_name]
            geom_attributes: dict[str, str] = {
                "type": "mesh", "mesh": asset_name,
                "contype": VISUAL_COLLISION_TYPE,
                "conaffinity": VISUAL_COLLISION_TYPE,
            }
            geom_attributes.update(origin_attributes(visual_element.find("origin")))
            color = visual_element.find("material/color")
            if color is not None and color.get("rgba") is not None:
                geom_attributes["rgba"] = color.attrib["rgba"]
            body.append(ET.Element("geom", geom_attributes))

        for child_joint in children[link_name]:
            add_body(child_joint.find("child").attrib["link"], body)

    add_body(root_links[0], worldbody)

    # Collision capsules along articulated link centrelines.
    for geom_name, body_name, fromto in COLLISION_CAPSULES:
        ET.SubElement(
            body_elements[body_name], "geom",
            name=geom_name, type="capsule", fromto=fromto,
            size=f"{COLLISION_PROXY_RADIUS_M:g}",
            contype=COLLISION_PROXY_TYPE, conaffinity=COLLISION_PROXY_TYPE,
            group="3", rgba="0.2 0.8 0.2 0.18",
        )

    contact = ET.SubElement(mujoco_xml, "contact")
    for body1, body2 in COLLISION_EXCLUDE_PAIRS:
        ET.SubElement(contact, "exclude", body1=body1, body2=body2)

    # Four-bar linkage closure constraints (same as the table version).
    equality = ET.SubElement(mujoco_xml, "equality")
    ET.SubElement(equality, "weld",
        name="revolute_14_closed_loop",
        body1="part_3", body2="part_3__1__loop_closure",
        solref="0.004 1", solimp="0.999 0.999 0.001", torquescale="0.05")
    for name, j1, j2 in (
        ("revolute_9_opposes_revolute_7",  "revolute_9",              "revolute_7"),
        ("revolute_6_opposes_revolute_8",  "revolute_6",              "revolute_8"),
        ("revolute_5_opposes_revolute_4",  "revolute_5",              "revolute_4"),
        ("revolute_13_opposes_revolute_8", "revolute_13",             "revolute_8"),
        ("revolute_14_opposes_revolute_8", "revolute_14_loop_closure","revolute_8"),
    ):
        ET.SubElement(equality, "joint", name=name, joint1=j1, joint2=j2,
            polycoef="0 -1 0 0 0", solref="0.004 1", solimp="0.999 0.999 0.001")

    # Actuators.
    actuators = ET.SubElement(mujoco_xml, "actuator")
    ET.SubElement(actuators, "position",
        name="ak60_revolute_1", joint="revolute_1", gear="1",
        kp=f"{AK60_6_POSITION_KP:g}", dampratio="1",
        ctrllimited="false", ctrlrange=CONTINUOUS_JOINT_SLIDER_RANGE,
        forcelimited="true", forcerange=f"{-AK60_6_PEAK_TORQUE_NM:g} {AK60_6_PEAK_TORQUE_NM:g}")
    ET.SubElement(actuators, "position",
        name="ak70_revolute_2", joint="revolute_2", gear="1",
        kp=f"{AK70_10_POSITION_KP:g}", dampratio="1",
        ctrllimited="true", ctrlrange=REVOLUTE_2_CTRL_RANGE,
        forcelimited="true", forcerange=f"{-AK70_10_PEAK_TORQUE_NM:g} {AK70_10_PEAK_TORQUE_NM:g}")
    ET.SubElement(actuators, "position",
        name="ak70_revolute_3", joint="revolute_3", gear="1",
        kp=f"{AK70_10_POSITION_KP:g}", dampratio="1",
        ctrllimited="false", ctrlrange=CONTINUOUS_JOINT_SLIDER_RANGE,
        forcelimited="true", forcerange=f"{-AK70_10_PEAK_TORQUE_NM:g} {AK70_10_PEAK_TORQUE_NM:g}")
    ET.SubElement(actuators, "position",
        name="ak40_revolute_4", joint="revolute_4",
        gear=f"{AK40_10_REDUCTION:g}",
        kp=f"{AK40_10_ACTUATOR_POSITION_KP:g}", dampratio="1",
        ctrllimited="true",
        ctrlrange=format_values([v * AK40_10_REDUCTION for v in AK40_10_JOINT_RANGE_RAD]),
        forcelimited="true", forcerange=f"{-AK40_10_PEAK_TORQUE_NM:g} {AK40_10_PEAK_TORQUE_NM:g}")

    # Hardware-zero keyframe.
    hardware_zero_ctrl = [
        HARDWARE_ZERO_ACTUATOR_CTRL_RAD["ak60_revolute_1"],
        HARDWARE_ZERO_ACTUATOR_CTRL_RAD["ak70_revolute_2"],
        HARDWARE_ZERO_ACTUATOR_CTRL_RAD["ak70_revolute_3"],
        HARDWARE_ZERO_ACTUATOR_CTRL_RAD["ak40_revolute_4"],
    ]
    keyframe = ET.SubElement(mujoco_xml, "keyframe")
    hardware_zero_key = ET.SubElement(
        keyframe, "key",
        name=HARDWARE_ZERO_KEYFRAME_NAME,
        qpos=format_values(hardware_zero_joint_positions(model_joint_names)),
        ctrl=format_values(hardware_zero_ctrl),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ET.indent(mujoco_xml, space="  ")
    ET.ElementTree(mujoco_xml).write(MJCF, encoding="utf-8", xml_declaration=True)

    # Solve the passive four-bar joints for the hardware-zero pose.
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data  = mujoco.MjData(model)
    data.ctrl[:] = hardware_zero_ctrl
    mujoco.mj_forward(model, data)
    for _ in range(round(HARDWARE_ZERO_SETTLE_TIME_S / model.opt.timestep)):
        mujoco.mj_step(model, data)
    if not all(math.isfinite(v) for v in data.qpos) or not all(math.isfinite(v) for v in data.qvel):
        raise RuntimeError("The hardware_zero keyframe solve became non-finite")
    hardware_zero_key.set("qpos", format_values([float(v) for v in data.qpos]))
    ET.indent(mujoco_xml, space="  ")
    ET.ElementTree(mujoco_xml).write(MJCF, encoding="utf-8", xml_declaration=True)

    # Validate.
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data  = mujoco.MjData(model)
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HARDWARE_ZERO_KEYFRAME_NAME)
    if keyframe_id < 0:
        raise RuntimeError(f"Generated MJCF is missing the {HARDWARE_ZERO_KEYFRAME_NAME!r} keyframe")
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)
    visual_count = sum(len(link.findall("visual")) for link in links.values())
    print(f"Wrote: {MJCF}")
    print(f"Copied {len(mesh_assets)} OBJ meshes.")
    print(f"Validated MJCF: {model.nbody} bodies, {model.njnt} joints, {model.ngeom} geoms, {model.nmesh} meshes.")
    print(f"Solved {HARDWARE_ZERO_KEYFRAME_NAME!r} over {HARDWARE_ZERO_SETTLE_TIME_S:g} s of constrained dynamics.")
    print(
        f"Mapped {visual_count} URDF visual elements plus {len(COLLISION_CAPSULES)} collision capsules "
        f"and {len(COLLISION_EXCLUDE_PAIRS)} hinge-pair exclusions."
    )


if __name__ == "__main__":
    main()
