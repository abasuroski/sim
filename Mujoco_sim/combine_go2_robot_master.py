"""Combine the Unitree Go2 MJCF and the robot_master arm MJCF into one model.

The arm's root body is welded to the Go2's base_link, positioned on top of
the back with a small Z offset so it sits above the torso.

Usage:
    python3 combine_go2_robot_master.py

Output:
    /home/andrea/sim/Mujoco_sim/robot_master_urdf_quadruped/mjcf/go2_with_arm.xml
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GO2_MJCF   = Path("/home/andrea/sim/Mujoco_sim/unitree_mujoco/unitree_robots/go2/go2.xml")
ARM_MJCF   = Path("/home/andrea/sim/Mujoco_sim/robot_master_urdf_quadruped/mjcf/robot_master.xml")
OUTPUT     = ARM_MJCF.parent / "go2_with_arm.xml"

# Where on the Go2 back to mount the arm (metres, in base_link frame).
# X=0 is centre, slightly toward the back; Z=0.05 sits just above the torso.
MOUNT_POS  = "-0.15 0 0.07"
# Rotate arm so its base faces forward (identity = no rotation).
MOUNT_QUAT = "1 0 0 0"

# ---------------------------------------------------------------------------
# Load both trees
# ---------------------------------------------------------------------------
go2_tree = ET.parse(GO2_MJCF)
go2_root = go2_tree.getroot()

arm_tree = ET.parse(ARM_MJCF)
arm_root = arm_tree.getroot()

# ---------------------------------------------------------------------------
# Merge <asset> sections
# ARM assets use relative paths like  assets/001_foo.obj
# Go2 assets use bare names like      base_0.obj  (resolved via meshdir)
# We prefix all arm mesh files with the arm's asset directory (absolute).
# ---------------------------------------------------------------------------
arm_asset_dir = ARM_MJCF.parent  # directory that contains the 'assets/' folder

go2_assets = go2_root.find("asset")
arm_assets = arm_root.find("asset")

if arm_assets is not None:
    for elem in arm_assets:
        # Rewrite relative file paths to absolute so Go2's meshdir doesn't
        # interfere.
        if "file" in elem.attrib:
            rel = elem.attrib["file"]
            elem.attrib["file"] = str((arm_asset_dir / rel).resolve())
        go2_assets.append(elem)

# ---------------------------------------------------------------------------
# Attach the arm body tree to base_link
# ---------------------------------------------------------------------------
# Find base_link in the Go2 worldbody
worldbody = go2_root.find("worldbody")
base_link  = worldbody.find(".//body[@name='base_link']")
if base_link is None:
    raise RuntimeError("Could not find 'base_link' in go2.xml")

# Get the arm's worldbody children (there should be one root body: 'root')
arm_worldbody = arm_root.find("worldbody")
arm_root_bodies = list(arm_worldbody)
if not arm_root_bodies:
    raise RuntimeError("No bodies found in robot_master.xml worldbody")

# Wrap arm root bodies in a mount body attached to base_link
mount = ET.SubElement(base_link, "body",
    name="arm_mount",
    pos=MOUNT_POS,
    quat=MOUNT_QUAT,
)
for body in arm_root_bodies:
    mount.append(body)

# ---------------------------------------------------------------------------
# Merge <actuator> sections
# ---------------------------------------------------------------------------
go2_actuators = go2_root.find("actuator")
arm_actuators = arm_root.find("actuator")
if arm_actuators is not None:
    for elem in arm_actuators:
        go2_actuators.append(elem)

# ---------------------------------------------------------------------------
# Merge <equality> (arm four-bar linkage constraints)
# ---------------------------------------------------------------------------
arm_equality = arm_root.find("equality")
if arm_equality is not None:
    go2_equality = go2_root.find("equality")
    if go2_equality is None:
        go2_equality = ET.SubElement(go2_root, "equality")
    for elem in arm_equality:
        go2_equality.append(elem)

# ---------------------------------------------------------------------------
# Merge <contact> exclusions
# ---------------------------------------------------------------------------
arm_contact = arm_root.find("contact")
if arm_contact is not None:
    go2_contact = go2_root.find("contact")
    if go2_contact is None:
        go2_contact = ET.SubElement(go2_root, "contact")
    for elem in arm_contact:
        go2_contact.append(elem)

# ---------------------------------------------------------------------------
# Merge <keyframe>  — combine go2 'home' qpos/ctrl with arm 'hardware_zero'
# ---------------------------------------------------------------------------
# Go2 home: 19 qpos values (7 free-joint + 12 leg joints), 12 ctrl values
# Arm hardware_zero: 12 qpos, 4 ctrl
# Combined key: concatenate both
go2_key  = go2_root.find(".//keyframe/key[@name='home']")
arm_key  = arm_root.find(".//keyframe/key[@name='hardware_zero']")

go2_keyframe = go2_root.find("keyframe")
if go2_keyframe is None:
    go2_keyframe = ET.SubElement(go2_root, "keyframe")

if go2_key is not None and arm_key is not None:
    combined_qpos = go2_key.attrib["qpos"] + " " + arm_key.attrib["qpos"]
    combined_ctrl = go2_key.attrib["ctrl"]  + " " + arm_key.attrib["ctrl"]
    # Update the existing home key in place
    go2_key.attrib["qpos"] = combined_qpos
    go2_key.attrib["ctrl"] = combined_ctrl
    go2_key.attrib["name"] = "home"

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
ET.indent(go2_root, space="  ")
ET.ElementTree(go2_root).write(OUTPUT, encoding="utf-8", xml_declaration=True)
print(f"Wrote combined MJCF: {OUTPUT}")

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
import mujoco
model = mujoco.MjModel.from_xml_path(str(OUTPUT))
print(f"Validated: {model.nbody} bodies, {model.njnt} joints, "
      f"{model.ngeom} geoms, {model.nq} qpos, {model.nu} actuators")
print(f"\nOpen with:")
print(f"  python3 -m mujoco.viewer --mjcf {OUTPUT}")
