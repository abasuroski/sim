"""Combine the robot_master arm MJCF and the Unitree Go2 MJCF into one model.

Starts from robot_master.xml as the base, appends the Go2 quadruped body
into its worldbody, hides the table, and outputs go2_with_arm.xml.

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
ARM_MJCF = Path("/home/andrea/sim/Mujoco_sim/robot_master_urdf_table/mjcf/robot_master.xml")
GO2_MJCF = Path("/home/andrea/sim/Mujoco_sim/unitree_mujoco/unitree_robots/go2/go2.xml")
OUTPUT   = Path("/home/andrea/sim/Mujoco_sim/robot_master_urdf_quadruped/mjcf/go2_with_arm.xml")

# Where to mount the arm on the Go2 back (in base_link frame)
MOUNT_POS  = "0.15 -0.08 0.09"
MOUNT_QUAT = "1 0 0 0"

# ---------------------------------------------------------------------------
# Load both trees — arm is the base
# ---------------------------------------------------------------------------
arm_tree = ET.parse(ARM_MJCF)
arm_root = arm_tree.getroot()

go2_tree = ET.parse(GO2_MJCF)
go2_root = go2_tree.getroot()

arm_asset_dir = ARM_MJCF.parent  # contains the 'assets/' folder

# ---------------------------------------------------------------------------
# 1. Hide the table: remove geoms from part_1_13, zero its position
# ---------------------------------------------------------------------------
table_body = arm_root.find(".//body[@name='part_1_13']")
if table_body is not None:
    for g in table_body.findall("geom"):
        table_body.remove(g)
    table_body.attrib["pos"] = "0 0 0"
    print("  Table hidden (part_1_13 geoms removed)")

# Hide the blue base plate (part_1_12)
base_plate = arm_root.find(".//body[@name='part_1_12']")
if base_plate is not None:
    for g in base_plate.findall("geom"):
        base_plate.remove(g)
    print("  Blue base plate hidden (part_1_12 geoms removed)")

# Zero planar_1_1 offset (was relative to table centre)
planar = arm_root.find(".//body[@name='planar_1_1']")
if planar is not None:
    planar.attrib["pos"] = "0 0 0"
    print("  planar_1_1 offset zeroed")

# ---------------------------------------------------------------------------
# 2. Rewrite arm asset paths to absolute so Go2's meshdir doesn't interfere
# ---------------------------------------------------------------------------
arm_assets = arm_root.find("asset")
if arm_assets is not None:
    for elem in arm_assets:
        if "file" in elem.attrib:
            elem.attrib["file"] = str((arm_asset_dir / elem.attrib["file"]).resolve())

# ---------------------------------------------------------------------------
# 3. Merge Go2 assets into arm's <asset>
# ---------------------------------------------------------------------------
go2_assets = go2_root.find("asset")
go2_meshdir = GO2_MJCF.parent  # bare mesh names resolve relative to go2.xml

if go2_assets is not None:
    for elem in go2_assets:
        # Rewrite bare go2 mesh paths to absolute
        if "file" in elem.attrib:
            elem.attrib["file"] = str((go2_meshdir / "assets" / elem.attrib["file"]).resolve())
        arm_assets.append(elem)

# Remove meshdir from arm compiler since we now use absolute paths
compiler = arm_root.find("compiler")
if compiler is not None:
    compiler.attrib.pop("meshdir", None)

# ---------------------------------------------------------------------------
# 4. Attach Go2 body tree under arm's worldbody as a top-level body
#    The arm's 'root' body stays; Go2's base_link becomes a sibling.
#    We wrap Go2 in an 'go2_base' body so we can position it.
# ---------------------------------------------------------------------------
arm_worldbody = arm_root.find("worldbody")
go2_worldbody = go2_root.find("worldbody")

go2_bodies = list(go2_worldbody)
if not go2_bodies:
    raise RuntimeError("No bodies found in go2.xml worldbody")

for body in go2_bodies:
    arm_worldbody.append(body)

# ---------------------------------------------------------------------------
# 5. Attach arm mount onto Go2's base_link
# ---------------------------------------------------------------------------
base_link = arm_root.find(".//body[@name='base_link']")
if base_link is None:
    raise RuntimeError("Could not find 'base_link' after merging go2")

arm_root_body = arm_root.find(".//body[@name='root']")
if arm_root_body is None:
    raise RuntimeError("Could not find 'root' body in robot_master.xml")

# Detach 'root' from arm worldbody and re-attach under base_link
arm_worldbody.remove(arm_root_body)
mount = ET.SubElement(base_link, "body",
    name="arm_mount",
    pos=MOUNT_POS,
    quat=MOUNT_QUAT,
)
mount.append(arm_root_body)

# ---------------------------------------------------------------------------
# 6. Merge <actuator> — Go2 actuators first, then arm
# ---------------------------------------------------------------------------
arm_actuators = arm_root.find("actuator")
go2_actuators = go2_root.find("actuator")
if arm_actuators is None:
    arm_actuators = ET.SubElement(arm_root, "actuator")

if go2_actuators is not None:
    # Prepend go2 actuators before arm actuators
    for i, elem in enumerate(go2_actuators):
        arm_actuators.insert(i, elem)

# ---------------------------------------------------------------------------
# 7. Merge <equality>
# ---------------------------------------------------------------------------
go2_equality = go2_root.find("equality")
if go2_equality is not None:
    arm_equality = arm_root.find("equality")
    if arm_equality is None:
        arm_equality = ET.SubElement(arm_root, "equality")
    for elem in go2_equality:
        arm_equality.append(elem)

# ---------------------------------------------------------------------------
# 8. Merge <contact> exclusions
# ---------------------------------------------------------------------------
go2_contact = go2_root.find("contact")
if go2_contact is not None:
    arm_contact = arm_root.find("contact")
    if arm_contact is None:
        arm_contact = ET.SubElement(arm_root, "contact")
    for elem in go2_contact:
        arm_contact.append(elem)

# ---------------------------------------------------------------------------
# 9. Merge <keyframe> — Go2 'home' + arm 'hardware_zero' → combined 'home'
# ---------------------------------------------------------------------------
go2_key = go2_root.find(".//keyframe/key[@name='home']")
arm_key = arm_root.find(".//keyframe/key[@name='hardware_zero']")

arm_keyframe = arm_root.find("keyframe")
if arm_keyframe is None:
    arm_keyframe = ET.SubElement(arm_root, "keyframe")

if go2_key is not None and arm_key is not None:
    combined_qpos = go2_key.attrib["qpos"] + " " + arm_key.attrib["qpos"]
    combined_ctrl = go2_key.attrib["ctrl"]  + " " + arm_key.attrib["ctrl"]
    # Keep 'hardware_zero' so the controller can find it, also add 'home'
    arm_key.attrib["qpos"] = combined_qpos
    arm_key.attrib["ctrl"] = combined_ctrl
    # arm_key retains name='hardware_zero'
    home_key = ET.SubElement(arm_keyframe, "key",
        name="home", qpos=combined_qpos, ctrl=combined_ctrl)
    _ = home_key  # satisfy linters

# ---------------------------------------------------------------------------
# 10. Merge Go2 <default> into arm root
# ---------------------------------------------------------------------------
go2_default = go2_root.find("default")
if go2_default is not None:
    arm_default = arm_root.find("default")
    if arm_default is None:
        arm_default = ET.SubElement(arm_root, "default")
    for elem in go2_default:
        arm_default.append(elem)

# ---------------------------------------------------------------------------
# 11. Merge Go2 <option> settings (cone, impratio) into arm <option>
# ---------------------------------------------------------------------------
go2_option = go2_root.find("option")
arm_option = arm_root.find("option")
if go2_option is not None and arm_option is not None:
    for k, v in go2_option.attrib.items():
        arm_option.attrib.setdefault(k, v)

# ---------------------------------------------------------------------------
# 12. Floor, lights, visual settings
# ---------------------------------------------------------------------------
visual = arm_root.find("visual")
if visual is None:
    visual = ET.Element("visual")
    arm_root.insert(2, visual)

if visual.find("global") is None:
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")

map_elem = visual.find("map")
if map_elem is None:
    map_elem = ET.SubElement(visual, "map")
map_elem.attrib.setdefault("znear", "0.01")
map_elem.attrib.setdefault("zfar", "50")

if visual.find("headlight") is None:
    ET.SubElement(visual, "headlight",
        diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0 0 0")

if arm_worldbody.find(".//geom[@name='floor']") is None:
    arm_worldbody.insert(0, ET.Element("geom",
        name="floor", type="plane", size="0 0 0.01",
        rgba="0.8 0.8 0.8 1", condim="3",
    ))

if arm_worldbody.find(".//light[@name='spotlight']") is None:
    arm_worldbody.insert(0, ET.Element("light",
        name="fill", pos="1 -1 2", dir="-1 1 -1",
        diffuse="0.4 0.4 0.4", specular="0 0 0", castshadow="false",
    ))
    arm_worldbody.insert(0, ET.Element("light",
        name="spotlight", pos="0 0 3", dir="0 0 -1",
        diffuse="0.8 0.8 0.8", specular="0.2 0.2 0.2", castshadow="true",
    ))

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ET.indent(arm_root, space="  ")
ET.ElementTree(arm_root).write(OUTPUT, encoding="utf-8", xml_declaration=True)
print(f"\nWrote combined MJCF: {OUTPUT}")

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
import mujoco
model = mujoco.MjModel.from_xml_path(str(OUTPUT))
print(f"Validated: {model.nbody} bodies, {model.njnt} joints, "
      f"{model.ngeom} geoms, {model.nq} qpos, {model.nu} actuators")
print(f"\nOpen with:")
print(f"  python3 -m mujoco.viewer --mjcf {OUTPUT}")
