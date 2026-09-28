"""Convert robot_master's visual URDF and GLTF assets to a loadable MJCF."""

from __future__ import annotations

import base64
import json
import math
import struct
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parent
URDF = ROOT / "robot_master" / "urdf" / "robot_master.urdf"
MESH_DIR = ROOT / "robot_master" / "meshes"
OUTPUT_DIR = ROOT / "robot_master" / "mjcf"
ASSET_DIR = OUTPUT_DIR / "assets"
MJCF = OUTPUT_DIR / "robot_master.xml"

# CAD-only helper frames and a loop-closure link have near-zero inertia.  Each
# body receives an inertia safely above MuJoCo's minimum thresholds.
MIN_MASS = 1e-4
MIN_INERTIA = 1e-9

# Direct-drive output-stage torque limits for the selected CubeMars modules.
# These values are peak torques, in N*m at the actuator output.
AK60_6_PEAK_TORQUE_NM = 9.0
AK70_10_PEAK_TORQUE_NM = 24.8
AK40_10_PEAK_TORQUE_NM = 4.1

# Position-control stiffnesses.  At a 1-radian tracking error, each servo
# requests its respective actuator's peak output torque before saturation.
AK60_6_POSITION_KP = AK60_6_PEAK_TORQUE_NM
AK70_10_POSITION_KP = AK70_10_PEAK_TORQUE_NM
AK40_10_POSITION_KP = AK40_10_PEAK_TORQUE_NM

# The CAD exporter represents the table-to-robot fastening as a planar mate:
# two ±10 km slide joints plus a hinge.  In simulation that leaves the bottom
# assembly unconstrained, so lock the mate as the requested fixed attachment.
# Revolutes 10 and 11 are rigidly attached to their parents.
LOCKED_JOINTS = frozenset({"planar_1", "planar_1_1", "planar_1_2", "revolute_10", "revolute_11"})


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
    return format_values(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def origin_attributes(origin: ET.Element | None) -> dict[str, str]:
    if origin is None:
        return {}
    attributes: dict[str, str] = {}
    if origin.get("xyz") is not None:
        attributes["pos"] = origin.attrib["xyz"]
    if origin.get("rpy") is not None:
        attributes["quat"] = rpy_to_quat(origin.attrib["rpy"])
    return attributes


def identity_matrix() -> list[list[float]]:
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [[sum(left[row][i] * right[i][column] for i in range(4)) for column in range(4)] for row in range(4)]


def gltf_node_matrix(node: dict) -> list[list[float]]:
    if "matrix" in node:
        values = node["matrix"]
        # glTF stores 4x4 matrices in column-major order.
        return [[values[column * 4 + row] for column in range(4)] for row in range(4)]

    tx, ty, tz = node.get("translation", [0.0, 0.0, 0.0])
    sx, sy, sz = node.get("scale", [1.0, 1.0, 1.0])
    x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    rotation = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
        [0, 0, 0, 1],
    ]
    scale = [[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]]
    translation = identity_matrix()
    translation[0][3], translation[1][3], translation[2][3] = tx, ty, tz
    return matmul(matmul(translation, rotation), scale)


def load_gltf_buffer(gltf: dict, gltf_path: Path, index: int) -> bytes:
    uri = gltf["buffers"][index].get("uri")
    if uri is None:
        raise ValueError(f"{gltf_path.name}: binary GLB buffers are not supported")
    if uri.startswith("data:"):
        return base64.b64decode(uri.split(",", 1)[1])
    return (gltf_path.parent / uri).read_bytes()


def accessor_values(gltf: dict, buffers: list[bytes], accessor_index: int) -> list[tuple[float, ...]]:
    accessor = gltf["accessors"][accessor_index]
    if "sparse" in accessor:
        raise ValueError("Sparse GLTF accessors are not supported")
    view = gltf["bufferViews"][accessor["bufferView"]]
    component_info = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
    component_format, component_size = component_info[accessor["componentType"]]
    components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[accessor["type"]]
    stride = view.get("byteStride", component_size * components)
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    unpack = struct.Struct("<" + component_format * components).unpack_from
    data = buffers[view["buffer"]]
    return [unpack(data, start + offset * stride) for offset in range(accessor["count"])]


def primitive_triangles(indices: list[int], mode: int) -> list[tuple[int, int, int]]:
    if mode == 4:  # TRIANGLES
        return [tuple(indices[i : i + 3]) for i in range(0, len(indices) - 2, 3)]
    if mode == 5:  # TRIANGLE_STRIP
        return [(indices[i], indices[i + 1 + (i % 2)], indices[i + 2 - (i % 2)]) for i in range(len(indices) - 2)]
    if mode == 6:  # TRIANGLE_FAN
        return [(indices[0], indices[i], indices[i + 1]) for i in range(1, len(indices) - 1)]
    raise ValueError(f"Unsupported GLTF primitive mode {mode}")


def convert_gltf_to_obj(gltf_path: Path, obj_path: Path) -> tuple[int, int]:
    """Flatten a GLTF scene to a mesh-only OBJ suitable for MuJoCo."""
    gltf = json.loads(gltf_path.read_text(encoding="utf-8"))
    buffers = [load_gltf_buffer(gltf, gltf_path, index) for index in range(len(gltf["buffers"]))]
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    def add_mesh(mesh_index: int, transform: list[list[float]]) -> None:
        for primitive in gltf["meshes"][mesh_index]["primitives"]:
            positions = accessor_values(gltf, buffers, primitive["attributes"]["POSITION"])
            offset = len(vertices)
            for x, y, z in positions:
                vertices.append(
                    (
                        transform[0][0] * x + transform[0][1] * y + transform[0][2] * z + transform[0][3],
                        transform[1][0] * x + transform[1][1] * y + transform[1][2] * z + transform[1][3],
                        transform[2][0] * x + transform[2][1] * y + transform[2][2] * z + transform[2][3],
                    )
                )
            if "indices" in primitive:
                indices = [int(item[0]) for item in accessor_values(gltf, buffers, primitive["indices"])]
            else:
                indices = list(range(len(positions)))
            faces.extend(tuple(offset + index for index in face) for face in primitive_triangles(indices, primitive.get("mode", 4)))

    def walk_node(node_index: int, parent_transform: list[list[float]]) -> None:
        node = gltf["nodes"][node_index]
        transform = matmul(parent_transform, gltf_node_matrix(node))
        if "mesh" in node:
            add_mesh(node["mesh"], transform)
        for child in node.get("children", []):
            walk_node(child, transform)

    scene_index = gltf.get("scene", 0)
    for node_index in gltf["scenes"][scene_index].get("nodes", []):
        walk_node(node_index, identity_matrix())

    if not vertices or not faces:
        raise ValueError(f"{gltf_path.name}: scene has no triangle mesh")
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    with obj_path.open("w", encoding="utf-8", newline="\n") as output:
        output.writelines(f"v {x:.9g} {y:.9g} {z:.9g}\n" for x, y, z in vertices)
        output.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces)
    return len(vertices), len(faces)


def inertial_attributes(link: ET.Element) -> dict[str, str]:
    inertial = link.find("inertial")
    if inertial is None:
        return {"pos": "0 0 0", "mass": format_values([MIN_MASS]), "diaginertia": format_values([MIN_INERTIA] * 3)}

    mass = max(float(inertial.find("mass").attrib["value"]), MIN_MASS)
    inertia = inertial.find("inertia")
    diagonal = [max(abs(float(inertia.attrib[key])), MIN_INERTIA) for key in ("ixx", "iyy", "izz")]
    # A valid principal inertia must satisfy triangle inequalities.  A diagonal
    # approximation is stable for viewing and avoids non-positive tensors.
    largest = max(diagonal)
    if sum(diagonal) - largest <= largest:
        diagonal = [max(value, largest * 0.5001) for value in diagonal]
    attributes = {"pos": "0 0 0", "mass": format_values([mass]), "diaginertia": format_values(diagonal)}
    attributes.update(origin_attributes(inertial.find("origin")))
    return attributes


def joint_element(joint: ET.Element) -> ET.Element:
    joint_type = joint.attrib["type"]
    attributes = {"name": joint.attrib["name"]}
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
    limit = joint.find("limit")
    if limit is not None and joint_type != "continuous":
        attributes["range"] = f"{limit.attrib['lower']} {limit.attrib['upper']}"
    return ET.Element("joint", attributes)


def main() -> None:
    robot = ET.parse(URDF).getroot()
    links = {link.attrib["name"]: link for link in robot.findall("link")}
    joints = robot.findall("joint")
    children: dict[str, list[ET.Element]] = defaultdict(list)
    child_joints: dict[str, ET.Element] = {}
    for joint in joints:
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        children[parent].append(joint)
        child_joints[child] = joint
    root_links = sorted(set(links) - set(child_joints))
    if len(root_links) != 1:
        raise ValueError(f"Expected exactly one URDF root link, got {root_links}")

    # Convert each source GLTF once, even when a part is used by several links.
    source_files = sorted(
        {
            Path(mesh.attrib["filename"].removeprefix("package://robot_master/meshes/")).name
            for link in links.values()
            for mesh in link.findall("visual/geometry/mesh")
        }
    )
    mesh_assets: dict[str, tuple[str, str, int, int]] = {}
    for index, source_name in enumerate(source_files, start=1):
        source_path = MESH_DIR / source_name
        output_name = f"{index:03d}_{source_path.stem}.obj"
        vertex_count, face_count = convert_gltf_to_obj(source_path, ASSET_DIR / output_name)
        mesh_assets[source_name] = (f"mesh_{index:03d}", f"assets/{output_name}", vertex_count, face_count)

    mujoco_xml = ET.Element("mujoco", model=robot.attrib.get("name", "robot_master"))
    ET.SubElement(mujoco_xml, "compiler", angle="radian", autolimits="true")
    ET.SubElement(mujoco_xml, "option", integrator="implicitfast")
    visual = ET.SubElement(mujoco_xml, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    assets = ET.SubElement(mujoco_xml, "asset")
    for source_name in source_files:
        asset_name, asset_file, _, _ = mesh_assets[source_name]
        ET.SubElement(assets, "mesh", name=asset_name, file=asset_file)

    worldbody = ET.SubElement(mujoco_xml, "worldbody")

    def add_body(link_name: str, parent: ET.Element) -> None:
        link = links[link_name]
        attributes = {"name": link_name}
        parent_joint = child_joints.get(link_name)
        if parent_joint is not None:
            attributes.update(origin_attributes(parent_joint.find("origin")))
        body = ET.SubElement(parent, "body", attributes)
        ET.SubElement(body, "inertial", inertial_attributes(link))
        if (
            parent_joint is not None
            and parent_joint.attrib["type"] != "fixed"
            and parent_joint.attrib["name"] not in LOCKED_JOINTS
        ):
            body.append(joint_element(parent_joint))

        for visual_element in link.findall("visual"):
            mesh = visual_element.find("geometry/mesh")
            if mesh is None:
                continue
            source_name = Path(mesh.attrib["filename"].removeprefix("package://robot_master/meshes/")).name
            asset_name, _, _, _ = mesh_assets[source_name]
            geom_attributes = {"type": "mesh", "mesh": asset_name, "contype": "0", "conaffinity": "0"}
            geom_attributes.update(origin_attributes(visual_element.find("origin")))
            # All source visuals use unit scale.  Mesh scaling belongs on the
            # MJCF asset (not on a geom), so no per-geom scale is required.
            color = visual_element.find("material/color")
            if color is not None and color.get("rgba") is not None:
                geom_attributes["rgba"] = color.attrib["rgba"]
            body.append(ET.Element("geom", geom_attributes))

        for child_joint in children[link_name]:
            add_body(child_joint.find("child").attrib["link"], body)

    add_body(root_links[0], worldbody)

    # Joint equality uses joint1 = a0 + a1 * joint2 around the model's
    # reference pose.  Each paired joint below uses a 1:1 inverse gear ratio.
    # The URDF cuts the physical
    # four-bar loop at the massless part_3__1__loop_closure placeholder; weld
    # it back to part_3 to restore the missing closed-loop attachment.
    equality = ET.SubElement(mujoco_xml, "equality")
    ET.SubElement(
        equality,
        "weld",
        name="revolute_14_closed_loop",
        body1="part_3",
        body2="part_3__1__loop_closure",
        solref="0.004 1",
        solimp="0.999 0.999 0.001",
        torquescale="0.05",
    )
    ET.SubElement(
        equality,
        "joint",
        name="revolute_9_opposes_revolute_7",
        joint1="revolute_9",
        joint2="revolute_7",
        polycoef="0 -1 0 0 0",
        solref="0.004 1",
        solimp="0.999 0.999 0.001",
    )
    ET.SubElement(
        equality,
        "joint",
        name="revolute_6_opposes_revolute_8",
        joint1="revolute_6",
        joint2="revolute_8",
        polycoef="0 -1 0 0 0",
        solref="0.004 1",
        solimp="0.999 0.999 0.001",
    )
    ET.SubElement(
        equality,
        "joint",
        name="revolute_5_opposes_revolute_4",
        joint1="revolute_5",
        joint2="revolute_4",
        polycoef="0 -1 0 0 0",
        solref="0.004 1",
        solimp="0.999 0.999 0.001",
    )
    ET.SubElement(
        equality,
        "joint",
        name="revolute_13_opposes_revolute_8",
        joint1="revolute_13",
        joint2="revolute_8",
        polycoef="0 -1 0 0 0",
        solref="0.004 1",
        solimp="0.999 0.999 0.001",
    )
    ET.SubElement(
        equality,
        "joint",
        name="revolute_14_opposes_revolute_8",
        joint1="revolute_14_loop_closure",
        joint2="revolute_8",
        polycoef="0 -1 0 0 0",
        solref="0.004 1",
        solimp="0.999 0.999 0.001",
    )

    # Position servo controls are target angles in radians because gear=1.  A
    # critically damped servo requests kp * (target - position), and forcerange
    # guarantees that applied torque stays within the motor's output rating.
    actuators = ET.SubElement(mujoco_xml, "actuator")
    ET.SubElement(
        actuators,
        "position",
        name="ak60_revolute_1",
        joint="revolute_1",
        gear="1",
        kp=f"{AK60_6_POSITION_KP:g}",
        dampratio="1",
        forcelimited="true",
        forcerange=f"{-AK60_6_PEAK_TORQUE_NM:g} {AK60_6_PEAK_TORQUE_NM:g}",
    )
    ET.SubElement(
        actuators,
        "position",
        name="ak70_revolute_2",
        joint="revolute_2",
        gear="1",
        kp=f"{AK70_10_POSITION_KP:g}",
        dampratio="1",
        ctrllimited="true",
        ctrlrange="-6.28319 0.153038",
        forcelimited="true",
        forcerange=f"{-AK70_10_PEAK_TORQUE_NM:g} {AK70_10_PEAK_TORQUE_NM:g}",
    )
    ET.SubElement(
        actuators,
        "position",
        name="ak70_revolute_3",
        joint="revolute_3",
        gear="1",
        kp=f"{AK70_10_POSITION_KP:g}",
        dampratio="1",
        forcelimited="true",
        forcerange=f"{-AK70_10_PEAK_TORQUE_NM:g} {AK70_10_PEAK_TORQUE_NM:g}",
    )
    ET.SubElement(
        actuators,
        "position",
        name="ak40_revolute_4",
        joint="revolute_4",
        gear="1",
        kp=f"{AK40_10_POSITION_KP:g}",
        dampratio="1",
        ctrllimited="true",
        ctrlrange="-0.00381777 1.54953",
        forcelimited="true",
        forcerange=f"{-AK40_10_PEAK_TORQUE_NM:g} {AK40_10_PEAK_TORQUE_NM:g}",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ET.indent(mujoco_xml, space="  ")
    ET.ElementTree(mujoco_xml).write(MJCF, encoding="utf-8", xml_declaration=True)

    # Validate the persisted MJCF including all OBJ assets.
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    total_vertices = sum(item[2] for item in mesh_assets.values())
    total_faces = sum(item[3] for item in mesh_assets.values())
    visual_count = sum(len(link.findall("visual")) for link in links.values())
    print(f"Wrote: {MJCF}")
    print(f"Converted {len(mesh_assets)} GLTF files to OBJ ({total_vertices} vertices, {total_faces} triangles).")
    print(f"Validated MJCF: {model.nbody} bodies, {model.njnt} joints, {model.ngeom} visual geoms, {model.nmesh} meshes.")
    print(f"Mapped {visual_count} URDF visual elements; collision is disabled on visual-only geoms.")


if __name__ == "__main__":
    main()
