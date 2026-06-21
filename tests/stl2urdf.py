import os
import xml.etree.ElementTree as ET
from xml.dom import minidom


def ply_to_urdf(ply_file, output_urdf, robot_name="mesh_robot", package_name="your_package"):
    """
    将PLY文件转换为URDF格式

    参数:
    - ply_file: 输入的PLY文件路径
    - output_urdf: 输出的URDF文件路径
    - robot_name: 机器人名称
    - package_name: ROS包名称
    """

    # 创建根元素
    robot = ET.Element("robot", name=robot_name)

    # 创建link
    link = ET.SubElement(robot, "link", name="base_link")

    # 视觉部分
    visual = ET.SubElement(link, "visual")
    geometry_visual = ET.SubElement(visual, "geometry")

    # 获取PLY文件名（不含路径）
    ply_filename = os.path.basename(ply_file)
    mesh_visual = ET.SubElement(geometry_visual, "mesh",
                                filename=f"package://{package_name}/meshes/{ply_filename}",
                                scale="1 1 1")

    # 材质
    material = ET.SubElement(visual, "material", name="blue")
    color = ET.SubElement(material, "color", rgba="0 0 0.8 1.0")

    # 碰撞部分
    collision = ET.SubElement(link, "collision")
    geometry_collision = ET.SubElement(collision, "geometry")
    mesh_collision = ET.SubElement(geometry_collision, "mesh",
                                   filename=f"package://{package_name}/meshes/{ply_filename}",
                                   scale="1 1 1")

    # 惯性部分（简化）
    inertial = ET.SubElement(link, "inertial")
    mass = ET.SubElement(inertial, "mass", value="1.0")
    inertia = ET.SubElement(inertial, "inertia",
                            ixx="0.001", ixy="0.0", ixz="0.0",
                            iyy="0.001", iyz="0.0", izz="0.001")

    # 美化XML输出
    rough_string = ET.tostring(robot, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    # 写入文件
    with open(output_urdf, 'w') as f:
        f.write(pretty_xml)

    print(f"URDF文件已生成: {output_urdf}")

if __name__ == "__main__":
    stl_file = "/mnt/disk/hy/code/SurRoL-main/SurRoL-main/surrol/assets/organ/右肾.stl"
    mesh = trimesh.load_mesh(stl_file)
    # 使用示例
    ply_to_urdf("robot_mesh.ply", "robot.urdf", "my_robot", "my_robot_package")