# Project EmbodiedGen
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


import logging
import os
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
from xml.dom.minidom import parseString

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from embodied_gen.data.convex_decomposer import decompose_convex_mesh
from embodied_gen.utils.gpt_clients import GPT_CLIENT, GPTclient
from embodied_gen.utils.process_media import render_asset3d
from embodied_gen.utils.tags import VERSION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = ["URDFGenerator"]


URDF_TEMPLATE = """
<robot name="template_robot">
    <link name="template_link">
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="mesh.obj" scale="1.0 1.0 1.0"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="mesh.obj" scale="1.0 1.0 1.0"/>
            </geometry>
            <gazebo>
                <mu1>0.8</mu1> <!-- Main friction coefficient -->
                <mu2>0.6</mu2> <!-- Secondary friction coefficient -->
            </gazebo>
        </collision>
        <inertial>
            <mass value="1.0"/>
            <origin xyz="0 0 0"/>
            <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
        </inertial>
        <extra_info>
            <scale>1.0</scale>
            <version>"0.0.0"</version>
            <category>"unknown"</category>
            <description>"unknown"</description>
            <min_height>0.0</min_height>
            <max_height>0.0</max_height>
            <real_height>0.0</real_height>
            <min_mass>0.0</min_mass>
            <max_mass>0.0</max_mass>
            <generate_time>"-1"</generate_time>
            <gs_model>""</gs_model>
        </extra_info>
    </link>
</robot>
"""


class URDFGenerator(object):
    def __init__(
        self,
        gpt_client: GPTclient,
        mesh_file_list: list[str] = ["material_0.png", "material.mtl"],
        prompt_template: str = None,
        attrs_name: list[str] = None,
        render_dir: str = "urdf_renders",
        render_view_num: int = 4,
        decompose_convex: bool = False,
        rotate_xyzw: list[float] = (0.7071, 0, 0, 0.7071),
    ) -> None:
        if mesh_file_list is None:
            mesh_file_list = []
        self.mesh_file_list = mesh_file_list
        self.output_mesh_dir = "mesh"
        self.output_render_dir = render_dir
        self.gpt_client = gpt_client
        self.render_view_num = render_view_num
        if render_view_num == 4:
            view_desc = "This is orthographic projection showing the front, left, right and back views "  # noqa
        else:
            view_desc = "This is the rendered views "

        if prompt_template is None:
            prompt_template = (
                view_desc
                + """of the 3D object asset,
            category: {category}.
            You are an expert in 3D object analysis and physical property estimation.
            Give the category of this object asset (within 3 words), (if category is
            already provided, use it directly), accurately describe this 3D object asset (within 15 words),
            Determine the pose of the object in the first image and estimate the true vertical height
            (vertical projection) range of the object (in meters), i.e., how tall the object appears from top
            to bottom in the first image. also weight range (unit: kilogram), the average
            static friction coefficient of the object relative to rubber and the average dynamic friction
            coefficient of the object relative to rubber. Return response in format as shown in Output Example.

            Output Example:
            Category: cup
            Description: shiny golden cup with floral design
            Pose: <short_description_within_10_words>
            Height: 0.10-0.15 m
            Weight: 0.3-0.6 kg
            Static friction coefficient: 0.6
            Dynamic friction coefficient: 0.5

            IMPORTANT: Estimating Vertical Height from the First (Front View) Image and pose estimation based on all views.
            - The "vertical height" refers to the real-world vertical size of the object
            as projected in the first image, aligned with the image's vertical axis.
            - For flat objects like plates or disks or book, if their face is visible in the front view,
            use the diameter as the vertical height. If the edge is visible, use the thickness instead.
            - This is not necessarily the full length of the object, but how tall it appears
            in the first image vertically, based on its pose and orientation estimation on all views.
            - For objects(e.g., spoons, forks, writing instruments etc.) at an angle showing in images,
                e.g., tilted at 45° will appear shorter vertically than when upright.
            Estimate the vertical projection of their real length based on its pose.
            For example:
              - A pen standing upright in the first image (aligned with the image's vertical axis)
                full body visible in the first image: → vertical height ≈ 0.14-0.20 m
              - A pen lying flat in the first image or either the tip or the tail is facing the image
                (showing thickness or as a circle) → vertical height ≈ 0.018-0.025 m
              - Tilted pen in the first image (e.g., ~45° angle): vertical height ≈ 0.07-0.12 m
            - Use the rest views to help determine the object's 3D pose and orientation.
            Assume the object is in real-world scale and estimate the approximate vertical height
            based on the pose estimation and how large it appears vertically in the first image.
            """
            )

        self.prompt_template = prompt_template
        if attrs_name is None:
            attrs_name = [
                "category",
                "description",
                "min_height",
                "max_height",
                "real_height",
                "min_mass",
                "max_mass",
                "version",
                "generate_time",
                "gs_model",
            ]
        self.attrs_name = attrs_name
        self.decompose_convex = decompose_convex
        # Rotate 90 degrees around the X-axis from blender to align with simulators.
        self.rotate_xyzw = rotate_xyzw

    def parse_response(self, response: str) -> dict[str, any]:
        lines = response.split("\n")
        lines = [line.strip() for line in lines if line]
        category = lines[0].split(": ")[1]
        description = lines[1].split(": ")[1]
        min_height, max_height = map(
            lambda x: float(x.strip().replace(",", "").split()[0]),
            lines[3].split(": ")[1].split("-"),
        )
        min_mass, max_mass = map(
            lambda x: float(x.strip().replace(",", "").split()[0]),
            lines[4].split(": ")[1].split("-"),
        )
        mu1 = float(lines[5].split(": ")[1].replace(",", ""))
        mu2 = float(lines[6].split(": ")[1].replace(",", ""))

        return {
            "category": category.lower(),
            "description": description.lower(),
            "min_height": round(min_height, 4),
            "max_height": round(max_height, 4),
            "min_mass": round(min_mass, 4),
            "max_mass": round(max_mass, 4),
            "mu1": round(mu1, 2),
            "mu2": round(mu2, 2),
            "version": VERSION,
            "generate_time": datetime.now().strftime("%Y%m%d%H%M%S"),
        }

    def generate_urdf(
        self,
        input_mesh: str,
        output_dir: str,
        attr_dict: dict,
        output_name: str = None,
    ) -> str:
        """Generate a URDF file for a given mesh with specified attributes.

        Args:
            input_mesh (str): Path to the input mesh file.
            output_dir (str): Directory to store the generated URDF
                and processed mesh.
            attr_dict (dict): Dictionary containing attributes like height,
                mass, and friction coefficients.
            output_name (str, optional): Name for the generated URDF and robot.

        Returns:
            str: Path to the generated URDF file.
        """

        # 1. Load and normalize the mesh
        mesh = trimesh.load(input_mesh)
        mesh_scale = np.ptp(mesh.vertices, axis=0).max()
        mesh.vertices /= mesh_scale  # Normalize to [-0.5, 0.5]
        raw_height = np.ptp(mesh.vertices, axis=0)[1]

        # 2. Scale the mesh to real height
        real_height = attr_dict["real_height"]
        scale = round(real_height / raw_height, 6)
        mesh = mesh.apply_scale(scale)

        # 3. Prepare output directories and save scaled mesh
        mesh_folder = os.path.join(output_dir, self.output_mesh_dir)
        os.makedirs(mesh_folder, exist_ok=True)

        obj_name = os.path.basename(input_mesh)
        mesh_output_path = os.path.join(mesh_folder, obj_name)
        mesh.export(mesh_output_path)

        # 4. Copy additional mesh files, if any
        input_dir = os.path.dirname(input_mesh)
        for file in self.mesh_file_list:
            src_file = os.path.join(input_dir, file)
            dest_file = os.path.join(mesh_folder, file)
            if os.path.isfile(src_file):
                shutil.copy(src_file, dest_file)

        # 5. Determine output name
        if output_name is None:
            output_name = os.path.splitext(obj_name)[0]

        # 6. Load URDF template and update attributes
        robot = ET.fromstring(URDF_TEMPLATE)
        robot.set("name", output_name)

        link = robot.find("link")
        if link is None:
            raise ValueError("URDF template is missing 'link' element.")
        link.set("name", output_name)

        if self.rotate_xyzw is not None:
            rpy = Rotation.from_quat(self.rotate_xyzw).as_euler(
                "xyz", degrees=False
            )
            rpy = [str(round(num, 4)) for num in rpy]
            link.find("visual/origin").set("rpy", " ".join(rpy))
            link.find("collision/origin").set("rpy", " ".join(rpy))

        # Update visual geometry
        visual = link.find("visual/geometry/mesh")
        if visual is not None:
            visual.set(
                "filename", os.path.join(self.output_mesh_dir, obj_name)
            )
            visual.set("scale", "1.0 1.0 1.0")

        # Update collision geometry
        collision = link.find("collision/geometry/mesh")
        if collision is not None:
            collision_mesh = os.path.join(self.output_mesh_dir, obj_name)
            if self.decompose_convex:
                try:
                    d_params = dict(
                        threshold=0.05, max_convex_hull=100, verbose=False
                    )
                    filename = f"{os.path.splitext(obj_name)[0]}_collision.ply"
                    output_path = os.path.join(mesh_folder, filename)
                    decompose_convex_mesh(
                        mesh_output_path, output_path, **d_params
                    )
                    obj_filename = filename.replace(".ply", ".obj")
                    trimesh.load(output_path).export(
                        f"{mesh_folder}/{obj_filename}"
                    )
                    collision_mesh = f"{self.output_mesh_dir}/{obj_filename}"
                except Exception as e:
                    logger.warning(
                        f"Convex decomposition failed for {output_path}, {e}."
                        "Use original mesh for collision computation."
                    )
            collision.set("filename", collision_mesh)
            collision.set("scale", "1.0 1.0 1.0")

        # Update friction coefficients
        gazebo = link.find("collision/gazebo")
        if gazebo is not None:
            for param, key in zip(["mu1", "mu2"], ["mu1", "mu2"]):
                element = gazebo.find(param)
                if element is not None:
                    element.text = f"{attr_dict[key]:.2f}"

        # Update mass
        inertial = link.find("inertial/mass")
        if inertial is not None:
            mass_value = (attr_dict["min_mass"] + attr_dict["max_mass"]) / 2
            inertial.set("value", f"{mass_value:.4f}")

        # Add extra_info element to the link
        extra_info = link.find("extra_info/scale")
        if extra_info is not None:
            extra_info.text = f"{scale:.6f}"

        for key in self.attrs_name:
            extra_info = link.find(f"extra_info/{key}")
            if extra_info is not None and key in attr_dict:
                extra_info.text = f"{attr_dict[key]}"

        # 7. Write URDF to file
        os.makedirs(output_dir, exist_ok=True)
        urdf_path = os.path.join(output_dir, f"{output_name}.urdf")
        tree = ET.ElementTree(robot)
        tree.write(urdf_path, encoding="utf-8", xml_declaration=True)

        logger.info(f"URDF file saved to {urdf_path}")

        return urdf_path

    @staticmethod
    def get_attr_from_urdf(
        urdf_path: str,
        attr_root: str = ".//link/extra_info",
        attr_name: str = "scale",
    ) -> float:
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        mesh_attr = None
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        extra_info = root.find(attr_root)
        if extra_info is not None:
            scale_element = extra_info.find(attr_name)
            if scale_element is not None:
                mesh_attr = scale_element.text
                try:
                    mesh_attr = float(mesh_attr)
                except ValueError as e:
                    pass

        return mesh_attr

    @staticmethod
    def add_quality_tag(
        urdf_path: str, results: list, output_path: str = None
    ) -> None:
        if output_path is None:
            output_path = urdf_path

        tree = ET.parse(urdf_path)
        root = tree.getroot()
        custom_data = ET.SubElement(root, "custom_data")
        quality = ET.SubElement(custom_data, "quality")
        for key, value in results:
            checker_tag = ET.SubElement(quality, key)
            checker_tag.text = str(value)

        rough_string = ET.tostring(root, encoding="utf-8")
        formatted_string = parseString(rough_string).toprettyxml(indent="   ")
        cleaned_string = "\n".join(
            [line for line in formatted_string.splitlines() if line.strip()]
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_string)

        logger.info(f"URDF files saved to {output_path}")

    def get_estimated_attributes(self, asset_attrs: dict):
        estimated_attrs = {
            "height": round(
                (asset_attrs["min_height"] + asset_attrs["max_height"]) / 2, 4
            ),
            "mass": round(
                (asset_attrs["min_mass"] + asset_attrs["max_mass"]) / 2, 4
            ),
            "mu": round((asset_attrs["mu1"] + asset_attrs["mu2"]) / 2, 4),
            "category": asset_attrs["category"],
        }

        return estimated_attrs

    def __call__(
        self,
        mesh_path: str,
        output_root: str,
        text_prompt: str = None,
        category: str = "unknown",
        **kwargs,
    ):
        if text_prompt is None or len(text_prompt) == 0:
            text_prompt = self.prompt_template
            text_prompt = text_prompt.format(category=category.lower())

        image_path = render_asset3d(
            mesh_path,
            output_root,
            num_images=self.render_view_num,
            output_subdir=self.output_render_dir,
            no_index_file=True,
        )

        response = self.gpt_client.query(text_prompt, image_path)
        # logger.info(response)
        if response is None:
            asset_attrs = {
                "category": category.lower(),
                "description": category.lower(),
                "min_height": 1,
                "max_height": 1,
                "min_mass": 1,
                "max_mass": 1,
                "mu1": 0.8,
                "mu2": 0.6,
                "version": VERSION,
                "generate_time": datetime.now().strftime("%Y%m%d%H%M%S"),
            }
        else:
            asset_attrs = self.parse_response(response)
        for key in self.attrs_name:
            if key in kwargs:
                asset_attrs[key] = kwargs[key]

        asset_attrs["real_height"] = round(
            (asset_attrs["min_height"] + asset_attrs["max_height"]) / 2, 4
        )

        self.estimated_attrs = self.get_estimated_attributes(asset_attrs)

        urdf_path = self.generate_urdf(mesh_path, output_root, asset_attrs)

        logger.info(f"response: {response}")

        return urdf_path


if __name__ == "__main__":
    # Rotate 90 degrees around the X-axis to align with simulators.
    urdf_gen = URDFGenerator(GPT_CLIENT, render_view_num=4)
    urdf_path = urdf_gen(
        mesh_path="outputs/layout2/asset3d/marker/result/mesh/marker.obj",
        output_root="outputs/test_urdf",
        category="marker",
        # min_height=1.0,
        # max_height=1.2,
        version=VERSION,
    )

    URDFGenerator.add_quality_tag(
        urdf_path, [[urdf_gen.__class__.__name__, "OK"]]
    )

    # zip_files(
    #     input_paths=[
    #         "scripts/apps/tmp/2umpdum3e5n/URDF_sample/mesh",
    #         "scripts/apps/tmp/2umpdum3e5n/URDF_sample/sample.urdf"
    #     ],
    #     output_zip="zip.zip"
    # )
