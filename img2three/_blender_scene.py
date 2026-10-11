"""Blender-only scene builder, invoked by blender.py (no project dependencies)."""
from array import array
import json
import math
from pathlib import Path
import shutil
import sys

import bpy
from mathutils import Matrix, Vector


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def area_light(name, location, target, energy, size):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    aim(obj, target)


def main(job):
    if bpy.app.version < (4, 2, 0):
        raise RuntimeError("Blender 4.2 or newer is required.")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=job["mesh_path"])
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes or not any(len(obj.data.polygons) for obj in meshes):
        raise RuntimeError("The exported mesh has no faces to render. Try increasing the mesh edge threshold.")

    # Our trimesh exporters write source-image sRGB bytes directly into COLOR_0.
    # glTF imports these as linear values; reinterpret them as sRGB so Blender's
    # display transform does not brighten the colors a second time.
    for mesh in {obj.data for obj in meshes}:
        for attribute in mesh.color_attributes:
            colors = array("f", [0.0]) * (len(attribute.data) * 4)
            attribute.data.foreach_get("color", colors)
            attribute.data.foreach_set("color_srgb", colors)

    if job["mode"] == "object":
        # TripoSR exports raw Z-up positions in its GLB. Undo the importer's
        # Y-up -> Z-up rotation to recover the model's original coordinate system.
        undo_gltf = Matrix.Rotation(-math.pi / 2, 4, "X")
        transforms = [(obj, undo_gltf @ obj.matrix_world) for obj in meshes]
        for obj, transform in transforms:
            obj.parent = None
            obj.matrix_world = transform
        for obj in meshes:
            for polygon in obj.data.polygons:
                polygon.use_smooth = True
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    lo = Vector(tuple(min(p[i] for p in corners) for i in range(3)))
    hi = Vector(tuple(max(p[i] for p in corners) for i in range(3)))
    center = (lo + hi) / 2
    span = max((hi - lo).length, 0.01)

    scene = bpy.context.scene
    engine = job["engine"]
    if engine == "BLENDER_EEVEE_NEXT" and bpy.app.version >= (5, 0, 0):
        engine = "BLENDER_EEVEE"
    scene.render.engine = engine
    if job["engine"] == "CYCLES":
        scene.cycles.device = "CPU"
        scene.cycles.samples = job["samples"]
        scene.cycles.use_denoising = True
    else:
        scene.eevee.taa_render_samples = job["samples"]
    ratio = job["aspect_ratio"]
    scene.render.resolution_x = max(1, round(job["resolution"] * min(1, ratio)))
    scene.render.resolution_y = max(1, round(job["resolution"] * min(1, 1 / ratio)))
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.fps = 8
    scene.view_settings.view_transform = "AgX" if job["mode"] == "object" else "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.world = bpy.data.worlds.new("Studio world")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.22, 0.22, 0.22, 1)
    background.inputs["Strength"].default_value = 0.7

    camera_data = bpy.data.cameras.new("Final camera")
    camera = bpy.data.objects.new("Final camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = "PERSP"
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.angle = math.radians(job["fov"])
    camera_data.clip_start = max(0.0001, span / 10000)
    camera_data.clip_end = max(100, span * 100)

    if job["mode"] == "depth":
        # geometry.unproject uses x right, y up, -z forward. glTF import maps
        # that to x right, z up, +y forward; preserve the source pinhole camera.
        camera.location = (0, 0, 0)
        camera.rotation_euler = (math.pi / 2, 0, 0)
        # Photographs already contain lighting. Use the imported base color
        # (including COLOR_0 / textures) as emission for faithful reprojection.
        for material in {m for obj in meshes for m in obj.data.materials if m}:
            if not material.use_nodes:
                continue
            nodes, links = material.node_tree.nodes, material.node_tree.links
            shader = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
            output = next((n for n in nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
            if shader and output:
                color = shader.inputs["Base Color"]
                emission = nodes.new("ShaderNodeEmission")
                emission.inputs["Color"].default_value = color.default_value
                if color.is_linked:
                    links.new(color.links[0].from_socket, emission.inputs["Color"])
                links.new(emission.outputs[0], output.inputs["Surface"])
    else:
        # A sphere bound fits every turntable angle, including tall/thin meshes.
        half_angle = min(camera_data.angle / 2, math.atan(math.tan(camera_data.angle / 2) / ratio))
        distance = span * 0.5 / math.sin(half_angle) * 1.15
        elevation = math.radians(15)

        def camera_pose(angle):
            camera.location = center + Vector((distance * math.cos(elevation) * math.cos(angle),
                                               distance * math.cos(elevation) * math.sin(angle),
                                               distance * math.sin(elevation)))
            aim(camera, center)

        camera_pose(0)
        area_light("Key", center + Vector((span, -span, 1.5 * span)), center, 110 * span**2, span)
        area_light("Fill", center + Vector((0.5 * span, span, 0.6 * span)), center, 55 * span**2, span)
        area_light("Rim", center + Vector((-span, 0, span)), center, 90 * span**2, span * 0.75)
        bpy.ops.mesh.primitive_plane_add(size=span * 200, location=(center.x, center.y, lo.z - span * 0.005))
        floor = bpy.context.object
        floor.name = "Studio floor"
        material = bpy.data.materials.new("Matte floor")
        material.diffuse_color = (0.18, 0.18, 0.18, 1)
        material.use_nodes = True
        shader = material.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = material.diffuse_color
        shader.inputs["Roughness"].default_value = 0.85
        floor.data.materials.append(material)

    scene.render.filepath = job["render_path"]
    bpy.ops.render.render(write_still=True)
    if job["frames_dir"]:
        frames_dir = Path(job["frames_dir"])
        shutil.copyfile(job["render_path"], frames_dir / "frame_0000.png")
        for i in range(1, job["n_views"]):
            camera_pose(2 * math.pi * i / job["n_views"])
            scene.render.filepath = str(frames_dir / f"frame_{i:04d}.png")
            bpy.ops.render.render(write_still=True)
        camera_pose(0)
    # Keep the scene self-contained and usable after the temporary job is gone.
    scene.render.filepath = job["final_render_path"]
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=job["blend_path"])


if __name__ == "__main__":
    job_file = Path(sys.argv[sys.argv.index("--") + 1])
    main(json.loads(job_file.read_text(encoding="utf-8")))
