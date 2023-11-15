import os, sys, glob
import torch
from pytorch3d.io import IO
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat

# for complex dependency in Text2Tex
sys.path.append("..")
sys.path.append("/root/CS479-3D_ML/Project/baselines/Text2Tex/")

from baselines.Text2Tex.lib.camera_helper import init_viewpoints
from baselines.Text2Tex.lib.projection_helper import render_one_view
from torchvision import transforms

from PIL import Image

# Setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    print("no gpu avaiable")
    exit()

def apply_scale_to_mesh(mesh, scale):
    new_mesh = mesh.scale_verts(scale)

    return new_mesh

def apply_offsets_to_mesh(mesh, offsets):
    new_mesh = mesh.offset_verts(offsets)

    return new_mesh

def init_mesh_glb(io, glb_file_path, include_textures=True):
    mesh = io.load_mesh(glb_file_path, include_textures=True, device=DEVICE)
    
    num_verts = mesh.verts_packed().shape[0]
    
    # make sure mesh center is at origin
    bbox = mesh.get_bounding_boxes()
    mesh_center = bbox.mean(dim=2).repeat(num_verts, 1)
    mesh = apply_offsets_to_mesh(mesh, -mesh_center)
    print(f"pre-updated mesh center : {mesh_center}")
    
    # make sure mesh size is normalized
    box_size = bbox[..., 1] - bbox[..., 0]
    box_max = box_size.max(dim=1, keepdim=True)[0].repeat(num_verts, 3)
    mesh = apply_scale_to_mesh(mesh, 1 / box_max)
    print(f"pre-updated mesh size : {box_max}")
    
    return io, mesh


if __name__=="__main__":
    
    objaverse_subdata_location = "../data/objaverse"
    objaverse_real_data_location = "/root/.objaverse/hf-objaverse-v1/glbs"
    
    for data in os.listdir("../data/objaverse"):
        glb_file_path = glob.glob(f"{objaverse_real_data_location}/**/{data}.glb")[0]
        if not glb_file_path:
            print(f"There is no {data}")

    # # Replace with the output directory for the exported files
    # output_directory = "./"


        io = IO()
        io.register_meshes_format(MeshGlbFormat())
        # print(mesh) : <pytorch3d.structures.meshes.Meshes object at 0x7fb8840fd820>
        io, mesh = init_mesh_glb(io, glb_file_path)
    
        (
            dist_list, 
            elev_list, 
            azim_list, 
            sector_list,
            view_punishments
        ) = init_viewpoints("predefined", 1, 1, 0, None, 
                                use_principle=True, 
                                use_shapenet=False,
                                use_objaverse=True)
    
        generate_dir = os.path.join("./outputs", "generate")
        os.makedirs(generate_dir, exist_ok=True)
        
        init_image_dir = os.path.join(generate_dir, "rendering")
        os.makedirs(init_image_dir, exist_ok=True)
        
        normal_map_dir = os.path.join(generate_dir, "normal")
        os.makedirs(normal_map_dir, exist_ok=True)