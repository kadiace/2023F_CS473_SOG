import os, sys, glob, shutil
import torch
from pytorch3d.io import IO
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat

# for complex dependency in Text2Tex
sys.path.append("..")
sys.path.append("/root/CS479-3D_ML/Project/Shoulders-of-Giants/Baseline/Text2Tex/")
from Baseline.Text2Tex.lib.camera_helper import init_viewpoints
from Baseline.Text2Tex.lib.projection_helper import render_one_view

from torchvision import transforms
from PIL import Image
import objaverse

# Setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    print("no gpu avaiable")
    exit()
    
SUBSET = "../Baseline/Text2Tex/data/objaverse_subset.txt"
    
def get_objaverse_subset():
    with open(SUBSET) as f:
        ids = [l.rstrip().split("_")[-1] for l in f.readlines()]

    return ids

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
    # print(f"pre-updated mesh center : {mesh_center}")
    
    # make sure mesh size is normalized
    box_size = bbox[..., 1] - bbox[..., 0]
    box_max = box_size.max(dim=1, keepdim=True)[0].repeat(num_verts, 3)
    mesh = apply_scale_to_mesh(mesh, 1 / box_max)
    # print(f"pre-updated mesh size : {box_max}")
    
    return io, mesh


if __name__=="__main__":
    
    objaverse_subset = get_objaverse_subset()

    # cache objects to ~/.objaverse/
    objects = objaverse.load_objects(objaverse_subset)
    print(f"objects: {len(objects)}")
    
    
    objaverse_subdata_location = "../Baseline/Text2Tex/data/objaverse"
    objaverse_real_data_location = "/root/.objaverse/hf-objaverse-v1/glbs"
    
    actual_data = 0
    
    for data_id in os.listdir(objaverse_subdata_location):
        glb_file_path = glob.glob(f"{objaverse_real_data_location}/**/{data_id}.glb")
        if not glb_file_path:
            print(f"There is no {data_id}")
        glb_file_path = glb_file_path[0]
        

        # glb_file_path = "/root/.objaverse/hf-objaverse-v1/glbs/000-087/52d850660e1d4358ad96b922ea75f62c.glb"
        
        io = IO()
        io.register_meshes_format(MeshGlbFormat())
        # print(mesh) : <pytorch3d.structures.meshes.Meshes object at 0x7fb8840fd820>
        try:
            io, mesh = init_mesh_glb(io, glb_file_path)
        except Exception as e:
            print(f"{data_id} error occur! : {e}")
            continue
        
        print(f"generate {data_id}")

        (
            dist_list, 
            elev_list, 
            azim_list, 
            sector_list,
            view_punishments
        ) = init_viewpoints("predefined", 4, 1, 0, None, 
                                use_principle=False, 
                                use_shapenet=False,
                                use_objaverse=False)
    
        generate_dir = os.path.join(f"{objaverse_subdata_location}/{data_id}", "fid_label")
        if os.path.isdir(generate_dir):
            shutil.rmtree(generate_dir)
        os.makedirs(generate_dir, exist_ok=True)
        
        init_image_dir = os.path.join(generate_dir, "rendering")
        os.makedirs(init_image_dir, exist_ok=True)
        
        normal_map_dir = os.path.join(generate_dir, "normal")
        os.makedirs(normal_map_dir, exist_ok=True)
        
        depth_map_dir = os.path.join(generate_dir, "depth")
        os.makedirs(depth_map_dir, exist_ok=True)
        
        error_occur = False
        
        for view_idx in range(len(dist_list)):

            # sequentially pop the viewpoints
            dist, elev, azim, sector = dist_list[view_idx], elev_list[view_idx], azim_list[view_idx], sector_list[view_idx]
            
            try:
                (
                    cameras, renderer,
                    init_images_tensor, normal_maps_tensor, similarity_tensor, depth_maps_tensor, fragments
                ) = render_one_view(mesh,
                    dist, elev, azim,
                    768, 1, DEVICE)
                
                init_image = init_images_tensor[0].cpu()
                init_image = init_image.permute(2, 0, 1)
                init_image = transforms.ToPILImage()(init_image).convert("RGB")

                normal_map = normal_maps_tensor[0].cpu()
                normal_map = normal_map.permute(2, 0, 1)
                normal_map = transforms.ToPILImage()(normal_map).convert("RGB")

                depth_map = depth_maps_tensor[0].cpu().numpy()
                depth_map = Image.fromarray(depth_map).convert("L")
                
                init_image.save(os.path.join(init_image_dir, "{}.png".format(view_idx)))
                normal_map.save(os.path.join(normal_map_dir, "{}.png".format(view_idx)))
                depth_map.save(os.path.join(depth_map_dir, "{}.png".format(view_idx)))

            except Exception as e:
                print(f"{data_id} error occur! : {e}")
                error_occur = True
                break
        if not error_occur:
            actual_data += 1
    print(f"actual_data : {actual_data}")

            