from pathlib import Path

import numpy as np
import point_cloud_utils as pcu
import polyscope as ps
import torch
from fvdb.utils.examples import load_car_1_mesh, load_car_2_mesh

voxel_size_1 = 0.02
voxel_size_2 = 0.03

if __name__ == "__main__":
    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_navigation_style("free")

    base_path = Path(__file__).parent.parent

    mesh_1_v, mesh_1_f = load_car_1_mesh(mode="vf", device=torch.device("cpu"))