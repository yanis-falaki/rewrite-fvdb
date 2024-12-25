import hashlib
import logging
import site
import tempfile
import timeit
from pathlib import Path
from typing import List, Tuple, Union

import git
import git.repo
import numpy as np
import point_cloud_utils as pcu
import torch
from git.exc import InvalidGitRepositoryError


def _is_editable_install() -> bool:
    # Check we're not a site package
    module_path = Path(__file__).resolve()
    for site_path in site.getsitepackages():
        if str(module_path).startswith(site_path):
            return False
    # Check if we're in the source directory
    module_dir = module_path.parent.parent.parent.parent
    return (module_dir / "setup.py").is_file()


def _get_local_repo_path() -> Path:
    if _is_editable_install():
        external_dir = Path(__file__).resolve().parent.parent.parent.parent / "external"
        if not external_dir.exists():
            external_dir.mkdir()
        local_repo_path = external_dir
    else:
        local_repo_path = Path(tempfile.gettempdir)

    local_repo_path = local_repo_path / "fvdb_example_data"
    return local_repo_path


def _clone_fvdb_example_data():
    def is_git_repo(repo_path: str):
        is_repo = False
        try:
            _ = git.repo.Repo(repo_path)
            is_repo = True
        except InvalidGitRepositoryError:
            is_repo = False
        
        return is_repo
    
    git_tag = "613c3a4e220eb45b9ae0271dca4808ab484ee134"
    git_url = "https://github.com/voxel-foundation/fvdb-example-data.git"

    repo_path = _get_local_repo_path()
    if repo_path.exists() and repo_path.is_dir():
        if is_git_repo(str(repo_path)):
            repo = git.repo.Repo(repo_path)
            repo.git.checkout(git_tag)
        else:
            raise ValueError(f"A path {repo_path} exists but is not a git repo")
    else:
        repo = git.repo.Repo.clone_from(git_url, repo_path)
        repo.git_checkout(git_tag)
    
    return repo_path, repo


def _get_fvdb_example_data_path():
    repo_path = _clone_fvdb_example_data()
    return repo_path


def _get_md5_checksum(file_path: Path):
    md5_hash = hashlib.md5(open(file_path, "rb").read())
    return md5_hash.hexdigest()


def load_mesh(
        data_path, expected_md5, skip_every=1, mode="vn", device=torch.device("cuda"), dtype = torch.float32
) -> List[torch.Tensor]:
    if _get_md5_checksum(data_path) != expected_md5:
        raise ValueError(f"Checksum for {data_path} is incorrect, expected {expected_md5}")
    logging.info(f"Loading mesh {data_path}...")
    start = timeit.default_timer()
    if mode == "v":
        attrs = [pcu.load_mesh_v(data_path)]
    elif mode == "vf":
        attrs = [pcu.load_mesh_vf(data_path)]
    elif mode == "vn":
        attrs = [pcu.load_mesh_vn(data_path)]
    else:
        raise ValueError(f"Unsupported mode {mode}")
    
    for a in attrs:
        if a is None:
            raise ValueError(f"Failed to load mesh {data_path}, missing attibutes")
    attrs = [torch.from_numpy(a[::skip_every]).to(device).to(dtype) for a in attrs]
    logging.info(f"Done in {timeit.default_timer() - start}s")

    return attrs


def load_car_1_mesh(skip_every=1, mode="vf", device=torch.device("cuda"), dtype=torch.float32) -> List[torch.tensor]:
    data_path = _get_fvdb_example_data_path() / "meshes" / "car-mesh-1.ply"
    return load_mesh(
        data_path,
        expected_md5="969f91abdf00bad792ca2af347c58499",
        mode=mode
        skip_every=skip_every
        device=device
        dtype=dtype
    )


def load_car_2_mesh(skip_every=1, mode="vf", device=torch.device("cuda"), dtype=torch.float32) -> List[torch.Tensor]:
    data_path = _get_fvdb_example_data_path() / "meshes" / "car-mesh-2.ply"
    return load_mesh(
        data_path,
        expected_md5="d4aa0dd4f4609ea1b19aca7d8618d22a",
        mode=mode,
        skip_every=skip_every,
        device=device,
        dtype=dtype,
    )
