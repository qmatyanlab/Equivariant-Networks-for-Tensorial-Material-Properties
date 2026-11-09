from tqdm import tqdm
from mp_api.client import MPRester
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Union, List
import warnings
from piezo_utils import real_harmonic_components_piezo_model_basis as real_harmonic_components_piezo
import warnings
import pickle
#warnings.filterwarnings("ignore", category=np.ComplexWarning)

np.random.seed(7)

CACHE = Path("./piezo/raw_data.json")
TRAIN = Path("./piezo/train")
VALIDATE = Path("./piezo/validate")
TEST = Path("./piezo/test")

LIMITING_FACTOR = 5000

def get_crystal_structure_dict(mats, mpr: MPRester):
    """
    Structure of the DFPT calculation each tensor comes from: the task named in the piezoelectric document's `origins`.
    Both GGA and GGA+U DFPT tasks are accepted (MP applies +U to oxides and fluorides of Co, Cr, Fe, Mn, Mo, Ni, V, W).
    Until 2026-11-05 only tasks labelled 'GGA DFPT Dielectric' were accepted, which silently dropped every GGA+U
    material and, for materials with several DFPT tasks, could pair a tensor with another task's structure.

    Args:
        mats: piezoelectric documents with `material_id` and `origins`
        mpr: The mpr api instance

    Returns:
        A dictionary of material ids to structure dictionaries
    """
    mat_to_task = {}
    for m in mats:
        tids = [str(o.task_id) for o in (m.origins or []) if o.name == "piezoelectric"]
        if tids:
            mat_to_task[str(m.material_id)] = tids[0]
    task_ids = sorted(set(mat_to_task.values()))
    tasks = {}
    for i in range(0, len(task_ids), 5000):
        for task in mpr.materials.tasks.search(task_ids=task_ids[i:i+5000], fields=["calc_type", "structure", "task_id"]):
            tasks[str(task.task_id)] = task  # str(): MP task ids are AlphaIDs since 2026
    mat_id_to_dfpt_task_id, mat_id_to_struc, calc_types = {}, {}, {}
    for mid, tid in mat_to_task.items():
        task = tasks.get(tid)
        if task is None or "DFPT" not in str(task.calc_type):
            continue
        mat_id_to_dfpt_task_id[mid] = tid
        mat_id_to_struc[mid] = task.structure.as_dict()
        calc_types[mid] = str(task.calc_type)

    n_unique = len({str(m.material_id) for m in mats})  # the arrow-backed MPDataset can yield each document twice
    if n_unique != len(mat_id_to_dfpt_task_id):
        print(f"Missing {n_unique-len(mat_id_to_dfpt_task_id)} material structures for piezo!")
    with open("./piezo/mat_id_to_task_id.pkl", "wb") as f:
        pickle.dump(mat_id_to_dfpt_task_id, f)
    with open("./piezo/calc_types.json", "wt+") as f:
        json.dump(calc_types, f, indent=0)
    return mat_id_to_struc

def gather_data(mpr: MPRester, remove_deprecated = True):
    """
    Gets the piezo tensors and material_id from Materials Project

    Args:
        mpr: The mpr api instance.
    """
    data = []
    if remove_deprecated:
        mats = mpr.materials.piezoelectric.search(fields=["material_id", "origins", "total","ionic","electronic","deprecated"])
        non_dep_mats = []
        for mat in mats:
            if not mat.deprecated:
                non_dep_mats.append(mat)
        mats = non_dep_mats

    else:
        mats = mpr.materials.piezoelectric.search(fields=["material_id", "origins", "total","ionic","electronic"])
    
    mat_ids_to_struc = get_crystal_structure_dict(mats, mpr)
    for mat in tqdm(mats, "Processi" \
    "ng Materials"):
        if str(mat.material_id) in mat_ids_to_struc and mat.total is not None:
            data.append({
                "material_id": str(mat.material_id),
                "material": mat_ids_to_struc[str(mat.material_id)],
                "piezo_tensor": mat.total,
                "ionic_contribution": mat.ionic,
                "electric_contribution": mat.electronic
            })
    return data


def compute_spherical_harmonic_coefficients(data: List[dict]):
    """
    Computes the spherical harmonic coefficients of the voigt tensors in the data.

    Args:
        data: Data to compute spherical harmonics of

    Returns the data with the computed spherical harmonics
    """
    for d in tqdm(data, "Computing Real Spherical Harmonic Coefficients"):
        sph_coefs = real_harmonic_components_piezo(d["electric_contribution"])
        sph_coefs = sph_coefs.astype(float)
        d["sph_coefs"] = sph_coefs

def prune_outliers(data: List[dict]):
    """
    Prunes any spherical harmonics which seem to be extremely far outside the norm

    Args:
        data: The data to prune extreme spherical harmonics from

    Returns the data with all the removed outliers
    """
    new_data = []
    for d in tqdm(data, "Pruning Outliers"):
        if np.all(np.abs(d["sph_coefs"]) < LIMITING_FACTOR):
            new_data.append(d)
    return new_data

def collect_stats(data: List[dict]):
    """
    Computes mean and stdev of the data spherical harmonic coefficients
    and saves them to a file

    Args:
        data: Data to compute mean and stdev of.
    """
    sph_coefs = []

    for d in data:
        sph_coefs.append(d["sph_coefs"])

    sph_coefs = np.vstack(sph_coefs)

    sph_mean = np.mean(sph_coefs, axis=0)
    sph_std = np.std(sph_coefs, axis=0)

    coef_stats = {
        "sph_std": sph_std.tolist(),
        "sph_mean": sph_mean.tolist(),
    }

    with open("./piezo/coef_stats.json", "wt+") as f:
        json.dump(coef_stats, f, indent=2)

def save_data(data: List[dict], dest: Union[str, Path]):
    """
    Saves the data to a certain destination writing down the material_ids

    Args:
        data: The data to save.
        dest: The folder to save the data to.
    """
    dest = Path(dest)
    dest.mkdir(exist_ok=True)
    material_ids = [d["material_id"] for d in data]
    useful_data = [
        {
            "material": d["material"], 
            "piezo_tensor": d["electric_contribution"],
            "sph_coefs": d["sph_coefs"].tolist(),
        } 
    for d in data]
    with open(dest.joinpath("material_ids.csv"), "wt+") as f:
        f.write("\n".join(material_ids))
    with open(dest.joinpath("data.json"), "wt+") as f:
        json.dump(useful_data, f)

def get_by_ids(data: List[dict]):
    """
    Sorts the data into train, validate and test by the split of the original work (<split>/original2024_material_ids.csv,
    from the 2024 repository). Materials not in that split (new in the database) are not used, as for the elasticity
    data, so that the splits stay comparable with the original work.
    """
    pre = {s: set(p.joinpath("original2024_material_ids.csv").read_text().split()) for s, p in
           [("train", TRAIN), ("validate", VALIDATE), ("test", TEST)]}
    new = sorted(d["material_id"] for d in data if not any(d["material_id"] in v for v in pre.values()))
    print(f"Split: {len(data) - len(new)} materials from the 2024 split, {len(new)} new ones not used")
    out = {"train": [], "validate": [], "test": []}
    for d in data:
        for s in out:
            if d["material_id"] in pre[s]:
                out[s].append(d)
                break
    return out["train"], out["validate"], out["test"]

def main():
    Path("piezo").mkdir(exist_ok=True)

    if CACHE.exists():
        print("Using cached data.")
        with CACHE.open("r") as f:
            data = json.load(f)
    else:
        with open("APIKEY") as f:
            mpr = MPRester(f.read().strip())
        data = gather_data(mpr)
        with CACHE.open("wt+") as f:
            json.dump(data, f)

    # mp-api's arrow-backed MPDataset can yield each document twice; keep the first record per material_id
    seen = set()
    data = [d for d in data if not (d["material_id"] in seen or seen.add(d["material_id"]))]
    print(f"Unique materials after de-duplication: {len(data)}")

    # Put each tensor and its structure in one Cartesian frame (see frame_align.py); records that fit neither the
    # original nor the IEEE orientation are dropped and counted.
    from frame_align import align_frames
    data, frame_stats = align_frames(data, "electric_contribution", "piezo")
    print(f"Frame alignment: {frame_stats}")
    with open("./piezo/frame_stats.json", "wt+") as f:
        json.dump(frame_stats, f, indent=2)

    compute_spherical_harmonic_coefficients(data)

    data = prune_outliers(data)

    print(f"Number of total usable datapoints: {len(data)}")

    if all([p.joinpath("original2024_material_ids.csv").exists() for p in [TRAIN, VALIDATE, TEST]]):
        print("Using preset IDs.")
        train, validate, test = get_by_ids(data)
    else:
        train, test_and_validate = train_test_split(data, test_size=0.3)
        test, validate = train_test_split(test_and_validate, test_size=0.5) 

    collect_stats(train + validate + test)

    print(f"Number Training Samples: {len(train)}\nNumber Validation Samples: {len(validate)}\nNumber Testing Samples: {len(test)}")

    save_data(train, TRAIN)
    save_data(validate, VALIDATE)
    save_data(test, TEST)

if __name__ == "__main__":
    main()
