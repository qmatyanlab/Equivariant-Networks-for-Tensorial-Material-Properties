from tqdm import tqdm
from mp_api.client import MPRester
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Union, List
import warnings
from elastic_utils import voigt2realsphvectors
import warnings
warnings.filterwarnings("ignore", category=np.ComplexWarning)

np.random.seed(7)

CACHE = Path("./elastic/raw_data.json")
TRAIN = Path("./elastic/train")
VALIDATE = Path("./elastic/validate")
TEST = Path("./elastic/test")

LIMITING_FACTOR = 5000

def get_crystal_structure_dict(mat_ids: List[str], mpr: MPRester):
    """
      Gets the dictionary of the unit cells of the material ids. 

      Args:
        mat_ids: The material ids to get the structures of
        mpr: The mpr api instance

      Returns: 
        A dictionary of material ids to structure dictionaries
    """
    mats = []
    for i in range(0, len(mat_ids), 10000):
        mats.extend(mpr.materials.search(material_ids=mat_ids[i:i+10000], fields=["material_id", "structure"]))
    mat_id_to_struc = {}
    for mat in mats:
        mat_id_to_struc[mat.material_id.string] = mat.structure.as_dict()
    return mat_id_to_struc

def gather_data(mpr: MPRester):
    """
    Gets the elasticity tensors and material_id from Materials Project

    Args:
        mpr: The mpr api instance.
    """
    data = []
    mats = mpr.materials.elasticity.search(fields=["material_id", "elastic_tensor"])
    mat_ids_to_struc = get_crystal_structure_dict([m.material_id for m in mats], mpr)
    for mat in tqdm(mats, "Processing Materials"):
        if mat.material_id.string in mat_ids_to_struc and mat.elastic_tensor is not None:
            data.append({
                "material_id": mat.material_id.string,
                "material": mat_ids_to_struc[mat.material_id.string],
                "elasticity_tensor": mat.elastic_tensor.ieee_format
            })
    return data

def remove_impossibilities(data: List[dict]):
    """
    Removes all impossible elastic tensors in the data.
    These are elastic tensors with 0 or negative eigenvalues
    as those shouldn't exist

    Args:
        data: The data to prune impossible elastic tensors

    Returns the new data with removed elastic tensors.
    """
    new_data = []
    for d in tqdm(data, "Removing Illegal Elastic Tensors"):
        if np.all(np.linalg.eigvals(d["elasticity_tensor"]) > 0):
            new_data.append(d)
    return new_data

def compute_spherical_harmonic_coefficients(data: List[dict]):
    """
    Computes the spherical harmonic coefficients of the voigt tensors in the data.

    Args:
        data: Data to compute spherical harmonics of

    Returns the data with the computed spherical harmonics
    """
    for d in tqdm(data, "Computing Real Spherical Harmonic Coefficients"):
        asymm_coefs, symm_coefs = voigt2realsphvectors(d["elasticity_tensor"])
        asymm_coefs = asymm_coefs.astype(float)
        symm_coefs = symm_coefs.astype(float)
        d["asymmetric_coefs"] = asymm_coefs
        d["symmetric_coefs"] = symm_coefs

def prune_outliers(data: List[dict]):
    """
    Prunes any spherical harmonics which seem to be extremely far outside the norm

    Args:
        data: The data to prune extreme spherical harmonics from

    Returns the data with all the removed outliers
    """
    new_data = []
    for d in tqdm(data, "Pruning Outliers"):
        if np.all(np.abs(d["asymmetric_coefs"]) < LIMITING_FACTOR) and np.all(np.abs(d["symmetric_coefs"]) < LIMITING_FACTOR):
            new_data.append(d)
    return new_data

def collect_stats(data: List[dict]):
    """
    Computes mean and stdev of the data spherical harmonic coefficients
    and saves them to a file

    Args:
        data: Data to compute mean and stdev of.
    """
    symmetric_coefs = []
    asymmetric_coefs = []

    for d in data:
        symmetric_coefs.append(d["symmetric_coefs"])
        asymmetric_coefs.append(d["asymmetric_coefs"])

    symmetric_coefs = np.vstack(symmetric_coefs)
    asymmetric_coefs = np.vstack(asymmetric_coefs)

    symmetric_mean = np.mean(symmetric_coefs, axis=0)
    symmetric_std = np.std(symmetric_coefs, axis=0)
    asymmetric_mean = np.mean(asymmetric_coefs, axis=0)
    asymmetric_std = np.std(asymmetric_coefs, axis=0)

    coef_stats = {
        "symmetric_std": symmetric_std.tolist(),
        "symmetric_mean": symmetric_mean.tolist(),
        "asymmetric_std": asymmetric_std.tolist(),
        "asymmetric_mean": asymmetric_mean.tolist(),
    }

    with open("./elastic/coef_stats.json", "wt+") as f:
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
            "elasticity_tensor": d["elasticity_tensor"],
            "symmetric_coefs": d["symmetric_coefs"].tolist(),
            "asymmetric_coefs": d["asymmetric_coefs"].tolist()
        } 
    for d in data]
    with open(dest.joinpath("material_ids.csv"), "wt+") as f:
        f.write("\n".join(material_ids))
    with open(dest.joinpath("data.json"), "wt+") as f:
        json.dump(useful_data, f)

def get_by_ids(data: List[dict]):
    """
    Sorts the data into train, test, and validate by the IDs.

    Args:
        data: The data dictionary to sort

    Returns:
        The train, validation, and test sets by the pregotten ids.
    """
    train, validate, test = [], [], []
    train_ids = TRAIN.joinpath("material_ids.csv").read_text().split("\n") 
    validate_ids = VALIDATE.joinpath("material_ids.csv").read_text().split("\n") 
    test_ids = TEST.joinpath("material_ids.csv").read_text().split("\n") 
    for d in data:
        if d["material_id"] in train_ids:
            train.append(d)
        elif d["material_id"] in validate_ids:
            validate.append(d)
        elif d["material_id"] in test_ids:
            test.append(d)
        else:
            warnings.warn("Task ID not in used IDs found. Left unused.")
    return train, validate, test

def main():
    Path("elastic").mkdir(exist_ok=True)

    if CACHE.exists():
        print("Using cached data.")
        with CACHE.open("r") as f:
            data = json.load(f)
    else:
        with open("APIKEY") as f:
            mpr = MPRester(f.read())
        data = gather_data(mpr)
        with CACHE.open("wt+") as f:
            json.dump(data, f)

    data = remove_impossibilities(data)

    compute_spherical_harmonic_coefficients(data)

    data = prune_outliers(data)

    print(f"Number of total usable datapoints: {len(data)}")

    if all([p.joinpath("material_ids.csv").exists() for p in [TRAIN, VALIDATE, TEST]]):
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