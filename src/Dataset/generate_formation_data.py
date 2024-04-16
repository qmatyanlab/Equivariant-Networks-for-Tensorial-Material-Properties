from tqdm import tqdm
from mp_api.client import MPRester
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Union, List
import warnings
warnings.filterwarnings("ignore", category=np.ComplexWarning)

np.random.seed(7)

CACHE = Path("./formation/raw_data.json")
TRAIN = Path("./formation/train")
VALIDATE = Path("./formation/validate")
TEST = Path("./formation/test")

def gather_data(mpr: MPRester):
    """
    Gets the elasticity tensors and material_id from Materials Project

    Args:
        mpr: The mpr api instance.
    """
    data = []
    mats = mpr.materials.summary.search(fields=["material_id", "structure", "formation_energy_per_atom"])
    for mat in tqdm(mats, "Processing Materials"):
        if mat.structure is not None and mat.formation_energy_per_atom is not None:
            data.append({
                "material_id": mat.material_id.string,
                "material": mat.structure.as_dict(),
                "formation_energy": mat.formation_energy_per_atom
            })
    return data

def collect_stats(data: List[dict]):
    """
    Computes mean and stdev of the data spherical harmonic coefficients
    and saves them to a file

    Args:
        data: Data to compute mean and stdev of.
    """
    formation_energy = []

    for d in data:
        formation_energy.append(d["formation_energy"])

    formation_energy = np.vstack(formation_energy)

    formation_energy_mean = np.mean(formation_energy, axis=0)
    formation_energy_std = np.std(formation_energy, axis=0)

    coef_stats = {
        "formation_energy_std": formation_energy_std.tolist(),
        "formation_energy_mean": formation_energy_mean.tolist(),
    }

    with open("./formation/coef_stats.json", "wt+") as f:
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
            "formation_energy": d["formation_energy"],
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
    Path("formation").mkdir(exist_ok=True)

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