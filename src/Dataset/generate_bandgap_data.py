from tqdm import tqdm
from mp_api.client import MPRester
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Union, List
import warnings
import warnings
warnings.filterwarnings("ignore", category=np.ComplexWarning)

np.random.seed(7)

CACHE = Path("./bandgap/raw_data.json")
TRAIN = Path("./bandgap/train")
VALIDATE = Path("./bandgap/validate")
TEST = Path("./bandgap/test")

def gather_data(mpr: MPRester):
    """
    Gets the elasticity tensors and material_id from Materials Project

    Args:
        mpr: The mpr api instance.
    """
    data = []
    mats = mpr.materials.summary.search(fields=["material_id", "structure", "band_gap"])
    for mat in tqdm(mats, "Processing Materials"):
        if mat.structure is not None and mat.band_gap is not None:
            data.append({
                "material_id": mat.material_id.string,
                "material": mat.structure,
                "band_gap": mat.band_gap
            })
    return data

def collect_stats(data: List[dict]):
    """
    Computes mean and stdev of the data spherical harmonic coefficients
    and saves them to a file

    Args:
        data: Data to compute mean and stdev of.
    """
    band_gap = []

    for d in data:
        band_gap.append(d["band_gap"])

    band_gap = np.vstack(band_gap)

    band_gap_mean = np.mean(band_gap, axis=0)
    band_gap_std = np.std(band_gap, axis=0)

    coef_stats = {
        "band_gap_std": band_gap_std.tolist(),
        "band_gap_mean": band_gap_mean.tolist(),
    }

    with open("./bandgap/coef_stats.json", "wt+") as f:
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
            "band_gap": d["band_gap"],
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
    Path("bandgap").mkdir(exist_ok=True)

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