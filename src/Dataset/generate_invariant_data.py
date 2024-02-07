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

CACHE = Path("./scalar/raw_data.json")
TRAIN = Path("./scalar/train")
VALIDATE = Path("./scalar/validate")
TEST = Path("./scalar/test")

LIMITING_FACTOR = 5000

def gather_data(mpr: MPRester):
    """
    Gets the invariant (scalar) targets (for now band gap and formation energy), material_id, and structure from Materials Project

    Args:
        mpr: The mpr api instance.
    """
    data = []
    mats = mpr.materials.search(fields=["material_id", "band_gap","formation_energy_per_atom","structure"])
    mat_ids_to_struc = {}
    for mat in mats:
        mat_ids_to_struc[mat.material_id.string] = mat.structure.as_dict()
    for mat in tqdm(mats, "Processing Materials"):
        data.append({
            "material_id": mat.material_id.string,
            "material": mat_ids_to_struc[mat.material_id.string],
            "band_gap": mat.band_gap,
            "formation_energy": mat.formation_energy_per_atom
        })
    return data


def collect_stats(data: List[dict]):
    """
    Computes mean and stdev of the scalar data sets (band gap and formation energy)
    and saves them to a file

    Args:
        data: Data to compute mean and stdev of.
    """
    band_gaps = []
    formation_energies = []

    for d in data:
        band_gaps.append(d["band_gap"])
        formation_energies.append(d["formation_energy"])


    band_gaps = np.vstack(band_gaps)
    formation_energies = np.vstack(formation_energies)

    bandgap_mean = np.mean(band_gaps, axis=0)
    bandgap_std = np.std(band_gaps, axis=0)

    formen_mean = np.mean(formation_energies, axis=0)
    formen_std = np.std(formation_energies, axis=0)

    coef_stats = {
        "band_gap_std": bandgap_std.tolist(),
        "band_gap_mean": bandgap_mean.tolist(),
        "form_en_std": formen_std.tolist(),
        "form_en_mean": formen_mean.tolist(),
    }

    with open("./scalar/coef_stats.json", "wt+") as f:
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
            "fomation_energy": d["formation_energy"],
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
    Path("scalar").mkdir(exist_ok=True)

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
