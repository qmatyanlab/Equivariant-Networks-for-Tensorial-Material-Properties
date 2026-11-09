"""Make each tensor and its structure share one Cartesian frame.

Materials Project tensors (piezoelectric, elastic `ieee_format`) are given in the IEEE standard frame of the
structure, which is not always the frame of the structure that comes with them. An equivariant model needs both in
the same frame. For each record we compare the tensor with the point-group projector of the structure in its original
orientation and after rotation into the IEEE frame (spglib via pymatgen, symprec 1e-3), keep the orientation in which
the tensor obeys the structure's symmetry, and drop records for which neither does (relative residual > TOL).
"""
import numpy as np
from pymatgen.core import Structure
from pymatgen.core.tensors import Tensor
from pymatgen.core.operations import SymmOp
from pymatgen.analysis.elasticity import ElasticTensor
from pymatgen.analysis.piezo import PiezoTensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

TOL = 0.01
SYMPREC = 1e-3


def full_tensor(kind, raw):
    a = np.array(raw, dtype=float)
    if kind == "dielectric":
        return Tensor(a.reshape(3, 3))
    if kind == "piezo":
        return PiezoTensor.from_voigt(a.reshape(3, 6))
    return ElasticTensor.from_voigt(a.reshape(6, 6))


def symmetry_residual(tensor, structure):
    """||T - P_G T|| / ||T|| with P_G the average over the structure's Cartesian point-group operations."""
    ops = SpacegroupAnalyzer(structure, symprec=SYMPREC).get_point_group_operations(cartesian=True)
    proj = np.mean([tensor.transform(op) for op in ops], axis=0)
    n = np.linalg.norm(tensor)
    return float(np.linalg.norm(tensor - proj) / n) if n > 0 else 0.0


def ieee_rotated(structure):
    rot = Tensor(np.zeros((3, 3))).get_ieee_rotation(structure)
    s = structure.copy()
    s.apply_operation(SymmOp.from_rotation_and_translation(rot, [0, 0, 0]))
    return s


def align_frames(data, tensor_key, kind):
    """Returns (kept records, stats). Each kept record gets 'frame' ('original' or 'ieee') and 'frame_residual'."""
    kept, stats = [], {"original": 0, "ieee": 0, "dropped": 0, "error": 0}
    for d in data:
        try:
            s = Structure.from_dict(d["material"])
            t = full_tensor(kind, d[tensor_key])
            r0 = symmetry_residual(t, s)
            best = ("original", r0, s)
            if r0 > TOL:
                s1 = ieee_rotated(s)
                r1 = symmetry_residual(t, s1)
                if r1 < r0:
                    best = ("ieee", r1, s1)
        except Exception:
            stats["error"] += 1
            continue
        if best[1] > TOL:
            stats["dropped"] += 1
            continue
        stats[best[0]] += 1
        d = dict(d, material=best[2].as_dict(), frame=best[0], frame_residual=best[1])
        kept.append(d)
    return kept, stats
