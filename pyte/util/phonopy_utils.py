import numpy as np
import math
import warnings

from ase import Atoms
import spglib

from phonopy.structure.atoms import PhonopyAtoms


def phonoatoms2aseatoms(phonoatoms):
    atoms = Atoms(
        phonoatoms.symbols,
        cell=phonoatoms.cell,
        positions=phonoatoms.positions,
        pbc=True
    )
    return atoms


def aseatoms2phonoatoms(atoms):
    phonoatoms = PhonopyAtoms(
        atoms.symbols,
        cell=atoms.cell,
        positions=atoms.positions,
    )
    return phonoatoms


def get_supercell_matrix(approx_length, cell):
    a, b, c = cell.lengths()

    if not isinstance(approx_length, list):
        approx_length = [approx_length] * 3
    mula = math.ceil(approx_length[0]/a)
    mulb = math.ceil(approx_length[1]/b)
    mulc = math.ceil(approx_length[2]/c)
    return np.diag([mula, mulb, mulc])


def get_mesh(density, cell):
    rec_cell = cell.reciprocal()
    a, b, c = rec_cell.lengths()*2*np.pi

    if not isinstance(density, list):
        density = [density] * 3
    mesha = math.ceil(a*density[0])
    meshb = math.ceil(b*density[1])
    meshc = math.ceil(c*density[2])
    return [mesha, meshb, meshc]


def check_imaginary_freqs(frequencies: np.ndarray) -> bool:
    try:
        if np.all(np.isnan(frequencies)):
            return True

        if np.any(frequencies[0, 3:] < 0):
            return True

        if np.any(frequencies[0, :3] < -1e-2):
            return True

        if np.any(frequencies[1:] < 0):
            return True
    except Exception as e:
        warnings.warn(f"Failed to check imaginary frequencies: {e!r}")

    return False


def get_spgnum(atoms, symprec=1e-5):
    cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    spgdat = spglib.get_symmetry_dataset(cell, symprec=symprec)
    return spgdat.number


def wrap_atoms(atoms, symprec=1e-5):
    spos = atoms.get_scaled_positions()
    spos[spos>1-symprec] = 0.
    spos[spos<symprec] = 0.
    atoms.set_scaled_positions(spos)
    return atoms


def rotate_atoms(orig_atoms, prec=1e-10):
    orig_spgnum = orig_atoms.info['spg_num']
    atoms = orig_atoms.copy()
    cell = atoms.get_cell()
    a, b, c = cell.lengths()
    alpha, beta, gamma = cell.angles() * np.pi / 180

    c2_factor = np.cos(alpha)-np.cos(beta)*np.cos(gamma)
    c2_factor /= np.sin(gamma)
    c3_factor = (1-np.cos(beta)**2-c2_factor**2)**(0.5)

    rot_cell = np.array(
        [
            [a, 0., 0.],
            [b*np.cos(gamma), b*np.sin(gamma), 0.],
            [c*np.cos(beta), c*c2_factor, c*c3_factor],
        ]
    )
    rot_cell[np.abs(rot_cell)<prec] = 0.
    atoms.set_cell(rot_cell, scale_atoms=True, apply_constraint=False)
    spgnum = get_spgnum(atoms)
    if spgnum == orig_spgnum:
        return atoms

    warnings.warn(f'Rotating atoms change symmetry (from {orig_spgnum} to {spgnum}')
    warnings.warn(f'We will not rotate atoms, but be careful when using ShengBTE')
    return orig_atoms
