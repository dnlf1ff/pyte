import numpy as np
import os
import sys
from tqdm import tqdm

from ase import Atoms

from phono3py import Phono3py
from phono3py import file_IO as ph3_IO
from phonopy import file_IO as ph_IO

from pyte.thirdorder.thirdorder_ase import thirdorder_main, from_atoms
from pyte.thirdorder.thirdorder_common import gen_SPOSCAR, calc_dists, calc_frange
from pyte.util.logger import Logger
from pyte.util.calc import SevenNetBatchCalculator, single_point_calculate_list
from pyte.util.phonopy_utils import aseatoms2phonoatoms, get_supercell_matrix


def calculate_fc2(ph3, calc, symmetrize_fc2):
    desc = 'fc2 calculation'
    forces = []
    nat = len(ph3.phonon_supercell)
    indices = []
    atoms_list = []
    for idx, sc in enumerate(ph3.phonon_supercells_with_displacements):
        if sc is not None:
            atoms_list.append(Atoms(sc.symbols, cell=sc.cell, positions=sc.positions, pbc=True))
            indices.append(idx)

    if isinstance(calc, SevenNetBatchCalculator):
        result = calc.batch_calculate(atoms_list, desc=desc)
    else:
        result = single_point_calculate_list(atoms_list, calc, desc=desc)

    for idx, sc in enumerate(ph3.phonon_supercells_with_displacements):
        if sc is not None:
            atoms = result[indices.index(idx)]
            f = atoms.get_forces()
        else:
            f = np.zeros((nat, 3))
        forces.append(f)

    # append forces
    force_set = np.array(forces)
    ph3.phonon_forces = force_set
    ph3.produce_fc2(symmetrize_fc2=symmetrize_fc2)

    return ph3


def calculate_fc3_phono3py(ph3, calc, symmetrize_fc3):
    desc = 'fc3 calculation'
    forces = []
    nat = len(ph3.supercell)
    indices = []
    atoms_list = []
    for idx, sc in enumerate(ph3.supercells_with_displacements):
        if sc is not None:
            atoms_list.append(Atoms(sc.symbols, cell=sc.cell, positions=sc.positions, pbc=True))
            indices.append(idx)

    if isinstance(calc, SevenNetBatchCalculator):
        result = calc.batch_calculate(atoms_list, desc=desc)
    else:
        result = single_point_calculate_list(atoms_list, calc, desc=desc)

    for idx, sc in enumerate(ph3.supercells_with_displacements):
        if sc is not None:
            atoms = result[indices.index(idx)]
            f = atoms.get_forces()
        else:
            f = np.zeros((nat, 3))
        forces.append(f)

    # append forces
    force_set = np.array(forces)
    ph3.forces = force_set
    ph3.produce_fc3(symmetrize_fc3r=symmetrize_fc3)

    return ph3


def process_fcs_for_ph3(config, relaxed_atoms_list, calc):
    logger = Logger()
    ph3_list = []

    config_fc = config['force_constant']
    load_fc2 = config_fc['load_fc2']
    load_fc3 = config_fc['load_fc3']
    fc3_type = config_fc['fc3_type'].lower()

    save_fc2 = config['data']['save_fc2']
    save_fc3 = config['data']['save_fc3']

    symmetrize_fc2 = config_fc['symmetrize_fc2']
    symmetrize_fc3 = config_fc['symmetrize_fc3']
    if save_fc2:
        os.makedirs(save_fc2, exist_ok=True)
    if save_fc3:
        os.makedirs(save_fc3, exist_ok=True)

    for idx, atoms in enumerate(tqdm(relaxed_atoms_list, desc='processing fcs')):
        error = False
        logger.log_progress_bar(idx, len(relaxed_atoms_list), 'processing fcs')
        unit_cell = aseatoms2phonoatoms(atoms)
        fc2_supercell = (
            atoms.info[config_fc['fc2_supercell']]
            if isinstance(config_fc['fc2_supercell'], str)
            else get_supercell_matrix(config_fc['fc2_supercell'], atoms.get_cell())
        )
        fc3_supercell = (
            atoms.info[config_fc['fc3_supercell']]
            if isinstance(config_fc['fc3_supercell'], str)
            else get_supercell_matrix(config_fc['fc3_supercell'], atoms.get_cell())
        )

        fc2_super_info = f'[{",".join(map(str, np.diagonal(fc2_supercell)))}]'
        fc3_super_info = f'[{",".join(map(str, np.diagonal(fc3_supercell)))}]'

        ph3 = Phono3py(
            unitcell=unit_cell,
            supercell_matrix=fc3_supercell,
            phonon_supercell_matrix=fc2_supercell,
            symprec=1e-5,
        )

        cutoff = config_fc['fc3_cutoff']
        if isinstance(cutoff, str):
            cutoff = atoms.info[cutoff]
        if cutoff < 0 and config_fc['fc3_type'].lower() == 'phonopy':
            poscar = from_atoms(atoms)
            sposcar = gen_SPOSCAR(poscar, *np.diag(fc2_supercell))
            dmin, _, _ = calc_dists(sposcar)
            cutoff = calc_frange(poscar, sposcar, -cutoff, dmin) * 10  # nm to Ang
        if cutoff > 0 and config_fc['fc3_type'].lower() == 'shengbte':
            cutoff /= 10  # Ang to nm
        ph3.generate_displacements(
            distance=config_fc['displacement'],
            cutoff_pair_distance=cutoff,
        )

        if load_fc2:
            fc2 = ph_IO.parse_FORCE_CONSTANTS(f'{load_fc2}/FORCE_CONSTANTS_2ND_{idx}')
            ph3.fc2 = fc2
        else:
            try:
                ph3 = calculate_fc2(ph3, calc, symmetrize_fc2)
                if save_fc2:
                    ph_IO.write_FORCE_CONSTANTS(
                        ph3.fc2,
                        filename=f'{save_fc2}/FORCE_CONSTANTS_2ND_{idx}',
                    )
            except Exception as e:
                sys.stderr.write(f'FC2 calc error at {idx}: {e}\n')
                error = True

        num_fc2 = sum(
            [1 for sc in ph3.phonon_supercells_with_displacements if sc is not None]
        )

        if load_fc3:
            fc3 = ph3_IO.read_fc3_from_hdf5(f'{load_fc3}/fc3_{idx}.hdf5')
            ph3.fc3 = fc3
        elif fc3_type == 'phonopy':
            try:
                ph3 = calculate_fc3_phono3py(ph3, calc, symmetrize_fc3)
                if save_fc3:
                    ph3_IO.write_fc3_to_hdf5(
                        ph3.fc3,
                        filename=f'{save_fc3}/fc3_{idx}.hdf5',
                    )
            except Exception as e:
                sys.stderr.write(f'FC3 calc error at {idx}: {e}\n')
                error = True

            num_fc3 = sum(
                [1 for sc in ph3.supercells_with_displacements if sc is not None]
            )
        else:
            assert np.all(fc3_supercell == np.diag(np.diag(fc3_supercell)))
            fc3_supercell = np.diag(fc3_supercell)
            try:
                num_fc3 = thirdorder_main(
                    *fc3_supercell,
                    cutoff,
                    atoms,
                    calc,
                    f'{save_fc3}/FORCE_CONSTANTS_3RD_{idx}'
                )
            except Exception as e:
                num_fc3 = 0
                sys.stderr.write(f'FC3 calc error at {idx}: {e}\n')
                error = True
        ph3_list.append(ph3)
        logger.recorder.update_recorder(
            idx, 'FC2_super', fc2_super_info+f'*{num_fc2}'
        )
        logger.recorder.update_recorder(
            idx, 'FC3_super', fc3_super_info+f'*{num_fc3}'
        )
        logger.recorder.update_recorder(idx, 'FC_calc_error', error)
    logger.finalize_progress_bar()
    return  ph3_list 
