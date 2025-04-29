import os
import sys
from tqdm import tqdm
import warnings
import yaml

from ase.io import read, write

from pyte.util.calc import calc_from_config
from pyte.util.relax import aar_from_config
from pyte.util.phonopy_utils import wrap_atoms, rotate_atoms, get_spgnum
from pyte.util.logger import Logger, LOG_ORDER
from pyte.scripts.parse_input import parse_config
from pyte.scripts.process_fcs import process_fcs_for_ph3
from pyte.scripts.process_conductivity import (
    process_shengbte_control,
    process_phono3py_conductivity
)


def relax_atoms_list(config, atoms_list, calc):
    logger = Logger()
    ase_atom_relaxer = aar_from_config(config, calc)
    relaxed_atoms_list = []
    for idx, atoms in enumerate(tqdm(atoms_list, desc='atom relax')):
        logger.log_progress_bar(idx, len(atoms_list), 'atom relax')
        logger.recorder.update_recorder(
            idx, 'Formula', atoms.get_chemical_formula(empirical=True)
        )
        atoms.info['init_spg_num'] = init_spg = get_spgnum(atoms)

        atoms, conv = ase_atom_relaxer.relax_atoms(atoms)
        atoms.calc = None
        spg_num = atoms.info['spg_num'] = get_spgnum(atoms)
        spg_same = spg_num == init_spg

        logger.recorder.update_recorder(idx, 'Conv', bool(conv))
        logger.recorder.update_recorder(idx, 'SPG_num', spg_num)
        logger.recorder.update_recorder(idx, 'SPG_same', spg_same)
        if not spg_same:
            warnings.warn(
                f'{idx}-th structure {atoms} changed spg '
                + f'from {init_spg} to {spg_num} while relaxing'
            )
        if not conv:
            step = config['relax']['steps']
            warnings.warn(
                f'{idx}-th structure {atoms} did not converged with in {step} steps!'
            )
        relaxed_atoms_list.append(atoms)
    logger.finalize_progress_bar()

    return relaxed_atoms_list


def main():
    logger = Logger('log.pyte')
    logger.greetings()
    logger.writeline('\nStarting pyte\n')

    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    logger.writeline(f'Reading config file {config_path} for pyte.')
    config = parse_config(config)
    logger.writeline('Reading config successful!')
    logger.writeline('\nConfigs for pyte')
    for key in ['data', 'calculator', 'relax', 'force_constant', 'conductivity']:
        logger.writeline(f'--------------  {key}  -------------')
        logger.log_config(config[key])
        logger.writeline('')

    calc = calc_from_config(config)
    if config['relax']['relaxed_input_path'] is None:
        ase_read_kwargs = config['data']['input_args']
        input_atoms_list = read(config['data']['input_path'], **ase_read_kwargs)
        logger.init_recorder(len(input_atoms_list))
        relaxed_atoms_list = relax_atoms_list(config, input_atoms_list, calc)

    else:
        relaxed_atoms_list = read(config['relax']['relaxed_input_path'], ':')
        logger.init_recorder(len(relaxed_atoms_list))

    # rotate, wrap ions pass pbc while relaxing (to avoid bug in thirdorder.py)
    relaxed_atoms_list = [
        wrap_atoms(rotate_atoms(atoms)) for atoms in relaxed_atoms_list
    ]

    if relax_path := config['data']['save_relax']:
        write(relax_path, relaxed_atoms_list)

    ph3_list = process_fcs_for_ph3(config, relaxed_atoms_list, calc)
    
    if config['conductivity']['solver_type'].lower() == 'shengbte':
        process_shengbte_control(config, relaxed_atoms_list, ph3_list)
    else:
        process_phono3py_conductivity(config, relaxed_atoms_list, ph3_list)

    logger.log_results()
    logger.log_terminate()

if __name__=='__main__':
    main()
