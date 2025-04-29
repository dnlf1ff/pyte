import os
import sys
from tqdm import tqdm
import warnings

from pyte.util.logger import Logger
from pyte.util.phonopy_utils import get_mesh, check_imaginary_freqs


def _get_mesh_from_config(atoms, config):
    mesh = (
        atoms.info[config['conductivity']['q_points']]
        if isinstance(config['conductivity']['q_points'], str)
        else get_mesh(
            config['conductivity']['q_points'],
            atoms.get_cell(),
        )
    )
    return mesh


def write_shengbte_control(ph3, atoms, config, filename):
    fp = open(filename, 'w')
    fp.write('&allocations\n')
    ordered_elems = []
    for elem in atoms.get_chemical_symbols():
        if elem not in ordered_elems:
            ordered_elems.append(elem)

    fp.write(f'        nelements={len(ordered_elems)},\n')
    fp.write(f'        natoms={len(atoms)},\n')

    mesh = _get_mesh_from_config(atoms, config)
    na, nb, nc = mesh
    fp.write(f'        ngrid(:)={na} {nb} {nc}\n')
    fp.write('&end\n')
    fp.write('&crystal\n')
    fp.write('        lfactor = 0.1\n')
    cell_T = atoms.get_cell()
    for i in range(3):
        fp.write(f'        lattvec(:,{i+1}) = {cell_T[i][0]} {cell_T[i][1]} {cell_T[i][2]}\n')
    fp.write('        elements =')
    for elem in ordered_elems:
        fp.write(f' "{elem}"')
    fp.write('\n')
    fp.write(f'        types =')
    for elem in atoms.get_chemical_symbols():
        fp.write(f' {ordered_elems.index(elem) + 1}')
    fp.write('\n')

    for i, spos in enumerate(atoms.get_scaled_positions()):
        fp.write(f'        positions(:,{i+1}) =')
        for sp in spos:
            fp.write(f' {sp}')
        fp.write('\n')

    scell = ph3.phonon_supercell.supercell_matrix
    fp.write(f'        scell(:) = {scell[0][0]} {scell[1][1]} {scell[2][2]}\n')
    fp.write('&end\n')
    fp.write('&parameters\n')
    if isinstance(temp := config['conductivity']['temperature'], list):
        fp.write(f'        T_min = {temp[0]}\n')
        fp.write(f'        T_max = {temp[1]}\n')
        fp.write(f'        T_step = {temp[2]}\n')

    else:
        fp.write(f'        T = {temp}\n')
    fp.write('        scalebroad=0.1\n')
    fp.write('&end\n')
    fp.write('&flags\n')
    fp.write('        nonanalytic = .false.\n')
    conv = '.true.' if config['conductivity']['convergence'] else '.false.'
    fp.write(f'        convergence = {conv}\n')
    isotope = '.true.' if config['conductivity']['is_isotope'] else '.false.'
    fp.write(f'        isotopes = {isotope}\n')
    fp.write('&end\n')
    fp.close()


def process_shengbte_control(config, relaxed_atoms_list, ph3_list):
    # TODO: check imaginary mode with phonopy?
    logger = Logger()
    ctrl_path = config['data']['save_control']
    os.makedirs(ctrl_path, exist_ok=True)

    for idx, (ph3, atoms) in enumerate(zip(ph3_list, relaxed_atoms_list)):
        mesh = _get_mesh_from_config(atoms, config)
        logger.recorder.update_recorder(
            idx, 'Q_mesh', f'[{",".join(map(str, mesh))}]'
        )
        write_shengbte_control(
            ph3,
            atoms,
            config,
            filename=f'{ctrl_path}/CONTROL_{idx}'
        )


def postprocess_kappa_to_csv(file, idx, temps, kappas):
    for temp, kappa in zip(temps, kappas):
        if kappa is None:
            kappa_join = 'NaN'
        else:
            kappa = kappa.reshape(-1)
            kappa_join = ','.join(map(str,kappa))
        file.write(f'{idx},{temp},{kappa_join}\n')


def process_phono3py_conductivity(config, relaxed_atoms_list, ph3_list):
    logger = Logger()
    save_path = config['data']['save_cond']
    os.makedirs(save_path, exist_ok=True)

    if isinstance(temp := config['conductivity']['temperature'], list):
        temperatures = list(range(temp[0], temp[1]+1, temp[2]))
    else:
        temperatures = [temp]
    if config['conductivity']['cond_type'].lower() == 'bte':
        conductivity_type = None
    else:
        conductivity_type = 'wigner'

    csv_tot = open(f'{save_path}/kappa_total.csv', 'w', buffering=1)
    csv_tot.write(f'index,temperature,xx,yy,zz,yz,xz,xy\n')
    if conductivity_type == 'wigner':
        csv_p = open(f'{save_path}/kappa_p.csv', 'w', buffering=1)
        csv_p.write(f'index,temperature,xx,yy,zz,yz,xz,xy\n')
        csv_c = open(f'{save_path}/kappa_c.csv', 'w', buffering=1)
        csv_c.write(f'index,temperature,xx,yy,zz,yz,xz,xy\n')

    KAPPA_KEYS = ['kappa', 'kappa_TOT_RTA', 'kappa_P_RTA', 'kappa_C']
    for idx, (ph3, atoms) in tqdm(enumerate(zip(
        ph3_list, relaxed_atoms_list)), desc='conductivity calculation'
    ):
        logger.log_progress_bar(
            idx, len(relaxed_atoms_list), 'conductivity calculation'
        )
        mesh = _get_mesh_from_config(atoms, config)
        ph3.mesh_numbers = mesh
        logger.recorder.update_recorder(
            idx, 'Q_mesh', f'[{",".join(map(str, mesh))}]'
        )
        try:
            ph3.init_phph_interaction(symmetrize_fc3q=False)
            ph3.run_phonon_solver()
            freqs, eigvecs, grid = ph3.get_phonon_data()
            has_imag = check_imaginary_freqs(freqs)
            logger.recorder.update_recorder(idx, 'Imaginary', has_imag)
            if has_imag:
                warnings.warn(f'{idx}-th structure {atoms} has imaginary frequencies!')

            ph3.run_thermal_conductivity(
                temperatures=temperatures,
                is_isotope=config['conductivity']['is_isotope'],
                conductivity_type=conductivity_type,
                boundary_mfp=1e6,  # kSRME
            )
            cond = ph3.thermal_conductivity
            cond_dict = {
                k: getattr(cond, k) for k in KAPPA_KEYS if hasattr(cond, k)
            }
            
        except Exception as e:
            sys.stderr.write(f'Conductivity error in {idx}: {e}\n')
            nones = [None for _ in temperatures]
            cond_dict = {key: nones for key in KAPPA_KEYS}

        total_key = 'kappa_TOT_RTA' if conductivity_type == 'wigner' else 'kappa'
        postprocess_kappa_to_csv(csv_tot, idx, temperatures, cond_dict[total_key])
        if conductivity_type == 'wigner':
            postprocess_kappa_to_csv(
                csv_p, idx, temperatures, cond_dict['kappa_P_RTA']
            )
            postprocess_kappa_to_csv(csv_c, idx, temperatures, cond_dict['kappa_C'])
    logger.finalize_progress_bar()
    csv_tot.close()
    if conductivity_type == 'wigner':
        csv_p.close()
        csv_c.close()
