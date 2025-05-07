import os


class Essential:
    pass


DEFAULT_DATA_CONFIG = {
    'input_path': Essential(),
    'input_args': {'index': ':'},
    'save_relax': False,
    'save_fc2': False,
    'save_fc3': False,
    'save_cond': False,
    'save_control': False,
}


DEFAULT_CALC_CONFIG = {
    'calc_type': 'sevennet',
    'path': Essential(),
    'calc_args': {},
    'batch_size': None,
    'avg_atom_num': None,
}


DEFAULT_RELAX_CONFIG = {
    'relaxed_input_path': None,
    'fmax': 0.0001,
    'steps': 1000,
    'opt': 'fire',
    'cell_filter': 'frechet',
    'fix_symm': True,
    'log': '-',
}


DEFAULT_FC_CONFIG = {
    'displacement': 0.03,  # phono3py default
    'fc2_supercell': 25.,
    'fc3_supercell': 15.,
    'fc2_type': 'phonopy',
    'fc3_type': 'shengbte',
    'fc3_cutoff': 10000000,
    'symmetrize_fc2': False,  # phonopy default
    'symmetrize_fc3': True,
    'load_fc2': None,
    'load_fc3': None,
}


DEFAULT_COND_CONFIG = {
    'solver_type': 'shengbte',
    'cond_type': 'bte',
    'q_points': 19,
    'temperature': 300,
    'is_isotope': True,
    'convergence': False,
}


def update_config_with_defaults(config):
    key_parse_pair = {
        'data': DEFAULT_DATA_CONFIG,
        'calculator': DEFAULT_CALC_CONFIG,
        'relax': DEFAULT_RELAX_CONFIG,
        'force_constant': DEFAULT_FC_CONFIG,
        'conductivity': DEFAULT_COND_CONFIG,
    }

    for key, default_config in key_parse_pair.items():
        config_parse = default_config.copy()
        config_parse.update(config[key])

        for k, v in config_parse.items():
            if not isinstance(v, Essential):
                continue
            raise ValueError(f'{key}: {k} must be given')
        config[key] = config_parse

    return config


def _isinstance_in_list(inp, insts):
    return any([isinstance(inp, inst) for inst in insts])


def _islistinstance(inps, insts):
    return all([_isinstance_in_list(inp, insts) for inp in inps])


def check_calc_config(config):
    config_calc = config['calculator']
    assert config_calc['calc_type'].lower() in ['sevennet', 'sevennet-batch', 'custom']
    assert isinstance(config_calc['path'], str)
    assert _isinstance_in_list(config_calc['batch_size'], [int, type(None)])
    assert _isinstance_in_list(config_calc['avg_atom_num'], [int, type(None)])


def check_relax_config(config):
    config_relax = config['relax']
    if (relaxed_path := config_relax['relaxed_input_path']) is not None:
        os.makedirs(relaxed_path, exist_ok=True)
#        assert os.path.isfile(relaxed_path)
        return

    assert os.path.isfile(config['data']['input_path'])
    assert isinstance(config_relax['fmax'], float)
    assert isinstance(config_relax['steps'], int)
    assert config_relax['opt'].lower() in ['lbfgs', 'fire']
    assert config_relax['cell_filter'].lower() in ['unitcell', 'frechet']
    assert isinstance(config_relax['fix_symm'], bool)
    assert isinstance(config_relax['log'], str)


def check_fc_config(config):
    config_fc = config['force_constant']
    pass_fc2, pass_fc3 = False, False
    if (load_fc2 := config_fc['load_fc2']) is not None:
        os.makedirs(load_fc2, exist_ok=True)
#        assert os.path.isdir(load_fc2)
        pass_fc2 = True

    else:
        assert (
            _isinstance_in_list(config_fc['fc2_supercell'], [float, int])
            or _islistinstance(config_fc['fc2_supercell'], [float, int])
            or isinstance(config_fc['fc2_supercell'], str)
        )
        assert isinstance(config_fc['symmetrize_fc2'], bool)

    if (load_fc3 := config_fc['load_fc3']) is not None:
        os.makedirs(load_fc3, exist_ok=True)
#        assert os.path.isdir(load_fc3)
        assert config_fc['fc3_type'].lower() == 'phonopy'
        pass_fc3 = True

    else:
        assert (
            _isinstance_in_list(config_fc['fc3_supercell'], [float, int])
            or _islistinstance(config_fc['fc3_supercell'], [float, int])
            or isinstance(config_fc['fc3_supercell'], str)
        )
        assert isinstance(config_fc['symmetrize_fc3'], bool)
    if not(pass_fc2 and pass_fc3):
        assert isinstance(config_fc['displacement'], float)

    assert config_fc['fc2_type'].lower() == 'phonopy'
    assert config_fc['fc3_type'].lower() in ['phonopy', 'shengbte']

    if not pass_fc3:
        if _isinstance_in_list(config_fc['fc3_cutoff'], [float, int]):
            if config_fc['fc3_cutoff'] < 0:
                assert isinstance(config_fc['fc3_cutoff'], int)
        else:
            assert isinstance(config_fc['fc3_cutoff'], str)


def check_cond_config(config):
    config_cond = config['conductivity']
    assert (
        _isinstance_in_list(config_cond['q_points'], [float, int])
        or _islistinstance(config_cond['q_points'], [float, int])
        or isinstance(config_cond['q_points'], str)
    )
    if (solver_type := config_cond['solver_type'].lower()) == 'shengbte':
        assert config['data']['save_fc2']
        assert config['data']['save_fc3']
        assert config['data']['save_control']
        assert config['force_constant']['fc3_type'].lower() == 'shengbte'
        assert config_cond['cond_type'].lower() == 'bte'
        assert isinstance(config_cond['convergence'], bool)
    elif solver_type == 'phonopy':
        assert config['data']['save_cond']
        assert config_cond['cond_type'].lower() in ['bte', 'wte']
        assert config_cond['convergence'] is False

    assert (
        _isinstance_in_list(config_cond['temperature'], [float, int])
        or _islistinstance(config_cond['temperature'], [float, int])
    )
    assert isinstance(config_cond['is_isotope'], bool)


def parse_config(config):
    config = update_config_with_defaults(config)
    check_calc_config(config)
    check_relax_config(config)
    check_fc_config(config)
    check_cond_config(config)

    return config
