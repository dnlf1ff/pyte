from ase.constraints import FixSymmetry
from ase.filters import UnitCellFilter, FrechetCellFilter
from ase.optimize import LBFGS, FIRE

OPT_DICT = {'fire': FIRE, 'lbfgs': LBFGS}
FILTER_DICT = {'frechet': FrechetCellFilter, 'unitcell': UnitCellFilter}


class AseAtomRelax:
    def __init__(
        self,
        calc,
        opt,
        cell_filter=None,
        fix_symm=True,
        fmax=0.0001,
        steps=1000,
        log='-'
    ):
        self.calc = calc
        self.opt = opt
        self.cell_filter = cell_filter
        self.fix_symm = fix_symm
        self.fmax = fmax
        self.steps = steps
        self.log = log

    def relax_atoms(self, atoms):
        atoms = atoms.copy()
        atoms.calc = self.calc
        if self.fix_symm:
            atoms.set_constraint(FixSymmetry(atoms, symprec=1e-5))

        if self.cell_filter is not None:
            cf = self.cell_filter(atoms)
            opt = self.opt(cf, logfile=self.log)
        else:
            opt = self.opt(atoms, logfile=self.log)

        conv = opt.run(fmax=self.fmax, steps=self.steps)
        return atoms, conv


def aar_from_config(config, calc):
    arr_args = config['relax']
    arr_args.pop('relaxed_input_path', None)

    opt = OPT_DICT[arr_args['opt'].lower()]
    cell_filter = arr_args.get('cell_filter', None)
    if isinstance(cell_filter, str):
        cell_filter = FILTER_DICT[cell_filter.lower()]

    arr_args['calc'] = calc
    arr_args['opt'] = opt
    arr_args['cell_filter'] = cell_filter

    return AseAtomRelax(**arr_args)
    
    
