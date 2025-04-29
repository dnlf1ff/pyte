import time
from datetime import timedelta

import pyte

GREETINGS = ""
GREETINGS += "  _ __  _   _   _    ___\n"
GREETINGS += " | '_ \| | | |_| |_ / _ \\\n"
GREETINGS += " | |_) | |_| |_   _|  __/\n"
GREETINGS += " | .__/ \__, | | |_ \___|\n"
GREETINGS += " |_|    |___/  |___|\n"
GREETINGS += f"                    {pyte.__version__}\n"

LOG_ORDER = ['Index', 'Formula', 'SPG_num', 'SPG_same', 'Conv', 'FC2_super', 'FC3_super', 'FC_calc_error', 'Q_mesh', 'Imaginary']
RES_INTERVAL = 3

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ProgressBar:
    def __init__(self, total, desc):
        self.total = total
        self.desc = desc
        self.init_time = time.time()
        self.bar_length = 100 - len(desc)


    def get_progress(self, idx):
        current_time = time.time()
        percent = (idx / self.total) * 100
        block = int(round(self.bar_length * idx / self.total))

        if idx == 0:
            elapsed_seconds = 0
            eta_time = '?'
        else:
            elapsed_seconds = int(current_time - self.init_time)
            eta_seconds = int(elapsed_seconds*(self.total-idx) / idx)
            eta_time = timedelta(seconds=eta_seconds)
        elapsed_time = timedelta(seconds=elapsed_seconds)
        progress = f"{self.desc}: {percent:.2f}%"
        progress += f" [{block * '='}{(self.bar_length - block) * ' '}] {idx}/{self.total}"
        progress += f" [{elapsed_time}<{eta_time}]"

        return progress


    def get_end_progress(self):
        percent = 100.
        elapsed_seconds = int(time.time() - self.init_time)
        eta_seconds = 0
        elapsed_time = timedelta(seconds=elapsed_seconds)
        eta_time = timedelta(seconds=eta_seconds)
        progress = f"{self.desc}: {percent:.2f}%"
        progress += f" [{self.bar_length * '='}] {self.total}/{self.total}"
        progress += f" [{elapsed_time}<{eta_time}]"

        return progress


class Recorder:
    def __init__(self, total_num):
        result_dict = {k: None for k in LOG_ORDER if k != 'Index'}
        self.result_dicts = [result_dict.copy() for _ in range(total_num)]


    def update_recorder(self, idx, key, val):
        self.result_dicts[idx].update({key: val})


class Logger(metaclass=Singleton):
    def __init__(self, filename):
        self.fp = open(filename, 'w+', buffering=1)
        self.init_time = time.time()
        self._erase_final = []


    def init_recorder(self, total_num):
        self.recorder = Recorder(total_num)


    def writeline(self, line):
        self.fp.write(line + '\n')


    def greetings(self):
        self.fp.write(GREETINGS)


    def log_config(self, config):
        max_len = max([len(k) for k in config])
        for k, v in config.items():
            self.writeline(f'{k:<{max_len}} : {v}')


    def log_progress_bar(self, idx, total_len, desc='pbar'):
        if idx == 0:
            self._epos_init = self.fp.tell()
            self.pbar = ProgressBar(total_len, desc)
        progress = self.pbar.get_progress(idx)
        self.fp.write(f'\r{progress}')


    def finalize_progress_bar(self):
        progress = self.pbar.get_end_progress()
        self._erase_final += list(range(self._epos_init, self.fp.tell()+1))
        del self._epos_init
        del self.pbar
        self.fp.write(f'\r{progress}')
        self.fp.write('\n')


    def log_bar(self, size=100):
        self.writeline('-'*size)


    def _make_string_with_space(self, strings, widths):
        final = ''
        for string, width in zip(strings, widths):
            final += f'{string:>{width+RES_INTERVAL}}'
        return final


    def log_results(self):
        self.writeline('')
        res_dicts = self.recorder.result_dicts
        max_len_dict = {k: [len(f'{v}')] for k, v in res_dicts[0].items()}
        for idx, res_dict in enumerate(res_dicts):
            res_dict['Index'] = idx
            if idx == 0:
                max_len_dict = {k: [] for k in res_dict.keys()}

            for key in LOG_ORDER:
                v = res_dict[key]
                res_dict[key] = f'{v}'
                max_len_dict[key].append(len(f'{v}'))

        max_len_dict = {k: max(max(v), len(k)) for k, v in max_len_dict.items()}
        widths = [max_len_dict[key] for key in LOG_ORDER]
        bar_len = RES_INTERVAL*(len(LOG_ORDER)+1) + sum(max_len_dict.values())

        self.log_bar(bar_len)
        self.writeline(self._make_string_with_space(LOG_ORDER, widths))
        for idx, res_dict in enumerate(res_dicts):
            if idx%10==0:
                self.log_bar(bar_len)
            strings = [res_dict[key] for key in LOG_ORDER]
            self.writeline(self._make_string_with_space(strings, widths))
        self.log_bar(bar_len)


    def log_terminate(self):
        total = time.time() - self.init_time
        self.writeline(f'Total elapsed time: {timedelta(seconds=total)}')
        self.writeline('pyte terminated.')
        self.fp.seek(0)
        final = ''
        for idx, char in enumerate(self.fp.read()):
            if idx in self._erase_final:
                continue
            final += char
        self.fp.seek(0)
        self.fp.write(final)
        self.fp.truncate()
        self.fp.close()

