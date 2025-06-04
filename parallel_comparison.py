import os
from compare_nlt import _run, loaders
from joblib import Parallel, delayed


day = '20250520_True'

for loader in loaders:

    _ = Parallel(n_jobs=-1)(
        delayed(_run)(
            f'log/{day}/' + path,
            loader
        ) for path in list(filter(lambda x: loader.__name__.split("load_")[-1] in x, os.listdir(f'log/{day}')))
    )
