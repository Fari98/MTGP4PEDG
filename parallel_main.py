from main import _run, loaders, seeds
from joblib import Parallel, delayed


for loader in loaders:

    _ = Parallel(n_jobs=1, prefer='threads')(
        delayed(_run)(
            seed,
            loader
        ) for seed in range(seeds)
    )

