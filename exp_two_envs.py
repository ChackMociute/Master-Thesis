import os
import torch
import numpy as np

from experiment import Experiment
from analysis import Analysis
from utils import parser, eval_position
from gc import collect



kwargs = vars(parser.parse_args())
name = kwargs.pop('name')
batches = kwargs.pop('batches')
pf_epochs = kwargs.pop('pf_epochs')
scheduler_updates = kwargs.pop('scheduler_updates')
batches_env2 = kwargs.pop('batches_env2')
batches_env2 = batches if batches_env2 is None else batches_env2

data_path = os.path.join('data', name)
eval_batches = [0, 1, 2, 3, 5, 10, 20, 50, 100, 500, 1000, 10000]

for i in range(5):
    if os.path.exists(os.path.join(data_path, str(i))):
        continue

    exp = Experiment(str(i), **kwargs)
    
    # First run
    exp.compile_grid_cells(1)
    exp.fit_positions(batches)
    print(f"Position loss: {eval_position(exp.agent, exp.coords, exp.grid_cells):.03e}")
    grid_cells = exp.grid_cells # Required for evaluation

    # First PFs fitting
    exp.fit_place_fields(pf_epochs, scheduler_updates=scheduler_updates)
    torch.cuda.empty_cache(); collect()
    exp.compile_grid_cells(2)
    exp.fit_place_fields(pf_epochs, scheduler_updates=scheduler_updates)

    # First save
    anl = Analysis(exp, initialized_pc=True)
    anl.collect_stats()
    exp.save(path=data_path)
    anl.save_stats(os.path.join(data_path, exp.name))
    
    # These are needed for AA' and BB' remapping statistics
    pfs_per_env = {k + len(exp.pfs_per_env): v for k, v in exp.pfs_per_env.items()}
    active_cells = {k + len(anl.active_per_env): v for k, v in anl.active_per_env.items()}
    place_cells = {k + len(anl.place_cells_per_env): v for k, v in anl.place_cells_per_env.items()}

    # Second run
    exp.rename(exp.name + '_env2')
    torch.cuda.empty_cache(); collect()
    eval_env2, eval_env1 = exp.fit_and_evaluate_positions(eval_batches, grid_cells, batches_env2)
    print(f"Position loss: {eval_position(exp.agent, exp.coords, exp.grid_cells):.03e}")

    # Second PFs fitting
    exp.fit_place_fields(pf_epochs, scheduler_updates=scheduler_updates)
    torch.cuda.empty_cache(); collect()
    exp.compile_grid_cells(1)
    exp.fit_place_fields(pf_epochs, scheduler_updates=scheduler_updates)

    # Second save with additional data
    anl = Analysis(exp, initialized_pc=True)
    anl.collect_stats(train_env=2)
    exp.save(path=data_path)
    anl.save_stats(os.path.join(data_path, exp.name))
    anl.save_retrain_remap(os.path.join(data_path, exp.name), pfs_per_env, active_cells, place_cells)
    np.save(os.path.join(data_path, exp.name, 'evals.npy'), [eval_env2, eval_env1])

    # Reset
    del exp; del anl; del grid_cells
    torch.cuda.empty_cache(); collect()