import os
import torch
import numpy as np

from utils import parser, eval_position
from experiment import Experiment
from analysis import Analysis
from copy import deepcopy


kwargs = vars(parser.parse_args())
name = kwargs.pop('name')
batches = kwargs.pop('batches')
pf_epochs = kwargs.pop('pf_epochs')
scheduler_updates = kwargs.pop('scheduler_updates')

eval_batches = np.diff([500, 3000, 10000, batches], prepend=0)


for i in os.listdir('data/baseline'):
    exp = Experiment.load_experiment('data', f'{name}/{i}')
    exp.agent.actor.change_activation(torch.cos)
    exp.compile_grid_cells(1)
    
    exp.load_pfs()
    exp.pfs_per_env = dict(old=deepcopy(exp.pfs))
    
    for b in eval_batches:
        exp.rename(f'baseline_cos{b}/{i}')
        
        exp.fit_positions(b)
        print(f"Position loss: {eval_position(exp.agent, exp.coords, exp.grid_cells):.03e}")
        exp.fit_place_fields(pf_epochs, scheduler_updates=scheduler_updates)
        
        anl = Analysis(exp, initialized_pc=True)
        anl.collect_stats()

        exp.save('data')
        anl.save_stats(os.path.join('data', exp.name))