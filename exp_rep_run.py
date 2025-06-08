import os
import torch

from experiment import Experiment
from analysis import Analysis
from utils import parser, eval_position
from gc import collect



kwargs = vars(parser.parse_args())
name = kwargs.pop('name')
batches = kwargs.pop('batches')
pf_epochs = kwargs.pop('pf_epochs')
scheduler_updates = kwargs.pop('scheduler_updates')
train_env2 = kwargs.pop('train_env2')
batches_env2 = kwargs.pop('batches_env2')
batches_env2 = batches if batches_env2 is None else batches_env2

data_path = os.path.join('data', name)


for i in range(5):
    exp = Experiment(str(i), **kwargs)
    exp.compile_grid_cells(1)

    exp.fit_positions(batches)
    exp.fit_place_fields(pf_epochs, scheduler_updates=scheduler_updates)

    print(f"Position loss: {eval_position(exp.agent, exp.coords, exp.grid_cells):.03f}")

    # 4 remappings for each run
    for env in range(2, 6):
        torch.cuda.empty_cache()
        exp.compile_grid_cells(env)
        exp.fit_place_fields(pf_epochs, scheduler_updates=scheduler_updates)

    anl = Analysis(exp, initialized_pc=True)
    anl.collect_stats()

    exp.save(path=data_path)
    anl.save_stats(os.path.join(data_path, exp.name))

    del exp
    del anl
    torch.cuda.empty_cache()