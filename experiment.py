import os
import json
import torch
import numpy as np

from agent import Agent
from grid_cells import GridCells
from tqdm import tqdm
from utils import (PlaceFields, to_tensor,
                   get_coords, get_flanks,
                   get_loc_batch, eval_position,
                   parser)


class Experiment:
    def __init__(self, name,
                 resolution=400,
                 coord_bounds=(-1, 1),
                 gc_scale_min=140,
                 gc_scale_max=400,
                 n_modules=10,
                 n_per_module=100, # Must be a square number (4, 9, 16, ...)
                 wd_l1=0,
                 wd_l2=0,
                 hidden_penalty=2e-2,
                 save_losses=True,
                 heterogeneous=False,
                 modular_peaks=False,
                 individual=False,
                 **agent_kwargs
                 ):
        self.name = name
        self.res = resolution
        self.wd = (wd_l1, wd_l2)
        self.hidden_penalty = hidden_penalty
        
        self.coords = get_coords(resolution, *coord_bounds)
        self.scales = np.linspace(gc_scale_min, gc_scale_max, n_modules, dtype=int)
        
        self.n_per_module = n_per_module
        self.gcs = GridCells(self.scales, n_per_module=n_per_module, res=resolution,
                             heterogeneous=heterogeneous, modular_peaks=modular_peaks,
                             individual=individual)
        self.agent = Agent(n_modules * n_per_module, 2, **agent_kwargs)

        self.pfs_per_env = dict()
        self.current_env = None

        self.heterogeneous = heterogeneous
        self.modular_peaks = modular_peaks
        self.individual = individual

        self.save_losses = save_losses
        self.pfs_losses = dict()

        self.store_experiment_kwargs()
        self.agent_kwargs = agent_kwargs
    
    def store_experiment_kwargs(self):
        self.experiment_kwargs = dict(
            resolution=self.res,
            coord_bounds=(self.coords.min().item(), self.coords.max().item()),
            gc_scale_min=self.scales.min().item(),
            gc_scale_max=self.scales.max().item(),
            n_modules=len(self.scales),
            n_per_module=self.n_per_module,
            wd_l1=self.wd[0],
            wd_l2=self.wd[1],
            hidden_penalty=self.hidden_penalty,
            save_losses=self.save_losses, 
            heterogeneous = self.heterogeneous
        )
    
    def rename(self, name):
        self.name = name
    
    def compile_grid_cells(self, env):
        self.current_env = env
        self.gcs.reset_modules(env)
        self.gcs.compile_numpy()
        self.grid_cells = to_tensor(self.gcs.grid_cells.transpose(1, 2, 0))
    
    def fit_positions(self, batches=50000, bs=256, progress=True):
        losses = [self.fit_position_batch(*get_loc_batch(self.coords, self.grid_cells, bs=bs))
                  for _ in tqdm(range(batches), disable=not progress)]
        if self.save_losses:
            self.pos_losses = losses
        return losses
    
    def fit_position_batch(self, x, y):
        self.agent.optim_actor.zero_grad()
        self.agent.actor.clamp_weights()
        
        _, pred = self.agent.actor(x)
        loss = torch.sum((pred - y)**2, dim=-1).mean()
        loss += self.agent.actor.hidden_loss(self.hidden_penalty)
        loss += self.agent.actor.regularization_loss(*self.wd)
        
        loss.backward()
        self.agent.optim_actor.step()
        
        return loss.detach().cpu().item()
    
    def fit_place_fields(self, epochs=3000, scheduler_updates=6, progress=True, N=None):
        N = self.agent_kwargs['actor_hidden'] if N is None else N
        
        self.initialize_place_fields(N)
        self.add_pfs_targets(reset=True)

        losses = self.pfs.fit(epochs=epochs, scheduler_updates=scheduler_updates, progress=progress)
        if self.save_losses:
            self.pfs_losses[self.current_env] = losses
        return losses
    
    def initialize_place_fields(self, N, env=None, state_dict=None):
        flanks = get_flanks(self.res // 6 + self.res // 30)
        self.pfs = PlaceFields(self.coords, flanks, N)
        self.pfs_per_env[self.current_env if env is None else env] = self.pfs
        if state_dict is not None:
            self.pfs.load_state_dict(state_dict)
    
    def add_pfs_targets(self, reset=False):
        if not hasattr(self, 'grid_cells'):
            raise AttributeError("Grid cells must be initialized to add targets")

        self.pfs.add_targets(self.calc_pfs_targets()[:self.pfs.N])
        if reset:
            self.pfs.informed_init()
    
    def load_pfs(self):
        self.pfs = self.pfs_per_env[self.current_env]
        self.add_pfs_targets()
    
    def calc_pfs_targets(self):
        return self.agent.actor.lin1(self.grid_cells).permute(-1, 0, 1).detach()
    
    def save(self, path='data', add_buffer=False):
        path = os.path.join(path, self.name)
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_pytorch(path, add_buffer)
        self.save_metadata(path)
    
    def save_pytorch(self, path, add_buffer):
        state_dicts = self.agent.collate_state_dicts(add_buffer)
        state_dicts['pfs'] = {env: pfs.state_dict() for env, pfs in self.pfs_per_env.items()}
        torch.save(state_dicts, os.path.join(path, 'models.pt'))
    
    def save_metadata(self, path):
        metadata = dict(
            name=self.name,
            envs=self.gcs.envs,
            experiment_kwargs=self.experiment_kwargs,
            agent_kwargs=self.agent_kwargs
        )

        if self.save_losses:
            metadata |= dict(pos_losses=self.pos_losses, pfs_losses=self.pfs_losses)
        
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            f.write(json.dumps(metadata))
    
    @staticmethod
    def load_experiment(path, name):
        path = os.path.join(path, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No experiment found at '{path}'")
        
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.loads(f.read())
        
        exp = Experiment(name, **metadata['experiment_kwargs'], **metadata['agent_kwargs'])
        exp.gcs.envs = metadata['envs']
        if exp.save_losses:
            exp.pos_losses = metadata['pos_losses']
            exp.pfs_losses = metadata['pfs_losses']

        state_dicts = torch.load(os.path.join(path, "models.pt"))
        pfs_state_dicts = state_dicts.pop('pfs', None)
        exp.agent.load_from_state_dicts(state_dicts)

        if pfs_state_dicts is not None:
            for env, state_dict in pfs_state_dicts.items():
                N = state_dict['means'].shape[0]
                exp.initialize_place_fields(N, env, state_dict)
        
        return exp
    

if __name__ == "__main__":
    from analysis import Analysis
    
    kwargs = vars(parser.parse_args())
    batches = kwargs.pop('batches')
    pf_epochs = kwargs.pop('pf_epochs')
    scheduler_updates = kwargs.pop('scheduler_updates')
    train_env2 = kwargs.pop('train_env2')
    batches_env2 = kwargs.pop('batches_env2')
    batches_env2 = batches if batches_env2 is None else batches_env2

    exp = Experiment(**kwargs)
    exp.compile_grid_cells(1)

    exp.fit_positions(batches)
    exp.fit_place_fields(pf_epochs, scheduler_updates=scheduler_updates)

    print(f"Position loss: {eval_position(exp.agent, exp.coords, exp.grid_cells):.03f}")
    
    exp.compile_grid_cells(2)
    exp.fit_place_fields(pf_epochs, scheduler_updates=scheduler_updates)

    anl = Analysis(exp, initialized_pc=True)
    anl.collect_stats()

    exp.save()
    anl.save_stats(os.path.join('data', exp.name))

    if train_env2:
        exp.rename(exp.name + '_env2')
        exp.fit_positions(batches_env2)
        exp.fit_place_fields(pf_epochs, scheduler_updates=scheduler_updates)
        exp.save()