import os
import json
import torch
import numpy as np

from agent import Agent
from grid_cells import GridCells
from utils import PlaceFields, to_tensor, get_coords, get_flanks, get_loc_batch
from tqdm import tqdm


class Experiment:
    def __init__(
        self,
        name,
        resolution=400,
        coord_bounds=(-1, 1),
        scale_bounds=(90, 300),
        n_modules=10,
        n_per_module=100, # Must be a square number (4, 9, 16, ...)
        wd_l1=0,
        wd_l2=0,
        hidden_penalty=2e-2,
        **agent_kwargs
    ):
        self.name = name
        self.agent = Agent(n_modules * n_per_module, 2, **agent_kwargs)
        
        self.res = resolution
        self.coords = get_coords(resolution, *coord_bounds)
        self.scales = np.linspace(*scale_bounds, n_modules, dtype=int)
        
        self.n_per_module = n_per_module
        self.gcs = GridCells(self.scales, n_per_module=n_per_module, res=resolution)
        
        self.wd = (wd_l1, wd_l2)
        self.hidden_penalty = hidden_penalty

        self.store_experiment_kwargs()
        self.agent_kwargs = agent_kwargs
    
    def store_experiment_kwargs(self):
        self.experiment_kwargs = dict(
            resolution=self.res,
            coord_bounds=(self.coords.min().item(), self.coords.max().item()),
            scale_bounds=(self.scales.min().item(), self.scales.max().item()),
            n_modules=len(self.scales),
            n_per_module=self.n_per_module,
            wd_l1=self.wd[0],
            wd_l2=self.wd[1],
            hidden_penalty=self.hidden_penalty
        )
    
    def compile_grid_cells(self, env):
        self.gcs.reset_modules(env)
        self.gcs.compile_numpy()
        self.grid_cells = to_tensor(self.gcs.grid_cells.transpose(1, 2, 0))
    
    def fit_positions(self, batches=50000, bs=256, progress=True):
        losses = [self.fit_position_batch(*get_loc_batch(self.coords, self.grid_cells, bs=bs))
                  for _ in tqdm(range(batches), disable=not progress)]
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
    
    def fit_gaussians(self, epochs=3000, scheduler_updates=6, progress=True, N=None):
        self.initialize_place_fields(N=N)
        return self.pfs.fit(epochs=epochs, scheduler_updates=scheduler_updates, progress=progress)
    
    def initialize_place_fields(self, N=None):
        gc_initialized = hasattr(self, 'grid_cells')
        assert gc_initialized or N is not None, "If grid cells have not been initialized, N should be specified manually"
        
        if gc_initialized:
            targets = self.calc_pfs_targets()[:N]
            N = len(targets)

        flanks = get_flanks(self.res + self.res // 10)
        self.pfs = PlaceFields(self.coords, flanks, N)
        
        if gc_initialized:
            self.pfs.add_targets(targets)
            self.pfs.informed_init()
    
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
        if hasattr(self, 'pfs'):
            state_dicts['pfs'] = self.pfs.state_dict()
        torch.save(state_dicts, os.path.join(path, 'models.pt'))
    
    def save_metadata(self, path):
        metadata = dict(
            name=self.name,
            envs=self.gcs.envs,
            experiment_kwargs=self.experiment_kwargs,
            agent_kwargs=self.agent_kwargs
        )
        
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

        state_dicts = torch.load(os.path.join(path, "models.pt"))
        pfs_state_dict = state_dicts.pop('pfs', None)
        exp.agent.load_from_state_dicts(state_dicts)

        if pfs_state_dict is not None:
            exp.initialize_place_fields(N=pfs_state_dict['means'].shape[0])
            exp.pfs.load_state_dict(pfs_state_dict)
        
        return exp