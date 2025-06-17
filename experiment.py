import os
import json
import torch
import numpy as np

from agent import Agent
from grid_cells import GridCells
from environment import GridCellWorld
from tqdm import tqdm
from gc import collect
from utils import (PlaceFields, parser,
                   get_coords, get_flanks,
                   get_loc_batch, eval_position,
                   eval_locomotion)


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
                 reinforcement=False, # Set to true for RL experiment
                 max_episode_len=30,
                 fake_cells=False,
                 **agent_kwargs
                 ):
        self.name = name
        self.res = resolution
        self.wd = (wd_l1, wd_l2)
        self.hidden_penalty = hidden_penalty
        
        self.coords = get_coords(resolution, *coord_bounds)
        self.scales = np.linspace(gc_scale_min, gc_scale_max, n_modules, dtype=int)
        
        self.n_per_module = n_per_module
        self.fake_cells = fake_cells
        self.gcs = GridCells(self.scales, n_per_module=n_per_module,
                             res=resolution, heterogeneous=heterogeneous)
        self.agent = Agent(n_modules * n_per_module, 2, **agent_kwargs)

        self.pfs_per_env = dict()
        self.current_env = None
        
        self.heterogeneous = heterogeneous

        self.save_losses = save_losses
        self.pfs_losses = dict()

        self.reinforcement = reinforcement
        self.max_episode_len = max_episode_len

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
            heterogeneous = self.heterogeneous,
            reinforcement=self.reinforcement,
            max_episode_len=self.max_episode_len,
            fake_cells=self.fake_cells
        )
    
    def rename(self, name):
        self.name = name
    
    def compile_grid_cells(self, env):
        self.current_env = env
        if self.fake_cells:
            self.gcs.fake_cells(self.coords, env)
            self.grid_cells = self.gcs.grid_cells
        else:
            self.gcs.reset_modules(env)
            self.grid_cells = self.gcs.grid_cells.permute(1, 2, 0)
    
    def run_experiment(self, *args, **kwargs):
        self.learn_rl(*args, **kwargs) if self.reinforcement else self.fit_positions(*args, **kwargs)
    
    def learn_rl(self, n_episodes):
        env = GridCellWorld(self.grid_cells, self.coords)
        rewards, lengths = list(), list()
        for ep in tqdm(range(n_episodes)):
            self.agent.exploration_update(ep)
            l, r = self.run_rl_episode(env)
            lengths.append(l)
            rewards.append(r)
        if self.save_losses:
            self.rl_results = {'lengths': lengths, 'rewards': rewards}
    
    def run_rl_episode(self, env):
        done = False
        step = 0
        reward = list()
        s, loc = env.reset(), env.state
        while not done and step < self.max_episode_len:
            a = self.agent.choose_action(s)
            s_new, r, done = env.next_state(a)
            self.agent.remember(s, a, r, s_new, done, loc)
            self.agent.learn(exp.wd, exp.hidden_penalty)
            s, loc = s_new, env.state
            step += 1
            reward.append(r)
        return step, sum(reward).cpu().item()


    def fit_positions(self, batches=50000, bs=256, progress=True):
        losses = [self.fit_position_batch(*get_loc_batch(self.coords, self.grid_cells, bs=bs))
                  for _ in tqdm(range(batches), disable=not progress)]
        if self.save_losses:
            self.pos_losses = losses
        return losses
    
    # Evaluate in two environments at specified timepoints during training
    def fit_and_evaluate_positions(self, eval_batches, env2_grid_cells, batches=50000, bs=256, progress=True):
        losses, eval_env1, eval_env2 = list(), list(), list()
        for i in tqdm(range(batches), disable=not progress):
            if i in eval_batches:
                eval_env1.append(eval_position(self.agent, self.coords, self.grid_cells))
                eval_env2.append(eval_position(self.agent, self.coords, env2_grid_cells))
            losses.append(self.fit_position_batch(*get_loc_batch(self.coords, self.grid_cells, bs=bs)))
        if self.save_losses:
            self.pos_losses = losses
        return eval_env1, eval_env2
    
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
            experiment_kwargs=self.experiment_kwargs,
            agent_kwargs=self.agent_kwargs
        )
        if self.fake_cells:
            torch.save(self.gcs.envs, os.path.join(path, 'envs.pt'))
        else:
            metadata['envs'] = self.gcs.envs

        if self.save_losses:
            if self.reinforcement:
                metadata['rl_results'] = self.rl_results
            else:
                metadata['pos_losses'] = self.pos_losses
            metadata['pfs_losses'] = self.pfs_losses
        
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
        
        exp.gcs.envs = torch.load(os.path.join(path, 'envs.pt')) \
            if exp.fake_cells else metadata['envs']
        
        if exp.save_losses:
            if exp.reinforcement:
                exp.rl_results = metadata['rl_results']
            else:
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
    name = kwargs.pop('name')
    batches = kwargs.pop('batches')
    pf_epochs = kwargs.pop('pf_epochs')
    scheduler_updates = kwargs.pop('scheduler_updates')
    kwargs.pop('batches_env2')
    
    exploration_std = eval(kwargs.pop('exploration_std'))
    if not (type(exploration_std) == float or type(exploration_std) == tuple):
        raise TypeError('Argument --exploration_std must be a float or a tuple')
    # If tuple, exploration std is of form (min, max, n) for np.logspace(min, max, n)
    kwargs['exploration_std'] = (*exploration_std, batches) if type(exploration_std) == tuple else exploration_std


    data_path = os.path.join('data', name)
    for i in range(5):
        if os.path.exists(os.path.join(data_path, str(i))):
            continue

        exp = Experiment(str(i), **kwargs)
        exp.compile_grid_cells(1)

        exp.run_experiment(batches)
        exp.fit_place_fields(pf_epochs, scheduler_updates=scheduler_updates)

        if exp.reinforcement:
            r, l = eval_locomotion(exp.agent, GridCellWorld(exp.grid_cells, exp.coords))
            print(f"Average reward: {r:.03f}\nAverage epsiode length: {l:.03f}")
        else:
            print(f"Position loss: {eval_position(exp.agent, exp.coords, exp.grid_cells):.03e}")

        # 4 remappings for each run
        for env in range(2, 6):
            torch.cuda.empty_cache(); collect()
            exp.compile_grid_cells(env)
            exp.fit_place_fields(pf_epochs, scheduler_updates=scheduler_updates)

        anl = Analysis(exp, initialized_pc=True)
        anl.collect_stats()

        exp.save(path=data_path)
        anl.save_stats(os.path.join(data_path, exp.name))

        del exp; del anl
        torch.cuda.empty_cache(); collect()