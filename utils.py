import torch
import torch.nn as nn
import numpy as np

from argparse import ArgumentParser
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR
from scipy.stats import norm
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

parser = ArgumentParser()
parser.add_argument('--name', required=True)
parser.add_argument('--batches', type=int, default=50000)
parser.add_argument('--pf_epochs', type=int, default=3000)
parser.add_argument('--scheduler_updates', type=int, default=6)
parser.add_argument('--n_modules', type=int, default=10)
parser.add_argument('--n_per_module', type=int, default=100, help="Must be a square (4, 9, 16, ...)")
parser.add_argument('--gc_scale_min', type=int, default=120)
parser.add_argument('--gc_scale_max', type=int, default=400)
parser.add_argument('--wd_l1', type=float, default=5e-3)
parser.add_argument('--wd_l2', type=float, default=5e-5)
parser.add_argument('--hidden_penalty', type=float, default=2e-2)
parser.add_argument('--save_losses', action='store_true')
parser.add_argument('--exploration_std', type=float, default=0.1)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--tau', type=float, default=5e-3)
parser.add_argument('--buffer_length', type=int, default=1000)
parser.add_argument('--actor_hidden', type=int, default=128)
parser.add_argument('--critic_hidden', type=int, default=256)
parser.add_argument('--lr_a', type=float, default=3e-4)
parser.add_argument('--wd_a', type=float, default=0)
parser.add_argument('--lr_c', type=float, default=5e-4)
parser.add_argument('--wd_c', type=float, default=1e-5)
parser.add_argument('--grad_norm', type=float, default=1e-1)
parser.add_argument('--heterogeneous', action='store_true')
parser.add_argument('--modular_peaks', action='store_true')
parser.add_argument('--individual', action='store_true')
parser.add_argument('--train_env2', action='store_true')
parser.add_argument('--batches_env2', type=int, default=None)


def gaussian_grid(coords, gaussians):
    shape = coords.shape[:2]
    return np.asarray([dist.pdf(coords.reshape(-1, 2)).reshape(*shape)
                       for dist in gaussians]).sum(axis=0) / len(gaussians)

def get_coords(resolution, MIN=-1, MAX=1):
    x = torch.linspace(MIN, MAX, resolution, device=device)
    x = torch.meshgrid(x, x, indexing='xy')
    return torch.stack(x).view(2, -1).T.view(resolution, resolution, 2)

# Only works for square coordinates centered around 0
def get_flanks(res, reach=1, bounds=1.1):
    flanks = torch.linspace(-bounds, bounds, res, device=device)
    flanks = torch.meshgrid(flanks, flanks, indexing='xy')
    flanks = torch.stack(flanks).view(2, -1).T
    mask = torch.any(flanks.abs() > reach, 1)
    return flanks[mask]
    
def get_loc_batch(coords, grid_cells, bs=64):
    states = torch.rand(bs, 2, device=device) * 2 - 1
    distances = torch.abs(torch.unsqueeze(coords, 2) - states).sum(axis=-1)
    idx = distances.view(-1, bs).argmin(0)
    rr = idx // coords.shape[0]
    cc = idx % coords.shape[1]
    return grid_cells[(rr, cc)], states

def eval_position(agent, coords, grid_cells, size=4096):
    losses = list()
    for _ in range(size // 256):
        x, y = get_loc_batch(coords, grid_cells, bs=256)
        x = agent.actor(x)[1]
        loss = torch.mean((x - y)**2).detach().cpu().numpy()
        losses.append(loss)
    return np.mean(losses)

def print_stats(w):
    print("min   |max  |mean |std  |shape")
    print(f"{w.min():.03f}|{w.max():.03f}|{w.mean():.03f}|{w.std():.03f}|{w.shape}")

def eval_locomotion(agent, env, n_ep=200, maxlen=50):
    rewards, lengths = list(), list()
    print("Starting evaluation")
    for _ in tqdm(range(n_ep)):
        reward = list()
        s = env.reset(end_radius=0.25)
        step = 0
        done = False
        while not done and step < maxlen:
            a = agent.choose_action(s, evaluate=True)
            s_new, r, done = env.next_state(a)
            s = s_new
            step += 1
            reward.append(r)
        rewards.append(sum(reward))
        lengths.append(step)
    return sum(rewards) / len(rewards), sum(lengths) / len(lengths)

to_tensor = lambda x: torch.tensor(x, device=device, dtype=torch.float32)


class ReplayBuffer:
    def __init__(self, inp_size, actions, maxlen=1000):
        self.max = maxlen
        self.counter = 0
        self.states = torch.zeros((maxlen, inp_size), device=device)
        self.states_ = torch.zeros_like(self.states, device=device)
        self.actions = torch.zeros((maxlen, actions), device=device)
        self.rewards = torch.zeros(maxlen, device=device)
        self.dones = torch.zeros(maxlen, dtype=torch.uint8, device=device)
        self.locs = torch.zeros_like(self.actions)
    
    def store_transition(self, state, action, reward, new_state, done, loc=None):
        i = self.counter % self.max
        self.states[i] = state
        self.states_[i] = new_state
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.locs[i] = np.nan if loc is None else loc
        self.counter += 1
    
    def sample(self, bs):
        i = torch.ones(min(self.max, self.counter), device=device).multinomial(min(bs, self.counter, self.max))
        return self.states[i], self.actions[i], self.rewards[i], self.states_[i], self.dones[i], self.locs[i]
    
    
class PlaceFields(nn.Module):
    def __init__(self, coords, flanks, N):
        super().__init__()
        self.coords = coords
        self.flanks = flanks
        self.bounds = (coords.min() * 1.1, coords.max() * 1.1)
        self.N = N

        self.means = nn.Parameter(torch.zeros(self.N, 2, device=device))
        self.cov_inv_diag = nn.Parameter(torch.ones(self.N, 2, device=device) * 2.3) # e^2.3 ~ 10
        self.cov_inv_off_diag = nn.Parameter(torch.zeros(self.N, 1, device=device))
        self.scales = nn.Parameter(torch.ones(self.N, 1, device=device))
    
    def add_targets(self, targets):
        assert targets.shape[0] == self.N
        self.targets = targets

    def informed_init(self):
        if not hasattr(self, 'targets'):
            raise AttributeError("Targets must be added before an informed initialization can be made")
        idx = self.targets.view(self.N, -1).argmax(1)
        xx = idx // self.coords.shape[0]
        yy = idx % self.coords.shape[1]

        self.scales.data = self.targets.view(self.N, -1).max(1, keepdim=True)[0].sqrt()
        self.means.data = self.coords[xx, yy]
        self.cov_inv_diag.data = torch.ones(self.N, 2, device=device) * 4.6 # e^4.6 ~ 100
    
    def fit(self, epochs=3000, optim=None, use_scheduler=True, gamma=0.8, scheduler_updates=6, progress=True):
        if optim is None:
            optim = RMSprop(self.parameters(), lr=1e-2)
        if use_scheduler:
            scheduler = ExponentialLR(optim, gamma=gamma)
            lr_epochs = epochs // (scheduler_updates + 1)
        
        losses = list()
        for i in tqdm(range(epochs), disable=not progress):
            optim.zero_grad()
            self.means.data = torch.clip(self.means, *self.bounds)
            loss = self()
            loss.backward()
            optim.step()
            losses.append(loss.detach().cpu().item())
            if use_scheduler and i % lr_epochs == 0 and i != 0:
                scheduler.step()
        
        return losses

    def forward(self):
        # High flank loss ensures that the Gaussian does not drift beyond the range
        # of the coordinates (at the cost of introducing some bias around the edges)
        flank_loss = self.calc_gaussian(self.flanks)
        smoothing = torch.exp(50 * (self.flanks.abs().max(1).values.unsqueeze(0) - 1.07))
        # Smoothing increases flank loss exponentially from the boundary
        flank_loss = (flank_loss * smoothing).sum() * 3
        pred_loss = torch.pow(self.predict() - self.targets, 2).sum()
        return pred_loss + flank_loss
    
    def calc_gaussian(self, coords):
        diff = coords.view(1, -1, 2) - self.means.view(-1, 1, 2)
        return torch.pow(self.scales, 2) * torch.e ** -((diff * (diff @ self.get_cov_inv())).sum(-1) / 2)
    
    # Calculate parametrized scaled Gaussian PDF over all coordinates
    def predict(self):
        pred = self.calc_gaussian(self.coords)
        return pred.view(self.N, *self.coords.shape[:-1])
    
    # The covariance is symmetric so off-diagonals are the same in the the 2D case
    def get_cov_inv(self):
        cov_inv = torch.diag_embed(self.cov_inv_diag.exp())
        cov_inv += torch.diag_embed(self.cov_inv_off_diag.tile(2)).flip(-1)
        return cov_inv
    
    def calc_fitness(self):
        if not hasattr(self, 'targets'):
            raise AttributeError("Targets must be added before fitness can be calculated")
        error = torch.pow(self.predict() - self.targets, 2).sum((1, 2))
        variance = torch.pow(self.targets, 2).sum((1, 2))
        return 1 - error / variance
    
    def get_place_cells(self, threshold=0.5):
        return torch.arange(self.N, device=device)[self.calc_fitness() > threshold]

    def get_active_cells(self, threshold=0.001):
        return torch.arange(self.N, device=device)[self.scales.squeeze().pow(2) >= threshold]
    
    def pairwise_distances(self, pairs):
        return torch.pow(self.means[pairs[:,0]] - self.means[pairs[:,1]], 2).sum(-1).sqrt()
    
    def get_coverage(self, p=0.99):
        diff = self.coords.view(1, -1, 2) - self.means.view(-1, 1, 2)
        dist = (diff * (diff @ self.get_cov_inv())).sum(-1)
        dist = dist.view(self.N, *self.coords.shape[:-1])
        return dist.sqrt() < norm.ppf(p)