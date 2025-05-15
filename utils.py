import torch
import torch.nn as nn
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


def gaussian_grid(coords, gaussians):
    shape = coords.shape[:2]
    return np.asarray([dist.pdf(coords.reshape(-1, 2)).reshape(*shape)
                       for dist in gaussians]).sum(axis=0) / len(gaussians)

def get_coords(resolution, MIN=-1, MAX=1):
    x = torch.linspace(MIN, MAX, resolution, device=device)
    x = torch.meshgrid(x, x, indexing='xy')
    return torch.stack(x).view(2, -1).T.view(resolution, resolution, 2)
    
def get_loc_batch(coords, grid_cells, bs=64):
    states = torch.rand(bs, 2, device=device) * 2 - 1
    distances = torch.abs(torch.unsqueeze(coords, 2) - states).sum(axis=-1)
    idx = distances.view(-1, bs).argmin(0)
    rr = idx // coords.shape[0]
    cc = idx % coords.shape[1]
    return grid_cells[(rr, cc)], states

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
    def __init__(self, coords, N):
        super().__init__()
        self.coords = coords
        self.N = N
        
        self.means = nn.Parameter(torch.zeros(N, 2, device=device))
        self.cov_inv_diag = nn.Parameter(torch.ones(N, 2, device=device))
        self.cov_inv_off_diag = nn.Parameter(torch.zeros(N, 1, device=device))
        self.scales = nn.Parameter(torch.ones(N, 1, device=device))
    
    def forward(self, real):
        return torch.pow(self.predict() - real, 2).sum()
    
    # The covariance is symmetric so off-diagonals are the same in the the 2D case
    def get_cov_inv(self):
        cov_inv = torch.diag_embed(self.cov_inv_diag)
        cov_inv += torch.diag_embed(self.cov_inv_off_diag.tile(2)).flip(-1)
        return cov_inv
    
    # Calculate parametrized scaled Gaussian PDF over all coordinates
    def predict(self):
        diff = self.coords.view(1, -1, 2) - self.means.view(-1, 1, 2)
        pred = self.scales * torch.e ** -((diff * (diff @ self.get_cov_inv())).sum(-1) / 2)
        return pred.view(self.N, *self.coords.shape[:-1])