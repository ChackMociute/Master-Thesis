import torch
import torch.nn as nn
import numpy as np

from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


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
        self.N = N
        
        self.means = nn.Parameter(torch.zeros(N, 2, device=device))
        self.cov_inv_diag = nn.Parameter(torch.ones(N, 2, device=device))
        self.cov_inv_off_diag = nn.Parameter(torch.zeros(N, 1, device=device))
        self.scales = nn.Parameter(torch.ones(N, 1, device=device))
    
    def forward(self, real):
        # High flank loss ensures that the Gaussian does not drift beyond the range
        # of the coordinates (at the cost of introducing some bias around the edges)
        flank_loss = self.calc_gaussian(self.flanks).sum() * 5
        pred_loss = torch.pow(self.predict() - real, 2).sum()
        return pred_loss + flank_loss
    
    # The covariance is symmetric so off-diagonals are the same in the the 2D case
    def get_cov_inv(self):
        cov_inv = torch.diag_embed(self.cov_inv_diag.exp())
        cov_inv += torch.diag_embed(self.cov_inv_off_diag.tile(2)).flip(-1)
        return cov_inv
    
    # Calculate parametrized scaled Gaussian PDF over all coordinates
    def predict(self):
        pred = self.calc_gaussian(self.coords)
        return pred.view(self.N, *self.coords.shape[:-1])
    
    def calc_gaussian(self, coords):
        diff = coords.view(1, -1, 2) - self.means.view(-1, 1, 2)
        return torch.pow(self.scales, 2) * torch.e ** -((diff * (diff @ self.get_cov_inv())).sum(-1) / 2)
    
    def fit(self, target, epochs=3000, optim=None, use_scheduler=True, gamma=0.8, scheduler_updates=6, progress=True):
        if optim is None:
            optim = RMSprop(self.parameters(), lr=1e-2)
        if use_scheduler:
            scheduler = ExponentialLR(optim, gamma=gamma)
            lr_epochs = epochs // scheduler_updates
        
        losses = list()
        for i in tqdm(range(epochs), disable=not progress):
            optim.zero_grad()
            loss = self(target)
            loss.backward()
            optim.step()
            losses.append(loss.detach().cpu().item())
            if use_scheduler and i % lr_epochs == 0 and i != 0:
                scheduler.step()
        
        return losses