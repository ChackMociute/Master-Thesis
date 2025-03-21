import torch
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


def gaussian_grid(coords, gaussians):
    shape = coords.shape[:2]
    return np.asarray([dist.pdf(coords.reshape(-1, 2)).reshape(*shape)
                       for dist in gaussians]).sum(axis=0) / len(gaussians)

def get_coords(resolution, MIN=-1, MAX=1):
    x, y = np.linspace(MIN, MAX, resolution), np.linspace(MIN, MAX, resolution)
    xx, yy = np.meshgrid(x, y)
    return np.concatenate([xx[:,:,np.newaxis], yy[:,:,np.newaxis]], axis=-1)
    
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
        self.dones = torch.zeros(maxlen, dtype=bool, device=device)
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
        i = np.random.choice(min(self.max, self.counter), min(bs, self.counter), replace=False)
        return self.states[i], self.actions[i], self.rewards[i], self.states_[i], self.dones[i], self.locs[i]