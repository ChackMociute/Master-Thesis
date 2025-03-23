import torch
import numpy as np

from abc import ABC, abstractmethod
from random import choice
from utils import to_tensor, device


class Environment(ABC):
    @abstractmethod
    def next_state(self):
        ...
    
    @abstractmethod
    def reset(self):
        ...


class GridWorld(Environment):
    VELOCITIES = [-1, 0, 1]
    
    def __init__(self, x, y):
        assert x >= 2 and y >= 2, f"GridWorld must be at least 2x2, got {x}x{y}"
        assert isinstance(x, int) and isinstance(y, int), f"Dimensions must be integers, got type(x)={type(x)} and type(y)={type(y)}"
        self.states = np.arange(x*y).reshape(x, y)
        self.reset()
    
    def reset(self):
        self.state = np.random.choice(self.states.flatten())
    
    def next_state(self):
        x, y = np.where(self.states == self.state)
        vx, vy = self.possible_velocity(x.item(), y.item())
        x += vx
        y += vy
        self.state = self.states[x, y].item()
    
    def possible_velocity(self, x, y):
        vx = choice(self.VELOCITIES[int(x == 0):-1 if x == self.states.shape[0] - 1 else len(self.VELOCITIES)])
        vy = choice(self.VELOCITIES[int(y == 0):-1 if y == self.states.shape[1] - 1 else len(self.VELOCITIES)])
        return vx, vy


class ToroidalGridWorld(GridWorld):
    def next_state(self):
        old = self.state
        x, y = np.where(self.states == old)
        vx, vy = self.possible_velocity()
        x = (x + vx) % self.states.shape[0]
        y = (y + vy) % self.states.shape[1]
        self.state = self.states[x, y].item()
        return old, vx, vy, self.state
    
    def possible_velocity(self):
        return choice(self.VELOCITIES), choice(self.VELOCITIES)


class GridCellWorld(Environment):
    def __init__(self, grid_cells, coords, max_velocity=None, debug=False):
        self.grid_cells = grid_cells
        self.coords = coords
        self.bounds = torch.tensor((self.coords.min(), self.coords.max())).numpy()
        self.max_velocity = self.bounds / 10 if max_velocity is None else max_velocity
        self.debug = debug
    
    def reset(self, end_point=None, end_radius=None):
        dim = self.coords.shape[-1]
        amp = np.diff(self.bounds).item()
        self.end_point = torch.zeros(dim, device=device) if end_point is None else end_point
        self.end_radius = amp / 20 if end_radius is None else end_radius
        # self.velocity = np.zeros(dim)
        self.state = torch.rand(dim, device=device) * amp + self.bounds[0]
        while self.done(self.distance()):
            self.state = torch.rand(dim, device=device) * amp + self.bounds[0]
        return self.get_state()
        
    # def next_state(self, acceleration):
        # self.velocity = np.clip(self.velocity + acceleration, *self.max_velocity)
    def next_state(self, action):
        self.state = torch.clip(self.state + action, *self.bounds)
        dist = self.distance()
        return self.get_state(), self.reward(dist), self.done(dist)
    
    def get_state(self):
        # return np.concatenate([self.velocity, self.grid_cells[self.closest_coord_index(self.state)]])
        if self.debug:
            return self.state
        return self.grid_cells[self.closest_coord_index(self.state)]
    
    def closest_coord_index(self, coord):
        idx = torch.abs(self.coords - coord).sum(axis=-1).argmin()
        row, col = idx // self.coords.shape[0], idx % self.coords.shape[1]
        return (row, col)
    
    def done(self, dist):
        return dist <= self.end_radius**2
    
    def reward(self, dist):
        reward = -torch.sqrt(dist)
        return to_tensor(1) if self.done(dist) else reward
    
    def distance(self):
        dist = self.state - self.end_point
        return torch.sum(dist * dist)