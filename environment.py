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
        self.end_radius = amp / 100 if end_radius is None else end_radius
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
        return dist <= self.end_radius
    
    def reward(self, dist):
        return to_tensor(1) if self.done(dist) else -dist
    
    def distance(self):
        dist = self.state - self.end_point
        return torch.sum(dist * dist).sqrt()