import numpy as np

from abc import ABC, abstractmethod
from random import choice


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
    def __init__(self, grid_cells, coords, max_velocity=None):
        self.grid_cells = grid_cells
        self.coords = coords
        self.bounds = (self.coords.min(), self.coords.max())
        self.max_velocity = np.array(self.bounds) / 10 if max_velocity is None else max_velocity
    
    def reset(self, end_point=None, end_radius=None):
        dim = self.coords.shape[-1]
        self.end_point = np.zeros(dim) if end_point is None else end_point
        self.end_radius = np.diff(self.bounds).item() / 20 if end_radius is None else end_radius
        # self.velocity = np.zeros(dim)
        self.state = np.random.uniform(*self.bounds, dim)
        while self.done():
            self.state = np.random.uniform(*self.bounds, dim)
        # self.state = np.random.choice([-1, 1], 2)
        return self.get_state()
        
    # def next_state(self, acceleration):
        # self.velocity = np.clip(self.velocity + acceleration, *self.max_velocity)
    def next_state(self, action):
        self.state = np.clip(self.state + action, *self.bounds)
        return self.get_state(), self.reward(), self.done()
    
    def get_state(self):
        # return np.concatenate([self.velocity, self.grid_cells[self.closest_coord_index(self.state)]])
        return self.grid_cells[self.closest_coord_index(self.state)]
        # return self.state
    
    def closest_coord_index(self, coord):
        return np.unravel_index(np.abs(self.coords - coord).sum(axis=-1).argmin(), self.coords.shape[:2])
    
    def done(self):
        return np.sum((self.state - self.end_point)**2) <= self.end_radius**2
    
    def reward(self):
        dim = self.coords.shape[-1]
        corners = np.array(np.meshgrid(*[self.bounds] * dim)).T.reshape(-1, dim)
        max_dist = np.sum((self.end_point - corners)**2, axis=-1).max()
        reward = -np.sqrt(np.sum((self.state - self.end_point)**2) / max_dist)
        return 1 if self.done() else reward