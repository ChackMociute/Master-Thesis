import torch
import numpy as np

from skimage.draw import line_aa
from scipy.ndimage import shift
from torchvision.transforms.functional import rotate, affine
from scipy.stats import multivariate_normal
from utils import gaussian_grid, device, to_tensor


class GridCellModule:
    def __init__(self, scale, radius, n_grid_cells, res=400):
        assert np.sqrt(n_grid_cells) % 1 == 0, "Number of grid cells should be a square"
        self.radius = radius
        self.set_scale(scale)
        self.n = n_grid_cells
        self.res = res
    
    def set_scale(self, scale):
        # The field size is quadratically proportional to spacing
        self.scale = scale
        self.period = np.sqrt(scale) * 2 + scale / 3
    
    def create_grid_cell(self, heterogeneous=False, mask=None):
        lines = self.lines_at_60()
        lines = [self.zeros_with_line(*line_aa(*l)) for l in lines]

        image = torch.stack(lines).sum(0) > 1
        image = self.clean_image(image).to(torch.float32)

        if heterogeneous:
            idx = image.to(bool)
            mask = torch.randn(idx.sum(), device=device, dtype=torch.half) / 3 + 1
            self.mask = mask.clip(0.1) if mask is None else mask
            image[idx] *= self.mask
        
        self.grid_cell = self.convolve(image, self.get_gaussian_kernel())
    
    def lines_at_60(self):
        lines, length = list(), 0
        while length < self.radius:
            shift = np.arcsin(length / self.radius)
            lines.extend(self.get_lines(shift, length==0))
            length += self.period * np.sin(np.pi / 3)
        return torch.tensor(lines, device=device) + self.radius
    
    def get_lines(self, shift, first):
        lines = list()
        angles = np.array([shift, np.pi - shift])
        lines.append(self.get_line(*angles))
        lines.append(self.get_line(*angles + np.pi / 3))
        if not first:
            lines.append(self.get_line(*-angles))
            lines.append(self.get_line(*-angles + np.pi / 3))
        return lines
    
    def get_line(self, angle1, angle2):
        x1, y1 = self.coords_on_circle(angle1)
        x2, y2 = self.coords_on_circle(angle2)
        return (x1, y1, x2, y2)

    def coords_on_circle(self, angle):
        lin = self.radius * np.e**(1j * angle)
        y, x = np.ceil([lin.real, lin.imag]).astype(int)
        return x, y
    
    def zeros_with_line(self, rr, cc, val=1):
        shape=[2 * self.radius + 1] * 2
        z = torch.zeros(shape, device=device, dtype=torch.float32)
        z[rr, cc] = torch.tensor(val, device=device, dtype=torch.float32)
        return z
    
    # Remove duplicated intersection points.
    # Bad intersections can be of the following forms:
    # [[1, 0],  |  [[0, 1],  \  [[1, 1]]  |  [[1],
    #  [0, 1]]  \   [1, 0]]  |            \   [1]]
    @staticmethod
    def clean_image(image):
        image[torch.where(image[:, :-1] & image[:, 1:])] = False
        image[torch.where(image[:-1] & image[1:])] = False
        image[torch.where(image[1:, 1:] & image[:-1, :-1])] = False
        rr, cc = torch.where(image[1:, :-1] & image[:-1, 1:])
        image[rr + 1, cc] = False
        return image
    
    def get_gaussian_kernel(self):
        range_ = np.linspace(-5, 5, self.scale)
        coords = np.stack(np.meshgrid(range_, range_)).transpose(1, 2, 0)
        return torch.tensor(gaussian_grid(coords, [multivariate_normal([0, 0])]), device=device, dtype=torch.float64)
    
    @staticmethod
    def convolve(image, kernel):
        kernel /= kernel.max()
        shape = tuple(np.asarray(kernel.shape) // 2 + image.shape)
        conv = torch.fft.rfft2(image, shape) * torch.fft.rfft2(kernel, shape)
        conv = torch.fft.irfft2(conv)
        conv = torch.where(torch.isclose(conv, torch.zeros_like(conv), atol=1e-4), 0, conv) # remove noise
        return conv[-image.shape[0]:, -image.shape[1]:]
    
    # Angle in degrees
    def reset_module(self, rot_angle, displacement, heterogeneous=False, mask=None):
        self.create_grid_cell(heterogeneous=heterogeneous, mask=mask)
        # gc = rotate(self.grid_cell.unsqueeze(0), rot_angle, expand=False)
        gcs = self.grid_cell.tile((self.n, 1, 1))
        self.grid_cells = self.add_phase(gcs, rot_angle, displacement)
        self.crop()
    
    def add_phase(self, grid_cells, rot_angle, displacement):
        p = torch.linspace(-self.period / 2, self.period / 2, int(np.sqrt(self.n)), device=device)
        phases = torch.stack(torch.meshgrid(p, p, indexing='xy')).view(2, -1).T + to_tensor(displacement)
        shear = self.shear2d(np.pi / 6).T
        rot = self.rot2d(rot_angle / 180 * np.pi).T
        phases = phases @ shear @ rot
        return torch.concat([affine(gc.unsqueeze(0), -rot_angle, phase, 1, 0)
                             for gc, phase in zip(grid_cells, phases.tolist())])
    
    @staticmethod
    def rot2d(angle):
        return torch.tensor([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ], device=device, dtype=torch.float32)
    
    @staticmethod
    def shear2d(angle):
        return torch.tensor([
            [np.cos(angle), 0],
            [np.sin(angle), 1]
        ], device=device, dtype=torch.float32)
    
    def crop(self):
        x, y = self.grid_cells.shape[1:]
        x, y = (x - self.res) // 2, (y - self.res) // 2
        self.grid_cells = self.grid_cells[:, x:x + self.res, y:y + self.res]


class GridCells:
    ROTATION_RANGE = (-180, 180)
    
    def __init__(self, scales, n_per_module=100, res=400, heterogeneous=False):
        self.res = res
        self.N = n_per_module
        radius = np.ceil(self.res * 1.2).astype(int)
        self.modules = [GridCellModule(scale, radius, n_per_module, res=res) for scale in scales]
        self.heterogeneous = heterogeneous
        self.envs = dict()
    
    def reset_modules(self, env='random'):
        # This ensures keys don't change after saving and loading with JSON
        env = str(env)
        # Initialize grid cell orientation in a new environment
        if env not in self.envs.keys() or env == 'random':
            self.envs[env] = dict(
                rotations=self.sample_rotations(),
                displacements=self.sample_displacements(),
                masks=[None] * len(self.modules)
            )
        
        angles = self.envs[env]['rotations']
        shifts = self.envs[env]['displacements']
        masks = self.envs[env]['masks']

        for i, (module, angle, displacement, mask) in enumerate(zip(self.modules, angles, shifts, masks)):
            module.reset_module(angle, displacement, heterogeneous=self.heterogeneous, mask=mask)
            if self.heterogeneous and self.envs[env]['masks'][i] is None:
                self.envs[env]['masks'][i] = module.mask.tolist()
    
    # Convert to list so JSON can dump it
    def sample_rotations(self):
        return np.random.randint(*self.ROTATION_RANGE, size=len(self.modules)).tolist()
    
    def sample_displacements(self):
        # The maximum observed displacement is close to half the period
        periods = np.asarray([m.period for m in self.modules]) / 2
        return np.random.uniform(-periods, periods, (2, len(periods))).T.tolist()

    def compile_numpy(self, crop=False):
        self.grid_cells = torch.concat([module.grid_cells for module in self.modules])
        if crop: self.crop()
        self.shape = self.grid_cells.shape
    
    def crop(self):
        x, y = self.grid_cells.shape[1:]
        x, y = (x - self.res) // 2, (y - self.res) // 2
        self.grid_cells = self.grid_cells[:, x:x + self.res, y:y + self.res]
    
    def __getitem__(self, i):
        if not hasattr(self, 'grid_cells'):
            raise AttributeError("Grid cells have not been initialized to any environment. Run reset_modules and compile_numpy first.")
        return self.grid_cells[i]


def purge_delinquent_cells(grid_cells):
    means = grid_cells.mean(axis=(1, 2))
    return grid_cells[~(means < means.mean() - means.std() * 4)]