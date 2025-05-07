import numpy as np

from skimage.draw import line_aa
from scipy.ndimage import rotate, shift
from scipy.stats import multivariate_normal
from utils import gaussian_grid


class GridCellModule:
    def __init__(self, scale, radius, n_grid_cells):
        assert np.sqrt(n_grid_cells) % 1 == 0, "Number of grid cells should be a square"
        self.radius = radius
        self.set_scale(scale)
        self.n = n_grid_cells
        self.create_grid_cell()
    
    def set_scale(self, scale):
        # The field size is quadratically proportional to spacing
        self.scale = scale
        self.period = np.sqrt(scale) * 2 + scale / 3
    
    def create_grid_cell(self):
        lines = self.lines_at_60()
        lines = [self.zeros_with_line(*line_aa(*l)) for l in lines]

        image = np.sum(lines, axis=0) > 1
        image = self.clean_image(image).astype(float)
        image *= np.random.uniform(0.5, 1.5, image.shape)
        image = self.convolve(image, self.get_gaussian_kernel())

        self.grid_cell = image / image.max()
    
    def lines_at_60(self):
        lines, length = list(), 0
        while length < self.radius:
            shift = np.arcsin(length / self.radius)
            lines.extend(self.get_lines(shift, length==0))
            length += self.period
        return np.asarray(lines) + self.radius
    
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
        z = np.zeros(shape)
        z[rr, cc] = val
        return z
    
    # Remove duplicated intersection points.
    # Bad intersections can be of the following forms:
    # [[1, 0],  |  [[0, 1],  \  [[1, 1]]  |  [[1],
    #  [0, 1]]  \   [1, 0]]  |            \   [1]]
    @staticmethod
    def clean_image(image):
        image[np.where(image[:, :-1] & image[:, 1:])] = False
        image[np.where(image[:-1] & image[1:])] = False
        image[np.where(image[1:, 1:] & image[:-1, :-1])] = False
        rr, cc = np.where(image[1:, :-1] & image[:-1, 1:])
        image[rr + 1, cc] = False
        return image
    
    def get_gaussian_kernel(self):
        range_ = np.linspace(-5, 5, self.scale)
        coords = np.stack(np.meshgrid(range_, range_)).transpose(1, 2, 0)
        return gaussian_grid(coords, [multivariate_normal([0, 0])])
    
    @staticmethod
    def convolve(image, kernel):
        shape = np.asarray(kernel.shape) // 2 + image.shape
        conv = np.fft.rfft2(image, shape) * np.fft.rfft2(kernel, shape)
        conv = np.fft.irfft2(conv)
        conv = np.where(np.isclose(conv, 0, atol=1e-4), 0, conv) # remove noise
        return conv[-image.shape[0]:, -image.shape[1]:]
    
    # Angle in degrees
    def reset_module(self, rot_angle):
        gc = rotate(self.grid_cell, rot_angle, reshape=False)
        gcs = np.tile(gc, (self.n, 1, 1))
        self.grid_cells = self.add_phase(gcs, rot_angle)
    
    def add_phase(self, grid_cells, rot_angle):
        p = np.linspace(-self.period / 2, self.period / 2, int(np.sqrt(self.n)))
        phases = np.stack(np.meshgrid(p, p)).reshape(2, -1).T
        shear = self.shear2d(np.pi / 6).T
        rot = self.rot2d(rot_angle / 180 * np.pi).T
        phases = phases @ shear @ rot
        return np.asarray([shift(gc, phase) for gc, phase in zip(grid_cells, phases)])
    
    @staticmethod
    def rot2d(angle):
        return np.asarray([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
    
    @staticmethod
    def shear2d(angle):
        return np.asarray([
            [np.cos(angle), 0],
            [np.sin(angle), 1]
        ])


class GridCells:
    ROTATION_RANGE = (0, 121)
    
    def __init__(self, scales, n_per_module=100, res=400):
        self.res = res
        radius = np.ceil(self.res * 0.9).astype(int)
        self.modules = [GridCellModule(scale, radius, n_per_module) for scale in scales]
        self.envs = dict()
    
    def reset_modules(self, env='random'):
        # Initialize grid cell orientation in a new environment
        if env not in self.envs.keys() or env == 'random':
            self.envs[env] = np.random.randint(*self.ROTATION_RANGE, size=len(self.modules))
        
        for module, angle in zip(self.modules, self.envs[env]):
            module.reset_module(angle)
    
    def compile_numpy(self):
        self.grid_cells = np.concatenate([module.grid_cells for module in self.modules])
        self.crop()
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