import numpy as np
import torch
import matplotlib.pyplot as plt
import abc

class PointwiseTransform(abc.ABC):

    @abc.abstractproperty
    def original_points(self):
        '''
        returns a tensor of points to transform
        '''
        pass

    def __init__(self, device='cuda', dtype=torch.float, dim=2):
        assert dim == 2
        self.dim = dim
        self.device = device
        self.dtype = dtype
    
    def show_transform(self, model, ax, grad=False, title="Trf"):
        points = self.original_points.type(self.dtype).to(self.device)
        assert(len(points.shape) == 2)
        assert(points.size(-1) == self.dim)
        np_orig_points = points.detach().cpu().numpy()
        if grad:
            points.requires_grad = True
            np_trf_points = model.push(points).detach().cpu().numpy()
        else:
            np_trf_points = model(points).detach().squeeze().cpu().numpy()
            assert(len(np_trf_points.shape) == 1)
            np_trf_points = np.stack([np_trf_points, np.zeros_like(np_trf_points)], axis=-1)
        s = [[2 + 2*n**2 for n in range(len(points))]]
        ax.scatter(np_orig_points[:, 0], np_orig_points[:, 1], s=s, color='g', label='original')
        ax.scatter(np_trf_points[:, 0], np_trf_points[:, 1], s=s, color='b', label='transformed')
        norms = np.linalg.norm(np_trf_points, axis=-1)
        max_norm = float(norms.max())
        mean_norm = float(norms.mean())
        std_norm = float(norms.std())
        _title="{}: max: {}, \nmean: {}, std: {}".format(title, '%.2g' % max_norm, '%.2g' % mean_norm, '%.2g' % std_norm)
        ax.set_title(_title)
        ax.grid()
        ax.legend()

class LinearPointwiseTransform(PointwiseTransform):

    @property
    def round_mtx(self):
        return np.array([
            [np.cos(self.phi), -np.sin(self.phi)], 
            [np.sin(self.phi), np.cos(self.phi)]])

    def __init__(self, n, phi=0. , space=1., device='cuda', dtype=torch.float, dim=2):
        super().__init__(device=device, dtype=dtype, dim=dim)
        self.n = n
        self.phi = phi
        self.space=space
    
    @property
    def original_points(self):
        pts = np.linspace(0., (self.n - 1) * self.space, num=self.n)
        pts -= (self.space * (self.n - 1)) / 2.
        pts = np.stack([pts, np.zeros_like(pts)], axis=-1)
        pts = pts @ self.round_mtx.T
        return torch.tensor(pts)

class CirclePointwiseTransform(PointwiseTransform):

    def __init__(self, n, r=1., device='cuda', dtype=torch.float, dim=2):
        super().__init__(device=device, dtype=dtype, dim=dim)
        self.n = n
        self.r = r
        angles = [(np.pi * 2. * i)/self.n for i in range(self.n)]
        self._points = np.asarray([
            [np.cos(ang) * self.r, np.sin(ang) * self.r] for ang in angles])
        
    @property
    def original_points(self):
        return torch.tensor(self._points.copy())

if __name__ == "__main__":
    trf = LinearPointwiseTransform(10, phi=np.pi/4., device='cpu')
    print(trf.round_mtx)
    print(trf.original_points)
    trf = CirclePointwiseTransform(6, device='cpu')
    print(trf.original_points)

