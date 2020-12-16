import random
import torch
import numpy as np
from scipy.linalg import sqrtm
import sklearn.datasets
from sklearn.mixture import GaussianMixture
import abc
from torch.utils.data import Dataset, DataLoader, IterableDataset
from multipledispatch import dispatch
import matplotlib.pyplot as plt

class DistributionSampler(IterableDataset):

    def _create_normalizer(self, n, eps=0.):
        batch = self.sample_distribution(n)
        if isinstance(batch, torch.Tensor):
            batch = batch.cpu().detach().numpy()
        
        np_mean, np_cov = np.mean(batch, axis=0), np.cov(batch.T)
        
        trc_mean = torch.tensor(
            np_mean, dtype=self.dtype
        )
        assert(len(np_cov.shape) != 1)

        if len(np_cov.shape) == 0:
            np_cov = np_cov[np.newaxis, np.newaxis]

        np_multiplier = sqrtm(np_cov)

        # for numerical stability
        if eps > 0:
            np_multiplier += np.eye(np_multiplier.shape[0]) * eps

        trc_multiplier = torch.tensor(
            np_multiplier, dtype=self.dtype
        )

        trc_inv_multiplier = torch.tensor(
            np.linalg.inv(np_multiplier), dtype=self.dtype
        )
            
        def _normalizer(self, batch):
            batch -= trc_mean
            # batch = batch - trc_mean
            batch @= trc_inv_multiplier
            # batch = batch @ trc_inv_multiplier
            return batch
        
        def _inverse_normalizer(self, batch):
            # batch = batch @ trc_multiplier
            batch @= trc_multiplier
            batch += trc_mean
            # batch = batch + trc_mean
            return batch
        
        setattr(
            DistributionSampler, 'mean_var_normalize', _normalizer)
        
        setattr(
            DistributionSampler, 'mean_var_inverse_normalize', _inverse_normalizer)


    @property
    def np_random_state(self):
        if hasattr(self, "_np_random_state"):
            return self._np_random_state
        else:
            raise Exception('np random state is not defined')
    
    @np_random_state.setter
    def np_random_state(self, value):
        if isinstance(value, int) or value is None:
            self._np_random_state = np.random.RandomState(seed=value)
            return
        if isinstance(value, np.random.RandomState):
            self._np_random_state = value
            return
        raise Exception(
            "np random state must be initialized"
            " by int, None or np.random.RandomState instance, "
            " got {} instead".format(type(value)))
    
    @property
    def torch_random_gen(self):
        if hasattr(self, "_torch_random_gen"):
            return self._torch_random_gen
        raise Exception("torch random generator is not defined")
    
    @torch_random_gen.setter
    def torch_random_gen(self, value):
        gen = torch.Generator()
        if value is None:
            gen.seed()
            self._torch_random_gen = gen
            return
        if isinstance(value, int):
            gen.manual_seed(value)
            self._torch_random_gen = gen
            return
        if isinstance(value, torch.Generator):
            self._torch_random_gen = value
            return
        raise Exception(
            "torch random generator must be initialized"
            " by int, None or via torch.Generator instance, "
            " got {} instead".format(type(value)))
        
    def __init__(
        self, device='cuda', dtype=torch.float, 
        requires_grad=False, transform=None):

        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.transform = transform
    
    def __iter__(self):
        def _distribution_generator():
            yield self.sample(1)
        return _distribution_generator
    
    @abc.abstractmethod
    def sample_distribution(self, batch_size):
        pass

    def sample(self, batch_size):
        batch = self.sample_distribution(batch_size)

        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(
                batch, dtype=self.dtype)
        batch = batch.type(self.dtype)

        if hasattr(self, 'mean_var_normalize'):
            batch = self.mean_var_normalize(batch)
        if self.transform is not None:
            batch = self.transform(batch)
        batch = batch.to(self.device)
        batch.requires_grad_()
        return batch
    
    def draw_samples(self, size=1000):

        X = self.sample(size).detach().cpu().numpy()
        plt.scatter(X[:, 0], X[:, 1], edgecolors='black', s=4.)
        plt.show()

class SwissRollSampler(DistributionSampler):

    def __init__(
        self, device='cuda', 
        dtype=torch.float, requires_grad=False, 
        random_seed=None, normalize=False, n_normalize=10000, transform=None):

        super(SwissRollSampler, self).__init__(
            device=device, dtype=dtype, requires_grad=requires_grad, transform=transform
        )
        self.dim = 2
        self.np_random_state = random_seed
        if normalize:
            self._create_normalizer(n_normalize)
    
    def sample_distribution(self, batch_size):
        return sklearn.datasets.make_swiss_roll(
            n_samples=batch_size,
            noise=0.8, random_state=self.np_random_state
        )[0].astype('float32')[:, [0, 2]] / 7.5
    
class StandartNormalSampler(DistributionSampler):
    def __init__(
        self, dim=1, device='cuda',
        dtype=torch.float, requires_grad=False, 
        random_seed=None, normalize=False, 
        n_normalize=10000, transform=None
    ):
        super(StandartNormalSampler, self).__init__(
            device=device, dtype=dtype, requires_grad=requires_grad, transform=transform
        )
        self.dim = dim
        self.torch_random_gen = random_seed
        if normalize:
            self._create_normalizer(n_normalize)
        
    def sample_distribution(self, batch_size):
        return torch.randn(
            batch_size, self.dim, dtype=self.dtype,
            generator=self.torch_random_gen
        )
    
class StandartUniformSampler(DistributionSampler):

    def __init__(
        self, dim=1, device='cuda',
        dtype=torch.float, requires_grad=False, 
        random_seed=None, normalize=False,
        n_normalize=10000, transform=None
    ):
        super(StandartUniformSampler, self).__init__(
            device=device, dtype=dtype, requires_grad=requires_grad, transform=transform
        )
        self.dim = dim
        self.torch_random_gen = random_seed
        if normalize:
            self._create_normalizer(n_normalize)
        
    def sample_distribution(self, batch_size):
        return torch.rand(
            batch_size, self.dim, dtype=self.dtype,
            generator=self.torch_random_gen
        )
    
class BallUniformSampler(DistributionSampler):

    def __init__(
        self, dim=1, device='cuda',
        dtype=torch.float, requires_grad=False, 
        random_seed=None, normalize=False, 
        n_normalize=10000, transform=None
    ):
        super(BallUniformSampler, self).__init__(
            device=device, dtype=dtype, requires_grad=requires_grad, transform=transform
        )
        self.dim = dim
        self.torch_random_gen = random_seed
        if normalize:
            self._create_normalizer(n_normalize)
        
    def sample_distribution(self, batch_size):
        batch = torch.randn(
            batch_size, self.dim,
            device=self.device, dtype=self.dtype, 
            generator=self.torch_random_gen
        )
        batch /= torch.norm(batch, dim=1)[:, None]
        r = torch.rand(
            batch_size, device=self.device, dtype=self.dtype,
            generator=self.torch_random_gen
        ) ** (1. / self.dim)
        return (batch.transpose(0, 1) * r).transpose(0, 1)

class LineGaussiansSampler(DistributionSampler):
    '''
    Creates Sampler of mixture of gausians located equidistantly
    on the line y = 0
    :Parameters:
    n : int : count of gausians
    l : float or int : distance betwee gausians centers 
    std : float : sqr of variance of each gausian
    :: Parameters of Distribution sampler ::
    '''

    def __init__(
        self, n, l=1, std=1., dim=2, device='cuda', dtype=torch.float, 
        requires_grad=False, random_seed=None, normalize=False, 
        n_normalize=10000, transform=None):

        super().__init__(
            device=device, dtype=dtype, requires_grad=requires_grad, 
            transform=transform)
        
        assert dim == 2
        assert n >= 1
        self.dim = 2
        self.n = n
        self.std, self.l = std, float(l)
        centers = torch.arange(
            - self.l * (self.n - 1) / 2., 
            self.l * self.n / 2., 
            self.l, dtype=self.dtype)
        self.centers = torch.stack([centers, torch.zeros_like(centers)]).T
        # print(self.centers.shape)
        
        self.torch_random_gen = random_seed
        if normalize:
            self._create_normalizer(n_normalize)

    def sample_distribution(self, batch_size):
        batch = torch.randn(
            batch_size, self.dim, dtype=self.dtype,
            generator=self.torch_random_gen
        )
        indices = torch.multinomial(
            torch.ones(len(self.centers))/len(self.centers), batch_size, 
            replacement=True, generator=self.torch_random_gen)
        # print(indices)
        # print(self.centers)
        batch *= self.std
        # print(batch.shape)
        # print(self.centers[indices, :].shape)
        batch += self.centers[indices, :]
        return batch
    
class Mix8GaussiansSampler(DistributionSampler):
    def __init__(
        self, with_central=False, std=1, r=12, dim=2, device='cuda',
        dtype=torch.float, requires_grad=False, 
        random_seed=None, normalize=False, 
        n_normalize=10000, transform=None
    ):
        super(Mix8GaussiansSampler, self).__init__(
            device=device, dtype=dtype, requires_grad=requires_grad, 
            transform=transform
        )
        assert dim == 2
        self.dim = 2
        self.std, self.r = std, r
        
        self.with_central = with_central
        centers = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        if self.with_central:
            centers.append((0, 0))
        self.centers = torch.tensor(
            centers, dtype=self.dtype
        )
        self.torch_random_gen = random_seed
        if normalize:
            self._create_normalizer(n_normalize)
        
    def sample_distribution(self, batch_size):
        batch = torch.randn(
            batch_size, self.dim, dtype=self.dtype,
            generator=self.torch_random_gen
        )
        indices = torch.multinomial(
            torch.ones(len(self.centers))/len(self.centers), batch_size, 
            replacement=True, generator=self.torch_random_gen)
        batch *= self.std
        batch += self.r * self.centers[indices, :]
        return batch
    
class MixN2GaussiansSampler(DistributionSampler):
    def __init__(self, n=5, dim=2, std=1, step=9, device='cuda',
        dtype=torch.float, requires_grad=False, 
        random_seed=None, normalize=False, 
        n_normalize=10000, transform=None
    ):
        super(MixN2GaussiansSampler, self).__init__(
            device=device, dtype=dtype, requires_grad=requires_grad, transform=transform
        )
        
        assert dim == 2
        self.dim = 2
        self.std, self.step = std, step
        
        self.n = n
        
        grid_1d = np.linspace(-(n-1) / 2., (n-1) / 2., n)
        xx, yy = np.meshgrid(grid_1d, grid_1d)
        centers = np.stack([xx, yy]).reshape(2, -1).T
        self.centers = torch.tensor(
            centers,
            dtype=self.dtype
        )
        self.torch_random_gen = random_seed
        if normalize:
            self._create_normalizer(n_normalize)
        
    def sample_distribution(self, batch_size):
        batch = torch.randn(
            batch_size, self.dim, dtype=self.dtype, 
            generator=self.torch_random_gen
        )
        indices = torch.multinomial(
            torch.ones(len(self.centers))/len(self.centers), batch_size, 
            replacement=True, generator=self.torch_random_gen)
        batch *= self.std
        batch += self.step * self.centers[indices, :]
        return batch

class DataLoaderWrapper:
    '''
    Helpful class for using the 
    DistributionSampler's in torch's 
    DataLoader manner
    '''

    class FiniteRepeatDSIterator:

        def __init__(self, sampler, batch_size, n_batches):
            dataset = sampler.sample(batch_size * n_batches)
            assert(len(dataset.shape) >= 2)
            new_size = (n_batches, batch_size) + dataset.shape[1:]
            self.dataset = dataset.view(new_size)
            self.batch_size = batch_size
            self.n_batches = n_batches
        
        def __iter__(self):
            for i in range(self.n_batches):
                yield self.dataset[i]
    
    class FiniteUpdDSIterator:

        def __init__(self, sampler, batch_size, n_batches):
            self.sampler = sampler
            self.batch_size = batch_size
            self.n_batches = n_batches
        
        def __iter__(self):
            for i in range(self.n_batches):
                yield self.sampler.sample(self.batch_size)
            
    class InfiniteDsIterator:

        def __init__(self, sampler, batch_size):
            self.sampler = sampler
            self.batch_size = batch_size
        
        def __iter__(self):
            return self
        
        def __next__(self):
            return self.sampler.sample(self.batch_size)


    @dispatch(DistributionSampler, int)
    def __init__(self, sampler, batch_size, n_batches=None, store_dataset=False):
        '''
        n_batches : count of batches before stop_iterations, if None, the dataset is infinite
        store_datset : if n_batches is not None and store_dataset is True, 
        during the first passage through the dataset the data will be stored,
        and all other epochs will use the same dataset, stored during the first pass
        '''
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.store_dataset = store_dataset
        self.sampler = sampler

        if self.n_batches is None:
            self.ds_iter = DataLoaderWrapper.InfiniteDsIterator(
                sampler, self.batch_size)
            return
        
        if not self.store_dataset:
            self.ds_iter = DataLoaderWrapper.FiniteUpdDSIterator(
                sampler, self.batch_size, self.n_batches)
            return
        
        self.ds_iter = DataLoaderWrapper.FiniteRepeatDSIterator(
            sampler, self.batch_size, self.n_batches)

    
    def __iter__(self):
        return iter(self.ds_iter)
    
    def __len__(self):
        return self.n_batches
    
if __name__ == "__main__":
    classes = [
        SwissRollSampler, 
        StandartNormalSampler, 
        StandartUniformSampler,
        BallUniformSampler,
        Mix8GaussiansSampler,
        MixN2GaussiansSampler]
    
    for _cls in classes:
        for rnd_seed in [None, 42]:
            sampler = _cls(
                device='cpu', 
                dtype=torch.float64,
                requires_grad=True,
                random_seed=None,
                normalize=True)
            batch = sampler.sample(17)
            assert(batch.size(0) == 17)
            assert(batch.dtype == torch.float64)
            assert(batch.requires_grad)
    
    sns = DataLoaderWrapper(StandartNormalSampler(
        device='cpu',
        dtype=torch.float64, 
        requires_grad=True,
        random_seed=None,
        normalize=True), 13)
    
    n_iterations = 37

    for i, batch in enumerate(sns):
        assert(batch.size(0) == 13)
        if i >= n_iterations:
            break
    
    sns = DataLoaderWrapper(StandartNormalSampler(
        device='cpu',
        dtype=torch.float64, 
        requires_grad=True,
        random_seed=None,
        normalize=True), 7, n_batches=30)
    
    n_iterations = 0
    for batch in sns:
        assert(batch.size(0) == 7)
        n_iterations += 1
    assert(n_iterations == 30)

    sns = DataLoaderWrapper(StandartNormalSampler(
        device='cpu',
        dtype=torch.float64, 
        requires_grad=True,
        random_seed=None,
        normalize=True), 7, n_batches=30, store_dataset=True)
    
    n_iterations = 0
    sp_batch = None
    for i, batch in enumerate(sns):
        assert(batch.size(0) == 7)
        if i == 13:
            sp_batch = batch
        n_iterations += 1
    assert(n_iterations == 30)

    for i, batch in enumerate(sns):
        if i == 13:
            assert((sp_batch == batch).all())


