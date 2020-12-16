import os
import abc
from .models import ToyCritic, ToyGenerator, ToyMover
from .distributions import DataLoaderWrapper, DistributionSampler
import p3wgan.distributions as distributions
from .state import State
from tqdm import tqdm
from .transport_costs import QuadraticTransportCost
from .utils import StatisticCollector
from IPython.display import clear_output
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
sns.set()

class NoAttributesException(Exception):

    def _message_one_attribute(self, arg):
        return "attribute: '{}' has not been provided".format(arg)
    
    def _message_many_attributes(self, *args):
        message = "attributes: "  
        message += ", ".join(map(lambda x: "'{}'".format(x), args))
        message += " have not bee provided"
        return message

    def __init__(self, obj, *args):
        message = "In class '{}': ".format(type(obj).__name__)
        if len(args) == 0:
            message = "<Incorrect NoAttributesException usage>"
        elif len(args) == 1:
            message += self._message_one_attribute(args[0])
        else:
            message += self._message_many_attributes(*args)
        super().__init__(self, message)

def check_presence(*args):
    def decorator(function):
        def wrapper(self, state, *_args, **_kwargs):
            self._check_attributes_presence(state, *args)
            return function(self, state, *_args, **_kwargs)
        return wrapper
    return decorator

def set_default(**kwargs):
    def decorator(function):
        def wrapper(self, state, *_args, **_kwargs):
            self._add_not_presented_attributes(state, **kwargs)
            return function(self, state, *_args, **_kwargs)
        return wrapper
    return decorator

class RunnableItem(abc.ABC):

    def _check_attributes_presence(self, state, *args):
        missed = []
        for arg in args:
            try:
                getattr(state, arg)
            except AttributeError:
                missed.append(arg)
        if len(missed) > 0:
            raise NoAttributesException(self, *missed)
    
    def _add_not_presented_attributes(self, state, **kwargs):
        for key, value in kwargs.items():
            try:
                getattr(state, key)
            except AttributeError:
                setattr(state, key, value)

    def __init__(self):
        pass
    

    def _preprocessing(self, state):
        self_class = self.__class__
        try:
            mro = self.__class__.__mro__[:-1]
            r_mro = tuple(reversed(mro))
            for _cls in r_mro:
                if 'preprocessing' in _cls.__dict__.keys():
                    self.__class__ = _cls
                    self.preprocessing(state)
        except:
            self.__class__ = self_class
            raise
        self.__class__ = self_class

    
    def _postprocessing(self, state):
        self_class = self.__class__
        try:
            mro = self.__class__.__mro__[:-1]
            for _cls in mro:
                if 'postprocessing' in  _cls.__dict__.keys():
                    self.__class__ = _cls
                    self.postprocessing(state)
        except:
            self.__class__ = self_class
            raise
        self.__class__ = self_class

    def preprocessing(self, state):
        pass

    def postprocessing(self, state):
        pass

    def run(self, state):
        state.attach_level() # level for local variables (created during preprocessing)

        try:
            self._preprocessing(state)
        except:
            state.detach_level()
            raise
        
        state.attach_level() # level for internal run results

        try:
            self.internal_run(state)
            self._postprocessing(state) # postprocessing works with both result and local arguments levels
        except:
            state.detach_level()
            state.detach_level()
            raise
    
        res_level_dict = state.get_level_dict() # get results level dict
        state.detach_level() # detach results level
        state.detach_level() # detach arguments level
        state.attach_current(**res_level_dict) # add results level arguments to the state
    
    @abc.abstractmethod
    def internal_run(self, state):
        pass

class ToyModelsCreator(RunnableItem):

    @set_default(
        data_dim=2, # dimensionality of the data
        n_layers=3, # count of layers in the models
        hid_dim=128, # hidden dimension in hidden layers
        z_dim=128) # hidden space dimensionality
    @check_presence('device')
    def preprocessing(self, state):
        pass

    def internal_run(self, state):
        s = state
        hiddens = [s.hid_dim for i in range(s.n_layers - 1)]
        s.generator = ToyGenerator(s.z_dim, s.data_dim, hiddens, device=s.device)
        s.critic = ToyCritic(s.data_dim, hiddens, device=s.device)
        s.mover = ToyMover(s.data_dim, hiddens, device=s.device)

class QuadraticTransportCostCreator(RunnableItem):

    def internal_run(self, state):
        state.cost = QuadraticTransportCost()


class ToyDistributionCreator(RunnableItem):

    @set_default(
        batch_size=2048,
        num_iter_per_epoch=5, # number of iterations per epoch
        distribution='8g')
    @check_presence('device')
    def preprocessing(self, state):
        pass

    def internal_run(self, state):
        s = state
        arg_dict = {'device':s.device, 'requires_grad':False, 'random_seed':None, 'normalize':False}

        if s.distribution == '8g':
            distrib = distributions.Mix8GaussiansSampler(std=0.2, r=2., **arg_dict)
        elif s.distribution == 'swiss':
            distrib = distributions.SwissRollSampler(**arg_dict)
        else:
            raise Exception('Unknown output distribution')

        s.dataloader = DataLoaderWrapper(
            distrib, s.batch_size, n_batches=s.num_iter_per_epoch)


class W3PPipeline(RunnableItem):

    def before_epoch(self, s, ps):
        pass

    def before_batch(self, s, ps):
        pass

    def before_train(self, s, ps):
        pass

    def before_mover(self, s, ps):
        pass

    def after_mover(self, s, ps):
        pass

    def before_critic(self, s, ps):
        pass

    def after_critic(self, s, ps):
        pass

    def before_generator(self, s, ps):
        pass

    def after_generator(self, s, ps):
        pass

    def after_batch(self, s, ps):
        pass

    def after_epoch(self, s, ps):
        pass

    def after_train(self, s, ps):
        pass
    
    @set_default(
        num_training_mover=5,
        num_training_generator=1,
        num_training_critic=1,
        learning_rate=0.0001,
        beta1=0.0,  # \beta1 in Adam optimiser
        beta2=0.9,  # \beta2 in Adam optimiser
        num_epochs=500)
    @check_presence(
        'generator', 'critic', 'mover', 'dataloader')
    def preprocessing(self, state):
        s = state
        s.generator_optim = torch.optim.Adam(s.generator.parameters(), lr=s.learning_rate, betas=(s.beta1, s.beta2))
        s.critic_optim = torch.optim.Adam(s.critic.parameters(), lr=s.learning_rate, betas=(s.beta1, s.beta2))
        s.mover_optim = torch.optim.Adam(s.mover.parameters(), lr=s.learning_rate, betas=(s.beta1, s.beta2))
        s.process_state = State(show_warnings=False)
        s.steps_per_session = s.num_training_mover + s.num_training_generator + s.num_training_critic

    def internal_run(self, state):
        s = state
        ps = state.process_state
        ps.gamma=1. 
        self.before_train(s, ps)
        for i_epoch in range(s.num_epochs):
            ps.i_epoch = i_epoch
            self.before_epoch(s, ps)
            for i_batch, batch in tqdm(
                enumerate(s.dataloader), 
                total=len(s.dataloader), 
                desc="Epoch: {}".format(ps.i_epoch)):
                ps.i_batch = i_batch
                ps.batch = batch.to(s.device)
                self.before_batch(s, ps)
                i_session = ps.i_batch % s.steps_per_session
                if i_session < s.num_training_mover:
                    ps.i_session = i_session
                    self.before_mover(s, ps)
                    # mover step
                    s.mover_optim.zero_grad()
                    mover_batch = s.mover(ps.batch)
                    ps.cost_loss = s.cost(mover_batch, ps.batch)

                    ps.loss = (ps.gamma * ps.cost_loss - s.critic(mover_batch)).mean()
                    ps.loss.backward()
                    s.mover_optim.step()
                    self.after_mover(s, ps)
                elif i_session < s.num_training_mover + s.num_training_critic:
                    ps.i_session = i_session - s.num_training_mover
                    self.before_critic(s, ps)
                    # critic step
                    b_size = ps.batch.size(0)
                    s.critic_optim.zero_grad()
                    mover_batch = s.mover(ps.batch)
                    gen_batch = s.generator.sample(b_size)

                    ps.loss = (s.critic(mover_batch) - s.critic(gen_batch)).mean()
                    ps.loss.backward()
                    s.critic_optim.step()
                    self.after_critic(s, ps)
                else:
                    ps.i_session = i_session - s.num_training_mover - s.num_training_critic
                    self.before_generator(s, ps)
                    # generator step
                    b_size = ps.batch.size(0)
                    s.generator_optim.zero_grad()
                    gen_batch = s.generator.sample(b_size)
                    
                    ps.loss = s.critic(gen_batch).mean()
                    ps.loss.backward()
                    s.generator_optim.step()
                    self.after_generator(s, ps)
                self.after_batch(s, ps)
            self.after_epoch(s, ps)
        self.after_train(s, ps)

class GammaAnnealing(W3PPipeline):

    @set_default(
        gamma_min=0.1,
        gamma_max=100.)
    def preprocessing(self, state):
        assert(state.gamma_max >= state.gamma_min)

class  LinearGammaAnnealing(GammaAnnealing):

    def before_train(self, s, ps):
        super().before_train(s, ps)
        ps.gamma = s.gamma_min
        n_iterations = s.num_epochs * len(s.dataloader)
        ps.gamma_add_factor = (s.gamma_max - s.gamma_min)/n_iterations
    
    def after_batch(self, s, ps):
        ps.gamma += ps.gamma_add_factor
        super().after_batch(s, ps)

class ExponGammaAnnealing(GammaAnnealing):

    def before_train(self, s, ps):
        super().before_train(s, ps)
        ps.gamma = s.gamma_min
        n_iterations = s.num_epochs * len(s.dataloader)
        final_factor = float(s.gamma_max) / s.gamma_min
        ps.gamma_mul_factor = final_factor ** (1/n_iterations)
    
    def after_batch(self, s, ps):
        ps.gamma *= ps.gamma_mul_factor
        super().after_batch(s, ps)

class Drawer(W3PPipeline):

    def clear_output(self, s, ps):
        if ps.draw_ticket:
            clear_output(wait=True)
            ps.draw_ticket = False
        
    def before_train(self, s, ps):
        super().before_train(s, ps)
    
    def before_epoch(self, s, ps):
        super().before_epoch(s, ps)
        ps.draw_ticket=True

    @set_default(
        draw_images=True)
    def preprocessing(self, state):
        pass

class TransportVisualizer2D(Drawer):

    @set_default(
        n_batches_to_draw=1)
    def preprocessing(self, state):
        pass

    @torch.no_grad()
    def after_epoch(self, s, ps):
        if s.draw_images:
            s.generator.eval()
            s.mover.eval()
            original = []
            moved = []
            b_size = 0
            for i, batch in enumerate(s.dataloader):
                assert len(batch.shape) == 2 and batch.size(1) == 2 , 'only 2d can be visualized'
                batch = batch.to(s.device).detach()
                b_size = batch.size(0)
                moved_batch = s.mover(batch)
                if np.isnan(moved_batch).any():
                    print('moved batch contains nans!')
                original.append(batch.cpu().numpy())
                moved.append(moved_batch.cpu().numpy())
                if i >= s.n_batches_to_draw - 1:
                    break
            original = np.concatenate(original, axis=0)
            moved = np.concatenate(moved, axis=0)
            generated = s.generator.sample(b_size * s.n_batches_to_draw).cpu().numpy()

            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
            axs[0].scatter(original[:, 0], original[:, 1], s=2.)
            axs[0].set_title('Original')
            axs[1].scatter(moved[:, 0], moved[:, 1], s=2.)
            axs[1].set_title('Moved')
            axs[2].scatter(generated[:, 0], generated[:, 1], s=2.)
            axs[2].set_title('Generated')
            plt.show()
            s.generator.train()
            s.mover.train()
        super().after_epoch(s, ps)

class LossesCollector(Drawer):

    @set_default(
        mover_loss_n_averaging=20,
        critic_loss_n_averaging=4,
        generator_loss_n_averaging=4,
        plot_losses=True)
    def preprocessing(self, state):
        pass

    def before_train(self, s, ps):
        super().before_train(s, ps)
        ps.mover_loss = StatisticCollector(s.mover_loss_n_averaging)
        ps.critic_loss = StatisticCollector(s.critic_loss_n_averaging)
        ps.generator_loss = StatisticCollector(s.generator_loss_n_averaging)
    
    def after_mover(self, s, ps):
        ps.mover_loss.add(ps.loss.item())
        super().after_mover(s, ps)
    
    def after_critic(self, s, ps):
        ps.critic_loss.add(ps.loss.item())
        super().after_critic(s, ps)
    
    def after_generator(self, s, ps):
        ps.generator_loss.add(ps.loss.item())
        super().after_generator(s, ps)
    
    def after_epoch(self, s, ps):
        ps.mover_loss.reset_current()
        ps.critic_loss.reset_current()
        ps.generator_loss.reset_current()
        self.clear_output(s, ps)
        if s.draw_images:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
            ps.mover_loss.plot_statistic(axs[0], title='Mover Loss')
            ps.critic_loss.plot_statistic(axs[1], title='Critic Loss')
            ps.generator_loss.plot_statistic(axs[2], title='Gen Loss')
            plt.show()
        super().after_epoch(s, ps)

class ToyP3WGAN(ExponGammaAnnealing, LossesCollector, TransportVisualizer2D):
    pass


