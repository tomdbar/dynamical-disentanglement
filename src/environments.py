from abc import ABC, abstractmethod
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.representations import Representation


class BaseWorld(ABC):
    '''Base class for constructing environments.'''

    class action_space():
        def __init__(self, n_actions, batch_size):
            '''Action space of an environment.

            Args:
                n_actions: The number of allowed actions.
                batch_size: The number of parallel environments configured.
            '''
            self.n = n_actions
            self._batch_size = batch_size

        def sample(self, k=None):
            '''Randomly sample from the action space.

            By default, a single random action will be sampled for each
            parallel environment.

            Args:
                k: The number of actions to sample.

            Returns: A tensor of [batch_size, action].
            '''
            if k is None:
                k = self._batch_size
            return torch.randint(0, self.n, (k,))

    class observation_space():
        def __init__(self, shape):
            '''Observation space of an environment.

            Args:
                shape: The shape the observation tensor..
            '''
            self.shape = shape

    def __init__(self, n_actions, n_observations, batch_size=1, random_resets = False, device="cpu"):
        '''Base construction required for all environments.

        Args:
            n_actions: The number of allowed actions.
            n_observations: The shape of the observations.
            batch_size: The number of parallel environments to configure.
            random_resets: Whether environments reset to random initialisations, or always to the
                           same state.
            device: The device to which returned tensors are mapped.  Should be "cpu" or "cuda".
        '''
        self.action_space = self.action_space(n_actions, batch_size)
        self.observation_space = self.observation_space(n_observations)
        self.batch_size = batch_size
        self.random_resets = random_resets
        self.device = device

    @abstractmethod
    def reset(self, state):
        '''Reset the environment.

        Args:
            state: The state to which the environment is reset.

        Returns: The new observation of the environment.
        '''
        raise NotImplementedError()

    @abstractmethod
    def get_observation(self):
        '''Get the observation corresponding to the current state.

        Returns: The observation (as a tensor).
        '''
        raise NotImplementedError()

    @abstractmethod
    def step(self, actions, magntidues):
        '''Take an action to evolve the environment.

        Args:
            actions: The action(s) to take.  If multiple actions are passed, it is assumed
                     each corresponds to one of the parallel environments configured.
            magntidues: The size of each action.

        Returns: The new observation of the environment(s) after the action(s) is taken.
        '''
        raise NotImplementedError()

class DynamicWorld(BaseWorld):
    '''A dynamical environment corresponding to a FactorisedDataset.'''

    def __init__(self, dataset, batch_size=1, random_resets=False, device="cpu"):
        '''A dynamical environment corresponding to a FactorisedDataset.

        The dynamical environment corresponding to a FactorisedDataset considers
        each generative factor (that can take more than one value) to correspond
        to two actions: one to step in each direction along the cycle of allowed
        values.

        Args:
            dataset: A FactorisedDataset.
            batch_size: The number of parallel environments to configure.
            random_resets: Whether environments reset to random initialisations, or always to the
                           same state.
            device: The device to which returned tensors are mapped.  Should be "cpu" or "cuda".
        '''
        super().__init__(n_actions=int((dataset.factor_space._factor_sizes > 1).sum()) * 2,
                         n_observations=(64, 64, 3),
                         batch_size=batch_size,
                         random_resets=random_resets,
                         device=device)

        self.dataset = dataset
        self.dataset.to_torch(device="cpu")
        self.num_dims = self.dataset.factor_space.num_dims
        self.factor_sizes = self.dataset.factor_space._factor_sizes
        self.factor_vals = self.dataset.factor_space.factor_vals
        # As we only consider actionable dimensions as those with moret than one possible value,
        # create a mask to select only these.
        self._act_dims_mask = (self.factor_sizes > 1)

        self.reset()

    def reset(self, factors=None, random=None):
        '''Reset the environment.

        Args:
            state: The state to which the environment is reset.

        Returns: The new observation of the environment.
        '''
        if random is None:
            random = self.random_resets
        if random:
            self.state = torch.stack([torch.randint(0, int(size), size=(self.batch_size,))
                                      for size in self.factor_sizes], dim=1).to(self.device)
        else:
            self.state = torch.zeros(size=(self.batch_size, self.num_dims), device=self.device).int()
        self.factors = None
        return self.get_observation()

    def get_observation(self, channel_last=False):
        '''Get the observation corresponding to the current state.

        Args:
            channel_last : If true, returned tensor is of shape (W,H,C), otherwise (C,W,H) is returned.

        Returns: The observation (as a tensor).
        '''
        obs = self.dataset.get_data(self.get_factors())
        if not channel_last:
            obs = obs.permute(0, 3, 1, 2)
        return obs.squeeze().float().to(self.device)

    def get_factors(self):
        if self.factors is None:
            self.factors = np.array([np.take(vals, self.state[..., idx].cpu().numpy())
                                     for idx, vals in enumerate(self.factor_vals)]
                                    ).T.squeeze()
        return self.factors

    def step(self, actions, magntidues=None):
        '''Take an action to evolve the environment.

        Args:
            actions: The action(s) to take.  If multiple actions are passed, it is assumed
                     each corresponds to one of the parallel environments configured.
            magntidues: The size of each action.  By default all actions have a magnitude of 1.

        Returns: The new observation of the environment(s) after the action(s) is taken.
        '''
        try:
            assert len(actions) == self.batch_size, "Batch of actions does not match batch size."
        except TypeError:
            # Assume single action is passed.
            actions = [actions] * self.batch_size

        if magntidues is not None:
            try:
                assert len(magntidues) == self.batch_size, "Batch of magntidues does not match batch size."
            except TypeError:
                # Assume single magntidues is passed.
                magntidues = [magntidues] * self.batch_size
        else:
            magntidues = [1] * self.batch_size

        actions = np.array(actions)

        act_dims = actions // 2
        act_vals = np.ones_like(actions)
        #         act_vals = torch.ones(actions.shape)
        act_vals[actions % 2 != 0] = -1

        act_vals *= np.array(magntidues, dtype=np.int)

        actionable_state_dims = self.state[..., self._act_dims_mask].cpu().numpy()
        target_state_vals = actionable_state_dims[np.arange(len(actionable_state_dims)), act_dims]

        target_state_vals = (target_state_vals + act_vals) % self.factor_sizes[self._act_dims_mask][act_dims]

        actionable_state_dims[np.arange(len(actionable_state_dims)), act_dims] = target_state_vals
        self.state[..., self._act_dims_mask] = torch.from_numpy(actionable_state_dims).to(self.state)

        self.factors = None

        return self.get_observation()

    def view(self, batch_idx=0, obs=None, figsize=(3, 3)):
        '''Look at an image corresponding a state of the environment.

        This is a utility function to allow us to quickly vizualise the ground-truth or
        reconstructed states.

        Args:
            batch_idx: Index of environment to look at (if multiple are configured to run in parallel).
            obs: An observation to look at.  This overwrites the batch_idx argument if passed.
            figsize: The size of the image as (W,H) tuple.

        Returns: matplotlib.figure
        '''
        assert batch_idx < self.batch_size, f"batch_idx must be < batch size (={self.batch_size})"
        if obs is None:
            obs = self.get_observation(channel_last=True)
            factors = self.get_factors()
            if self.batch_size > 1:
                obs = obs[batch_idx]
                factors = factors[batch_idx]
        else:
            if obs.shape[-1] != 3:
                obs = obs.permute(1, 2, 0)

        fig = plt.figure(frameon=True, figsize=figsize)
        plt.imshow(obs)
        ax = plt.gca()
        title = self.dataset.factors2string(factors)
        if self.batch_size > 1:
            title = f"batch_idx : {batch_idx}\n" + title
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for pos in ['bottom', 'top', 'right', 'left']:
            ax.spines[pos].set_color('0')

        fig.tight_layout()

        return fig

class LatentWorld(BaseWorld):
    '''A spherical latent space evolved with unitary rotations as actions.'''

    def __init__(self,
                 dim=4,
                 n_actions=4,
                 action_reps=None,
                 batch_size=1,
                 random_resets=False,
                 device="cpu"):
        '''A spherical latent space evolved with unitary rotations as actions.

        Args:
            dim: Dimension of the spherical latent space.
            n_actions: The number of possible actions.
            action_reps: A list of Representation objects, one for each action.  If None, these will
                         be constructed and randomly initialised.
            batch_size: The number of parallel environments to configure.
            random_resets: Whether environments reset to random initialisations, or always to the
                           same state.
            device: The device to which returned tensors are mapped.  Should be "cpu" or "cuda".
        '''
        super().__init__(n_actions, dim, batch_size, random_resets, device)
        self.dim = dim

        if action_reps is None:
            self.action_reps = [Representation(dim=self.dim, device=self.device) for _ in range(n_actions)]
        else:
            if len(action_reps) != n_actions:
                raise Exception("Must pass an action representation for every action.")
            if not all([rep.dim == self.dim for rep in self.action_reps]):
                raise Exception("Action representations do not act on the dimension of the latent space.")
            self.action_reps = action_reps

        self.reset()

    def reset(self, state=None, random=None):
        '''Reset the environment.

        Args:
            state: The state to which the environment is reset.

        Returns: The new observation of the environment.
        '''
        if random is None:
            random = self.random_resets

        if state is None:
            # No state was passed, randomly reset.
            if random:
                state = torch.randint(0, 2, size=(self.batch_size, self.dim, 1), device=self.device).float()
                state /= torch.norm(state, p=2, dim=1, keepdim=True).clamp(min=1)
            else:
                state = torch.zeros(size=(self.batch_size, self.dim, 1), device=self.device).float()
                state[:,0] = 1
        else:
            state = state.to(self.device)
            # Explicit state passed, let's just make sure it is of the form: (batch, state, 1).
            if state.shape[0] != self.batch_size:
                # (state, [1]) --> (batch_size, state, [1])
                # Note: .clone() to ensure every expanded element has it's own memory so we
                # can modify in-place and still track gradients.
                state = state.expand(self.batch_size, *state.shape).clone()
            if state.dim() < 3:
                #  (batch_size, state) --> (batch_size, state, 1)
                state = state.unsqueeze(-1)
        self.state = state
        return self.get_observation()

    def get_observation(self):
        '''Get the observation corresponding to the current state.

        Returns: The observation (as a tensor).
        '''
        return self.state.squeeze()

    def step(self, actions, magntidues=None):
        '''Take an action to evolve the environment.

        Args:
            actions: The action(s) to take.  If multiple actions are passed, it is assumed
                     each corresponds to one of the parallel environments configured.
            magntidues: The size of each action.  By default all actions have a magnitude of 1.

        Returns: The new observation of the environment(s) after the action(s) is taken.
        '''
        if magntidues is None:
            magntidues = 1
        batch_act = True
        if isinstance(actions, Iterable) and isinstance(magntidues, Iterable):
            assert len(actions) == self.batch_size, "Batch of actions does not match batch size."
            assert len(magntidues) == self.batch_size, "Batch of magnitudes does not match batch size."

        elif isinstance(actions, Iterable):
            assert len(actions) == self.batch_size, "Batch of actions does not match batch size."
            magntidues = [magntidues] * self.batch_size

        elif isinstance(magntidues, Iterable):
            assert len(magntidues) == self.batch_size, "Batch of magnitudes does not match batch size."
            actions = [actions] * self.batch_size

        else:
            batch_act = False

        if batch_act:
            mats_list = [self.action_reps[act].get_matrix(mag) for act, mag in zip(actions, magntidues)]
            rep_mats = torch.stack(mats_list)
        else:
            rep_mats =  self.action_reps[actions].get_matrix(magntidues)

        self.state = torch.matmul(rep_mats, self.state)
        obs = self.get_observation()
        return obs

    def clear_representations(self):
        '''Clear the cached unitary matrices for all action representations.

        The action matrices are cached to avoid re-calculating them at every step.  However,
        if the underlying parameters are changed (e.g. after a step of SGD), this cache must
        be cleared so that the correct representations are re-calculated and cached in their
        place.
        '''
        for rep in self.action_reps:
            rep.clear_matrix()

    def get_representation_params(self, detach=False):
        '''Get the parameters of the action representations.

        Args:
            detach: Whether to detach these parameters from the computation graph.

        Returns: A list of tuples, each tuple the thetas of a single action representation.
        '''
        params = []
        for rep in self.action_reps:
            thetas = rep.thetas
            if detach:
                thetas = thetas.detach().cpu()
            params.append(thetas)
        return params

    def get_representations(self):
        '''Get the parameters of the action representations.

        Returns: A list of tuples, each tuple the thetas of a single action representation.
        '''

        return [rep.thetas for rep in self.action_reps]

    def set_representations(self, representations):
        '''Set the parameters of the action representations

        Args:
            representations: A list of tuples, each tuple the thetas of a single action representation.
        '''
        for rep in self.action_reps:
            rep.set_thetas(representations.pop(0))