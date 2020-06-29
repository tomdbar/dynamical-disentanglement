import os
import time
from datetime import datetime
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import gridspec

from src.coders.conv import EncoderConv, DecoderConv
from src.environments import DynamicWorld, LatentWorld
from src.utils import mk_dir


class LossTarget(Enum):
    '''The aim of training w.r.t. the  (entanglement) loss.

    MIN : minimise the loss (i.e. maximal disentanglement).
    MAX : maximise the loss (i.e. maximal entanglement)
    VALUE : aim for a specific value of loss (i.e. tailored entanglement).
    '''
    MIN = 0
    MAX = 1
    VALUE = 2

class DynamicRepresentationLearner():
    '''Prepare, train and test the representations learning in a dynamical environment.

    ---Preparation---

    This high level class abstracts as much of the set up as possible by automatically
    creating self-consistent instances of:
        - DynamicWorld (obs_env): The dynamical environment associated with the passed
          FactorisedDataset
        - LatentWorld (lat_env): The spherical latent space to which we map observations.
        - Representation(s) (lat_env.action_reps):  Unitary transformations corresponding
          to actions in the latent space.
        - Encoder/Decoder (encoder/decoder): Convolutional neural networks to map between
          the observation and latent space.

    ---Training---

    The training of the encoder, decoder and representations is also handled.  It can be
    configured by changing the learning rates of the corresponding optimizers, as well as
    how (if at all) the desired (dis)entanglement regularisation is applied.

    Intermediate and final training states can be saved automatically, as well as learning
    curves for the various loss contributions (total, reconstruction and entanglement).

    ---Testing---

    The representations can also be plotted and tested using the built in functions.
    '''

    def __init__(self,
                 dataset,
                 latent_dim,
                 episode_length=10,
                 num_parallel_episodes=1,
                 max_action_magnitude=1,
                 coder_args={"obs_channels": 3,
                             "obs_width": 64,
                             "kernels": [[32, 4, 2, 0]] * 2 + [[64, 4, 2, 0]] * 2,
                             # (out_channels, kernel_size, stride, padding)
                             "n_hid": [256],
                             "activation_fn": nn.ReLU,
                             "batch_norm": False},
                 lr_enc=5e-3,
                 lr_dec=5e-3,
                 lr_rep=5e-3,
                 ent_loss_weight=0,
                 final_ent_loss_weight=None,
                 final_ent_loss_weight_iter=None,
                 ent_loss_target=LossTarget.MIN,
                 device=None,
                 random_resets=False,
                 save_loc=None):
        '''Initialise the DynamicRepresentationLearner and underlying structures.

        To prepare the environments and representations, pass a FactorisedDataset and
        some key arguments to pass to the relevent constructors.

        The system can then be trained (learning rates etc are also passed to the
        initialiser) by initialising the observation and latent space to a known
        state(s), and then evolving both by taking randomly selected actions. After
        each step the ground-truth observation can be compared to the observation
        reconstructed from the latent space, providing the reconstruction loss.
        Additional disentanglment regularisation can also be applied.  After each
        episode, the environments are reset, a step of SGD is taken, and the next
        episode being.


        Args:
            dataset: A FactorisedDataset.
            latent_dim:  Dimension of the spherical latent space.
            episode_length: How many random actions to take per episode.
            num_parallel_episodes: The number of episodes to run in parallel.
            max_action_magnitude: The maximum number of positions a single action
                                  can change the associated generative factor in
                                  it's cycle.
            coder_args: A dictionary of keyword arguments to pass to both the EncoderConv
                        and DecoderConv constructors.
            lr_enc: Learning rate for the (Adam) optimizer of the encoder parameters.
            lr_dec: Learning rate for the (Adam) optimizer of the decoder parameters.
            lr_rep: Learning rate for the (Adam) optimizer of action representation parameters.
            ent_loss_weight: The initial weight of the entanglement regularisation.
            final_ent_loss_weight: The final weight of the entanglement regularisation.
            final_ent_loss_weight_iter: The number of steps over which the en entanglement
                                        regularisation is (linearly) increased from it's inital
                                        to final value.
            ent_loss_target: The target to which entanglement regularisation loss is trained.
                             Either LossTarget.MIN, LossTarget.MAX or a number.
            device: The device on which training is performed.  Should be "cpu" or "cuda".
            random_resets: Whether environments reset to random initialisations, or always to the
                           same state.
            save_loc: The directory to save training and testing output.  Will be created if it
                      does not alraedy exist.
        '''

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            # I trust that you know what you are doing!
            self.device = device

        # Cast dataset to torch, but on CPU to avoid storing everything on the GPU, even if one is available.
        dataset.to_torch("cpu")

        self.episode_length = episode_length
        self.num_parallel_episodes = num_parallel_episodes
        self.max_action_magnitude = int(max_action_magnitude)

        self.obs_env = DynamicWorld(dataset,
                                    batch_size=num_parallel_episodes,
                                    random_resets = random_resets,
                                    device=self.device)
        self.lat_env = LatentWorld(dim=latent_dim,
                                   n_actions=self.obs_env.action_space.n,
                                   batch_size=num_parallel_episodes,
                                   random_resets = random_resets,
                                   device=self.device)

        coder_args['lat_dim'] = latent_dim
        coder_args['device'] = self.device

        self.coder_args = coder_args
        self.lr_enc = lr_enc
        self.lr_dec = lr_dec
        self.lr_rep = lr_rep

        self.init_ent_loss_weight = ent_loss_weight
        self.final_ent_loss_weight = final_ent_loss_weight
        self.final_ent_loss_weight_iter = final_ent_loss_weight_iter
        if (self.final_ent_loss_weight is None) or (self.final_ent_loss_weight_iter is None):
            self.get_ent_loss_weight = lambda iter: self.init_ent_loss_weight
        else:
            def get_ent_loss_weight(iter):
                r = min(iter/self.final_ent_loss_weight_iter, 1)
                return self.init_ent_loss_weight + r*(self.final_ent_loss_weight - self.init_ent_loss_weight)
            self.get_ent_loss_weight = get_ent_loss_weight

        self.reset()

        if type(ent_loss_target) is LossTarget:
            if ent_loss_target is LossTarget.VALUE:
                raise ValueError(
                    "The loss targets should be either LossTarget.MIN, LossTarget.MAX or a numerical value."
                )
            self.ent_loss_target = self.ent_loss_target_type = ent_loss_target
        else:
            self.ent_loss_target = ent_loss_target
            self.ent_loss_target_type = LossTarget.VALUE

        # Create directories to save results.
        self.save_loc = save_loc
        if self.save_loc is None:
            self.save_loc = datetime.now().strftime("%y-%d-%m_%H-%M-%S")

        self.train_folder = os.path.join(self.save_loc, "train")
        self.test_folder = os.path.join(self.save_loc, "test")

        mk_dir(self.save_loc, quite=True)
        mk_dir(self.train_folder, quite=True)
        mk_dir(self.test_folder, quite=True)

    def reset(self):
        '''Reset to a random representation.

        Resets the encoder, decoder, action representations and optimizers.
        Also clears and saved progress, as if the DynamicRepresentationLearner
        has just been initialised.
        '''
        self.encoder = EncoderConv(**self.coder_args)
        self.decoder = DecoderConv(**self.coder_args)

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        
        self.reset_optimizers()

        self.ent_loss_weight = self.get_ent_loss_weight(0)

        # Set up lists to save training data.
        self.total_training_steps = 0
        self.total_training_time = 0
        self.losses = []
        self.reconstruction_losses = []
        self.entanglement_losses = []
        
    def reset_optimizers(self):
        '''Reset the optimizers for all trainable parameters.'''
        self.reset_encoder_optimizer()
        self.reset_decoder_optimizer()
        self.reset_reps_optimizer()

    def reset_encoder_optimizer(self):
        '''Reset the optimizer for the encoder network.'''
        self.optimizer_enc = optim.Adam(self.encoder.parameters(),
                                        lr=self.lr_enc,
                                        weight_decay=0)

    def reset_decoder_optimizer(self):
        '''Reset the optimizer for the decoder network.'''
        self.optimizer_dec = optim.Adam(self.decoder.parameters(),
                                        lr=self.lr_dec,
                                        weight_decay=0)

    def reset_reps_optimizer(self):
        '''Reset the optimizer for the action representations.'''
        self.optimizer_rep = optim.Adam(self.lat_env.get_representation_params(),
                                        lr=self.lr_rep,
                                        weight_decay=0)

    def set_random_resets(self, random_resets):
        '''Configure the reset behaviour of the observation and latent environments.

        Args:
            random_resets: Boolean flag. Whether environments reset to random
                           initialisations (True), or always to the same state (False).
        '''
        self.obs_env.random_resets = random_resets
        self.lat_env.random_resets = random_resets

    def _calc_entanglement(self):
        '''Calculate the entanglement of the action representation matrices.

        The entanglement is given by the sum of the L2-norms of the theta
        parameters (mixing angles) defining the unitary rotations: minus
        the single largest theta for each representation.

        Note that this does not directly penalise 'entanglement', but rather
        it is sufficient that any set of action matrices minimising this metric
        will be well disentangled.

        Returns: The entanglement loss.
        '''
        params_list = [r.thetas for r in self.lat_env.action_reps]
        total_entanglement = 0
        for params in params_list:
            params = params.abs().pow(2)
            # params = params.abs()
            total_entanglement += (params.sum() - params.max())
        return total_entanglement / len(params_list)

    def train(self, num_sgd_steps, log_freq=10):
        '''Train the representation.

        Each step of SGD uses the reconstruction errors accumulated
        over a single episode.  If multiple episodes are run in parallel,
        their contributions are averaged.  Any disentanglement regularisation
        is applied only once per SGD step, rather than at each time-step within
        the episodes.

        Args:
            num_sgd_steps: The number of steps to train for.
            log_freq: The frequency (in terms of number of SGD steps) with which
                      we save metrics and print output that monitor training progress.
        '''
        def run_episode():
            t_step = -1
            # reconstruction_loss = 0
            observation_grounds, observation_latent = [], []

            while t_step < self.episode_length:
                if t_step == -1:
                    obs_x = self.obs_env.reset()
                    obs_z = self.lat_env.reset(self.encoder(obs_x))
                else:
                    action = self.obs_env.action_space.sample()
                    if self.max_action_magnitude > 1:
                        mags = torch.randint_like(action, 1, self.max_action_magnitude+1)
                        # mags = self.max_action_magnitude*torch.rand_like(action)
                    else:
                        mags = None
                    obs_x = self.obs_env.step(action, mags)
                    obs_z = self.lat_env.step(action, mags)

                t_step += 1

                if obs_x.dim() == 3:
                    # (ch,X,Y) --> (B=1,ch,X,Y)
                    obs_x.unsqueeze_(0)
                if obs_z.dim() == 1:
                    # (Z) --> (B=1,Z)
                    obs_z.unsqueeze_(0)

                observation_grounds.append(obs_x)
                observation_latent.append(obs_z)

            observation_grounds = torch.cat(observation_grounds)
            observation_recons = self.decoder(torch.cat(observation_latent))

            reconstruction_loss = F.binary_cross_entropy(observation_recons.squeeze(), observation_grounds.squeeze(),
                                                         reduction="mean")

            return reconstruction_loss

        def sgd_step(reconstruction_loss):
            """
            Save loss, step all optmizers and clear cached representations.
            """
            entanglement_loss = self._calc_entanglement()
            if self.ent_loss_target_type == LossTarget.MIN:
                reg_loss = entanglement_loss
            elif self.ent_loss_target_type == LossTarget.MAX:
                reg_loss = -entanglement_loss
            else:  # self.ent_loss_target_type == LossTarget.VALUE:
                reg_loss = (entanglement_loss - self.ent_loss_target).abs()

            total_loss = reconstruction_loss + self.ent_loss_weight * reg_loss
            self.losses.append(total_loss.item())
            self.reconstruction_losses.append(reconstruction_loss.item())
            self.entanglement_losses.append(entanglement_loss.item())

            self.optimizer_enc.zero_grad()
            self.optimizer_rep.zero_grad()
            self.optimizer_dec.zero_grad()

            total_loss.backward()

            self.optimizer_enc.step()
            self.optimizer_rep.step()
            self.optimizer_dec.step()

            # Rember to clear the cached action representations after we update the parameters!
            self.lat_env.clear_representations()

        def _num2str(num):
            return f"{num:{'.5f' if num > 1e-4 else '.4e'}}"

        self.encoder.train()
        self.decoder.train()

        t_start = time.time()
        num_steps = 0
        while num_steps < num_sgd_steps:
            t0 = time.time()
            reconstruction_loss = run_episode()
            sgd_step(reconstruction_loss)

            num_steps += 1
            self.total_training_steps += 1

            self.ent_loss_weight = self.get_ent_loss_weight(self.total_training_steps)

            t1 = time.time()
            self.total_training_time += (t1 - t0)

            if (num_steps % log_freq == 0):
                log_str = f"iter {self.total_training_steps} : loss={_num2str(self.losses[-1])}"
                if self.ent_loss_weight > 0:
                    for label, val in zip(["recon.", "ent."],
                                          [self.reconstruction_losses[-1], self.entanglement_losses[-1]]):
                        log_str += f" : {label} loss : {_num2str(val)}"
                log_str += f" : last {log_freq} iters in {time.time() - t_start:.3f}s"
                # print(log_str, end="\r" if num_steps%int((log_freq*10)) else "\n")
                print(log_str)
                t_start = time.time()

    def plot_training(self, save=True):
        '''Plot the training curves.

        Plots the (log) of total loss, and underlying reconstruction and entanglement losses.

        Args:
            save: Boolean. Whether to save the figure to file.

        Returns: A matplotlib.figure
        '''
        with plt.style.context('seaborn-paper', after_reset=True):
            fig = plt.figure(figsize=(5, 4), constrained_layout=True)
            gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

            # Set up sliding average window.
            window = min(25, len(self.losses) // 5)
            avg_mask = np.ones(window) / window

            #####
            # Plot total losses
            #####
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(np.convolve(range(len(self.losses)), avg_mask, 'valid'),
                     np.convolve(np.log(self.losses), avg_mask, 'valid'),
                     linewidth=0.75,
                     alpha=0.8)

            ax1.set_xlabel("Num. param. updates")
            ax1.set_ylabel(r"$\log(\mathcal{L})$")

            #####
            # Plot reconstruction losses
            #####
            ax2 = fig.add_subplot(gs[1, 0])

            ax2.plot(np.convolve(range(len(self.reconstruction_losses)), avg_mask, 'valid'),
                     np.convolve(np.log(self.reconstruction_losses), avg_mask, 'valid'),
                     linewidth=0.75,
                     alpha=0.8)

            ax2.set_xlabel("Num. param. updates")
            ax2.set_ylabel(r"$\log(\mathcal{L}_{\mathrm{recon}})$")

            #####
            # Plot entanglement losses
            #####
            ax3 = fig.add_subplot(gs[1, 1])

            ax3.plot(np.convolve(range(len(self.entanglement_losses)), avg_mask, 'valid'),
                     np.convolve(np.log(self.entanglement_losses), avg_mask, 'valid'),
                     linewidth=0.75,
                     alpha=0.8)

            ax3.set_xlabel("Num. param. updates")
            ax3.set_ylabel(r"$\log(\mathcal{L}_{\mathrm{ent}})$")

        if save:
            fig_fname = os.path.join(self.train_folder, "training_curves")
            plt.savefig(fig_fname + ".pdf", bbox_inches='tight')
            plt.savefig(fig_fname + ".png", bbox_inches='tight')
            print("Training curves saved to", fig_fname + "(.pdf/.png)")

        return fig

    def plot_representations(self, save=False, num_highlight=1):
        '''Plot the thetas of the action representations.

        Each action representation is plotted as an individual bar-chart, showing
        the value of each rotation angle mixing two dimensions in latent space.

        The largest 'num_highlight' rotations will have their bars highlighted.

        Args:
            save: Boolean. Whether to save the figure to file.
            num_highlight: Number of rotation angles to highlight per representation
                           plot.

        Returns: A matplotlib.figure
        '''
        # Get the names of all factors which can be acted upon.
        all_factor_names = self.obs_env.dataset.factor_space.factor_names
        all_factor_sizes = self.obs_env.dataset.factor_space._factor_sizes
        actions_factor_names = [f for f, act in zip(all_factor_names, self.obs_env._act_dims_mask) if act]
        actions_factor_sizes = [f for f, act in zip(all_factor_sizes, self.obs_env._act_dims_mask) if act]
        num_factors = len(actions_factor_names)

        # The thetas that parameterise the action representations.
        rep_thetas = self.lat_env.get_representation_params(detach=True)

        # Generate the labels for the latent dimensions that each theta mixes.
        dim = self.lat_env.dim
        theta_dims = []
        for i in range(1, dim + 1):
            for j in range(i + 1, dim + 1):
                theta_dims.append(f"({i},{j})")

        with plt.style.context('seaborn-paper', after_reset=True):

            fig = plt.figure(figsize=(4, 2 * num_factors), constrained_layout=True)
            gs = gridspec.GridSpec(ncols=2, nrows=num_factors, figure=fig)

            width = 0.75

            for idx, thetas in enumerate(rep_thetas):
                row, col = idx // 2, idx % 2
                factor_name = actions_factor_names[idx // 2]
                factor_size = actions_factor_sizes[idx // 2]
                action_name = '+1' if idx % 2 == 0 else '-1'

                title = f"{factor_name} : {action_name}"
                plt_lim = max(1.05 / factor_size, max([max(t) for t in rep_thetas]) / (2 * np.pi))

                ax = fig.add_subplot(gs[row, col])

                x = np.arange(len(thetas))
                bar_list = ax.bar(x - width / 2, thetas / (2 * np.pi), width, label='Rep {}'.format(i))

                theta_arg_sort = np.argsort(np.abs(thetas))
                xticks, xticklabels = [], []
                for arg_idx in theta_arg_sort[-num_highlight:]:
                    bar_list[arg_idx].set_color('r')
                    xticks.append(x[arg_idx] - 0.25)
                    xticklabels.append(theta_dims[arg_idx])

                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)

                if row == num_factors - 1:
                    ax.set_xlabel("$(i,j)$")
                if col == 0:
                    ax.set_ylabel(r"$\theta / 2\pi$")

                ax.set_ylim(-plt_lim, plt_lim)
                ax.set_title(title)

        if save:
            fig_fname = os.path.join(self.test_folder, "thetas")
            plt.savefig(fig_fname + ".pdf", bbox_inches='tight')
            plt.savefig(fig_fname + ".png", bbox_inches='tight')
            print("Representation thetas plots saved to", fig_fname + "(.pdf/.png)")

        return fig

    def save(self, fname='rep_learner'):
        '''Save the state of the DynamicRepresentationLearner to file.

        To fully save the state we need to save:
            history of total loss, reconstruction loss and entanglement losses,
            action representations,
            decoder network parameters,
            encoder  network parameters,
            optimizer states for the enocder, decorder and representations.

        Args:
            fname: File name of the saved state.

        Returns: File name of the saved state
        '''
        if os.path.splitext(fname)[-1] != '.pth':
            fname += '.pth'
        fname = os.path.join(self.save_loc, fname)

        state = {}

        state['total_training_steps'] = self.total_training_steps
        state['total_training_time'] = self.total_training_time
        state['losses'] = self.losses
        state['reconstruction_losses'] = self.reconstruction_losses
        state['entanglement_losses'] = self.entanglement_losses

        state['encoder'] = self.encoder.state_dict()
        state['representations'] = self.lat_env.get_representations()
        state['decoder'] = self.decoder.state_dict()

        state['optimizer_end'] = self.optimizer_enc.state_dict()
        state['optimizer_rep'] = self.optimizer_rep.state_dict()
        state['optimizer_dec'] = self.optimizer_dec.state_dict()

        torch.save(state, fname)

        print("Saved RepresentationLearner state to", fname)

        return fname

    def load(self, path):
        '''Load the state of the DynamicRepresentationLearner from file.

        To fully replicate the state we need to load:
            history of total loss, reconstruction loss and entanglement losses,
            action representations,
            decoder network parameters,
            encoder  network parameters,
            optimizer states for the enocder, decorder and representations.

        Args:
            path: Path to the saved state.
        '''
        state = torch.load(path, map_location=self.device)

        self.total_training_steps = state['total_training_steps']
        self.total_training_time = state['total_training_time']
        self.losses = state['losses']
        self.reconstruction_losses = state['reconstruction_losses']
        self.entanglement_losses = state['entanglement_losses']

        self.encoder.load_state_dict(state['encoder'])
        self.lat_env.set_representations(state['representations'])
        self.decoder.load_state_dict(state['decoder'])

        self.optimizer_enc.load_state_dict(state['optimizer_end'])
        self.optimizer_rep.load_state_dict(state['optimizer_rep'])
        self.optimizer_dec.load_state_dict(state['optimizer_dec'])

        print("Loaded RepresentationLearner state from", path)

    @torch.no_grad()
    def test(self,
             episode_length=None,
             num_episodes=None,
             save_scores=False,  # Save pd.Dataframe of scores from each episode.
             save_plot=False,  # Save plot of scores from each episode.
             show_imgs=False,  # Show ground truth and reconstructed images (from only a single episode).
             save_imgs=False):  # Save ground truth and reconstructed images (from all episodes)
        '''Test the representation.

        To test the representation, we initialise the observation and latent space such that
        they correspond to a know starting state.  We then evolve each independently taking (the same)
        random actions in both environments.  The quality of the representation then corresponds to
        how well observations reconstructed from the latent state correspond to the ground-truth at
        each time-step.

        Args:
            episode_length: How long the test episodes should be (in terms of number of actions per
                            environment).
            num_episodes: How many test episodes to run (results will be averaged).
            save_scores: Boolean.  Whether to save the scores from each episode as a pandas dataframe.
            save_plot: Boolean.  Whether to save a plot of averaged scores as function of time-steps.
            show_imgs: Boolean.  Whether to show ground truth and reconstructed images (from only a
                                 single episode -- typically this is only useful in interactive
                                 environments like Jupyter notebooks).
            save_imgs: Boolean.  Whether to save ground truth and reconstructed images from all episodes.
        '''
        if episode_length is None:
            episode_length = self.episode_length
        if num_episodes is None:
            num_episodes = self.obs_env.batch_size

        def run_episode_batch(episode_length):
            obs_x = self.obs_env.reset()
            obs_z = self.lat_env.reset(self.encoder(obs_x))

            def _append(list, obs):
                if obs.dim() == 3:
                    # (ch,X,Y) --> (B=1,ch,X,Y)
                    obs.unsqueeze_(0)
                list.append(obs)

            def _append_f(list, f):
                if f.ndim == 1:
                    # (ch,X,Y) --> (B=1,ch,X,Y)
                    f = np.expand_dims(f, 0)
                list.append(f)

            # obs_ground = [obs_x]
            # obs_recon = [self.decoder(obs_z)]
            obs_ground, obs_recon, facts = [], [], []
            _append(obs_ground, obs_x)
            _append(obs_recon, self.decoder(obs_z))
            _append_f(facts, self.obs_env.get_factors())

            # facts = [self.obs_env.get_factors()]

            for _ in range(episode_length):
                action = self.obs_env.action_space.sample()
                # obs_ground.append(self.obs_env.step(action))
                # obs_recon.append(self.decoder(self.lat_env.step(action)))
                # facts.append(self.obs_env.get_factors())
                _append(obs_ground, self.obs_env.step(action))
                _append(obs_recon, self.decoder(self.lat_env.step(action)))
                _append_f(facts, self.obs_env.get_factors())

            return obs_ground, obs_recon, facts

        def plot_loss(obs_recon, obs_ground):
            X_LABEL, Y_LABEL, EPS_LABEL = "Step", "Recon. loss", "Eps."

            dfs_loss = []
            for step, (obs_g, obs_r) in enumerate(zip(obs_ground, obs_recon)):
                recon_loss_per_pixel = F.binary_cross_entropy(obs_r, obs_g,
                                                              reduction="none").cpu()
                recon_loss_per_frame = recon_loss_per_pixel.reshape(recon_loss_per_pixel.size(0), -1).mean(1)

                data_step = np.array([np.arange(len(recon_loss_per_frame), dtype=float),
                                      [step] * len(recon_loss_per_frame),
                                      recon_loss_per_frame.numpy()]).T
                dfs_loss.append(pd.DataFrame(data_step, columns=[EPS_LABEL, X_LABEL, Y_LABEL]))

            df_loss = pd.concat(dfs_loss)

            with plt.style.context('seaborn-paper', after_reset=True):
                fig, (ax) = plt.subplots(1, 1, figsize=(4, 2.5))
                sns.lineplot(x=X_LABEL, y=Y_LABEL, data=df_loss, ax=ax)

            return df_loss, fig

        def plot_img(ground_obs, recon_obs, factors=None):
            def factors2string(f):
                f_strings = []
                for factor_name, value, valid in zip(self.obs_env.dataset.factor_space.factor_names,
                                                     f,
                                                     self.obs_env._act_dims_mask):
                    if valid:
                        f_strings.append(f"{factor_name} : {value}")
                return "\n".join(f_strings)

            h = 2 if factors is None else 3
            fig = plt.figure(figsize=(4, h), constrained_layout=True)
            gs = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

            ax_g = fig.add_subplot(gs[0])
            ax_r = fig.add_subplot(gs[1])

            ax_g.imshow(ground_obs)
            ax_r.imshow(recon_obs)
            ax_g.set_title("ground truth", fontsize=12)
            ax_r.set_title("reconstruction", fontsize=12)

            fig.suptitle(factors2string(factors), fontsize=12, va="top")

            for ax in [ax_g, ax_r]:
                ax.set_xticks([])
                ax.set_yticks([])
                for pos in ['bottom', 'top', 'right', 'left']:
                    ax.spines[pos].set_color('0')

            return fig

        self.encoder.eval()
        self.decoder.eval()

        obs_ground, obs_recon, facts = None, None, None

        complete_episode = 0

        while complete_episode < num_episodes:
            obs_ground_batch, obs_recon_batch, facts_batch = run_episode_batch(episode_length)
            complete_episode += self.obs_env.batch_size
            if obs_ground is None:
                obs_ground, obs_recon, facts = obs_ground_batch, obs_recon_batch, facts_batch
            else:
                obs_ground = [torch.cat([all_g, batch_g]) for all_g, batch_g in zip(obs_ground, obs_ground_batch)]
                obs_recon = [torch.cat([all_r, batch_r]) for all_r, batch_r in zip(obs_recon, obs_recon_batch)]
                facts = [np.concatenate([all_f, batch_f]) for all_f, batch_f in zip(facts, facts_batch)]

        df_loss, fig = plot_loss(obs_recon, obs_ground)

        if save_scores:
            scores_fname = os.path.join(self.test_folder, "scores.pkl")
            df_loss.to_pickle(scores_fname)
            print("Test scores saved to", scores_fname)

        if save_plot:
            fig_fname = os.path.join(self.test_folder, "scores")
            plt.savefig(fig_fname + ".pdf", bbox_inches='tight')
            plt.savefig(fig_fname + ".png", bbox_inches='tight')
            print("Test score plot saved to", fig_fname + "(.pdf/.png)")

        if show_imgs or save_imgs:
            # channels last for plotting: (batch, ch, x, y) --> (batch, x, y, ch)
            obs_ground_ch_last = [obs.permute(0, 2, 3, 1).cpu() for obs in obs_ground]
            obs_recon_ch_last = [obs.permute(0, 2, 3, 1).cpu() for obs in obs_recon]

            if save_imgs:
                img_dirs = [os.path.join(self.test_folder, f"images/episode_{i + 1}") for i in range(num_episodes)]
                for d in img_dirs:
                    mk_dir(d, quite=True)

            # For every time step...
            for idx_step, (step_obs_g, step_obs_r, step_factors) in enumerate(
                    zip(obs_ground_ch_last, obs_recon_ch_last, facts)):
                # For every observation pair (episode)...
                for idx_eps, (obs_g, obs_r) in enumerate(zip(step_obs_g, step_obs_r)):
                    fig = plot_img(obs_g, obs_r, step_factors[idx_eps])
                    if not save_imgs:
                        # If we aren't saving everything, just show the first episode and move to the next step.
                        fig.show()
                        break
                    if idx_eps < num_episodes:
                        fig_fname = os.path.join(img_dirs[idx_eps], f"step_{idx_step}")
                        print(f"Saving {fig_fname}(.pdf/.png)...", end="\r")
                        fig.savefig(fig_fname + ".pdf", bbox_inches='tight')
                        fig.savefig(fig_fname + ".png", bbox_inches='tight')
                        plt.close(fig)

            if save_imgs:
                obs_plt_path = os.path.join(self.test_folder, 'images/episode_X/step_X(.pdf/.png)')
                print(f"Observation plots saved to {obs_plt_path}")
