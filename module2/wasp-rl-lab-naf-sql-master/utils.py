import numpy as np
import torch
import torch.autograd as autograd
import copy


def transform_action(action, env):
    """
    Gym action shape is different for envs with action in R^1 and R^(n > 1)
    """
    if env.action_space.shape[0] == 1:
        # return action.detach().cpu().squeeze(0).numpy()
        return action.detach().cpu().numpy()
    else:
        return action.detach().cpu().squeeze().numpy()



def rbf_kernel2(input_1, input_2, h_min=1e-3):
    k_fix, out_dim1 = input_1.size()[-2:]
    k_upd, out_dim2 = input_2.size()[-2:]
    assert out_dim1 == out_dim2

    # Compute the pairwise distances of left and right particles.
    diff = input_1.unsqueeze(-2) - input_2.unsqueeze(-3)
    dist_sq = diff.pow(2).sum(-1)
    dist_sq = dist_sq.unsqueeze(-1)

    # Get median.
    median_sq = torch.median(dist_sq, dim=1)[0]
    median_sq = median_sq.unsqueeze(1)

    h = median_sq / np.log(k_fix + 1.) + .001

    kappa = torch.exp(-dist_sq / h)

    # Construct the gradient
    kappa_grad = -2. * diff / h * kappa
    return kappa, kappa_grad


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.3):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state


def presample_env(env, buffer, steps=1000):
    """
    Samples random transitions from the environment and adds those to replay buffer.
    This has a stabilizing effect on learning due to more heterogenous buffer at the beginning.
    """
    counter = 0
    while counter < steps:
        obs, info = env.reset()
        done, trunc = False, False
        while not (done or trunc):
            action = env.action_space.sample()
            new_obs, reward, done, trunc, info = env.step(action)
            buffer.add(obs, action, reward, new_obs, done)

            obs = new_obs

            counter += 1

            if counter == steps:
                break

    print("Presampling done")


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth


def obs_transform(state, n_particles=1):
    if state.shape[0] == 1:
        # environment interaction
        inp = state.repeat(n_particles, 1)
    else:
        # batch update
        inp = state.repeat(n_particles, 1, 1).movedim(1, 0)

    return inp

