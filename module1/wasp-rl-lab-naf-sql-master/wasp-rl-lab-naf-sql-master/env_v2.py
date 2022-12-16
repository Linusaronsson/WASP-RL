from gym.utils import EzPickle
from gym import spaces
from gym import Env

import numpy as np
import matplotlib.pyplot as plt


class MultiGoalEnv(Env, EzPickle):
    """
    Move a 2D point mass to one of the goal positions. Cost is the distance to
    the closest goal.

    State: position.
    Action: velocity.
    """
    def __init__(self,
                 goal_reward=10,
                 actuation_cost_coeff=30.0,
                 distance_cost_coeff=1.0,
                 init_sigma=0.1,
                 render_mode=None
                 ):
        EzPickle.__init__(**locals())

        self.dynamics = PointDynamics(dim=2, sigma=0)
        self.init_mu = np.zeros(2, dtype=np.float32)
        self.init_sigma = init_sigma
        self.goal_positions = np.array(
            (
                (5, 0),
                (-5, 0),
                (0, 5),
                (0, -5)
            ),
            dtype=np.float32)
        self.goal_threshold = 0.05
        self.goal_reward = goal_reward
        self.action_cost_coeff = actuation_cost_coeff
        self.distance_cost_coeff = distance_cost_coeff
        # making the env larger makes it so that SQL does not overshoot, since it can learn that the area behind
        # the goal is also bad...
        self.xlim = (-10, 10)
        self.ylim = (-10, 10)
        self.vel_bound = 1.
        self.reset()
        self.observation = None

        self._ax = None
        self._env_lines = []
        self.fixed_plots = None
        self.dynamic_plots = []

        self.step_counter = 0

        self.render_mode = render_mode
        if render_mode == "human":
            self.render_fig, self.render_ax = plt.subplots()
            self.render_fig.show()

    def reset(self, init_state=None, **kwargs):

        self.step_counter = 0
        if init_state:
            unclipped_observation = init_state
        else: 
            unclipped_observation = (self.init_mu + self.init_sigma * np.random.normal(size=self.dynamics.s_dim))

        self.observation = np.clip(
            unclipped_observation,
            self.observation_space.low,
            self.observation_space.high)
        return self.observation, "info"

    @property
    def observation_space(self):
        return spaces.Box(
            low=np.array((self.xlim[0], self.ylim[0])),
            high=np.array((self.xlim[1], self.ylim[1])),
            dtype=np.float32,
            shape=None)

    @property
    def action_space(self):
        return spaces.Box(
            low=-self.vel_bound,
            high=self.vel_bound,
            shape=(self.dynamics.a_dim, ),
            dtype=np.float32)

    def get_current_obs(self):
        return np.copy(self.observation)

    def step(self, action):
        action = action.ravel()

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high).ravel()

        observation = self.dynamics.forward(self.observation, action)
        observation = np.clip(
            observation,
            self.observation_space.low,
            self.observation_space.high)

        reward = self.compute_reward(observation, action)
        dist_to_goal = np.amin([
            np.linalg.norm(observation - goal_position)
            for goal_position in self.goal_positions
        ])
        done = dist_to_goal < self.goal_threshold
        if done:
            reward += self.goal_reward

        self.observation = np.copy(observation)
        trunc = self.step_counter >= 100

        self.step_counter += 1

        return observation, reward, done, trunc, {'pos': observation}

    def render(self, mode='human', *args, **kwargs):
        self.render_fig.gca().cla()

        for tup in self.goal_positions:
            self.render_ax.scatter(tup[0], tup[1], marker="*", s=400, c="gold")

        self.render_ax.scatter(self.observation[0], self.observation[1], s=400, c="red")

        self.render_ax.set_xlim(self.xlim[0], self.xlim[1])
        self.render_ax.set_ylim(self.ylim[0], self.ylim[1])

        self.render_fig.canvas.draw()
        self.render_fig.canvas.flush_events()

    def compute_reward(self, observation, action):
        # penalize the L2 norm of acceleration
        action_cost = np.sum(action ** 2) * self.action_cost_coeff

        # penalize squared dist to goal
        cur_position = observation
        goal_cost = self.distance_cost_coeff * np.amin([
            np.sum((cur_position - goal_position) ** 2)
            for goal_position in self.goal_positions
        ])

        # penalize staying with the log barriers
        costs = [action_cost, goal_cost]
        reward = -np.sum(costs)
        return reward

    def plot_reward_landscape(self):
        f = plt.figure(figsize=(8, 8))

        x_min, x_max = tuple(np.array(self.xlim))
        y_min, y_max = tuple(np.array(self.ylim))

        img = np.zeros((100, 100))
        for x_idx, x in enumerate(np.linspace(x_min, x_max, 100)):
            for y_idx, y in enumerate(np.linspace(y_min, y_max, 100)):

                r = self.compute_reward(np.array([x, y]), 0)
                img[x_idx, y_idx] = r

        # plot the goals
        plt.scatter(50, 15, c="r", s=100, marker="*")
        plt.scatter(50, 85, c="r", s=100, marker="*")
        plt.scatter(15, 50, c="r", s=100, marker="*")
        plt.scatter(85, 50, c="r", s=100, marker="*")

        plt.imshow(img)
        cbar = plt.colorbar()
        cbar.set_label("Reward")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()


class PointDynamics(object):
    """
    State: position.
    Action: velocity.
    """
    def __init__(self, dim, sigma):
        self.dim = dim
        self.sigma = sigma
        self.s_dim = dim
        self.a_dim = dim

    def forward(self, state, action):
        mu_next = state + action
        state_next = mu_next + self.sigma * \
            np.random.normal(size=self.s_dim)
        return state_next
    
    
if __name__ == "__main__":
    env = MultiGoalEnv()
    # env.plot_position_cost()
    env.plot_reward_landscape()



