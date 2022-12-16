import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import smooth


def plot_loss(losses):
    plt.plot(losses)
    plt.ylabel("Loss")
    plt.xlabel("Updates")
    plt.show()


def plot_reward(train_reward, eval_reward, eval_episodes):
    plt.plot(smooth(train_reward, 5), label="train")
    # smooth_eval_rewards = smooth(eval_reward, 2)
    # plt.plot(eval_episodes[:len(smooth_eval_rewards)], smooth_eval_rewards, label="eval", marker=".")
    plt.plot(eval_episodes, eval_reward, label="eval", marker=".")
    plt.ylabel("Reward")
    plt.xlabel("Episodes")
    plt.legend()
    plt.show()

