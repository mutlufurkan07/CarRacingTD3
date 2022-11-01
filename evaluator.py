import time

import numpy as np
import torch
from TD3 import Actor
from env_binder_lidar import car_racer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_action(policy_network, state):
    state = torch.FloatTensor(state.reshape(1, -1)).to(device)
    return policy_network(state).cpu().data.numpy().flatten()

if __name__ == "__main__":

    eval_env = car_racer(num_of_lidar_points=7, render_opencv=True)

    policy = Actor(state_dim=13, action_dim=2, max_action=1).to(device=device)
    policy.load_state_dict(torch.load("results/checkpoint_best_actor"))
    # policy.load_state_dict(torch.load("results/checkpoint_last_actor"))

    eval_episodes = 40
    avg_reward = 0.0
    for _ in range(eval_episodes):
        episode_reward = 0.0
        s_, d_ = eval_env.reset(), False
        while not d_:
            a_ = select_action(policy_network=policy, state=np.array(s_))
            s_, r_, d_, _ = eval_env.step(a_, test=True)
            eval_env.env.render()
            print(f"\rsteering:{a_[0]:5.2f}, gas: {a_[1]:5.2f}, reward: {r_:5.2f}", end="")
            episode_reward += r_
            time.sleep(0.005)
        print(f"episode reward = {episode_reward}")
        avg_reward += episode_reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")








