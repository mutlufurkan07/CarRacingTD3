import numpy as np
import torch
import argparse
import os

from env_binder_lidar import car_racer
import utils
import TD3
from tqdm import tqdm
import gym

from gym.envs.box2d import car_racing


# Runs policy for X episodes and returns average reward
def eval_policy(policy_,eval_env, eval_episodes=5):
    avg_reward = 0.0
    for _ in range(eval_episodes):
        s_, d_ = eval_env.reset(), False
        while not d_:
            a_ = policy_.select_action(np.array(s_), test=True)
            print(f"action: {a_}")
            s_, r_, d_, _ = eval_env.step(a_)
            avg_reward += r_

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="TD3")  # Algorithm nameu
    parser.add_argument("--env", default="CarRacing-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=32, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=5e7, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.15)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--alpha", default=0.4)  # Priority = TD^alpha (only used by LAP/PAL)
    parser.add_argument("--min_priority", default=1,
                        type=int)  # Minimum priority (set to 1 in paper, only used by LAP/PAL)
    args = parser.parse_args()

    file_name = "%s_%s_%s" % (args.algorithm, args.env, str(args.seed))
    print("---------------------------------------")
    print(f"Settings: {file_name}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    env = car_racer(num_of_lidar_points=7, render_opencv=False)
    eval_env = car_racer(num_of_lidar_points=7, render_opencv=True)
    eval_env.env.seed(12 + 100)
    max_episode_step = 1000

    # Set seeds
    env.env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = 7 + 6
    action_dim = 2
    max_action = 1

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq
    }

    policy = TD3.TD3(**kwargs)
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    # evaluations = [eval_policy(policy, eval_env)]
    evaluations = []

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in tqdm(range(int(args.max_timesteps))):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.sample_random_action()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < max_episode_step else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:  # >=
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t + 1} Episode Num: {episode_num + 1} "
                  f"Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0 and t > args.start_timesteps:
            evaluations.append(eval_policy(policy, eval_env))
            np.save("./results/%s" % file_name, evaluations)

        if t % 100000 == 0:
            policy.save(f"checkpoint_last")

