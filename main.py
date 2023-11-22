import gym
from Brain import SACAgent
from Common import Play, Logger, get_params
import numpy as np
from tqdm import tqdm
import mujoco_py
import wandb


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])

def main():
    params = get_params()

    test_env = gym.make(params["env_name"])
    n_states = test_env.observation_space.shape[0]
    n_actions = test_env.action_space.shape[0]
    action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]

    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds})
    print("params:", params)
    test_env.close()
    del test_env, n_states, n_actions, action_bounds

    env = gym.make(params["env_name"])

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = SACAgent(p_z=p_z, **params)
    logger = Logger(agent, **params)

    if params["do_train"]:
        if not params["train_from_scratch"]:
            episode, last_logq_zs = logger.load_weights()
            agent.hard_update_target_network()
            min_episode = episode
            print("Keep training from previous run.")
        else:
            min_episode = 0
            last_logq_zs = 0
            np.random.seed(params["seed"])
            env.observation_space.seed(params["seed"])
            env.action_space.seed(params["seed"])
            print("Training from scratch.")

        logger.on()
        for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
            z = np.random.choice(params["n_skills"], p=p_z)
            state = env.reset()[0]
            state = concat_state_latent(state, z, params["n_skills"])
            episode_reward = 0
            logq_zses = []

            max_n_steps = min(params["max_episode_len"], env.spec.max_episode_steps)
            for step in range(1, 1 + max_n_steps):
                action = agent.choose_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                next_state = concat_state_latent(next_state, z, params["n_skills"])
                agent.store(state, z, done, action, next_state, reward)
                losses, skill_reward, logq_zs = agent.train()
                logq_zses += [logq_zs] if logq_zs is not None else [last_logq_zs]
                episode_reward += reward
                state = next_state
                if truncated: assert ValueError
                if done: break

            logger.log(episode,
                       episode_reward,
                       z,
                       sum(logq_zses) / len(logq_zses),
                       step,
                       losses,
                       skill_reward,
                       )

    else:
        logger.load_weights()
        player = Play(env, agent, n_skills=params["n_skills"])
        player.evaluate()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("User interrupt, aborting")
        if get_params()["wandb"]:
            wandb.finish()
