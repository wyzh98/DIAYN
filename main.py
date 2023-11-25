import gym
from Brain import SACAgent
from Common import Play, Logger, get_params
import numpy as np
from tqdm import tqdm
import mujoco_py
import wandb
from mamujoco_env import MAMujocoEnv


def concat_state_latent(s, z_, n_a, n_z):
    z_one_hot = np.zeros((n_a, n_z))
    z_one_hot[:, z_] = 1
    return np.concatenate([s, z_one_hot], axis=-1)

def main():
    params = get_params()
    env = MAMujocoEnv(params["env_name"])
    n_agents = env.n_agents
    params.update({"n_agents": n_agents,
                   "n_states": env.state_spec[0],
                   'n_obs': env.observation_spec[-1],
                   "n_actions": env.action_spec[0],
                   "action_bounds": env.action_bounds})
    print("params:", params)

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    meta_agent = SACAgent(p_z=p_z, **params)
    logger = Logger(meta_agent, **params)

    if params["do_train"]:
        min_episode = 0
        last_logq_zs = 0
        if params["fix_skill"] is not None:
            assert params["do_diayn"] is False
            episode, last_logq_zs = logger.load_weights(policy_only=True)
            print("Policy initialized with skill: ", params["fix_skill"])
        else:
            np.random.seed(params["seed"])
            print("Training from scratch.")

        logger.on()
        for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
            z = np.random.choice(params["n_skills"], p=p_z) if params["fix_skill"] is None else params["fix_skill"]
            joint_obs, state = env.reset()
            joint_obs = concat_state_latent(joint_obs, z, n_agents, params["n_skills"])
            state = concat_state_latent(state, z, 1, params["n_skills"])
            episode_reward = 0
            logq_zses = []

            for step in range(1, 1 + params["max_episode_len"]):
                joint_action = meta_agent.choose_action(joint_obs)
                next_joint_obs, next_state, reward, done, _ = env.step(joint_action)
                next_joint_obs = concat_state_latent(next_joint_obs, z, n_agents, params["n_skills"])
                next_state = concat_state_latent(next_state, z, 1, params["n_skills"])
                for agent_id in range(n_agents):
                    meta_agent.store(joint_obs[agent_id], z, done, joint_action[agent_id], next_joint_obs[agent_id], state, next_state, reward)
                if step % params["steps_per_train"] == 0:
                    losses, skill_reward, logq_zs = meta_agent.train()
                    logq_zses += [logq_zs] if logq_zs is not None else [last_logq_zs]
                episode_reward += reward
                joint_obs = next_joint_obs
                state = next_state
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
        player = Play(env, meta_agent, n_skills=params["n_skills"])
        player.evaluate()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("User interrupt, aborting")
        if get_params()["wandb"]:
            wandb.finish()
