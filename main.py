import copy
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


def run_episode(episode, z, env, meta_agent, logger, params):
    joint_obs, state = env.reset()
    joint_obs = concat_state_latent(joint_obs, z, params["n_agents"], params["n_skills"])
    state = concat_state_latent(state, z, 1, params["n_skills"])
    episode_reward = 0
    logq_zses = []

    for step in range(params["max_episode_len"]):
        joint_action = meta_agent.choose_action(joint_obs)
        next_joint_obs, next_state, reward, done, _ = env.step(joint_action)
        next_joint_obs = concat_state_latent(next_joint_obs, z, params["n_agents"], params["n_skills"])
        next_state = concat_state_latent(next_state, z, 1, params["n_skills"])
        for agent_id in range(params["n_agents"]):
            meta_agent.store(joint_obs[agent_id], z, done, joint_action[agent_id], next_joint_obs[agent_id], state,
                             next_state, reward)
        if step % params["steps_per_train"] == 0:
            losses, skill_reward, logq_zs = meta_agent.train(params["do_diayn"])
            logq_zses += [logq_zs] if logq_zs is not None else [0]
        episode_reward += reward
        joint_obs = next_joint_obs
        state = next_state
        if done:
            break

    logger.log(episode, episode_reward, z, np.mean(logq_zses), step, losses, skill_reward, params["do_diayn"])


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

    np.random.seed(params["seed"])
    logger.on()

    # Pretrain with DIAYN
    print("\n============== Pretraining ==============\n")
    for episode in tqdm(range(params["pretrain_episodes"])):
        z = np.random.choice(params["n_skills"], p=p_z) if params["do_diayn"] else 0
        run_episode(episode, z, env, meta_agent, logger, params)

    # Evaluation
    print("\n============== Evaluating ==============\n")
    if params["do_diayn"]:
        player = Play(env, copy.deepcopy(meta_agent), n_skills=params["n_skills"], log_dir=logger.log_dir)
        episode_rewards = []
        for z in range(params["n_skills"]):
            r = player.evaluate(z)
            episode_rewards.append(r)
    else:
        episode_rewards = [0]

    # Finetuning
    print("\n============== Finetuning ==============\n")
    params["do_diayn"] = False
    meta_agent.reset_buffer()
    z = np.argmax(episode_rewards)
    for episode in tqdm(range(params["pretrain_episodes"], params["pretrain_episodes"] + params["finetune_episodes"])):
        run_episode(episode, z, env, meta_agent, logger, params)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("User interrupt, aborting")
        if get_params()["wandb"]:
            wandb.finish()
