# from mujoco_py.generated import const
from mujoco_py import GlfwContext
import cv2
import numpy as np
import os


# GlfwContext(offscreen=True)

class Play:
    def __init__(self, env, agent, n_skills):
        self.env = env
        self.agent = agent
        self.n_agents = self.env.n_agents
        self.n_skills = n_skills
        self.max_episode_len = int(1e+6)
        self.agent.set_policy_net_to_cpu_mode()
        self.agent.set_policy_net_to_eval_mode()
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        os.makedirs("Video", exist_ok=True)

    @staticmethod
    def concat_state_latent(s, z_, n_a, n_z):
        z_one_hot = np.zeros((n_a, n_z))
        z_one_hot[:, z_] = 1
        return np.concatenate([s, z_one_hot], axis=-1)

    def play_episode(self, z):
        video_writer = cv2.VideoWriter(f"Video/skill_{z}" + ".avi", self.fourcc, 50.0, (250, 250))
        joint_obs, state = self.env.reset()
        joint_obs = self.concat_state_latent(joint_obs, z, self.n_agents, self.n_skills)
        episode_reward = 0
        for step in range(self.max_episode_len):
            joint_action = self.agent.choose_action(joint_obs)
            next_joint_obs, _, reward, done, _ = self.env.step(joint_action)
            next_joint_obs = self.concat_state_latent(next_joint_obs, z, self.n_agents, self.n_skills)
            episode_reward += reward
            joint_obs = next_joint_obs
            I = self.env.render()
            I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
            I = cv2.putText(I, f"Skill {z} Reward: {int(episode_reward)}", org=(20, 50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            I = cv2.resize(I, (250, 250))
            video_writer.write(I)
            if done:
                break
        print(f"skill: {z}, episode reward:{episode_reward:.1f}, episode length:{step + 1}")
        video_writer.release()
        return episode_reward

    def evaluate(self, skill=None):
        ep_r = []
        if skill is None:
            print("Evaluating all skills...")
            for z in range(self.n_skills):
                r = self.play_episode(z)
                ep_r.append(r)
        else:
            print(f"Evaluating skill {skill}...")
            for _ in range(1):
                r = self.play_episode(skill)
                ep_r.append(r)

        self.env.close()
        cv2.destroyAllWindows()

        print(f"Average episode reward: {np.mean(ep_r):.1f}")
        print(f"Std of episode reward: {np.std(ep_r):.1f}")
        print(f"Total number: {len(ep_r)}")
