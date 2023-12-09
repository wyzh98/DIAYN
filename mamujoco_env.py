import gymnasium_robotics
import numpy as np


class MAMujocoEnv:
    def __init__(self, name):
        if name == 'HalfCheetah':
            self.name = name
            self.env = gymnasium_robotics.mamujoco_v0.parallel_env("HalfCheetah", "6x1", render_mode="rgb_array")
        elif name == 'Walker2d':
            self.name = name
            self.env = gymnasium_robotics.mamujoco_v0.parallel_env("Walker2d", "2x3", render_mode="rgb_array")
        else:
            raise NotImplementedError

        self.agents = self.env.agents
        self.n_agents = len(self.env.agents)
        self.observation_spec = [self.env.observation_space(a).shape[0] for a in self.agents]
        state = self.env.state()
        self.state_spec = state.shape[0]
        self.action_spec = [self.env.action_space(a).shape[0] for a in self.agents]
        self.action_bounds = [(self.env.action_space(a).low[0], self.env.action_space(a).high[0]) for a in self.agents]

    def reset(self):
        joint_obs, _ = self.env.reset()
        joint_obs = [joint_obs[x] for x in self.agents]
        state = self.get_current_state()
        return joint_obs, state

    def step(self, joint_action):
        joint_action_set = dict()
        for agent, action in zip(self.agents, joint_action):
            joint_action_set[agent] = action
        joint_obs, reward, terminated, truncated, info = self.env.step(joint_action_set)
        # if np.any([*truncated.values()]): print("Warning: environment truncated", truncated.values())
        assert len(set(reward.values())) == 1, "Reward should be the same for all agents"
        joint_obs = [joint_obs[x] for x in self.agents]
        reward = reward[self.agents[0]]
        terminated = terminated[self.agents[0]]
        truncated = truncated[self.agents[0]]
        done = terminated or truncated
        next_state = self.get_current_state()
        return joint_obs, next_state, reward, done, info

    def get_current_state(self):
        return self.env.state().reshape(1, -1).astype(np.float32)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


if __name__ == '__main__':
    env = MAMujocoEnv('HalfCheetah')
    env.reset()
