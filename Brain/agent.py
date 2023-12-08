import numpy as np
from .model import PolicyNetwork, QvalueNetwork, ValueNetwork, Discriminator
import torch
from .replay_memory import Memory, Transition
from torch.optim.adam import Adam
from torch.nn.functional import log_softmax


class SACAgent:
    def __init__(self, p_z, **config):
        self.config = config
        self.n_obs = self.config["n_obs"]
        self.n_states = self.config["n_states"]
        self.n_skills = self.config["n_skills"]
        self.batch_size = self.config["batch_size"]
        self.p_z = np.tile(p_z, self.batch_size).reshape(self.batch_size, self.n_skills)
        self.memory = Memory(self.config["mem_size"], self.config["seed"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device on:", self.device)

        torch.manual_seed(self.config["seed"])
        self.policy_network = PolicyNetwork(n_obs=self.n_obs + self.n_skills, n_actions=self.config["n_actions"],
                                            action_bounds=self.config["action_bounds"],
                                            n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.q_value_network1 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.config["n_actions"],
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.q_value_network2 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.config["n_actions"],
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.value_network = ValueNetwork(n_states=self.n_states + self.n_skills,
                                          n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.value_target_network = ValueNetwork(n_states=self.n_states + self.n_skills,
                                                 n_hidden_filters=self.config["n_hiddens"]).to(self.device)
        self.hard_update_target_network()

        self.discriminator = Discriminator(n_states=self.n_states, n_skills=self.n_skills,
                                           n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.mse_loss = torch.nn.MSELoss()
        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=self.config["lr"])
        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.config["lr"])
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.config["lr"])
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.config["lr"])
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=self.config["lr"])

    def choose_action(self, joint_obs_skill):
        joint_action = []
        for obs_skill in joint_obs_skill:
            obs_skill = np.expand_dims(obs_skill, axis=0)
            obs_skill = torch.from_numpy(obs_skill).float().to(self.device)
            action, _, _ = self.policy_network.sample_or_likelihood(obs_skill)
            joint_action.append(action.detach().cpu().numpy()[0])
        return joint_action

    def store(self, obs, z, done, action, next_obs, state, next_state, reward):
        obs = torch.from_numpy(obs).reshape(1, -1).float().to("cpu")
        state = torch.from_numpy(state).reshape(1, -1).float().to("cpu")
        z = torch.ByteTensor([z]).to("cpu")
        done = torch.BoolTensor([done]).to("cpu")
        action = torch.Tensor(np.array(action)).reshape(1, -1).to("cpu")
        next_obs = torch.from_numpy(next_obs).reshape(1, -1).float().to("cpu")
        next_state = torch.from_numpy(next_state).reshape(1, -1).float().to("cpu")
        reward = torch.Tensor([reward]).to("cpu")
        self.memory.add(state, z, done, action, next_state, obs, next_obs, reward)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        obs = torch.cat(batch.obs).view(self.batch_size, self.n_obs + self.n_skills).to(self.device)
        states = torch.cat(batch.state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)
        zs = torch.cat(batch.z).view(self.batch_size, 1).long().to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
        actions = torch.cat(batch.action).view(-1, self.config["n_actions"]).to(self.device)
        next_obs = torch.cat(batch.next_obs).view(self.batch_size, self.n_obs + self.n_skills).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)
        reward = torch.cat(batch.reward).view(self.batch_size, 1).to(self.device)

        return states, zs, dones, actions, next_states, obs, next_obs, reward

    def train(self, do_diayn):
        if len(self.memory) < self.batch_size:
            return None, None, None

        self.config["do_diayn"] = do_diayn
        batch = self.memory.sample(self.batch_size)
        states, zs, dones, actions, next_states, obs, next_obs, env_reward = self.unpack(batch)
        p_z = torch.from_numpy(self.p_z).to(self.device)

        # Calculating the value target
        reparam_actions, log_probs, dist_entropy = self.policy_network.sample_or_likelihood(obs)
        q1 = self.q_value_network1(states, reparam_actions)
        q2 = self.q_value_network2(states, reparam_actions)
        q = torch.min(q1, q2)
        target_value = q.detach() - self.config["alpha"] * log_probs.detach()

        value = self.value_network(states)
        value_loss = self.mse_loss(value, target_value)

        logits = self.discriminator(torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0])
        p_z = p_z.gather(-1, zs)  # b, z -> b, 1, zs: skill index
        logq_z_ns = log_softmax(logits, dim=-1)
        skill_reward = logq_z_ns.gather(-1, zs).detach() - torch.log(p_z + 1e-6)  # detached

        # Calculating the Q-Value target
        reward = skill_reward if self.config["do_diayn"] else env_reward
        with torch.no_grad():
            next_value = self.value_target_network(next_states)
            target_q = self.config["reward_scale"] * reward.float() + self.config["gamma"] * next_value * (~dones)
        q1 = self.q_value_network1(states, actions)
        q2 = self.q_value_network2(states, actions)
        q1_loss = self.mse_loss(q1, target_q)
        q2_loss = self.mse_loss(q2, target_q)

        policy_loss = (self.config["alpha"] * log_probs - q).mean()
        logits = self.discriminator(torch.split(states, [self.n_states, self.n_skills], dim=-1)[0])
        discriminator_loss = self.cross_ent_loss(logits, zs.squeeze(-1))

        self.policy_opt.zero_grad()
        policy_loss.backward()
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=100)
        self.policy_opt.step()

        self.value_opt.zero_grad()
        value_loss.backward()
        value_grad_norm = torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=2000)
        self.value_opt.step()

        self.q_value1_opt.zero_grad()
        q1_loss.backward()
        q_grad_norm = torch.nn.utils.clip_grad_norm_(self.q_value_network1.parameters(), max_norm=2000)
        self.q_value1_opt.step()

        self.q_value2_opt.zero_grad()
        q2_loss.backward()
        q_grad_norm = torch.nn.utils.clip_grad_norm_(self.q_value_network2.parameters(), max_norm=2000)
        self.q_value2_opt.step()

        if self.config["do_diayn"]:
            self.discriminator_opt.zero_grad()
            discriminator_loss.backward()
            discriminator_grad_norm = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=10)
            self.discriminator_opt.step()
        else:
            discriminator_loss = torch.zeros_like(discriminator_loss)
            discriminator_grad_norm = torch.zeros_like(discriminator_loss)
            logq_z_ns = torch.zeros_like(logq_z_ns)
            skill_reward = torch.zeros_like(skill_reward)

        self.soft_update_target_network(self.value_network, self.value_target_network)

        losses = {'policy_loss': policy_loss.item(),
                  'value_loss': value_loss.item(),
                  'q_loss': q2_loss.item(),
                  'discriminator_loss': discriminator_loss.item(),
                  'policy_grad_norm': policy_grad_norm.item(),
                  'value_grad_norm': value_grad_norm.item(),
                  'q_grad_norm': q_grad_norm.item(),
                  'discriminator_grad_norm': discriminator_grad_norm.item(),
                  'entropy': dist_entropy.mean().item(),
                  }

        return losses, skill_reward.mean().item(), logq_z_ns.mean().item()

    def reset_buffer(self):
        self.memory.clean()

    def soft_update_target_network(self, local_network, target_network):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(self.config["tau"] * local_param.data + (1 - self.config["tau"]) * target_param.data)

    def hard_update_target_network(self):
        self.value_target_network.load_state_dict(self.value_network.state_dict())
        self.value_target_network.eval()

    def set_policy_net_to_eval_mode(self):
        self.policy_network.eval()

    def set_policy_net_to_cpu_mode(self):
        self.device = torch.device("cpu")
        self.policy_network.to(self.device)
