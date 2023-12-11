import numpy as np
from .model import PolicyNetwork, QvalueNetwork, ValueNetwork, Discriminator
import torch
from .replay_memory import Memory, Transition
from torch.optim.adam import Adam
from torch.nn.functional import log_softmax


class SACAgent:
    def __init__(self, p_z, **config):
        self.config = config
        self.n_agents = self.config["n_agents"]
        self.n_obs = self.config["n_obs"]
        self.n_states = self.config["n_states"]
        self.n_skills = self.config["n_skills"]
        self.batch_size = self.config["batch_size"]
        self.p_z = np.tile(p_z, self.batch_size).reshape(self.batch_size, self.n_skills)
        self.memory = Memory(self.config["mem_size"], self.n_agents, self.config["seed"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device on:", self.device)
        assert len(set(self.config["n_actions"])) == 1, "You need to seperate Q network."

        torch.manual_seed(self.config["seed"])
        self.policy_networks = [PolicyNetwork(n_obs=self.n_obs[a] + self.n_skills, n_actions=self.config["n_actions"][a],
                                              action_bounds=self.config["action_bounds"][a],
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device)
                                for a in range(self.n_agents)]

        self.q_value_network1 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.config["n_actions"][0],
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.q_value_network2 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.config["n_actions"][0],
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
        self.policy_opts = [Adam(p_net.parameters(), lr=self.config["lr"]) for p_net in self.policy_networks]
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=self.config["lr"])

    def choose_action(self, joint_obs_skill):
        joint_action = []
        for i, obs_skill in enumerate(joint_obs_skill):
            obs_skill = np.expand_dims(obs_skill, axis=0)
            obs_skill = torch.from_numpy(obs_skill).float().to(self.device)
            action, _, _ = self.policy_networks[i].sample_or_likelihood(obs_skill)
            joint_action.append(action.detach().cpu().numpy()[0])
        return joint_action

    def store(self, obs, z, done, action, next_obs, state, next_state, reward, agent_id):
        obs = torch.from_numpy(obs).reshape(1, -1).float().to("cpu")
        state = torch.from_numpy(state).reshape(1, -1).float().to("cpu")
        z = torch.ByteTensor([z]).to("cpu")
        done = torch.BoolTensor([done]).to("cpu")
        action = torch.Tensor(np.array(action)).reshape(1, -1).to("cpu")
        next_obs = torch.from_numpy(next_obs).reshape(1, -1).float().to("cpu")
        next_state = torch.from_numpy(next_state).reshape(1, -1).float().to("cpu")
        reward = torch.Tensor([reward]).to("cpu")
        self.memory.add(agent_id, state, z, done, action, next_state, obs, next_obs, reward)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        obs = torch.cat(batch.obs).view(self.batch_size, -1).to(self.device)  # b, o+z
        states = torch.cat(batch.state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)
        zs = torch.cat(batch.z).view(self.batch_size, 1).long().to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
        actions = torch.cat(batch.action).view(-1, self.config["n_actions"][0]).to(self.device)
        next_obs = torch.cat(batch.next_obs).view(self.batch_size, -1).to(self.device)  # b, o+z
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)
        reward = torch.cat(batch.reward).view(self.batch_size, 1).to(self.device)

        return states, zs, dones, actions, next_states, obs, next_obs, reward

    def train(self, do_diayn):
        if self.memory.min_mem_len < self.batch_size:
            return None, None, None

        self.config["do_diayn"] = do_diayn
        pl, vl, ql, dl, pg, vg, qg, dg, et, sr, logqzns = [], [], [], [], [], [], [], [], [], [], []

        for agent_id in range(self.n_agents):
            batch = self.memory.sample(self.batch_size, agent_id)
            states, zs, dones, actions, next_states, obs, next_obs, env_reward = self.unpack(batch)
            p_z = torch.from_numpy(self.p_z).to(self.device)

            # Calculating the value target
            reparam_actions, log_probs, dist_entropy = self.policy_networks[agent_id].sample_or_likelihood(obs)
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

            self.policy_opts[agent_id].zero_grad()
            policy_loss.backward()
            policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_networks[agent_id].parameters(), max_norm=100)
            self.policy_opts[agent_id].step()

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

            pl += [policy_loss.item()]
            vl += [value_loss.item()]
            ql += [q1_loss.item()]
            dl += [discriminator_loss.item()]
            pg += [policy_grad_norm.item()]
            vg += [value_grad_norm.item()]
            qg += [q_grad_norm.item()]
            dg += [discriminator_grad_norm.item()]
            et += [dist_entropy.mean().item()]
            sr += [skill_reward.mean().item()]
            logqzns += [logq_z_ns.mean().item()]

        losses = {'policy_loss': np.mean(pl),
                  'value_loss': np.mean(vl),
                  'q_loss': np.mean(ql),
                  'discriminator_loss': np.mean(dl),
                  'policy_grad_norm': np.mean(pg),
                  'value_grad_norm': np.mean(vg),
                  'q_grad_norm': np.mean(qg),
                  'discriminator_grad_norm': np.mean(dg),
                  'entropy': np.mean(et),
                  }
        skill_rewards = np.mean(sr)
        logqzns = np.mean(logqzns)

        return losses, skill_rewards, logqzns

    def reset_buffer(self):
        self.memory.clean()

    def soft_update_target_network(self, local_network, target_network):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(self.config["tau"] * local_param.data + (1 - self.config["tau"]) * target_param.data)

    def hard_update_target_network(self):
        self.value_target_network.load_state_dict(self.value_network.state_dict())
        self.value_target_network.eval()

    def set_policy_net_to_eval_mode(self):
        for a in range(self.n_agents):
            self.policy_networks[a].eval()

    def set_policy_net_to_train_mode(self):
        for a in range(self.n_agents):
            self.policy_networks[a].train()

    def set_policy_net_to_cpu_mode(self):
        device = torch.device("cpu")
        for a in range(self.n_agents):
            self.policy_networks[a].to(device)

    def set_policy_net_to_device_mode(self):
        for a in range(self.n_agents):
            self.policy_networks[a].to(self.device)
