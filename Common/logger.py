import time
import numpy as np
import psutil
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import datetime
import wandb


class Logger:
    def __init__(self, agent, **config):
        self.config = config
        self.agent = agent
        self.log_dir = self.config["env_name"] + "/" + self.config["run_name"] + datetime.datetime.now().strftime("_%Y-%m-%d-%H-%M-%S")
        self.log_writer = SummaryWriter("Logs/" + self.log_dir)
        self.start_time = 0
        self.duration = 0
        self.max_episode_reward = -np.inf
        self.device = agent.device
        self._turn_on = False
        self.to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024

        if self.config["wandb"]:
            wandb.init(project="SkillDiscovery", name=self.config["run_name"] + "_" + self.config["env_name"], entity="ezo", config=self.config,
                       notes="", id=None, resume="allow")

        self._create_wights_folder(self.log_dir)
        self._log_params()

    @staticmethod
    def _create_wights_folder(dir):
        os.makedirs("Checkpoints/" + dir, exist_ok=True)

    def _log_params(self):
        for k, v in self.config.items():
            self.log_writer.add_text(k, str(v))

    def on(self):
        self.start_time = time.time()
        self._turn_on = True

    def _off(self):
        self.duration = time.time() - self.start_time

    def log(self, *args):
        if not self._turn_on:
            print("First you should turn the logger on once, via on() method to be able to log parameters.")
            return
        self._off()

        episode, episode_reward, skill, logq_zs, step, losses, skill_reward, do_diayn = args

        self.max_episode_reward = max(self.max_episode_reward, episode_reward)
        self.config["do_diayn"] = do_diayn

        ram = psutil.virtual_memory()
        assert self.to_gb(ram.used) < 0.98 * self.to_gb(ram.total), "RAM usage exceeded permitted limit!"

        if episode % self.config["interval"] == 0:
            self._save_weights(episode)
            print("E: {}| "
                  "Skill: {}| "
                  "E_Reward: {:.1f}| "
                  "EP_Duration: {:.2f}| "
                  "Memory_Length: {}| "
                  "Mean_steps_time: {:.3f}| "
                  "{:.1f}/{:.1f} GB RAM| "
                  "Time: {} ".format(episode,
                                     skill,
                                     episode_reward,
                                     self.duration,
                                     self.agent.memory.min_mem_len,
                                     self.duration / step,
                                     self.to_gb(ram.used),
                                     self.to_gb(ram.total),
                                     datetime.datetime.now().strftime("%H:%M:%S"),
                                     ))

        metrics = {"Perf/Max episode reward": self.max_episode_reward,
                   "Perf/Episode reward": episode_reward,
                   "Perf/Step Length": step,
                   }
        if self.config["do_diayn"]:
            metrics.update({"Perf/Running logq(z|s)": logq_zs})

        if losses is not None:
            metrics = {**metrics,
                       "Perf/Entropy": losses["entropy"],
                       "Loss/Policy loss": losses["policy_loss"],
                       "Loss/Value loss": losses["value_loss"],
                       "Loss/Q loss": losses["q_loss"],
                       "Loss/Policy grad norm": losses["policy_grad_norm"],
                       "Loss/Value grad norm": losses["value_grad_norm"],
                       "Loss/Q grad norm": losses["q_grad_norm"],
                       }

            if self.config["do_diayn"]:
                metrics = {**metrics,
                           "Perf/Skill reward": skill_reward,
                           "Loss/Discriminator loss": losses["discriminator_loss"],
                           "Loss/Discriminator grad norm": losses["discriminator_grad_norm"],
                           }

        for k, v in metrics.items():
            self.log_writer.add_scalar(k, v, episode)
        self.log_writer.add_histogram(f"Skill {skill}", episode_reward, episode)
        self.log_writer.add_histogram("Total Rewards", episode_reward, episode)
        if self.config["wandb"]:
            wandb.log(metrics, step=episode)
            wandb.log({f"Skill {skill}": wandb.Histogram(episode_reward)}, step=episode)
            wandb.log({"Total Rewards": wandb.Histogram(episode_reward)}, step=episode)

        self.on()

    def _save_weights(self, episode):
        file_name = "/params_pretrain.pth" if self.config["do_diayn"] else "/params_finetune.pth"
        torch.save({"policy_network_state_dicts": [self.agent.policy_networks[a].state_dict() for a in range(self.config["n_agents"])],
                    "q_value_network1_state_dict": self.agent.q_value_network1.state_dict(),
                    "q_value_network2_state_dict": self.agent.q_value_network2.state_dict(),
                    "value_network_state_dict": self.agent.value_network.state_dict(),
                    "discriminator_state_dict": self.agent.discriminator.state_dict(),
                    "q_value1_opt_state_dict": self.agent.q_value1_opt.state_dict(),
                    "q_value2_opt_state_dict": self.agent.q_value2_opt.state_dict(),
                    "policy_opt_state_dicts": [self.agent.policy_opts[a].state_dict() for a in range(self.config["n_agents"])],
                    "value_opt_state_dict": self.agent.value_opt.state_dict(),
                    "discriminator_opt_state_dict": self.agent.discriminator_opt.state_dict(),
                    "episode": episode,
                    "max_episode_reward": self.max_episode_reward,
                    },
                   "Checkpoints/" + self.log_dir + file_name)

    def load_weights(self):
        model_dir = f"Checkpoints/{self.config['env_name']}/{self.config['pretrain_name']}"
        checkpoint = torch.load(model_dir + "/params.pth", map_location=self.device)
        for a in range(self.config["n_agents"]):
            self.agent.policy_networks[a].load_state_dict(checkpoint["policy_network_state_dicts"][a])
            self.agent.policy_opts[a].load_state_dict(checkpoint["policy_opt_state_dicts"][a])
        self.agent.q_value_network1.load_state_dict(checkpoint["q_value_network1_state_dict"])
        self.agent.q_value_network2.load_state_dict(checkpoint["q_value_network2_state_dict"])
        self.agent.value_network.load_state_dict(checkpoint["value_network_state_dict"])
        self.agent.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.agent.q_value1_opt.load_state_dict(checkpoint["q_value1_opt_state_dict"])
        self.agent.q_value2_opt.load_state_dict(checkpoint["q_value2_opt_state_dict"])
        self.agent.value_opt.load_state_dict(checkpoint["value_opt_state_dict"])
        self.agent.discriminator_opt.load_state_dict(checkpoint["discriminator_opt_state_dict"])

        self.max_episode_reward = checkpoint["max_episode_reward"]
        self.agent.hard_update_target_network()

        print("Model loaded from: ", model_dir)
        return checkpoint["episode"]
