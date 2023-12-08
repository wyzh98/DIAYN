import argparse


def get_params():
    parser = argparse.ArgumentParser(description="Variable parameters based on the configuration of the machine or user's choice")

    parser.add_argument("--run_name", default="example", type=str, help="Name of the run.")
    parser.add_argument("--env_name", default="Walker2d", type=str, help="Name of the environment.")
    parser.add_argument("--do_diayn", action="store_false", default=True, help="Train with DIAYN/intrinsic reward.")
    parser.add_argument("--pretrain_episodes", default=int(2e+4), type=int, help="Number of DIAYN pretraining episodes.")
    parser.add_argument("--finetune_episodes", default=int(2e+4), type=int, help="Number of finetuning episodes.")
    parser.add_argument("--interval", default=100, type=int, help="The interval specifies how often different parameters should be saved and printed, counted by episodes.")
    parser.add_argument("--mem_size", default=int(1e+6), type=int, help="The memory size.")
    parser.add_argument("--n_skills", default=20, type=int, help="The number of skills to learn.")
    parser.add_argument("--reward_scale", default=1, type=float, help="The reward scaling factor introduced in SAC.")
    parser.add_argument("--steps_per_train", default=10, type=float, help="Train every n steps.")
    parser.add_argument("--seed", default=123, type=int, help="The randomness' seed for torch, numpy, random & gym[env].")
    parser.add_argument("--wandb", action="store_true", default=True, help="Use wandb.")

    parser_params = parser.parse_args()

    #  Parameters based on the DIAYN and SAC papers.
    # region default parameters
    default_params = {"lr": 3e-4,
                      "batch_size": 512,
                      "max_episode_len": 1000,
                      "gamma": 0.99,
                      "alpha": 0.1,
                      "tau": 0.005,
                      "n_hiddens": 512
                      }
    # endregion
    total_params = {**vars(parser_params), **default_params}
    return total_params
