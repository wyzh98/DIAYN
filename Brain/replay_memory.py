import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'z', 'done', 'action', 'next_state', 'obs', 'next_obs', 'reward'))


class Memory:
    def __init__(self, buffer_size, n_agents, seed):
        self.buffer_size = buffer_size
        self.n_agents = n_agents
        self.buffer = [[] for _ in range(self.n_agents)]
        self.seed = seed
        random.seed(self.seed)

    def add(self, agent_id, *transition):
        self.buffer[agent_id].append(Transition(*transition))
        if self.mem_len_of(agent_id) > self.buffer_size:
            self.buffer[agent_id].pop(0)
        assert self.mem_len_of(agent_id) <= self.buffer_size

    def sample(self, size, agent_id):
        return random.sample(self.buffer[agent_id], size)

    def clean(self):
        self.buffer = [[] for _ in range(self.n_agents)]

    def mem_len_of(self, agent_id):
        return len(self.buffer[agent_id])

    @property
    def min_mem_len(self):
        return min([self.mem_len_of(a) for a in range(self.n_agents)])

    @staticmethod
    def get_rng_state():
        return random.getstate()

    @staticmethod
    def set_rng_state(random_rng_state):
        random.setstate(random_rng_state)
