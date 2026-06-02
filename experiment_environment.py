import copy
import numpy as np

from client import Client
from config import CONFIG
from data_utils import get_client_loaders


class ExperimentEnvironment:
    """
    Shared experiment environment for fair comparisons.

    It owns the dataset split, client speed profiles, per-round client sampling,
    per-edge sampling, edge assignment, and latency trace. Algorithms can then
    consume the same heterogeneous world while keeping their own aggregation logic.
    """

    def __init__(self, config=None, seed=None):
        self.config = copy.deepcopy(config or CONFIG)
        self.seed = self.config.get('seed', 42) if seed is None else seed
        self.n_clients = self.config['n_clients']
        self.global_rounds = self.config['global_rounds']

        self.client_loaders = None
        self.test_loader = None
        self.profiles = None
        self.clients = None
        self.edge_clients = None

        self._selected_clients = {}
        self._selected_edge_clients = {}
        self._latencies = {}

    def build(self, device=None):
        self.client_loaders, self.test_loader = get_client_loaders(
            self.config['dataset'],
            self.config['data_root'],
            self.config['n_clients'],
            self.config['batch_size'],
            self.config['dirichlet_alpha'],
            seed=self.seed,
        )
        self.profiles = self._build_profiles()
        self._build_latency_trace()

        run_device = device or self.config['device']
        self.clients = {}
        for cid in range(self.n_clients):
            client = Client(
                cid,
                self.client_loaders[cid],
                profile_type=self.profiles[cid],
                device=run_device,
                environment=self,
            )
            self.clients[cid] = client
        return self

    def clone_clients(self, device=None):
        if self.clients is None:
            self.build(device=device)

        run_device = device or self.config['device']
        clients = {}
        for cid in range(self.n_clients):
            clients[cid] = Client(
                cid,
                self.client_loaders[cid],
                profile_type=self.profiles[cid],
                device=run_device,
                environment=self,
            )
        return clients

    def get_latency(self, round_num, client_id):
        if round_num not in self._latencies:
            self._build_latency_trace()
        return self._latencies[round_num][client_id]

    def select_clients(self, round_num, population=None, sample_fraction=None, force_all=False):
        population = list(range(self.n_clients)) if population is None else list(population)
        if force_all:
            return np.array(population, dtype=int)

        fraction = self.config.get('sample_fraction', 0.1) if sample_fraction is None else sample_fraction
        sample_size = max(1, int(fraction * len(population)))
        sample_size = min(sample_size, len(population))

        key = (round_num, tuple(population), sample_size)
        if key not in self._selected_clients:
            rng = np.random.RandomState(self.seed + 20_000 + round_num)
            self._selected_clients[key] = rng.choice(population, sample_size, replace=False)
        return self._selected_clients[key].copy()

    def assign_edge_clients(self, num_edge_servers):
        if self.edge_clients is not None and len(self.edge_clients) == num_edge_servers:
            return copy.deepcopy(self.edge_clients)

        rng = np.random.RandomState(self.seed + 30_000)
        cids = np.arange(self.n_clients)
        rng.shuffle(cids)

        self.edge_clients = {i: [] for i in range(num_edge_servers)}
        chunks = np.array_split(cids, num_edge_servers)
        for edge_id, chunk in enumerate(chunks):
            self.edge_clients[edge_id] = [int(cid) for cid in chunk.tolist()]
        return copy.deepcopy(self.edge_clients)

    def select_edge_clients(self, round_num, edge_id, available_clients, sample_size):
        available_clients = list(available_clients)
        sample_size = min(sample_size, len(available_clients))
        key = (round_num, edge_id, tuple(available_clients), sample_size)
        if key not in self._selected_edge_clients:
            rng = np.random.RandomState(self.seed + 40_000 + round_num * 101 + edge_id)
            self._selected_edge_clients[key] = rng.choice(available_clients, sample_size, replace=False)
        return self._selected_edge_clients[key].copy()

    def _build_profiles(self):
        ratios = self.config['profile_ratio']
        n_fast = int(self.n_clients * ratios[0])
        n_medium = int(self.n_clients * ratios[1])
        n_slow = self.n_clients - n_fast - n_medium

        profiles = ['fast'] * n_fast + ['medium'] * n_medium + ['slow'] * n_slow
        rng = np.random.RandomState(self.seed)
        rng.shuffle(profiles)
        return profiles

    def _build_latency_trace(self):
        if self.profiles is None:
            self.profiles = self._build_profiles()

        rng = np.random.RandomState(self.seed + 10_000)
        self._latencies = {}
        volatility_cfg = self.config.get('network_volatility', 0.0)

        for round_num in range(1, self.global_rounds + 1):
            round_latencies = {}
            for cid, profile_type in enumerate(self.profiles):
                params = self.config['profiles'][profile_type]
                base = max(10, rng.normal(params['mean'], params['std']))
                volatility = rng.uniform(1 - volatility_cfg, 1 + volatility_cfg)
                round_latencies[cid] = base * volatility
            self._latencies[round_num] = round_latencies


def create_environment(config=None, seed=None, device=None):
    return ExperimentEnvironment(config=config, seed=seed).build(device=device)
