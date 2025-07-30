import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Define observation and action spaces
        self.observation_space = spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2) # 0: decrease, 1: increase
        self.current_value = 5.0 # Initial state

    def _get_obs(self):
        return np.array([self.current_value], dtype=np.float32)

    def _get_info(self):
        return {"current_value": self.current_value}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_value = 5.0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        if action == 0:
            self.current_value -= 1
        elif action == 1:
            self.current_value += 1

        self.current_value = np.clip(self.current_value, 0, 10) # Clip value

        reward = -abs(self.current_value - 5.0) # Reward for staying near 5
        terminated = False
        truncated = False
        if self.current_value == 0 or self.current_value == 10:
            terminated = True # End episode if at extremes

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self):
        print(f"Current Value: {self.current_value}")

    def close(self):
        pass

import torchrl
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
custom_env = torchrl.envs.GymWrapper(CustomEnv())

# Wrap it with TorchRL's GymEnv
env = TransformedEnv(custom_env, StepCounter())

import torchrl 
import torch
from tensordict import TensorDict
from torchrl.data.tensor_specs import Bounded

action_spec = Bounded(-torch.ones(1), torch.ones(1))
actor = torchrl.envs.utils.RandomPolicy(action_spec=action_spec) 
td = actor(TensorDict({}, batch_size=[])) 

import time
import matplotlib.pyplot as plt
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
from tensordict.nn import TensorDictModule as TensorDict, TensorDictSequential as Seq
from torchrl.modules import EGreedyModule, MLP, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torch.optim import Adam
from torchrl._utils import logger as torchrl_logger

INIT_RAND_STEPS = 5000 
FRAMES_PER_BATCH = 100
OPTIM_STEPS = 10
EPS_0 = 0.5
BUFFER_LEN = 100_000
ALPHA = 0.05
TARGET_UPDATE_EPS = 0.95
REPLAY_BUFFER_SAMPLE = 128
LOG_EVERY = 1000
MLP_SIZE = 64

value_mlp = MLP(out_features=env.action_spec.shape[-1], num_cells=[MLP_SIZE, MLP_SIZE])
value_net = TensorDict(value_mlp, in_keys=["observation"], out_keys=["action_value"])
policy = Seq(value_net, QValueModule(spec=env.action_spec))

exploration_module = EGreedyModule(
    env.action_spec, annealing_num_steps=BUFFER_LEN, eps_init=EPS_0
)
policy_explore = Seq(policy, exploration_module)

collector = SyncDataCollector(
    env,
    policy_explore,
    frames_per_batch=FRAMES_PER_BATCH,
    total_frames=-1,
    init_random_frames=INIT_RAND_STEPS,
)

rb = ReplayBuffer(storage=LazyTensorStorage(BUFFER_LEN))


loss = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True)
optim = Adam(loss.parameters(), lr=ALPHA)
updater = SoftUpdate(loss, eps=TARGET_UPDATE_EPS)

total_count = 0
total_episodes = 0
t0 = time.time()
success_steps = []

for i, data in enumerate(collector):
    rb.extend(data)
    max_length = rb[:]["next", "step_count"].max()
    if len(rb) > INIT_RAND_STEPS:
        for _ in range(OPTIM_STEPS):
            sample = rb.sample(REPLAY_BUFFER_SAMPLE)
            loss_vals = loss(sample)
            loss_vals["loss"].backward()
            optim.step()

            optim.zero_grad()
            # Update exploration factor
            exploration_module.step(data.numel())

            # Update target params
            updater.step()
            total_count += data.numel()
            total_episodes += data["next", "done"].sum()
            print(total_count)
    success_steps.append(max_length)