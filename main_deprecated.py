from screen import capture, process
import cv2
import numpy as np
import pytesseract
import os
os.environ["TESSDATA_PREFIX"] = r'C:\Program Files (x86)\Tesseract-OCR\tessdata'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#from nn import net

import time 

map_coords = {
    "x":115,
    "y":925,
    "w":210-115,
    "h":1020-925
}

speed_coords = {
    "x":638,
    "y":976,
    "w":704-638,
    "h":1013-976
}
mp = capture.Capture(map_coords["x"], map_coords["y"], map_coords["w"], map_coords["h"])

spd = capture.Capture(speed_coords["x"], speed_coords["y"], speed_coords["w"], speed_coords["h"])

scale = 5
def image_processing():
    global scale
    frame = np.array(mp.frame())

    # Invert the mask to get non-black pixels
    #non_black_mask = ~black_mask

    # Set non-black pixels to white (255, 255, 255)
    #frame[non_black_mask] = [255, 255, 255]
    
    #frame = ~frame

    frame2 = cv2.resize(frame, ((map_coords["w"]*scale,map_coords["h"]*scale)))
    process.place_dot(frame2,(164-map_coords["x"])*scale,(986-map_coords["y"])*scale)
    frame2, inp = process.lines(frame2,(164-map_coords["x"])*scale,(986-map_coords["y"])*scale)

    frame = np.array(spd.frame())
    
    mask = cv2.inRange(frame, np.array([254,254,254]), np.array([255,255,255]))

    # Apply the mask to the original image
    # This will keep the pixels within the mask (the desired color range) and turn the rest to black
    result = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("hi2",result)
    
    s = time.time()

    digits_only_config = '--oem 3 --psm 6 outputbase digits' # OEM 3 for default engine, PSM 6 for single uniform block of text
    digits = pytesseract.image_to_string(result, config=digits_only_config)

    e = time.time()

    try:
        speed = int(digits)
        #print(speed)
    except:
        speed = 0
        #print(speed)
    #speed = 0
    return frame2, inp, speed

#while True:
frame2,inp,speed = image_processing()
    #cv2.imshow("hi",frame2)
    #if cv2.waitKey(25) & 0xFF == ord('q'):
    #    pass
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import keyboard
class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Define observation and action spaces
        self.observation_space = spaces.Dict({
            'pos': spaces.Box(low=0, high=120, shape=(7*4,), dtype=np.float32),
            'speed': spaces.Box(low=0, high=150, shape=(1,), dtype=np.float32)
        })
        self.action_space = spaces.MultiDiscrete([2,2,2,2]) # 0: w, 1: a, 2: s, 3: d
        self.state = np.array([0,0,0,0,0,0,0], dtype=np.float32)
        self.steps = 21
        self.last_reward = 300
        self.last_speed = 300
        self.curr_speed = 0
    def _get_obs(self):
        frame2,inp,speed = image_processing()
        self.curr_speed = speed
        #print(speed)
        cv2.imshow("hi",frame2)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            pass
        self.state = inp
        return {
            'pos': np.array(inp, dtype=np.float32),
            'speed': np.array(speed, dtype=np.float32)
        }

    def _get_info(self):
        return {"current_value": 0}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = self._get_obs()
        info = self._get_info()
        print('terminated')
        keyboard.press_and_release('esc')
        time.sleep(2)
        keyboard.press_and_release('s')
        time.sleep(1)
        keyboard.press_and_release('enter')
        time.sleep(1)
        keyboard.press_and_release('x')
        time.sleep(1)
        keyboard.press_and_release('s')
        time.sleep(1)
        keyboard.press_and_release('enter')
        time.sleep(1)
        keyboard.press_and_release('w')
        time.sleep(1)
        keyboard.press_and_release('enter')
        time.sleep(1)
        keyboard.press_and_release('enter')
        time.sleep(1)
        keyboard.press_and_release('enter')
        time.sleep(10)
        return observation, info

    def step(self, action):
        reward = 0
        print(action)
        if action[0] == 1:
            #print('w')
            keyboard.press('w')
        if action[1] == 1:
            #print('a')
            keyboard.press('a')
        if action[2] == 1:
            #print('s')
            keyboard.press('s')
        if action[3] == 1:
            #print('d')
            keyboard.press('d')
        
        time.sleep(0.3)
        
        if action[0] == 1:
            #print('w')
            keyboard.release('w')
        if action[1] == 1:
            #print('a')
            keyboard.release('a')
        if action[2] == 1:
            #print('s')
            keyboard.release('s')
        if action[3] == 1:
            #print('d')
            keyboard.release('d')
        

        observation = self._get_obs()
        info = self._get_info()
        terminated = False
        reward = (sum(self.state[:7]))*min(self.curr_speed,15)*(1/70)
        #print(action)
        if action[3] == 1 and (sum(self.state[7:14])/7) > 10:
            reward += 5
        if action[0] == 1 and (sum(self.state[:7])/7) > 10:
            reward += 5
            
        print(self.last_reward,reward, self.last_speed, self.curr_speed)
        
        if self.steps%8 == 0:
            if self.last_reward < 6 and reward < 6:
                terminated = True
                self.last_reward = 300
            else:
                self.last_reward = reward
        if self.steps%20 == 0:
            if self.last_speed < 15 and self.curr_speed < 15:
                terminated = True
                self.last_speed = 300  
            else:
                self.last_speed = self.curr_speed
        
        self.steps+=1
        truncated = False

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

import torchrl
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
custom_env = torchrl.envs.GymWrapper(CustomEnv())
# Wrap it with TorchRL's GymEnv
env = TransformedEnv(custom_env, StepCounter())

import torchrl 
import torch


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

INIT_RAND_STEPS = 0
FRAMES_PER_BATCH = 5
OPTIM_STEPS = 10
EPS_0 = 0.5
BUFFER_LEN = 100_000
ALPHA = 0.05
TARGET_UPDATE_EPS = 0.95
REPLAY_BUFFER_SAMPLE = 5
LOG_EVERY = 1000
MLP_SIZE = 512

value_mlp = MLP(out_features=env.action_spec.shape, num_cells=[MLP_SIZE, MLP_SIZE])
value_net = TensorDict(value_mlp, in_keys=["pos","speed"], out_keys=["action_value"])

class MultiDiscretePolicy(torch.nn.Module):
    def __init__(self, input_size, num_actions_per_dim):
        super().__init__()
        self.mlp = MLP(in_features=input_size, out_features=sum(num_actions_per_dim))
        self.num_actions_per_dim = num_actions_per_dim

    def forward(self, x):
        logits = self.mlp(x)  # shape: (batch_size, sum of all actions)
        # Split logits into chunks for each action dimension
        split_logits = torch.split(logits, self.num_actions_per_dim, dim=-1)
        return split_logits
        
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

#policy.load_state_dict(torch.load('policylatest.pth'))

total_count = 0
total_episodes = 0
t0 = time.time()
success_steps = []

for i, data in enumerate(collector):
    rb.extend(data)
    print('initial buffer')
    max_length = rb[:]["next", "step_count"].max()
    for _ in range(OPTIM_STEPS):
        print('sampling...')
        sample = rb.sample(REPLAY_BUFFER_SAMPLE)

        loss_vals = loss(sample)
        loss_vals["loss"].backward()
        optim.step()

        optim.zero_grad()
        # Update exploration factor
        exploration_module.step(data.numel())

        
        # Update target params
        updater.step()

        torch.save(policy.state_dict(), 'policy'+str(total_count)+'.pth')
        print('saved')
        total_count += data.numel()
        total_episodes += data["next", "done"].sum()

        if total_count > 0 and total_count % LOG_EVERY == 0:
            torchrl_logger.info(f"Successful steps in the last episode: {max_length}, rb length {len(rb)}, Number of episodes: {total_episodes}")
#success_steps.append(max_length)
