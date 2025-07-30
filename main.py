from screen import capture, process

import pytesseract

import cv2
import numpy as np
import os
import keyboard
os.environ["TESSDATA_PREFIX"] = r'C:\Program Files (x86)\Tesseract-OCR\tessdata'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#from nn import net
n_things = 8
import time 
import torch
import gym
from gym import spaces
import numpy as np
import random
from collections import deque

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


    frame2 = cv2.resize(frame, ((map_coords["w"]*scale,map_coords["h"]*scale)))
    process.place_dot(frame2,(164-map_coords["x"])*scale,(986-map_coords["y"])*scale)
    frame2, inp = process.lines(frame2,(164-map_coords["x"])*scale,(986-map_coords["y"])*scale)

    frame = np.array(spd.frame())
    mask = cv2.inRange(frame, np.array([254,254,254]), np.array([255,255,255]))
    result = cv2.bitwise_and(frame, frame, mask=mask)
    result = cv2.resize(result, ((int(speed_coords["w"]*0.5),int(speed_coords["h"]*0.5))))
    
    cv2.imshow("hi2",result)
    
    

    digits_only_config = '--oem 3 --psm 6 outputbase digits' # OEM 3 for default engine, PSM 6 for single uniform block of text
    digits = pytesseract.image_to_string(result, config=digits_only_config)

    try:
        speed = int(digits)
        #print(speed)
    except:
        speed = 0
        #print(speed)
    #speed = 0
    return frame2, inp, speed
import time
#while True:
frame2,inp,speed = image_processing()
    #print(inp[:n_things])
    #print(np.min(np.array(inp[:n_things], dtype=np.float32)))
    #cv2.imshow("hi",frame2)
    #if cv2.waitKey(25) & 0xFF == ord('q'):
    #    pass
    #time.sleep(0.1)
import torch.nn as nn
import torch.optim as optim

# Custom Environment
class ForzaEnv(gym.Env):
    def __init__(self):
        super(ForzaEnv, self).__init__()
        self.observation_space = spaces.Dict({
            'pos': spaces.Box(low=0, high=1, shape=(n_things,), dtype=np.float32),
            'speed': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
        self.action_space = spaces.Box(low=np.array([0,-1]), high=np.array([1,1]), shape=(2,), dtype=np.float32)
        self.state = None
        self.steps = 1
        self.last_reward = 300
        self.last_speed = 300
        self.curr_speed = 0
    def reset(self):

        self.state = {
            'pos': np.zeros(n_things, dtype=np.float32),
            'speed': np.zeros(1, dtype=np.float32)
        }
        print('terminated')
        keyboard.release('w')
        keyboard.release('a')
        keyboard.release('s')
        keyboard.release('d')
        keyboard.press_and_release('esc')
        time.sleep(2)
        keyboard.press_and_release('s')
        time.sleep(1)
        keyboard.press_and_release('enter')
        time.sleep(1)
        keyboard.press_and_release('x')
        time.sleep(0.5)
        keyboard.press_and_release('s')
        time.sleep(0.5)
        keyboard.press_and_release('enter')
        time.sleep(0.5)
        keyboard.press_and_release('w')
        time.sleep(0.5)
        keyboard.press_and_release('enter')
        time.sleep(0.5)
        keyboard.press_and_release('enter')
        time.sleep(0.5)
        keyboard.press_and_release('enter')
        time.sleep(7)

        return self.state

    def step(self, action):
        #print(round(action[0]), round(action[1]))
        action[0],action[1] = round(action[0]), round(action[1])
        frame2,inp,speed = image_processing()
        self.curr_speed = speed
        cv2.imshow("hi",frame2)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            pass
        
        if action[0] == 1:
            #print('w')
            keyboard.press('w')
            if self.curr_speed == 0:
                time.sleep(0.4)
            else:
                time.sleep(0.2)
            keyboard.release('w')
        if action[1] == 1:
            #print('a')
            keyboard.press('a')
        if action[1] == -1:
            #print('d')
            keyboard.press('d')
        
        time.sleep(abs(action[1])*0.1 + 0.1)
        if action[1] == 1:
            #print('a')
            keyboard.release('a')
        if action[1] == -1:
            #print('d')
            keyboard.release('d')
        
        #if action[0]   == -1:
            #print('s')
            #keyboard.press('s')
        
        
        #if action[0] == 1:
            #print('w')
        #    keyboard.release('w')
        
        #if action[0] == -1:
            #print('s')
            #keyboard.release('s')

        
        self.state['pos'] = np.array(inp[:5], dtype=np.float32)
        self.state['speed'] = np.array([speed], dtype=np.float32)
        #print(sum(self.state['pos'][:5])/5)
        #reward = ((sum(self.state['pos'][:5])/5)  - 5)*min(self.curr_speed,15)*(1/10)
        
        min_dist = np.min(self.state['pos'])
        
        mean_dist = np.mean(self.state['pos'])
        print("mean_dist:",mean_dist)
        out_of_bounds = mean_dist < 5  # or whatever threshold means "too close"

        if out_of_bounds:
            reward = -10.0
            done = True
        else:
            reward = mean_dist * 0.1  # scale as needed
            # Optionally, add a small bonus for higher speed if safe
            reward += 0.1*min(self.curr_speed, 30)
        print("reward:",reward)
        done = False
        if self.steps%8 == 0 and self.steps>0:
            if self.last_reward < 1 and reward < 1:
                done = True
                self.last_reward = 300
                self.steps = 1
            else:
                self.last_reward = reward
        if self.steps%10 == 0 and self.steps>0:
            if self.last_speed < 4 and self.curr_speed < 4:
                done = True
                self.last_speed = 300  
                self.steps = 1
            else:
                self.last_speed = self.curr_speed
        
        self.steps+=1
        
        #print(self.last_reward, self.last_speed, reward, self.curr_speed)
        
        info = {}
        state = {}
        state['pos'] = np.clip(np.array(inp[:n_things], dtype=np.float32) / 50.0, 0, 1)
        state['speed'] = np.clip(np.array([speed], dtype=np.float32) / 100.0, 0, 1)

        print(list(state['pos']),list(state['speed']))
        return state, reward, done, info

# Policy Network (MLP)
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_things+1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Tanh()
        )

    def forward(self, obs):
        pos = obs['pos'].view(obs['pos'].size(0), -1)
        speed = obs['speed'].view(obs['speed'].size(0), -1)
        x = torch.cat([pos, speed], dim=1)
        return self.fc(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(list, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
import os
import logging


# Training Loop with Replay Buffer and Epsilon-Greedy Exploration
env = ForzaEnv()
policy_net = PolicyNet()
target_net = PolicyNet()


MODEL_PATH = "forza_policy_net.pth"
SAVE_EVERY = 5  # steps

# Load model if exists
if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}")
    policy_net.load_state_dict(torch.load(MODEL_PATH))
    target_net.load_state_dict(policy_net.state_dict())
else:
    print("No saved model found, starting from scratch.")


target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
replay_buffer = ReplayBuffer(500000)
batch_size = 200
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 100
steps_done = 0
update_target_every = 1000

def select_action(obs_tensor, epsilon):
    global steps_done
    steps_done += 1
    if random.random() < epsilon:
        return env.action_space.sample()
    with torch.no_grad():
        action = policy_net(obs_tensor)
        return action.squeeze(0).cpu().numpy()
# Set up logging
logging.basicConfig(
level=logging.INFO,
format='[%(asctime)s] %(levelname)s: %(message)s',
handlers=[
    logging.FileHandler("forza_ai_training.log"),
    logging.StreamHandler()
]
)

num_episodes = 100
for episode in range(num_episodes):
    obs = env.reset()
    obs_tensor = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in obs.items()}
    total_reward = 0
    logging.info(f"=== Episode {episode+1} started ===")
    for t in range(100):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                    np.exp(-1. * steps_done / epsilon_decay)
        action = select_action(obs_tensor, epsilon)
        logging.info(f"Step {t+1} | Epsilon: {epsilon:.4f} | Action: {action}")
        next_obs, reward, done, _ = env.step(action)
        logging.info(f"Step {t+1} | Reward: {reward} | Done: {done}")
        next_obs_tensor = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in next_obs.items()}
        replay_buffer.push(
            {k: v.clone() for k, v in obs_tensor.items()},
            torch.tensor(action, dtype=torch.float32),
            torch.tensor([reward], dtype=torch.float32),
            {k: v.clone() for k, v in next_obs_tensor.items()},
            torch.tensor([done], dtype=torch.float32)
        )
        obs_tensor = next_obs_tensor
        total_reward += reward

        # Training step
        print("training - ",len(replay_buffer),batch_size)

        if len(replay_buffer) >= batch_size:
            for _ in range(3):
                batch = replay_buffer.sample(batch_size)
                state_batch = {k: torch.cat([s[k] for s in batch[0]], dim=0) for k in batch[0][0].keys()}
                action_batch = torch.stack(batch[1])
                reward_batch = torch.cat(batch[2])
                next_state_batch = {k: torch.cat([s[k] for s in batch[3]], dim=0) for k in batch[3][0].keys()}
                done_batch = torch.cat(batch[4])

                predicted_actions = policy_net(state_batch)
                loss = loss_fn(predicted_actions, action_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logging.info(f"Step {t+1} | Training | Loss: {loss.item():.6f}")

        if steps_done % update_target_every == 0:
            target_net.load_state_dict(policy_net.state_dict())
            tau = 0.005
            for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            logging.info(f"Step {t+1} | Target network updated.")

        # Save model every SAVE_EVERY steps
        if steps_done % SAVE_EVERY == 0:
            torch.save(policy_net.state_dict(), MODEL_PATH)
            logging.info(f"Model saved at step {steps_done}")

        if done:
            logging.info(f"Episode {episode+1} finished after {t+1} steps with total reward {total_reward}")
            break
    logging.info(f"Episode {episode+1}, Total Reward: {total_reward}")


'''
print("Running policy without training...")
for episode in range(200):
    obs = env.reset()
    obs_tensor = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in obs.items()}
    total_reward = 0
    for t in range(100):
        with torch.no_grad():
            action = policy_net(obs_tensor)
            action = action.squeeze(0).cpu().numpy()
        next_obs, reward, done, _ = env.step(action)
        obs_tensor = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in next_obs.items()}
        total_reward += reward
        if done:
            print(f"Episode {episode+1} finished after {t+1} steps with total reward {total_reward}")
            break
    print(f"Episode {episode+1} | Total Reward: {total_reward}")

'''