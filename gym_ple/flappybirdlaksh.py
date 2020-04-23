import os
from PIL import Image    
from ple import PLE
from ple.games.flappybird import FlappyBird
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import torchvision.transforms as T

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()
import logging
import os, sys

import gym
from gym.wrappers import Monitor
import gym_ple
from gym import logger as gymlogger
gymlogger.set_level(40) # error only
import numpy as np
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import glob
import io
import base64
from IPython.display import HTML
from IPython.display import clear_output
from IPython import display as ipythondisplay
plt.rcParams["figure.figsize"] = [16,9]
from collections import namedtuple
from itertools import count
from collections import deque

def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")

def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


base_dir="results/dqn"

ENV='FlappyBird-v0'

torch.manual_seed(0)
np.random.seed(0)

env = gym.make(ENV)
env = wrap_env(env)
env.seed(0)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from collections import deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class PriorityReplayMemory:
    # modified from https://github.com/susantamoh84/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter07/bench/prio_buffer_bench.py
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.min_prio = 0.1
        
    def beta_by_frame(self):
        self.frame += 1
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def __len__(self):
        return len(self.buffer)

    def push(self, sample):
        max_prio = max(self.priorities) if self.priorities else 1.0
        self.buffer.append(sample)
        self.priorities.append(max_prio ** self.alpha)

    def sample(self, batch_size):
        probs = np.array(self.priorities, dtype=np.float32) 
        probs /= probs.sum()
        total = len(self.buffer)
        indices = np.random.choice(total, batch_size, p=probs, replace=True)
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_frame()
        
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, torch.Tensor(weights).to(device)
    

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = (prio + self.min_prio)** self.alpha

class ReplayMemory:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, sample):
        """Saves a transition."""
        self.buffer.append(sample)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        samples = [self.buffer[idx] for idx in indices]
        return samples, None, torch.Tensor([1/len(self.buffer) for _ in range(batch_size)]).to(device)
    
    def update_priorities(self, batch_indices, batch_priorities):
        pass
    
    def __len__(self):
        return len(self.buffer)

class DDQN4(nn.Module):
    # 4 frames,

    def __init__(self, h, w):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 6, stride = 2, padding=0):
            return (size - kernel_size +2*padding ) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,8,4),4,2),3,1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,8,4),4,2),3,1)

        linear_input_size = convw * convh * 64
        fc_output_size=512
        self.fc_val=nn.Linear(linear_input_size, fc_output_size)
        self.fc_adv=nn.Linear(linear_input_size, fc_output_size)
        self.val = nn.Linear(fc_output_size, 1)
        self.adv = nn.Linear(fc_output_size, N_ACTION)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x_val = F.relu(self.fc_val(x.view(x.size(0), -1)))
        x_adv = F.relu(self.fc_adv(x.view(x.size(0), -1)))
        val=self.val(x_val)
        adv=self.adv(x_adv)
        
        
        x=val+adv-adv.mean(1,keepdim=True)
        return x

process_pic = T.Compose([
                    T.ToPILImage(),
                    T.Grayscale(),
                    T.Resize((84,84)),
                    T.ToTensor(),
                        ])

def get_screen_flappy():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = screen[:, :int(screen_height * 0.8)]
    screen = screen.mean(0).astype('uint8')

    screen = torch.from_numpy(screen)
    
    screen = process_pic(screen.unsqueeze(0))
    screen = (screen*255).type(torch.uint8)

    return screen.unsqueeze(0).to(device)


get_screen=get_screen_flappy


BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.005
ESP_END2 = 0.1
FRAME_SKIP = 2
EPS_DECAY = 300000
EPS_DECAY2 = 200000
LOG_DECAY = False
TARGET_UPDATE = 1000
PLOT_INTERVAL = 50
REPLAY_SIZE= 100000
SAVE_CHECKPOINT=500
FULL_RANDOM=40000
OBSERVE = 20000
LR = 1e-6
USE_PRIORITY_REPLAY = True
N_ACTION = env.action_space.n
USE_BONUS_REWARD = False
LIMIT_MAX_REWARD = 250
CLIP_NORM=0.5
TOTALTIME = 0
TOTALLIVES = 3
TOTALEPISODES = 0
TOTALNSTEPS = 0

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

policy_net = DDQN4(screen_height, screen_width).to(device)
target_net = DDQN4(screen_height, screen_width).to(device)
print(policy_net)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(),lr=LR)
# optimizer = optim.RMSprop(policy_net.parameters(),lr=LR)
if USE_PRIORITY_REPLAY:
    memory = PriorityReplayMemory(REPLAY_SIZE)
else:
    memory = ReplayMemory(REPLAY_SIZE)

def select_action(state):
    global action_q
    policy_net.eval()
    sample = random.random()
    if LOG_DECAY:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (steps_done - OBSERVE)/ EPS_DECAY)
    else:
        if not EPS_DECAY2:
            eps_threshold = max( EPS_START - (EPS_START - EPS_END)/EPS_DECAY *(steps_done - OBSERVE),EPS_END)
        else:
            # 0-EPS_DECAY: reduce threshold to EPS_END+ESP_END2, EPS_DECAY-EPS_DECAY2: reduce threshold to EPS_END
            if steps_done-OBSERVE<EPS_DECAY:
                eps_threshold = EPS_START - (EPS_START - EPS_END-ESP_END2)/EPS_DECAY *(steps_done - OBSERVE)
            else:
                eps_threshold = max( EPS_END+ESP_END2 - ESP_END2/EPS_DECAY2 *(steps_done - OBSERVE-EPS_DECAY),EPS_END)
                
    if not is_full_random() and sample > eps_threshold:
        with torch.no_grad():
            t=policy_net(state)
            action_q=t.max().item()
            return t.max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(N_ACTION)]], device=device, dtype=torch.long)

import time

#need to change
def plot_durations():
    if i_episode%PLOT_INTERVAL==0:
        global action_qs
        plt.clf()
        #display.clear_output(wait=True)
        rewards_t = torch.tensor(total_rewards, dtype=torch.float)
        bonus_rewards_t = torch.tensor(total_bonus_rewards, dtype=torch.float)
        action_qs_t = torch.tensor(action_qs, dtype=torch.float)
        plt.title('Training...episode:{},steps:{},time used:{}s'.format(i_episode,steps_done,round(time_used)))
        plt.xlabel('Episode')
        plt.ylabel('Duration')
#         plt.plot(action_qs_t.numpy(),label='Q')
        if USE_BONUS_REWARD:
            plt.plot(bonus_rewards_t.numpy(),label='bonus_reward')
        plt.plot(rewards_t.numpy(),label='reward')
        # Take 100 episode averages and plot them too
        if len(rewards_t) >= 100:
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(),label='reward_mean')
        if len(action_qs_t) >= 100:
            means = action_qs_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(),label='Q_mean')
        plt.legend()
        plt.savefig('plot2.png')
        plt.pause(0.00001)  # pause a bit so that plots are updated
        plt.close()

def optimize_model():
    policy_net.train()
    if len(memory) < BATCH_SIZE:
        return
    if steps_done<OBSERVE:
        return
        
    samples, ids, weights = memory.sample(BATCH_SIZE)
    
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*samples))
    

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    with torch.no_grad():
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    diff = state_action_values.squeeze() - expected_state_action_values.squeeze()
    loss=torch.zeros(diff.shape)
    for index,(i,j) in enumerate(zip(diff,weights)):
        if -1<i<1:
            loss[index]=(i * i)/2 * j
        else:
            loss[index]=(i.abs()-1/2)*j
    loss=loss.mean().to(device)
    
#     loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    delta = diff.abs().detach().cpu().numpy().tolist()
    memory.update_priorities(ids, delta)


    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(),CLIP_NORM)
    optimizer.step()

from collections import deque
import time
import os




checkpoint_file='flappy_checkpoint2.pt'
temp_policy='best_score.pt'

def save_checkpoint():
    torch.save({
                'i_episode': i_episode+1,
                'total_rewards':total_rewards,
                'total_bonus_rewards':total_bonus_rewards,
                'steps_done':steps_done,
                'policy_state_dict': policy_net.state_dict(),
                'target_state_dict': target_net.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'time_used' : time_used,
            },checkpoint_file)

def load_checkpoint():
    global i_episode,policy_net,optimizer,steps_done,memory,total_rewards,time_used,total_bonus_rewards
    if os.path.exists(checkpoint_file):
        checkpoint=torch.load(checkpoint_file)
        i_episode=checkpoint['i_episode']
        total_rewards=checkpoint['total_rewards']
        total_bonus_rewards=checkpoint['total_bonus_rewards']
        steps_done=checkpoint['steps_done']
        time_used=checkpoint['time_used']
        optimizer.load_state_dict(checkpoint['optimizer'])
        policy_net.load_state_dict(checkpoint['policy_state_dict'])
        target_net.load_state_dict(checkpoint['target_state_dict'])


while TOTALLIVES > 0:
    resume=False

    i_episode = 0
    total_rewards = []
    total_bonus_rewards = []
    action_qs = []
    steps_done = 0
    time_used = 0
    best_avg_score=-5

    if resume:
        load_checkpoint()
    
    def is_key_frame(t):
        return t%FRAME_SKIP==0

    def is_full_random():
        return steps_done<FULL_RANDOM

    print(TOTALLIVES)
    for i_episode in range(10):
        print(i_episode)
    # Initialize the environment and state
        
        #print(i_episode)
        start_time=time.time()
        env.reset()
        total_reward=0
        total_bonus_reward=0
        last_screen = get_screen()
        current_screen = get_screen()
        screens = deque([current_screen] * 4, 4)
        state = torch.cat(list(screens), dim=1)
        last_action=None
        key_frame_reward=0.0
        action_q=None
        episode_q=[]
    
        for t in count():
        # Select and perform an action
            env.render(mode='human')
            steps_done += 1
            if is_key_frame(t):
                action = select_action(state)
                last_action=action
                if action_q is not None:
                    episode_q.append(action_q+total_reward)
            else:
                action=last_action
            _, reward, done, _ = env.step(action.item())
            print("reward for that frame is ", reward)
            #print(done)
            total_reward+=reward
            key_frame_reward+=reward
            total_bonus_reward+=reward+0.05
            print("reward for the episode is ", total_reward)
        
            # Observe new state
            last_screen = current_screen
        
            if is_key_frame(t) or done:
                current_screen = get_screen()
                screens.append(current_screen)
            if not done:
                next_state = torch.cat(list(screens), dim=1)
            else:
                next_state = None
                

            # Store the transition in memory
            if is_key_frame(t) or done:
                if USE_BONUS_REWARD:
                    bonus_reward_t = torch.tensor([reward+0.05], device=device)
                    memory.push(Transition(state, action, bonus_reward_t, next_state, done))
                
                else:
                    key_frame_reward_t=torch.tensor([key_frame_reward], device=device)
                    memory.push(Transition(state, action, key_frame_reward_t, next_state, done))
                    key_frame_reward=0
            
                if LIMIT_MAX_REWARD and total_reward>LIMIT_MAX_REWARD:
                    True

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
        
            if done:
                total_rewards.append(total_reward)
                total_bonus_rewards.append(total_bonus_reward)
                if episode_q:
                    action_qs.append(sum(episode_q)/len(episode_q))
                else:
                    action_qs.append(0)
                plot_durations()
                break
            
            if steps_done % TARGET_UPDATE==0 and not is_full_random():
                target_net.load_state_dict(policy_net.state_dict())
        
#         if steps_done%100==0:
#             print(np.array(list(memory.priorities)).max(),np.array(list(memory.priorities)).mean(),len(memory.buffer))
        if i_episode and i_episode%SAVE_CHECKPOINT==0 and not is_full_random():
            save_checkpoint()
        if len(total_rewards)>100:
            last_100_avg_rewards=sum(total_rewards[-100:])/100
            if last_100_avg_rewards>best_avg_score:
                best_avg_score=last_100_avg_rewards
                torch.save(policy_net,temp_policy)
        
        
        time_used+=time.time()-start_time
        TOTALTIME += time_used
        TOTALEPISODES += i_episode
        TOTALNSTEPS += steps_done

    TOTALLIVES = TOTALLIVES - 1
    print('Complete')
    env.render()
    env.close()

print('TOTALLIVES: ',TOTALLIVES)
print('TOTALTIME: ',TOTALTIME)
print('TOTALEPISODES: ',TOTALEPISODES)
print('TOTALNSTEPS: ',TOTALNSTEPS)