import numpy as np
import torch
import torch.nn as nn

from utils import ReplayBuffer, to_tensor, device
from torch.optim import Adam


class Critic(nn.Module):
    def __init__(self, inp_size, hidden=256):
        super().__init__()
        self.lin1 = nn.Linear(inp_size, hidden)
        self.lin1_1 = nn.Linear(hidden, hidden)
        self.lin1_2 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, 1)
    
    def forward(self, state, action):
        # state = state if isinstance(state, torch.Tensor) else to_tensor(state)
        # action = action if isinstance(action, torch.Tensor) else to_tensor(action)
        x = torch.concat([state, action], dim=-1)
        x = nn.functional.relu(self.lin1(x))
        x = nn.functional.relu(self.lin1_1(x))
        x = nn.functional.relu(self.lin1_2(x))
        return self.lin2(x)

    
class Actor(nn.Module):
    def __init__(self, inp_size, actions, bounds, hidden=128, action_amp=1):
        super().__init__()
        self.lin1 = nn.Linear(inp_size, hidden)
        self.lin_positions = nn.Linear(hidden, actions)
        self.lin2 = nn.Linear(hidden, hidden)
        self.lin_actions = nn.Linear(hidden, actions)
        self.bounds = bounds
        self.amp = action_amp
    
    def forward(self, state):
        # state = state if isinstance(state, torch.Tensor) else to_tensor(state)
        x = nn.functional.sigmoid(self.lin1(state))
        positions = nn.functional.sigmoid(self.lin_positions(x))
        positions = positions * np.diff(self.bounds).item() + self.bounds[0]
        
        x = nn.functional.relu(self.lin2(x))
        actions = nn.functional.tanh(self.lin_actions(x)) * self.amp
        return actions, positions


class Agent:
    def __init__(self, inp_size, actions,
                 bounds=(-1, 1),
                 action_amp=1,
                 bs=64,
                 tau=0.005,
                 buffer_length=1000,
                 actor_hidden=128,
                 critic_hidden=256,
                 lr_a=1e-4, wd_a=1e-5,
                 lr_c=3e-4, wd_c=1e-5):
        self.bs = bs
        self.tau = tau
        self.amp = action_amp
        self.supervised = 0
        
        self.buffer = ReplayBuffer(inp_size, actions, maxlen=buffer_length)
        
        self.actor = Actor(inp_size, actions, bounds=bounds, action_amp=action_amp, hidden=actor_hidden).to(device)
        self.target_actor = Actor(inp_size, actions, bounds=bounds, action_amp=action_amp, hidden=actor_hidden).to(device)
        self.critic = Critic(inp_size + actions, hidden=critic_hidden).to(device)
        self.target_critic = Critic(inp_size + actions, hidden=critic_hidden).to(device)
        self.update_target_networks(tau=1)
        
        self.optim_critic = Adam([p for p in self.critic.parameters()], lr=lr_c, weight_decay=wd_c)
        self.optim_actor = Adam([p for p in self.actor.parameters()], lr=lr_a, weight_decay=wd_a)
        self.critic_criterion = nn.MSELoss()
    
    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau
        for name in ['actor', 'critic']:
            self.polyak_update(tau, getattr(self, name), getattr(self, f"target_{name}"))
    
    @staticmethod
    def polyak_update(tau, net, tnet):
        for p, tp in zip(net.parameters(), tnet.parameters()):
            tp.data.copy_(tau * p.data + tp.data * (1 - tau))
    
    def remember(self, state, action, reward, new_state, done, loc=None):
        self.buffer.store_transition(state, action, reward, new_state, done, loc)
    
    def choose_action(self, state, evaluate=False, numpy=False):
        action, _ = self.actor(state)
        if not evaluate:
            action += torch.randn(*action.shape, device=device) * self.amp / 20
            action = torch.clip(action, -self.amp, self.amp)
        return action.detach().cpu().numpy() if numpy else action
    
    def learn(self):
        if self.buffer.counter < self.bs:
            return
        
        states, actions, rewards, next_states, dones, locs = self.buffer.sample(self.bs)
        
        self.optim_actor.zero_grad()
        actions_, positions = self.actor(states)
        if (not self.supervised % 3 == 0) or self.supervised < 3000:
            loss_actor = torch.sum((positions - locs)**2, dim=-1).mean()
            loss_actor.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
            self.optim_actor.step()
        else:
            self.optim_critic.zero_grad()
            target_actions, _ = self.target_actor(next_states)
            q_next = self.target_critic(next_states, target_actions)
            q = self.critic(states, actions)
            targets = rewards + torch.squeeze(q_next) * (1 - dones.to(int))
            loss_critic = self.critic_criterion(q, targets.view(-1, 1))
            loss_critic.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)
            self.optim_critic.step()
        
            loss_actor = -self.critic(states, actions_)
            loss_actor = torch.mean(loss_actor)
            loss_actor.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
            self.optim_actor.step()
        
        self.update_target_networks()
        self.supervised += 1