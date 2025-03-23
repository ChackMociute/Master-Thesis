import torch
import torch.nn as nn
import numpy as np

from utils import ReplayBuffer, device
from torch.optim import Adam, RMSprop


class Critic(nn.Module):
    def __init__(self, inp_size, hidden=256):
        super().__init__()
        self.lin1 = nn.Linear(inp_size, hidden)
        self.lin2 = nn.Linear(hidden, 1)
    
    def forward(self, state, action):
        x = torch.concat([state, action], dim=-1)
        x = nn.functional.relu(self.lin1(x))
        return self.lin2(x)

    
class Actor(nn.Module):
    def __init__(self, inp_size, actions, bounds, hidden=128, action_amp=1):
        super().__init__()
        self.lin1 = nn.Linear(inp_size, hidden)
        self.lin2 = nn.Linear(hidden, 2 * actions)
        self.actions = actions
        self.bounds = bounds
        self.position_amp = np.diff(self.bounds).item()
        self.action_amp = action_amp
    
    def forward(self, state):
        x = nn.functional.sigmoid(self.lin1(state))
        x = self.lin2(x).view(-1, 2 * self.actions)
        actions, positions = x[:,:self.actions], x[:,self.actions:]
        
        actions = nn.functional.tanh(actions) * self.action_amp
        positions = nn.functional.sigmoid(positions)
        positions = positions * self.position_amp + self.bounds[0]
        
        return actions.squeeze(), positions.squeeze()


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
                 lr_c=3e-4, wd_c=1e-5,
                 grad_norm=0.1):
        self.bs = bs
        self.tau = tau
        self.amp = action_amp
        self.grad_norm = grad_norm
        
        self.buffer = ReplayBuffer(inp_size, actions, maxlen=buffer_length)
        
        self.actor = Actor(inp_size, actions, bounds=bounds, action_amp=action_amp, hidden=actor_hidden).to(device)
        self.target_actor = Actor(inp_size, actions, bounds=bounds, action_amp=action_amp, hidden=actor_hidden).to(device)
        self.critic = Critic(inp_size + actions, hidden=critic_hidden).to(device)
        self.target_critic = Critic(inp_size + actions, hidden=critic_hidden).to(device)
        self.update_target_networks(tau=1)
        
        self.optim_critic = RMSprop([p for p in self.critic.parameters()], lr=lr_c, weight_decay=wd_c)
        self.optim_actor = RMSprop([p for p in self.actor.parameters()], lr=lr_a, weight_decay=wd_a)
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
        action = self.actor(state)[0].detach()
        if not evaluate:
            action += torch.randn(*action.shape, device=device) * (self.amp / 3)
            action = torch.clip(action, -self.amp, self.amp)
        return action.detach().cpu().numpy() if numpy else action
    
    def learn(self):
        if self.buffer.counter < self.bs:
            return
        
        states, actions, rewards, next_states, dones, locs = self.buffer.sample(self.bs)
        
        self.optim_critic.zero_grad(set_to_none=True)
        target_actions, _ = self.target_actor(next_states)
        q_next = self.target_critic(next_states, target_actions)
        q = self.critic(states, actions)
        targets = rewards + torch.squeeze(q_next) * (1 - dones)
        loss_critic = self.critic_criterion(q, targets.view(-1, 1))
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)
        self.optim_critic.step()
        
        self.optim_actor.zero_grad(set_to_none=True)
        actions_, positions = self.actor(states)
        loss_actor = -self.critic(states, actions_)
        loss_actor = torch.mean(loss_actor)
        dists = positions - locs
        loss_actor += torch.sum(dists * dists, dim=-1).mean()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
        self.optim_actor.step()
        
        self.update_target_networks()