import gymnasium as gs
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


env = gs.make("Pendulum-v1") # makes the environment. 


"""
The class for the Actor Critic PPO
We are sampling actions a from a Gaussian N(mu, std^2)
This is an MDP based model, and this is a starting DRL project.
The critic model will find an optimal V(s)
The actor Model will find create an optimal policy pi_theta(s)
At the moment, I am having trouble running MuJoCo on here, so I do not have results.

"""
class ActorCriticPPO(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        # 
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )

        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        self.critic  = nn.Linear(256,1)

    def forward(self, obs):
        shared_stuff = self.shared(obs)
        mean = self.actor_mean(shared_stuff)
        std = self.actor_logstd.exp()

        value  = self.critic(shared_stuff)

        return mean,std,value
    

def get_action(model, obs):
        obs = torch.tensor(obs, dtype=torch.float32)

        mean, std, value = model(obs)

        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()

        logprob = dist.log_prob(action).sum()

        return action.detach().numpy(), logprob.detach(), value.detach()

    # Comples GAE (better than R_t - V_t)
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
        adv = []
        # T-1 o f gae should be 0, so that the recurisve definition of gae can work.
        # all of this is to compute the advantage better
        gae = 0
        
        values = values + [0]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            adv.insert(0, gae)

        return adv
    
def ppo_update(model, optimizer, batch, clip_eps=0.2):
        states, actions, old_logprobs, returns, advantages = batch

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        old_logprobs = torch.tensor(old_logprobs, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        mean, std, values = model(states)

        dist = torch.distributions.Normal(mean, std)
        logprobs = dist.log_prob(actions).sum(dim=-1)

        ratio = torch.exp(logprobs - old_logprobs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = F.mse_loss(values.squeeze(), returns)

        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

model = ActorCriticPPO(obs_dim, act_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for episode in range(1000):

    obs, _ = env.reset()

    states, actions, logprobs, rewards, values, dones = [], [], [], [], [], []

    for step in range(200):

        action, logprob, value = get_action(model, obs)

        next_obs, reward, done, _, _ = env.step(action)

        states.append(obs)
        actions.append(action)
        logprobs.append(logprob)
        rewards.append(reward)
        values.append(value)
        dones.append(done) # dones check if model terminated yet. 

        obs = next_obs

        if done:
            break

    advantages = compute_gae(rewards, values, dones)

    returns = np.array(advantages) + np.array(values) # critic is trying to learn returns
    # So the minimization is (V_theta(s_t) - (A_t + V_Theta_old(s_t)))^2

    batch = (states, actions, logprobs, returns, advantages)

    ppo_update(model, optimizer, batch)

    print(f"Episode {episode}, reward: {sum(rewards)}")