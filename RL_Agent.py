import numpy as np
import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones



class OUActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)



class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, env, directory):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name

        # self.checkpoint_dir = f'/media/bharath/New Volume/Bharath_AIRL/CBF/RL_CBF/trained_agents/{env}'
        self.checkpoint_dir = directory
        self.checkpoint_file = os.path.join(self.checkpoint_dir, env + '_' + name)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)


        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        
        self.q = nn.Linear(self.fc2_dims, 1)

        # f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        # self.fc1.weight.data.uniform_(-f1, f1)
        # self.fc1.bias.data.uniform_(-f1, f1)

        # f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        # self.fc2.weight.data.uniform_(-f2, f2)
        # self.fc2.bias.data.uniform_(-f2, f2)

        # f3 = 0.003
        # self.q.weight.data.uniform_(-f3, f3)
        # self.q.bias.data.uniform_(-f3, f3)

        # f4 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        # self.action_value.weight.data.uniform_(-f4, f4)
        # self.action_value.bias.data.uniform_(-f4, f4)

        self.optimizer = optim.Adam(self.parameters(), lr=beta,
                                    weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:0')
        # self.device = 'cpu'

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        # print('... saving best checkpoint ...')
        # checkpoint_file = os.path.join(self.checkpoint_dir, self.name +'_best')
        T.save(self.state_dict(), self.checkpoint_file)



class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, env, directory):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name

        # self.checkpoint_dir = f'/media/bharath/New Volume/Bharath_AIRL/CBF/RL_CBF/trained_agents/{env}'
        self.checkpoint_dir = directory
        self.checkpoint_file = os.path.join(self.checkpoint_dir, env + '_' + name)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        #self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        #self.bn2 = nn.BatchNorm1d(self.fc2_dims)

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        # f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        # self.fc2.weight.data.uniform_(-f2, f2)
        # self.fc2.bias.data.uniform_(-f2, f2)

        # f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        # self.fc1.weight.data.uniform_(-f1, f1)
        # self.fc1.bias.data.uniform_(-f1, f1)

        # f3 = 0.003
        # self.mu.weight.data.uniform_(-f3, f3)
        # self.mu.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:0')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = F.tanh(self.mu(x))
        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        # print('... saving best checkpoint ...')
        # checkpoint_file = os.path.join(self.checkpoint_dir, self.name +'_best')
        T.save(self.state_dict(), self.checkpoint_file)



class KappaNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, name, env):
        super(KappaNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.name = name
        self.checkpoint_dir = f'RL_CBF/trained_agents/{env}'
        self.checkpoint_file = os.path.join(self.checkpoint_dir, env + '_' + name)

        self.fc1 = nn.Linear(1, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.mu = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:0')

        self.to(self.device)

    def forward(self, h):
        x = self.fc1(h)
        x = F.softplus(x)
        # x = F.relu(x)
        x = self.fc2(x)
        x = F.softplus(x)
        # x = F.relu(x)
        x = self.mu(x)
        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        T.save(self.state_dict(), self.checkpoint_file)



class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                 max_size=1000000, fc1_dims=128, fc2_dims=64, 
                 batch_size=64,env_name=None,k=None,l=None,iter=None,directory=None):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.directory = directory

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        if k == None and l == None:
            self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                    n_actions=n_actions, name=f'actor_{iter}', env=env_name, directory=self.directory)
            self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                    n_actions=n_actions, name=f'critic_{iter}', env=env_name, directory=self.directory)

            self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                    n_actions=n_actions, name=f'target_actor_{iter}', env=env_name, directory=self.directory)

            self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                    n_actions=n_actions, name=f'target_critic_{iter}', env=env_name, directory=self.directory)

        else:
            self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                    n_actions=n_actions, name=f'actor_{iter}_{k}_{l}', env=env_name, directory=self.directory)
            self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                    n_actions=n_actions, name=f'critic_{iter}_{k}_{l}', env=env_name, directory=self.directory)

            self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                    n_actions=n_actions, name=f'target_actor_{iter}_{k}_{l}', env=env_name, directory=self.directory)

            self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                    n_actions=n_actions, name=f'target_critic_{iter}_{k}_{l}', env=env_name, directory=self.directory)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        # observation = np.array(observation,dtype=np.float32)
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), 
                                    dtype=T.float).to(self.actor.device)
        self.actor.train()

        return mu_prime.cpu().detach().numpy()[:]

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        self.actor.save_best()
        self.target_actor.save_best()
        self.critic.save_best()
        self.target_critic.save_best()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, done = \
                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        # actor_loss = self.gamma * rewards
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)

