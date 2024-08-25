import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

#### UTILS ####

NN_LAYER_1 = 100
NN_LAYER_2 = 100
MEMORY_SIZE = 100000

def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

class DoublePolicyNetQuant(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(DoublePolicyNetQuant, self).__init__()

        self.quant = torch.quantization.QuantStub()
        self.fc1 = nn.Linear(state_dim, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_mu = nn.Linear(NN_LAYER_2, act_dim)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))
        mu = self.dequant(mu)
        return mu
    
class DoubleQNet(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(DoubleQNet, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + act_dim, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_out = nn.Linear(NN_LAYER_2, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x2 = F.relu(self.fc1(x))
        x3 = F.relu(self.fc2(x2))
        q = self.fc_out(x3)
        return q

class OffPolicyBuffer(object):
    def __init__(self, state_dim, action_dim):     
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0

        self.states = np.empty((MEMORY_SIZE, state_dim))
        self.actions = np.empty((MEMORY_SIZE, action_dim))
        self.next_states = np.empty((MEMORY_SIZE, state_dim))
        self.rewards = np.empty((MEMORY_SIZE, 1))
        self.dones = np.empty((MEMORY_SIZE, 1))

    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr += 1
        
        if self.ptr == MEMORY_SIZE: self.ptr = 0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.ptr-1, size=batch_size)
        states = np.empty((batch_size, self.state_dim))
        actions = np.empty((batch_size, self.action_dim))
        next_states = np.empty((batch_size, self.state_dim))
        rewards = np.empty((batch_size, 1))
        dones = np.empty((batch_size, 1))

        for i, j in enumerate(ind): 
            states[i] = self.states[j]
            actions[i] = self.actions[j]
            next_states[i] = self.next_states[j]
            rewards[i] = self.rewards[j]
            dones[i] = self.dones[j]
            
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(1- dones)

        return states, actions, next_states, rewards, dones

    def size(self):
        return self.ptr
    
#### MODEL ####

# hyper parameters
BATCH_SIZE = 100
GAMMA = 0.99
tau = 0.005
# NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
POLICY_NOISE = 0.2

# Quant Backend for M1:
torch.backends.quantized.engine = 'qnnpack'

class TrainTD3Quant:
    def __init__(self, state_dim, action_dim):
        self.act_dim = action_dim
        
        self.actor = DoublePolicyNetQuant(state_dim, action_dim)
        # FOR QAT ONLY ####
        self.actor.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        torch.quantization.prepare_qat(self.actor, inplace=True)
        #####
        self.actor_target = DoublePolicyNetQuant(state_dim, action_dim)
        # FOR QAT ONLH ####
        self.actor_target.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        torch.quantization.prepare_qat(self.actor_target, inplace=True)
        #####
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic_1 = DoubleQNet(state_dim, action_dim)
        self.critic_target_1 = DoubleQNet(state_dim, action_dim)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_2 = DoubleQNet(state_dim, action_dim)
        self.critic_target_2 = DoubleQNet(state_dim, action_dim)
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer = torch.optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=1e-3)

        self.replay_buffer = OffPolicyBuffer(state_dim, action_dim)

    def act(self, state, noise=EXPLORE_NOISE):
        state = torch.FloatTensor(state.reshape(1, -1))

        action = self.actor(state).data.numpy().flatten()
        
        if noise != 0: 
            action = (action + np.random.normal(0, noise, size=self.act_dim))
            
        return action.clip(-1, 1)

    def train(self, iterations=2):
        if self.replay_buffer.size() < BATCH_SIZE:
            return
        for it in range(iterations):
            state, action, next_state, reward, done = self.replay_buffer.sample(BATCH_SIZE)
            self.update_critic(state, action, next_state, reward, done)
        
            if it % POLICY_FREQUENCY == 0:
                self.update_policy(state)
                
                soft_update(self.critic_1, self.critic_target_1, tau)
                soft_update(self.critic_2, self.critic_target_2, tau)
                soft_update(self.actor, self.actor_target, tau)
    
    def update_critic(self, state, action, next_state, reward, done):
        noise = torch.normal(torch.zeros(action.size()), POLICY_NOISE)
        noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
        next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)

        target_Q1 = self.critic_target_1(next_state, next_action)
        target_Q2 = self.critic_target_2(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (done * GAMMA * target_Q).detach()

        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_policy(self, state):
        actor_loss = -self.critic_1(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
                

    def save(self, filename, directory):
        print("Saving Actor's State Dict")
        torch.save(self.actor.state_dict(), directory + f'{filename}_actor.pth')

    
class TestTD3Quant:
    
    def __init__(self, filename, directory, state_dim, act_dim):
        self.actor = DoublePolicyNetQuant(state_dim, act_dim)

        # FOR QAT ONLY ####
        self.actor.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        torch.quantization.prepare_qat(self.actor, inplace=True)
        #####

        self.actor.load_state_dict(torch.load(directory + f'{filename}_actor.pth'))

        # Convert the model to a quantized version
        torch.quantization.convert(self.actor, inplace=True)
        self.actor.eval()

        # Print the model architecture with data types
        self.print_model_architecture()
    
    def act(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).data.numpy().flatten()
        
        return action
    
    def print_model_architecture(self):
        def print_layer(module, indent=0):
            spaces = ' ' * indent
            if isinstance(module, nn.Module):
                print(f'{spaces}{module.__class__.__name__} ({module._get_name()})')
                for name, param in module.named_parameters(recurse=False):
                    print(f'{spaces}  Parameter: {name}, dtype: {param.dtype}')
                for name, submodule in module.named_children():
                    print(f'{spaces}  Submodule: {name}')
                    print_layer(submodule, indent + 4)
            else:
                print(f'{spaces}{module.__class__.__name__} ({module._get_name()})')

        print_layer(self.actor)

