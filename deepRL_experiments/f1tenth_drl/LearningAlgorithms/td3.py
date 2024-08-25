import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from f1tenth_drl.Utils.Networks import DoublePolicyNet, DoubleQNet
from f1tenth_drl.Utils.ReplayBuffers import OffPolicyBuffer
from f1tenth_drl.Utils.utils import soft_update

from .td3_quantizable import DoublePolicyNetQuant

# hyper parameters
BATCH_SIZE = 100
GAMMA = 0.99
tau = 0.005
# NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
POLICY_NOISE = 0.2


class TrainTD3:
    def __init__(self, state_dim, action_dim):
        self.act_dim = action_dim
        
        self.actor = DoublePolicyNet(state_dim, action_dim)
        self.actor_target = DoublePolicyNet(state_dim, action_dim)
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
        torch.save(self.actor.state_dict(), directory + f'{filename}_actor.pth')

    


class TestTD3:
    def __init__(self, filename, directory):
        self.actor = torch.load(directory + f'{filename}_actor.pth') # Original init for no quant

    PTDQ
    def __init__(self, filename, directory, state_dim, act_dim):
        self.actor = DoublePolicyNet(state_dim, act_dim).cpu()
        print("Loading Actor's State Dict for Tests")
        self.actor.load_state_dict(torch.load(directory + f'{filename}_actor.pth'))
        self.actor.eval()

        # Set the dynamic quantization qconfig
        print("Applying Dynamic Quantization QConfig")
        self.actor.qconfig = torch.quantization.default_dynamic_qconfig
        print("Dynamically Quantizing Linear Layers")
        self.actor = torch.quantization.quantize_dynamic(self.actor, {nn.Linear}, dtype=torch.qint8)
        self.print_model_architecture()

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

    # PTSQ
    # def __init__(self, filename, directory, state_dim, act_dim):
    #     self.actor = DoublePolicyNetQuant(state_dim, act_dim).cpu()
    #     print("Loading Actor's State Dict for Tests")
    #     self.actor.load_state_dict(torch.load(directory + f'{filename}_actor.pth'))
    #     self.actor.eval()

    #     # Set the static quantization qconfig
    #     print("Applying Staticic Quantization QConfig")
    #     self.actor.qconfig = torch.quantization.default_qconfig
    #     # Prepare for static quantization
    #     torch.quantization.prepare(self.actor, inplace=True)
    #     print("Statically Quantizing Linear Layers")
    #     # Convert the model to a quantized version
    #     torch.quantization.convert(self.actor, inplace=True)
    #     self.print_model_architecture()

    # def act(self, state):
    #     state = torch.FloatTensor(state.reshape(1, -1))
    #     action = self.actor(state).data.numpy().flatten()
        
    #     return action
    
    # def print_model_architecture(self):
    #     def print_layer(module, indent=0):
    #         spaces = ' ' * indent
    #         if isinstance(module, nn.Module):
    #             print(f'{spaces}{module.__class__.__name__} ({module._get_name()})')
    #             for name, param in module.named_parameters(recurse=False):
    #                 print(f'{spaces}  Parameter: {name}, dtype: {param.dtype}')
    #             for name, submodule in module.named_children():
    #                 print(f'{spaces}  Submodule: {name}')
    #                 print_layer(submodule, indent + 4)
    #         else:
    #             print(f'{spaces}{module.__class__.__name__} ({module._get_name()})')

    #     print_layer(self.actor)

        