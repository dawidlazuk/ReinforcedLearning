# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 20:30:30 2018

@author: Wazz
"""

import numpy as np
import matplotlib.pyplot as plt
import math

#n_arms = 2
LinearFactor = 0.01

RewardProbability = 0.5

def linear_reward(reward):
    return lambda it: LinearFactor * it + reward

class Arm:
    def __init__(self, actions):
        self.actions = actions
        self.pulled_times = 0
        
    def pull(self, action_index):
        self.pulled_times = self.pulled_times + 1
        return self.actions[action_index].pull(self.pulled_times);       
  
                        
class Action:
    def __init__(self, reward_f):
        self.reward_f = reward_f     
        
    def pull(self, pulled_times):
        if(np.random.random() < RewardProbability):      
            return self.reward_f(pulled_times)    
        return 0
            
class Bandit:
    def __init__(self, n_actions_per_arm):
        self.arms = [
                Arm(actions = [Action(linear_reward(0.1)), Action(linear_reward(0.8))]),
                Arm(actions = [Action(linear_reward(0.9)), Action(linear_reward(0.2))])
            ];
        self.actions_prob = np.array([0.5,0.5])
        
        self.n_actions_per_arm = n_actions_per_arm
        self.n_arms = 2
        
        self.current_action = 0;
        
    def init_step(self):
        rand = np.random.random()
        action_dist = 0
        for i in range(self.n_actions_per_arm):
            action_dist = action_dist + self.actions_prob[i]
            if(rand < action_dist):
                self.current_action = i
                break
            
    def pull_arm(self, arm_index):
        result = self.arms[arm_index].pull(self.current_action)
        self.init_step()
        return result
    
class Player_WithoutCaseKnowledge:
    
    def __init__(self, bandit):       
        self.bandit = bandit        
        
        self.Q = np.zeros(bandit.n_arms)
        self.N = np.zeros(bandit.n_arms)        
    
        self.epsilon = 0.5        
        self.steps_performed = 0
        
        
    def get_next_arm(self):
        if(np.random.random() > self.epsilon):
            #consume
            print('Consume')
            return np.argmax(self.Q)
        else:
            #explore
            print('Explore')
            return np.random.randint(0,bandit.n_arms)
        
    def perform_step(self):
        self.steps_performed = self.steps_performed + 1
        self.epsilon =  math.pow((StepsNumber - self.steps_performed) / StepsNumber,3)
        
        A = self.get_next_arm()
        R = bandit.pull_arm(A)
        self.N[A] = self.N[A] + 1
        self.Q[A] = self.Q[A] + (R - self.Q[A])/self.N[A]
        
        print('Q ', self.Q, 'N ', self.N, ' epsilon ', self.epsilon)
        return R
    
class Player_WithCaseKnowledge:
        
    def __init__(self, bandit):       
        self.bandit = bandit        
        
        self.Q = np.zeros([bandit.n_arms, bandit.n_actions_per_arm])
        self.N = np.zeros([bandit.n_arms, bandit.n_actions_per_arm])        
    
        #można zmieniać wraz z przebiegiem uczenia - początkowo eksplorować, a następnie wraz z przypływem wiedzy eksploatować
        self.epsilon = 0.5        
        self.steps_performed = 0
        
        
    def get_next_arm(self):
        if(np.random.random() > self.epsilon):
            #consume
            print('Consume')
            return np.argmax(self.Q[:,bandit.current_action])
        else:
            #explore
            print('Explore')
            #tutaj można wybierać te akcje, które mało razy zostały przeszukane
            #return np.argmin(self.N[:,bandit.current_action])
            #losowo (według algorytmu)
            return np.random.randint(0,bandit.n_arms)            
        
    def perform_step(self):
        self.steps_performed = self.steps_performed + 1
        self.epsilon =  math.pow((StepsNumber - self.steps_performed) / StepsNumber,3)

        curr_action = bandit.current_action
        A = self.get_next_arm()
        R = bandit.pull_arm(A)
        self.N[A, curr_action] = self.N[A, curr_action] + 1
        self.Q[A, curr_action] = self.Q[A, curr_action] + (R - self.Q[A, curr_action])/self.N[A, curr_action]
        
        print('Q ', self.Q, 'N ', self.N, ' epsilon ', self.epsilon)
        return R
             
StepsNumber = 1000
    
bandit = Bandit(2)          
player = Player_WithCaseKnowledge(bandit)

plotdata = []

for i in range(StepsNumber):
    reward = player.perform_step();
    plotdata.append([reward])
    print('Step ', i, ' reward ', reward)
    
plt.figure(figsize=(15,10))
plt.plot(plotdata)
    
    
    
    
    
    
