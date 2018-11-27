import numpy as np
import matplotlib.pyplot as plt
import math

###### Implementacja ######

#Liniowa funkcja wartosci akcji
def linear_reward(lin_factor):
    return lambda step: lin_factor * step

#Pojedyncze ramie maszyny
class Arm:
    def __init__(self, actions):
        self.actions = actions
        
    def pull(self, action_index, pulled_times):
        return self.actions[action_index].pull(pulled_times);       
        
#Akcja                 
class Action:
    def __init__(self, reward_fun, reward_prob):
        self.reward_fun = reward_fun    
        self.reward_prob = reward_prob
        
    def pull(self, pulled_times):
        if(np.random.random() < self.reward_prob):      
            return self.reward_fun(pulled_times)    
        return 0
            
#Maszyna
class Bandit:    
    #arms_with_actions macierz: rzędy - ramiona, kolumny - prawdopodobieństwo akcji
    def __init__(self, arms_with_actions, reward_fun):             
        self.n_arms = arms_with_actions.shape[0]
        self.n_actions_per_arm = arms_with_actions.shape[1]        
        self.actions_prob = np.ones(self.n_actions_per_arm)/self.n_actions_per_arm #rowne prawdopodobienstwo akcji
        
        self.arms = []
        for k in range(self.n_arms):
            actions = [];
            for a in range(self.n_actions_per_arm):
                actions.append(Action(reward_fun, arms_with_actions[k,a]))
            self.arms.append(Arm(actions))
        
        self.current_action = 0;
        self.pulled_times = 0                
        
    def init_step(self):
        rand = np.random.random()
        action_dist = 0
        for i in range(self.n_actions_per_arm):
            action_dist = action_dist + self.actions_prob[i]
            if(rand < action_dist):
                self.current_action = i
                break
            
    def pull_arm(self, arm_index):
        self.pulled_times = self.pulled_times + 1;
        result = self.arms[arm_index].pull(self.current_action, self.pulled_times)
        self.init_step()
        return result
    
#Gracz, który nie posiada wiedzy o obecnym stanie maszyny
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
        return R, self.epsilon
    
#Gracz, który posiada wiedzę o obecnym stanie maszyny    
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
        self.epsilon = math.pow((StepsNumber - self.steps_performed) / StepsNumber,3)

        curr_action = bandit.current_action
        A = self.get_next_arm()
        R = bandit.pull_arm(A)
        self.N[A, curr_action] = self.N[A, curr_action] + 1
        self.Q[A, curr_action] = self.Q[A, curr_action] + (R - self.Q[A, curr_action])/self.N[A, curr_action]
        
        for i in range(self.Q.shape[1]):
            print('State ', i, ': ', 'Q ', self.Q[:, i], 'N ', self.N[:, i])            
        print('Epsilon ', self.epsilon)
        return R, self.epsilon


###### Metody pomocnicze do eksperymentów ######

#Rysowanie wykresu liniowego
def printPlot_lines(subplot, array, title, x_label, y_label):
    subplot.plot(array)
    subplot.set_title(title)
    subplot.set_xlabel(x_label)
    subplot.set_ylabel(y_label)

#Rysowanie wykresu kropkowanego    
def printPlot_dots(subplot, array, title, x_label, y_label):
    subplot.plot(array, linestyle="", marker="o")
    subplot.set_title(title)
    subplot.set_xlabel(x_label)
    subplot.set_ylabel(y_label)

#Pobieranie danych danego klastra z tablicy
def GetBatch(array, index):
    return array[index*Batch_size : (index+1)*Batch_size-1]


###### Eksperymenty ######

#Inicjalizacja stałych
LinearFactor = 0.01
StepsNumber = 20000
Batch_size = 100
Bandit_array = np.array([[0.1, 0.4 , 0.1], [0.3, 0.2, 0.1]])

Batch_number = (int)(StepsNumber/Batch_size)
    
#Inicjalizacja maszyny
bandit = Bandit(Bandit_array, linear_reward(LinearFactor))          

#Inicjalizacja gracza (wybór klasy jako algorytmu)
player = Player_WithoutCaseKnowledge(bandit)

#Listy na dane do wykresów
rewards_per_step = []
epsilon_on_step = []

#Wykonaj iteracje algorytmu
for i in range(StepsNumber):
    reward, curr_eps = player.perform_step();
    rewards_per_step.append([reward])        
    epsilon_on_step.append([curr_eps])
    print('Step ', i, ' reward ', reward)

#Zlicz sumę zebranych nagród dla każdego kroku
total_reward_per_step = np.cumsum(rewards_per_step)

#Wyznaczenie sumy nagród zdobytych w klastrze
reward_value_in_batch = []
for i in range(Batch_number):
    value_in_batch = np.sum(rewards_per_step[i*Batch_size : (i+1)*Batch_size-1])
    reward_value_in_batch.append(value_in_batch)

#Wyznaczenie ilosci nagród zdobytych w klastrze    
reward_number_in_batch = []
for i in range(Batch_number):
    number_in_batch = np.count_nonzero(GetBatch(rewards_per_step, i))
    reward_number_in_batch.append(number_in_batch)      
    
#Wykres nagród w krokach    
fig, axs = plt.subplots(2, 1, constrained_layout=True)
printPlot_dots(axs[0], rewards_per_step, 'Nagroda w każdym kroku', 'Kroki', 'Wartosc nagrody')
printPlot_lines(axs[1], total_reward_per_step, 'Suma nagród w każdym kroku', 'Kroki', 'Suma nagrody')
fig.set_size_inches(10,10)
plt.show()

#Wykres nagród w klastrach
fig, axs = plt.subplots(2, 1, constrained_layout=True)
printPlot_dots(axs[0], reward_value_in_batch, 'Wartosć nagród w klastrach', 'Klastry', 'Wartosc nagrody')
printPlot_lines(axs[1], reward_number_in_batch, 'Ilosc sukcesow w klastrach', 'Klastry', 'Ilosc sukcesów')
fig.set_size_inches(10,10)
plt.show()

#wykres epsilon
fig, axs = plt.subplots(1, 1, constrained_layout=True)
printPlot_lines(axs, epsilon_on_step, 'Wartosć epsilon w każdym kroku', 'Kroki', 'Wartosć epsilon')
fig.set_size_inches(10,5)
plt.show()