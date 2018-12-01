import numpy as np
import matplotlib.pyplot as plt
import math

###### Implementacja ######

def linear_fun(a, b):
    return lambda x: a * x + b

def const_fun(val):
    return lambda x: val

def epsilon_reducing_fun(max_steps, power):
    return lambda x: math.pow( 1 - x / max_steps, power)

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
    def __init__(self, bandit, epsilon_fun, verbose=False):       
        self.bandit = bandit        
        self.verbose = verbose
        
        self.Q = np.zeros(bandit.n_arms)
        self.N = np.zeros(bandit.n_arms)        
    
        self.epsilon_fun = epsilon_fun
        self.steps_performed = 0        
          
    def get_epsilon(self):
        return self.epsilon_fun(self.steps_performed)

    def get_next_arm(self):
        if(np.random.random() > self.get_epsilon()):
            #consume
            if(self.verbose):
                print('Consume')
            return np.argmax(self.Q)
        else:
            #explore
            if(self.verbose):
                print('Explore')
            return np.random.randint(0, self.bandit.n_arms)
        
    def perform_step(self):
        self.steps_performed = self.steps_performed + 1
        epsilon = self.get_epsilon()
        
        A = self.get_next_arm()
        R = self.bandit.pull_arm(A)
        self.N[A] = self.N[A] + 1
        self.Q[A] = self.Q[A] + (R - self.Q[A])/self.N[A]
        
        if(self.verbose):
            print('Q ', self.Q, 'N ', self.N, ' epsilon ', epsilon)
        return R
    
#Gracz, który posiada wiedzę o obecnym stanie maszyny    
class Player_WithCaseKnowledge:        
    def __init__(self, bandit, epsilon_fun, verbose=False):       
        self.bandit = bandit        
        self.verbose = verbose
        
        self.Q = np.zeros([bandit.n_arms, bandit.n_actions_per_arm])
        self.N = np.zeros([bandit.n_arms, bandit.n_actions_per_arm])        
    
        self.epsilon_fun = epsilon_fun
        self.steps_performed = 0        
        
    def get_epsilon(self):
        return self.epsilon_fun(self.steps_performed)
        
    def get_next_arm(self):
        if(np.random.random() > self.get_epsilon()):
            #consume
            if(self.verbose):
                print('Consume')
            return np.argmax(self.Q[:, self.bandit.current_action])
        else:
            #explore
            if(self.verbose):
                print('Explore')
            #tutaj można wybierać te akcje, które mało razy zostały przeszukane
            #return np.argmin(self.N[:,bandit.current_action])
            #losowo (według algorytmu)
            return np.random.randint(0, self.bandit.n_arms)            
        
    def perform_step(self):
        self.steps_performed = self.steps_performed + 1
        epsilon = self.get_epsilon()

        curr_action = self.bandit.current_action
        A = self.get_next_arm()
        R = self.bandit.pull_arm(A)
        self.N[A, curr_action] = self.N[A, curr_action] + 1
        self.Q[A, curr_action] = self.Q[A, curr_action] + (R - self.Q[A, curr_action])/self.N[A, curr_action]
        
        if(self.verbose):
            for i in range(self.Q.shape[1]):
                print('State ', i, ': ', 'Q ', self.Q[:, i], 'N ', self.N[:, i])            
            print('Epsilon ', epsilon)
        return R

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
    
def print_actionstatevalue_subplot(subplot, array):
    subplot.plot(array)
    
def get_actionstatevalue_per_steps(Q_values, action, state):
    return Q_values[:,action, state];

def print_Q_plot(Q_values):
    Q_values = np.array(Q_values)
    while(Q_values.ndim < 3):
        Q_values = np.expand_dims(Q_values, axis=2)
    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    legend = []
    for state in range(Q_values.shape[2]):
        for action in range(Q_values.shape[1]):
            actionStateValue_per_steps = get_actionstatevalue_per_steps(Q_values, action, state)
            printPlot_lines(axs, actionStateValue_per_steps, 'Funkcja wartosci akcji w danym kroku', 'Krok', 'Wartosc')
            legend.append(f'Stan: {state+1}, akcja: {action+1}')
    plt.legend(legend)
    plt.show()

#Pobieranie danych danego klastra z tablicy
def GetBatch(array, index, batch_size):
    return array[index*batch_size : (index+1)*batch_size-1]

def PerformAnExperiment(player_type, bandit_array, reward_fun, epsilon_fun, steps_number, batch_size):    
    Batch_number = (int)(steps_number/batch_size)
    
    #Inicjalizacja maszyny
    bandit = Bandit(bandit_array, reward_fun)          
    
    #Inicjalizacja gracza (wybór klasy jako algorytmu)
    player = player_type(bandit, epsilon_fun)
    
    #Listy na dane do wykresów
    rewards_per_step = []
    epsilon_on_step = []
    Q_values_per_step = []
    
    #Wykonaj iteracje algorytmu
    for i in range(steps_number):
        reward = player.perform_step();
        rewards_per_step.append([reward])        
        epsilon_on_step.append([player.get_epsilon()])
        Q_values_per_step.append(np.copy(player.Q))
        print('Step ', i, ' reward ', reward)
    
    #Zlicz sumę zebranych nagród dla każdego kroku
    total_reward_per_step = np.cumsum(rewards_per_step)
    
    #Wyznaczenie sumy nagród zdobytych w klastrze
    reward_value_in_batch = []
    for i in range(Batch_number):
        value_in_batch = np.sum(rewards_per_step[i*batch_size : (i+1)*batch_size-1])
        reward_value_in_batch.append(value_in_batch)
    
    #Wyznaczenie ilosci nagród zdobytych w klastrze    
    reward_number_in_batch = []
    for i in range(Batch_number):
        number_in_batch = np.count_nonzero(GetBatch(rewards_per_step, i, batch_size))
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
    
    print_Q_plot(Q_values_per_step)
        
    
###### Eksperymenty ######
    
def Case_WithoutKnowledge_ConstReward_EpsilonConst():
    LinearFactor = 0.0
    LinearStartValue = 1
    Steps_Number = 20000
    Batch_size = 100
    Bandit_array = np.array([[0.1, 0.4], [0.3, 0.2]])
    #Bandit_array = np.array([[0.3, 0.3], [0.3, 0.3]])
    
    player_type = Player_WithoutCaseKnowledge
    reward_fun = linear_fun(LinearFactor, LinearStartValue)
    epsilon_fun = const_fun(0.1)
        
    PerformAnExperiment(player_type,
                        Bandit_array,
                        reward_fun,
                        epsilon_fun,
                        Steps_Number,
                        Batch_size)
    
def Case_WithoutKnowledge_LinearReward_EpsilonConst():
    LinearFactor = 0.01
    LinearStartValue = 1
    Steps_Number = 20000
    Batch_size = 100
    Bandit_array = np.array([[0.1, 0.4], [0.3, 0.2]])
    #Bandit_array = np.array([[0.3, 0.3], [0.3, 0.3]])
    
    player_type = Player_WithoutCaseKnowledge
    reward_fun = linear_fun(LinearFactor, LinearStartValue)
    epsilon_fun = const_fun(0.1)
        
    PerformAnExperiment(player_type,
                        Bandit_array,
                        reward_fun,
                        epsilon_fun,
                        Steps_Number,
                        Batch_size)
    
def Case_WithKnowledge_ConstReward_EpsilonConst():
    LinearFactor = 0.0
    LinearStartValue = 1
    Steps_Number = 10000000
    Batch_size = 10000
    Bandit_array = np.array([[0.1, 0.4], [0.3, 0.2]])
    #Bandit_array = np.array([[0.3, 0.3], [0.3, 0.3]])
    
    player_type = Player_WithCaseKnowledge
    reward_fun = linear_fun(LinearFactor, LinearStartValue)
    epsilon_fun = const_fun(0.1)
        
    PerformAnExperiment(player_type,
                        Bandit_array,
                        reward_fun,
                        epsilon_fun,
                        Steps_Number,
                        Batch_size)
    
def Case_WithKnowledge_LinearReward_EpsilonConst():
    LinearFactor = 0.01
    LinearStartValue = 1
    Steps_Number = 20000
    Batch_size = 100
    Bandit_array = np.array([[0.1, 0.4], [0.3, 0.2]])
    #Bandit_array = np.array([[0.3, 0.3], [0.3, 0.3]])
    
    player_type = Player_WithCaseKnowledge
    reward_fun = linear_fun(LinearFactor, LinearStartValue)
    epsilon_fun = const_fun(0.1)
        
    PerformAnExperiment(player_type,
                        Bandit_array,
                        reward_fun,
                        epsilon_fun,
                        Steps_Number,
                        Batch_size)
    
def Case_WithoutKnowledge_ConstReward_ReducingEpsilon():
    LinearFactor = 0.0
    LinearStartValue = 1
    Steps_Number = 30000
    Batch_size = 100
    Bandit_array = np.array([[0.1, 0.4], [0.3, 0.2]])
    #Bandit_array = np.array([[0.3, 0.3], [0.3, 0.3]])
    
    player_type = Player_WithoutCaseKnowledge
    reward_fun = linear_fun(LinearFactor, LinearStartValue)
    epsilon_fun = epsilon_reducing_fun(Steps_Number, 3)
        
    PerformAnExperiment(player_type,
                        Bandit_array,
                        reward_fun,
                        epsilon_fun,
                        Steps_Number,
                        Batch_size)
    
def Case_WithoutKnowledge_LinearReward_ReducingEpsilon():
    LinearFactor = 0.01
    LinearStartValue = 1
    Steps_Number = 20000
    Batch_size = 100
    Bandit_array = np.array([[0.1, 0.4], [0.3, 0.2]])
    #Bandit_array = np.array([[0.3, 0.3], [0.3, 0.3]])
    
    player_type = Player_WithoutCaseKnowledge
    reward_fun = linear_fun(LinearFactor, LinearStartValue)
    epsilon_fun = epsilon_reducing_fun(Steps_Number, 3)
        
    PerformAnExperiment(player_type,
                        Bandit_array,
                        reward_fun,
                        epsilon_fun,
                        Steps_Number,
                        Batch_size)
    
def Case_WithKnowledge_ConstReward_ReducingEpsilon():
    LinearFactor = 0.0
    LinearStartValue = 1
    Steps_Number = 20000
    Batch_size = 100
    Bandit_array = np.array([[0.1, 0.4], [0.3, 0.2]])
    #Bandit_array = np.array([[0.3, 0.3], [0.3, 0.3]])
    
    player_type = Player_WithCaseKnowledge
    reward_fun = linear_fun(LinearFactor, LinearStartValue)
    epsilon_fun = epsilon_reducing_fun(Steps_Number, 3)        
    PerformAnExperiment(player_type,
                        Bandit_array,
                        reward_fun,
                        epsilon_fun,
                        Steps_Number,
                        Batch_size)
    
def Case_WithKnowledge_LinearReward_ReducingEpsilon():
    LinearFactor = 0.01
    LinearStartValue = 1
    Steps_Number = 20000
    Batch_size = 100
    Bandit_array = np.array([[0.1, 0.4], [0.3, 0.2]])
    #Bandit_array = np.array([[0.3, 0.3], [0.3, 0.3]])
    
    player_type = Player_WithCaseKnowledge
    reward_fun = linear_fun(LinearFactor, LinearStartValue)
    epsilon_fun = epsilon_reducing_fun(Steps_Number, 3)
        
    PerformAnExperiment(player_type,
                        Bandit_array,
                        reward_fun,
                        epsilon_fun,
                        Steps_Number,
                        Batch_size)
    