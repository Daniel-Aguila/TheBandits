import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

#Function to select one of the actions with the highest estimated value
class EpsilonGreedy:
    def __init__(self, Q_values, epsilon=0):
        self.Q_values = Q_values
        self.epsilon = epsilon

    def _argmax(self):
        max_Q_value = max(self.Q_values)
        max_Q_values_list = []
        for i,value in enumerate(self.Q_values):
            if value == max_Q_value:
                max_Q_values_list.append(i)
        return random.choice(max_Q_values_list) #randomly choose if multiple indixes have same value

    def select_action(self): #Incorporate epsilon to explore
        random_number = np.random.random()
        if random_number < self.epsilon:
            return random.choice(range(len(self.Q_values)))
        else:
            return self._argmax()

#implement QtA which gives the Q value at a current action given a certain time
def getQtA(action, time, rewards_list, action_record):
    #rewards_list keeps a sum of each rewards at a certain action until prior to a time
    if time == 0:
        return 0 #no action has been taken yet since time is 0
    else:
        sum_rewards_until_t = sum(rewards_list[t][action] for t in range(time))
        num_times_action_recorded = action_record[:time].count(action)
        if num_times_action_recorded == 0:
            return 0 #avoids a division by 0
        Qta = sum_rewards_until_t / num_times_action_recorded
    return Qta 

#more space efficient of updating Q
def updateQ(old_estimate, step_size_n, reward):
    new_estimate = old_estimate + (1/step_size_n) * (reward-old_estimate)
    return new_estimate

#upper confidence bound action selection action
def UCB(Q_values, action_counts, time, c):
    epsilon = 1e-9
    action = np.argmax(Q_values + c * np.sqrt(np.log(time)/(action_counts) + epsilon))
    return action

def stationaryBandit(action_type):
    k = 5
    Q_values_realistic = np.zeros(k)
    Q_values_optimistic = np.full(k,5.0)
    action_counts = np.zeros(k)

    steps = 1000
    alpha = 0.01 #for fixed learning rate if wanting to use that instead of a step size 1/n
    #for non-stationary a learning rate is usually more beneficial while for stationary step size is more beneficial
    true_action_values = np.random.normal(0,1,k)

    epsilon_greedy = EpsilonGreedy(Q_values_realistic,0.1)
    optimal_action = np.argmax(true_action_values)
    optimal_action_selections = np.zeros(steps)
    rewards = np.zeros(steps)

    for step in range(steps):
        if action_type == "egreedy":
            action = epsilon_greedy.select_action()
        elif action_type == "UCB":
            if step == 0:
                action = 0
            else:
                action = UCB(Q_values_realistic,action_counts,step,2)

        reward = np.random.normal(true_action_values[action], 1)
        rewards[step] = reward
        action_counts[action] += 1
        if action == optimal_action:
            optimal_action_selections[step] = 1
        else:
            optimal_action_selections[step] = 0

        Q_values_realistic[action] = updateQ(Q_values_realistic[action], action_counts[action], reward)
        
    print("Estimated action values:", Q_values_realistic)
    print("True action values:", true_action_values)
    print("Optimal action:", np.argmax(true_action_values))
    print("Most selected action:", np.argmax(action_counts))


    #Plot
    optimal_percentage = np.cumsum(optimal_action_selections) / (np.arange(steps) + 1)

    plt.figure(figsize=(12, 6))

    plt.plot(optimal_percentage, label='Optimal Action %')
    plt.title('% ' + action_type + 'Over Time')
    plt.xlabel('Step')
    plt.ylabel('Optimal Action %')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    stationaryBandit(action_type="UCB")

