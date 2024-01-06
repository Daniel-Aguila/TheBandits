import numpy as np
import cv2
import random

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
def updateQ(old_estimate, learning_rate, reward):
    new_estimate = old_estimate + learning_rate * (reward-old_estimate)
    return new_estimate

if __name__ == "__main__":
    k = 10
    Q_values = np.zeros(k)
    action_counts = np.zeros(k)

    steps = 1000
    alpha = 0.01
    true_action_values = np.random.normal(0,1,k)

    epsilon_greedy = EpsilonGreedy(Q_values,0.01)

    for step in range(steps):
        action = epsilon_greedy.select_action()

        reward = np.random.normal(true_action_values[action], 1)
        action_counts[action] += 1

        Q_values[action] = updateQ(Q_values[action], alpha, reward)
        
    print("Estimated action values:", Q_values)
    print("True action values:", true_action_values)
    print("Optimal action:", np.argmax(true_action_values))
    print("Most selected action:", np.argmax(action_counts))

