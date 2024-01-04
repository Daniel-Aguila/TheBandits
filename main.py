import numpy as np
import cv2
import random

#Function to select one of the actions with the highest estimated value
class EpsilonGreedy:
    def __init__(self, Qa, epsilon):
        self.Qa = Qa
        self.epsilon = epsilon

    def _argmax(self):
        max_Qa = max(self.Qa)
        max_Qa_list = []
        for i,value in enumerate(self.Qa):
            if value == max_Qa:
                max_Qa_list.append(i)
        return random.choice(max_Qa_list) #randomly choose if multiple indixes have same value

    def epsilon_greedy(self): #Incorporate epsilon to explore
        random_number = np.random.random()
        if random_number < self.epsilon:
            return random.choice(range(len(self.Qa)))
        else:
            return self._argmax(self.Qa)

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
