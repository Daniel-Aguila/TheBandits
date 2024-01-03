import numpy as np
import cv2
import random

#Function to select one of the actions with the highest estimated value
def argmax(Qa):
    max_Qa = max(Qa)
    max_Qa_list = []
    for i,value in enumerate(Qa):
        if value == max_Qa:
            max_Qa_list.append(i)
    return random.choice(max_Qa_list) #randomly choose if multiple indixes have same value

def epsilon_greedy(Qa,epsilon):
    random_number = np.random.random()
    if random_number < epsilon:
        return random.choice(range(len(Qa)))
    else:
        return argmax(Qa)