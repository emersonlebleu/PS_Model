import os
import pygame as ps
import numpy as np
from memory import Memory 

# A class called agent that will be used to control the stimuli and store experiences in the memory space
class Agent:
    def __init__(self, name: str, memory: Memory, actions: np.ndarray):
        self.name = name
        self.memory = memory
        self.actions = actions

    def input_coupler(self, input: np.ndarray):
        pass

    def observe(self, observations: np.ndarray):
        # will use the input_coupler to map input from the observations to the clips in memory space
        pass

    def reflect(self, starting_clip: np.ndarray):
        pass

    def output_coupler(self, actuators: np.ndarray):
        pass

    def act(self, actions: np.ndarray):
        # will use the output_coupler to map the actuators in memory space to actions available to the agent
        pass



            
        
