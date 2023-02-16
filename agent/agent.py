import os
import pygame as ps
import numpy as np
from memory import Memory 

# A class called agent that will be used to control the stimuli and store experiences in the memory space
class Agent:
    """
    The agent class will be used to control the stimuli and store experiences in the memory space

    """
    def __init__(self, name: str, memory: Memory, actions: list, reflection=1, dampening=0):
        self.name = name
        self.memory = memory
        self.actions = actions
        self.reflection = reflection

    def input_coupler(self, input):
        pass

    def observe(self, observations):
        # will use the input_coupler to map input from the observations to the clips in memory space
        pass

    def reflect(self, starting_clip):
        pass

    def output_coupler(self, actuators):
        pass

    def act(self, actions):
        # will use the output_coupler to map the actuators in memory space to actions available to the agent
        pass



            
        
