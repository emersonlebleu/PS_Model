import os
import numpy as np

# A class called agent that will be used to control the stimuli and store experiences in the memory space
class PSAgent:
    """
    Informed by: 
        Briegel & De las Cuevas (2012)
        Mautner et al. (2015)
        Bjerland (2015) --Thesis but good overview of the model
        Melnikov et al. (2017)
        Mofrad et al. (2020)

    Memory
        Clips 
        #Clips are the transitions themselves they do not appear to be literally embodied in the memory space
            Percepts (P)
            Actions (A)

    ##---Matrix Values---##
        Edges (h_c)
            The weight of the edge between two percepts
        Edges (h_a)
            The weight of the edge between a percept and an action
        Glow (g_e) Also Called "afterglow" by Mautner et al. (2015)
            There can be edge and clip glow depending on the type of glow
            Edge glow is the "glow" on each edge of a previously traversed and successful clip walk
        Glow (g_c)
            Clip glow is the "glow" on the starting clip and ending clip only of a previously traversed and successful clip walk
        Emotion (e)
            Whether or not the path traversed was successful recently or not
        
        &&---Formal Similarity
        Mautner et al. (2015) talks about similarity but theirs can differ by "exactly one" component
        
        If we could find a way to do a weighted matrix of similarity... based on how much clips "differ" that would be better
        Maybe we could figure out how to make it impacted by whether the "similarity" is relevant to feedback or not? R+ or not
        &&---
    
    ##---Scalar Values---##
        Reflection (r)
            The number of cycles the agent has to find the answers
        Dampening/Decay (d_h)
            Forgetting or connection discounting rate
        Glow_Decay/Dampening (d_g)
            Rate of glow decay
        Hopping Probability (p)
            For each edge it would be the sum the weight devided by the sum of all the weights of the edges connected to the same node

    """
    def __init__(self, g_edge=False, g_clip=False, emotion=False, reflection=0, decay_h=0, decay_g=0):
        self.g_edge = g_edge
        self.g_clip = g_clip
        self.emotion = emotion
        self.reflection = reflection
        self.decay_h = decay_h
        self.decay_g = decay_g

        self.memory_space = np.zeros((1,))
        self.action_space = np.zeros((1,))

        self.percept_percept_matrix = np.zeros((3, 1, 1))
        self.action_percept_matrix = np.zeros((3, 1, 1))


    def observe_environment(self, observations):
        """
        The agent observes accepts inputs from the environment and processes them

        observations[0] will be the percepts
        observations[1] will be the actions possible at present
        observations[2] will be the reward
        """
        pass

    def add_to_memory(self, percepts: list = [], actions: list = []):
        self.memory_space = np.append(self.memory_space, percepts)
        self.action_space = np.append(self.action_space, actions)
        
        #Add new fields to the percept_h_matrix
        if self.percept_percept_matrix.shape[1] == 1:
            self.percept_percept_matrix = np.full((3, len(percepts), len(percepts)), 0)
            self.percept_percept_matrix[0] = np.full((len(percepts), len(percepts)), 1)
        else:
            percept_row_index = self.percept_percept_matrix.shape[1]
            percept_column_index = self.percept_percept_matrix.shape[2]

            self.percept_percept_matrix = np.append(self.percept_percept_matrix, np.full((3, self.percept_percept_matrix.shape[1], len(percepts)), 0), axis=2)
            self.percept_percept_matrix = np.append(self.percept_percept_matrix, np.full((3, len(percepts), self.percept_percept_matrix.shape[2]), 0), axis=1)

            self.percept_percept_matrix[0, percept_row_index:, :] = 1
            self.percept_percept_matrix[0, :percept_row_index, percept_column_index:] = 1
        
        #Add new fields to the action_h_matrix
        if self.action_percept_matrix.shape[1] == 1:
            self.action_percept_matrix = np.full((3, len(percepts), len(actions)), 0)
            self.action_percept_matrix[0] = np.full((len(percepts), len(actions)), 1)
        else:
            action_row_index = self.action_percept_matrix.shape[1]
            action_column_index = self.action_percept_matrix.shape[2]

            self.action_percept_matrix = np.append(self.action_percept_matrix, np.full((3, self.action_percept_matrix.shape[1], len(actions)), 0), axis=2)
            self.action_percept_matrix = np.append(self.action_percept_matrix, np.full((3, len(percepts), self.action_percept_matrix.shape[2]), 0), axis=1)

            self.action_percept_matrix[0, action_row_index:, :] = 1
            self.action_percept_matrix[0, :action_row_index, action_column_index:] = 1


agent = PSAgent()
agent.add_to_memory(actions = ["+", "-"], percepts=[1, 2, 3])
print("Memory Space:")
print(agent.memory_space)
print("Percept H Matrix:")
print(agent.percept_percept_matrix)
print("Action H Matrix:")
print(agent.action_percept_matrix)

print("Adding One more:")
agent.add_to_memory(percepts=[4])
print("Memory Space:")
print(agent.memory_space)
print("Percept H Matrix:")
print(agent.percept_percept_matrix)
print("Action H Matrix:")
print(agent.action_percept_matrix)

print("Adding 3 more:")
agent.add_to_memory(percepts=[5, 6, 7])
print("Memory Space:")
print(agent.memory_space)
print("Percept H Matrix:")
print(agent.percept_percept_matrix)
print("Action H Matrix:")
print(agent.action_percept_matrix)