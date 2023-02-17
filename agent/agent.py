import os
import pygame as ps
import numpy as np

# A class called agent that will be used to control the stimuli and store experiences in the memory space
class PSAgent:
    """
    Memory
        Clips 
        #Clips are the transitions themselves they do not appear to be literally embodied in the memory space
            Percepts
            Actions (A)

    ##---Vector Values---##
        Edges (h_c)
            The weight of the edge between two percepts
        Edges (h_a)
            The weight of the edge between two actions
        Glow (g)

        Emotion (e)
            Whether or not the path traversed was successful recently or not
    
    ##---Scalar Values---##
        Reflection (r)
            The number of cycles the agent has to find the answers
        Dampening (d)
            Forgetting or connection discounting rate
        Hopping Probability (p)
            For each edge it would be the sum the weight devided by the sum of all the weights of the edges connected to the same node

    """
    def __init__(self):
        pass


            
        
