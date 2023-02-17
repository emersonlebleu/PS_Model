import os
import pygame as ps
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
    def __init__(self):
        pass


            
        
