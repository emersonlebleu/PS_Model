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
        #Clips are tuples of percepts feed together

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
    def __init__(self, g_edge=False, g_clip=False, emotion=False, softmax=False, reflection=0, decay_h=0, decay_g=0, actions = []):
        self.g_edge = g_edge
        self.g_clip = g_clip
        self.emotion = emotion
        self.reflection = reflection
        self.decay_h = decay_h
        self.decay_g = decay_g

        #The memory space of action and percepts is a dictionary of clips because we will be looking up clips frequently (O(1))
        self.clip_space = {}
        self.clip_index = 0
        self.clip_clip_matrix = np.zeros((3, 1, 1))

        self.action_space = {}
        self.action_index = 0        
        self.clip_action_matrix = np.zeros((3, 1, 1))
        self.__init_action_space(actions)

        self.last_path_taken = [] #A list of the indexes of the clips taken in the last walk

    def __init_action_space(self, actions):
        for action in actions: 
            self.action_space[action] = self.action_index
            self.action_index += 1
        
        self.clip_action_matrix = np.full((3, 1, len(actions)), 0)
        self.clip_action_matrix[0] = np.full((1, len(actions)), 1)

    def observe_environment(self, observations=(), reward=0):
        """
        The agent observes accepts inputs from the environment and processes them

        1. Check if the clip is in the clip space
            1.1 If it is not in the clip space, add it to the clip space
        2. Get the index of the clip in the clip space
        3. Reward the previous clip walk
        4. Take the next action

        """
        if len(observations) == 0:
            print("No observations were given to the agent")
            return
        
        if tuple(observations) not in self.clip_space:
            self.add_clip_to_memory(clip = tuple(observations))
            #in the future we may want to add a glow to the newly added clip before we move on to rewarding previous perception jumps
        else:
            #get the index of the clip in the clip space
            clip_index = self.clip_space[tuple(observations)]

        #Reward the previous clip walk
        #self.update_weights(percepts, actions, reward)

    def take_action(self, observations):
        """
        The agent takes an action based on the current state of the environment
        """
        return 0

    def add_clip_to_memory(self, clip = ()):
        #Add the clip to the clip space
        self.clip_space[clip] = self.clip_index
        self.clip_index += 1
        
        if self.clip_index <= 1:
            self.clip_clip_matrix = np.full((3, 1, 1), 0)
            self.clip_clip_matrix[0] = np.full((1, 1), 1)

            self.clip_action_matrix = np.full((3, 1, self.clip_action_matrix.shape[2]), 0)
            self.clip_action_matrix[0] = np.full((1, self.clip_action_matrix.shape[2]), 1)
        else:
            #get the current index of the clip_clip_matrix for reference
            clip_row_index = self.clip_clip_matrix.shape[1]
            clip_column_index = self.clip_clip_matrix.shape[2]

            self.clip_clip_matrix = np.append(self.clip_clip_matrix, np.full((3, self.clip_clip_matrix.shape[1], 1), 0), axis=2)
            self.clip_clip_matrix = np.append(self.clip_clip_matrix, np.full((3, 1, self.clip_clip_matrix.shape[2]), 0), axis=1)

            self.clip_clip_matrix[0, clip_row_index:, :] = 1
            self.clip_clip_matrix[0, :clip_row_index, clip_column_index:] = 1

            #Add new fields to the clip_action_matrix
            action_row_index = self.clip_action_matrix.shape[1]
            action_column_index = self.clip_action_matrix.shape[2]

            self.clip_action_matrix = np.append(self.clip_action_matrix, np.full((3, 1, self.clip_action_matrix.shape[2]), 0), axis=1)

            self.clip_action_matrix[0, action_row_index:, :] = 1

    def add_action_to_memory(self, action):
        self.action_space[action] = self.action_index
        self.action_index += 1

        #Add new fields to the action_h_matrix
        action_row_index = self.clip_action_matrix.shape[1]
        action_column_index = self.clip_action_matrix.shape[2]

        self.clip_action_matrix = np.append(self.clip_action_matrix, np.full((3, self.clip_action_matrix.shape[1], 1), 0), axis=2)
        self.clip_action_matrix = np.append(self.clip_action_matrix, np.full((3, 1, self.clip_action_matrix.shape[2]), 0), axis=1)

        self.clip_action_matrix[0, action_row_index:, :] = 1
        self.clip_action_matrix[0, :action_row_index, action_column_index:] = 1

    def update_weights(self, percept_indices: list, action_indices: list, reward: int, glowing_percepts: list = [], glowing_actions: list = []):
        """
        Update the weights of the agent's memory
            Takes a list of the visited percepts and actions during the last cycle and updates the weights based on the reward given
            uses the decay factor
            also uses the emotion factor
            uses clip/edge glow if applicable

            each weight is updated by the following formula traditionally if rewards are allways positive:
                h_t_plus_1 = h - (decay_h * (h - 1) + reward) ##if the clip was traversed

            Can use softmax as well to account for negative rewards as needed:

        """
        if self.g_edge or self.g_clip:
            #edge glow
                #update the glowing edges to some extent based on the reward? is that what they did? or is that just what i want to do because 
                #that makes sense to me?
            #clip glow
                #update the glowing clips weights to some extent based on the reward
            pass

agent = PSAgent(actions=["+", "-"])
agent.add_clip_to_memory(clip=(1, 2, 3))
print("Memory Space:")
print(agent.clip_space)
print(agent.action_space)
print("Percept H Matrix:")
print(agent.clip_clip_matrix)
print("Action H Matrix:")
print(agent.clip_action_matrix)

print("Adding One more:")
agent.add_action_to_memory("tap")
print("Memory Space:")
print(agent.clip_space)
print(agent.action_space)
print("Percept H Matrix:")
print(agent.clip_clip_matrix)
print("Action H Matrix:")
print(agent.clip_action_matrix)

print("Adding 3 more:")
agent.add_clip_to_memory(clip=(7, 8, 9))
print("Memory Space:")
print(agent.clip_space)
print(agent.action_space)
print("Percept H Matrix:")
print(agent.clip_clip_matrix)
print("Action H Matrix:")
print(agent.clip_action_matrix)