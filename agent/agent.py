import os
import numpy as np
import random

# A class called agent that will be used to control the stimuli and store experiences in the memory space
class PSAgent:
    """
    Based in part on ideas from:
        Briegel & De las Cuevas (2012)
        Mautner et al. (2015)
        Melnikov et al. (2017)
        Mofrad et al. (2020)

    NOTE: Matrix Values
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
    
    NOTE: Scalar Values
        Reflection (r)
            The number of cycles the agent has to find the answer
        Deliberation (d)
            The number of hops between clips the agent can take before it has to choose an action
        Dampening/Decay (d_h)
            Forgetting or connection discounting rate
        Glow_Decay/Dampening (d_g)
            Rate of glow decay
        Associative Groth (k)
    """
    #TODO: use matrix multiplication to speed up the process anywhere possible

    def __init__(self, g_edge=False, g_clip=False, emotion=False, probability_type="traditional", reflection=0, deliberation=0, decay_h=0, decay_g=0, k=.25, actions = []):
        self.g_edge = g_edge
        self.g_clip = g_clip
        self.emotion = emotion
        self.reflection = reflection
        self.decay_h = decay_h
        self.decay_g = decay_g
        self.deliberation = deliberation
        self.probability_type = probability_type
        self.k = k

        self.log_file = "log.txt"

        #The memory space of action and percepts is a dictionary of clips because we will be looking up clips frequently (O(1))
        self.clip_space = {}
        self.clip_index = 0
        self.clip_clip_matrix = np.zeros((3, 1, 1), dtype= float)

        self.action_space = {}
        self.action_index = 0        
        self.clip_action_matrix = np.zeros((3, 1, 1), dtype= float)
        self.__init_action_space(actions)

        self.last_path_taken = [] #A list of the indexes of the clips taken in the last walk

    def __init_action_space(self, actions):
        for action in actions: 
            self.action_space[action] = self.action_index
            self.action_index += 1
        
        self.clip_action_matrix = np.full((3, 1, len(actions)), 0)
        self.clip_action_matrix[0] = np.full((1, len(actions)), 1)

    def observe_environment(self, observations=(), reward=0, terminated=False, truncated=False, info={}):
        """
        The agent observes accepts inputs from the environment and processes them

        1. Check if the clip is in the clip space
            1.1 If it is not in the clip space, add it to the clip space
        2. Get the index of the clip in the clip space
        3. Reward the previous clip walk
        4. Take the next action

        provides terminated truncated and info in order to use with the OpenAI Gym API
        """
        if len(observations) == 0:
            print("No observations were given to the agent")
            return
        
        if type(observations) is not tuple or type(observation) is not list:
            observations = (observations,)
            self.observations = observations
        elif type(observations) is list:
            observations = tuple(observations)
            self.observations = observations

        if tuple(observations) not in self.clip_space:
            self.add_clip_to_memory(clip = self.observations)
            #in the future we may want to add a glow to the newly added clip before we move on to rewarding previous perception jumps
            percept_index = self.clip_space[self.observations]
        else:
            #get the index of the clip in the clip space name percept use to pick the next action later
            percept_index = self.clip_space[self.observations]
        
        #Reward the previous clip walk if there is one
        if len(self.last_path_taken) > 0:
            #pop will remove the item at the index and return it now we can use last_action_index and self.update_weights to update the weights
            last_clip_walk = self.last_path_taken
            last_action_index = last_clip_walk.pop()
            self.update_weights(last_clip_walk, last_action_index, float(reward))

        #Take the next action which returns the index of the action taken & the path taken
        action_index, self.last_path_taken = self.take_action(percept_index)
        action = list(self.action_space.keys())[list(self.action_space.values()).index(action_index)]
        
        return action

    def take_action(self, percept_index):
        """
        The agent takes an action based on the current state of the environment
        
        percept is the index of the percept in the percept space
        if deliberation is < 0 then a clip walk will be taken
        if deliberation is >= 0 then an action from the action space will be taken based on the percept

        """
        action_index = 0
        remaining_jumps = self.deliberation
        remaining_reflections = self.reflection
        last_path_taken = []
        #to use if deliberation is < 0
        clip_index = percept_index
        
        #Couple out immediately if there is no more deliberation and no more reflection
        if remaining_jumps == 0 and remaining_reflections == 0:
            action_index, emotion_tag, path_pair = self.get_action(percept_index) #pick an action and observe emotion tag
            
            for index in path_pair:
                last_path_taken.append(index) #add the percept and action to the path taken

            self.log_memory(list(self.action_space.keys())[action_index], last_path_taken)
            return action_index, last_path_taken

        elif remaining_jumps > 0 and remaining_reflections == 0:
            #see if there is a positive emotion on the action chosen by percept
            action_index, emotion_tag, path_pair = self.get_action(percept_index) #pick an action and observe emotion tag
            
            for index in path_pair:
                    last_path_taken.append(index)
            
            if emotion_tag:
                #if there is a positive emotion then we will take the action
                self.log_memory(list(self.action_space.keys())[action_index], last_path_taken)
                return action_index, last_path_taken
            else:
                last_path_taken.pop() #remove the action from the path taken
                clip_index = last_path_taken[-1]

            #Take only one clip walk after checking if initial percept has a +emotion action (because there is no reflection)
            while remaining_jumps > 0:
                clip_index = np.random.choice(list(self.clip_space.values()), p=self.get_clip_probabilities(clip_index))
                last_path_taken.append(clip_index)

                remaining_jumps -= 1

                if remaining_jumps == 0:
                    action_index, emotion_tag, path_pair = self.get_action(clip_index) #pick an action

                    last_path_taken.append(action_index)

                    self.log_memory(list(self.action_space.keys())[action_index], last_path_taken)
                    return action_index, last_path_taken
        #If there is reflection
        else:
            while remaining_reflections > 0:
                #see if there is a positive emotion on the action chosen by percept
                action_index, emotion_tag, path_pair = self.get_action(percept_index) #pick an action and observe emotion tag
                
                for index in path_pair:
                        last_path_taken.append(index)
                
                remaining_reflections -= 1  

                if emotion_tag:
                    #if there is a positive emotion then we will take the action
                    self.log_memory(list(self.action_space.keys())[action_index], last_path_taken)
                    return action_index, last_path_taken                   
                elif remaining_jumps != 0: #there are jumps left
                    last_path_taken.pop() #remove the action from the path taken
                    clip_index = last_path_taken[-1]

                    #take a clip walk if we have jumps
                    while remaining_jumps > 0:
                        clip_index = np.random.choice(list(self.clip_space.values()), p=self.get_clip_probabilities(clip_index))
                        last_path_taken.append(clip_index)

                        remaining_jumps -= 1
                    
                    #get an action for our end clip
                    action_index, emotion_tag, path_pair = self.get_action(clip_index) #pick an action and observe emotion tag
                    
                    last_path_taken.append(action_index)

                    if emotion_tag:
                        #if there is a positive emotion then we will take the action
                        self.log_memory(list(self.action_space.keys())[action_index], last_path_taken)
                        return action_index, last_path_taken
                    if remaining_reflections == 0:
                        self.log_memory(list(self.action_space.keys())[action_index], last_path_taken)
                        return action_index, last_path_taken
                    else:
                        last_path_taken = []#reset and try again
                        remaining_jumps = self.deliberation
                elif remaining_reflections > 0: #there are no jumps left but there are reflections left
                    last_path_taken = []
                    remaining_jumps = self.deliberation
                else: #if there are no reflections left and no jumps left then we will take the action chosen by the percept
                    self.log_memory(list(self.action_space.keys())[action_index], last_path_taken)
                    return action_index, last_path_taken
                    

    def add_clip_to_memory(self, clip = ()):
        if type(clip) != tuple:
            clip = tuple(clip)
        #Add the clip to the clip space
        self.clip_space[clip] = self.clip_index
        self.clip_index += 1
        
        if self.clip_index <= 1:
            self.clip_clip_matrix = np.full((3, 1, 1), 0.0)
            self.clip_clip_matrix[0] = np.full((1, 1), 1.0)

            for i in range(self.clip_clip_matrix[0].shape[0]):
                self.clip_clip_matrix[0, i, i] = 0.0

            self.clip_action_matrix = np.full((3, 1, self.clip_action_matrix.shape[2]), 0.0)
            self.clip_action_matrix[0] = np.full((1, self.clip_action_matrix.shape[2]), 1.0)
        else:
            #get the current index of the clip_clip_matrix for reference
            clip_row_index = self.clip_clip_matrix.shape[1]
            clip_column_index = self.clip_clip_matrix.shape[2]

            self.clip_clip_matrix = np.append(self.clip_clip_matrix, np.full((3, self.clip_clip_matrix.shape[1], 1), 0.0), axis=2)
            self.clip_clip_matrix = np.append(self.clip_clip_matrix, np.full((3, 1, self.clip_clip_matrix.shape[2]), 0.0), axis=1)

            self.clip_clip_matrix[0, clip_row_index:, :] = 1.0
            self.clip_clip_matrix[0, :clip_row_index, clip_column_index:] = 1.0
            self.clip_clip_matrix[0, clip_row_index, clip_column_index] = 0.0

            #Add new fields to the clip_action_matrix
            action_row_index = self.clip_action_matrix.shape[1]

            self.clip_action_matrix = np.append(self.clip_action_matrix, np.full((3, 1, self.clip_action_matrix.shape[2]), 0.0), axis=1)

            self.clip_action_matrix[0, action_row_index:, :] = 1.0

    def add_action_to_memory(self, action):
        self.action_space[action] = self.action_index
        self.action_index += 1

        self.clip_action_matrix = np.append(self.clip_action_matrix, np.full((3, self.clip_action_matrix.shape[1], 1), 0.0), axis=2)

        self.clip_action_matrix[0, :, self.action_index] = 1.0

    def update_weights(self, percept_indices: list, action_index: int, reward: float):
        """
        Update the weights of the agent's memory
            Takes a list of the visited percepts and actions during the last cycle and updates the weights based on the reward given
            uses the decay factor
            also uses the emotion factor
            uses clip/edge glow if applicable

            each weight is updated by the following formula traditionally if rewards are allways positive:
                h_t_plus_1 = h - (decay_h * (h - 1) + reward) ##if the clip was traversed

            Can use softmax as well to account for negative rewards as needed:

            NOTE: planning on getting glowing clips rather than passing them in the update weights function
        """
        if type(reward) != float:
            reward = float(reward)

        if not self.g_edge and not self.g_clip:
            #NOTE: Using mautner et. al 2015 weight updates, it is more readable. Adapting with Briegel et al. 2012's use of k for the indirect walk
            clip_update_matrix = self.decay_h * (self.clip_clip_matrix[0] - 1)
            action_update_matrix = self.decay_h * (self.clip_action_matrix[0] - 1)

            self.clip_clip_matrix[0] = self.clip_clip_matrix[0] - clip_update_matrix#decay the clip_clip_matrix
            self.clip_action_matrix[0] = self.clip_action_matrix[0] - action_update_matrix#decay the clip_action_matrix

            #update the direct connection
            self.clip_action_matrix[0, percept_indices[0], action_index] += reward
            
            #update the emotion matrix
            if reward > 0:
                self.clip_action_matrix[1, percept_indices[0], :] *= 0
                self.clip_action_matrix[1, percept_indices[0], action_index] = 1
            else: 
                self.clip_action_matrix[1, percept_indices[0], :] *= 0
            
            #using reward rather than unity (Briegel et al. 2012 uses unity) expecting reward will be 1 or 0 can look at other rewards as well
            #update the indirect clip walk with K factor
            if len(percept_indices) > 1:
                prev_clip_index = 0
                for i in range(1, len(percept_indices)):
                    self.clip_clip_matrix[0, percept_indices[prev_clip_index], percept_indices[i]] += self.k * reward
                    prev_clip_index = i

                #update the indirect action walk with K factor
                self.clip_action_matrix[0, percept_indices[-1], action_index] += self.k * reward

                #update the emotion matrix
                if reward > 0:
                    self.clip_action_matrix[1, percept_indices[0], :] *= 0
                    self.clip_action_matrix[1, percept_indices[0], action_index] = 1
                else: 
                    self.clip_action_matrix[1, percept_indices[0], :] *= 0

        elif self.g_edge:
            #if the edget is in the percept indices then update the edge glow
            #then we can use the multiplication strategy to update the weights with the glow and decay
            #edge glow
                #update the glowing edges to some extent based on the reward? is that what they did? or is that just what i want to do because 
                #that makes sense to me?
            #clip glow
                #update the glowing clips weights to some extent based on the reward
            pass
        else:
            #clip glow
            pass
        
    def get_action_probabilities(self, percept_index: int):
        """
        Returns the probabilities of each action given a percept uses the type of probability to determine how to calculate the probabilities
        """
        if self.probability_type == "traditional":
            action_probabilities = self.clip_action_matrix[0, percept_index, :]/sum(self.clip_action_matrix[0, percept_index, :])
        
        #TODO: add softmax
        elif self.probability_type == "softmax":
                pass

        return action_probabilities

    def get_clip_probabilities(self, percept_index: int):
        """
        Returns the probabilities of each clip given a percept uses the type of probability to determine how to calculate the probabilities
        """
        if self.probability_type == "traditional":
            clip_probabilities = self.clip_clip_matrix[0, percept_index, :]/sum(self.clip_clip_matrix[0, percept_index, :])
        
        #TODO: add softmax
        elif self.probability_type == "softmax":
                pass

        return clip_probabilities

    def get_action(self, percept_index: int):
        """
        Returns the action to be taken given a percept
        """
        action_index = None
        emotion_tag = None
        path_couple = []

        #choices is going to choose from the actions based on the probabilities
        action_index = np.random.choice(list(self.action_space.values()), p=self.get_action_probabilities(percept_index))
        emotion_tag = self.clip_action_matrix[1, percept_index, action_index]
        path_couple = [percept_index, action_index]

        if emotion_tag == 1:
            emotion_tag = True
        else:
            emotion_tag = False

        return action_index, emotion_tag, path_couple

    def log_memory(self, action: str, path):
        """
        Logs the memory of the agent
        """
        log = open(file=self.log_file, mode="a")
        log.write("\nClips:\n")
        log.write(str(self.clip_space))
        log.write("\nActions:\n")
        log.write(str(self.action_space))
        log.write("\nPercept H Matrix:\n")
        log.write(str(self.clip_clip_matrix))
        log.write("\nAction H Matrix:\n")
        log.write(str(self.clip_action_matrix))

        log.write("\nObservation:\n")
        log.write(str(self.observations))
        log.write("\nAction Chosen:\n")
        log.write(str(action))
        log.write("\nPath Taken:\n")
        log.write(str(path))
        log.close()

    def clear_log(self):
        """
        Clears the log file
        """
        log = open(file=self.log_file, mode="w")
        log.write("")
        log.close()