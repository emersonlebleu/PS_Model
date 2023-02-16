import numpy as np

class Sensor:
    def __init__(self):
        '''
        The sensor class will be used to collect stimlui from the environment and convert them into percepts.

        Eventually there may be different types of sensor classes that will be used to collect different types of stimuli.
        '''
        #should this be a list or a tuple?
        self.observations = []