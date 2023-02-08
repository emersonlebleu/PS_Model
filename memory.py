import os
import pygame as ps
import numpy as np

class Memory:
    def __init__(self, clips: np.ndarray, actuators: np.ndarray):
        self.clips_list = clips
        self.actuators_list = actuators

# Will not ultimately be implementing as a list, but for now it will be a list
    def add_clip(self, clip: np.ndarray):
        self.clips_list = np.append(self.clips_list, clip)

# Will not ultimately be implementing as a list, but for now it will be a list
    def add_actuator(self, actuator: np.ndarray):
        self.actuators_list = np.append(self.actuators_list, actuator)