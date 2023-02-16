import os
import pygame as ps
import numpy as np

class Clip:
    def __init__(self, percept, type="percept"):
        self.percept = percepts
        self.type = type
        self.connections = {}

    def add_connection(self, connection, edge):
        self.connections[connection] = edge