
# Projective Simulation (PS) Model

An implementation of the projective simulation model. The implementation is derived from readings of the following papers:
- Briegel & De las Cuevas (2012)
- Mautner et al. (2015)
- Melnikov et al. (2017)
- Mofrad et al. (2020)

**Projective Simulation** is a neural network based machine learning model that is particularly suited for reinforcement learning. The model is a directed graph network of percieved stimuli collections (percepts) and actions (available behavior choices) of the agent. The theoretical formulation is as follows: 

1. Agents can percieve (take in) collections of perceptions, called percepts, at each iteration of the environment
2. If the agent has not percieved the current percept it may add it to its memory, otherwise the agent takes a "walk" through its memory 
starting at the current percept until an action is chosen.
    - A walk is taken after novel percpets as well.
    - Walks are probablistic jumps from percept to percept and/or percept to action in memory.
3. Action is chosen. Based on the environment the agent will recieve a reward, and the next state of the environment.

**CURRENTLY:** Implementation is being built out fully. In future updates will include a refactor of the code base in order to better partition the agent into appropriate components for use.
