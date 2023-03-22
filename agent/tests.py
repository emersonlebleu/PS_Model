from agent import PSAgent
import random

agent = PSAgent(actions=["+", "-"], deliberation=0, reflection=1)
agent.add_clip_to_memory(clip=["happy"])
agent.add_clip_to_memory(clip=["sad"])

agent.clear_log()

reward = 0
for i in range(10):
    random_percept = random.choice([["happy"], ["sad"]])
    action = agent.observe_environment(observations=random_percept, reward=reward)
    if random_percept == ["happy"] and action == "+":
        reward = 1
    elif random_percept == ["sad"] and action == "-":
        reward = 1
    else:
        reward = 0
