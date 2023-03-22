from agent import PSAgent
import random

agent = PSAgent(actions=["+", "-"], deliberation=0, reflection=1)
agent.add_clip_to_memory(clip=["happy"])
agent.add_clip_to_memory(clip=["sad"])

agent.clear_log()

percent_correct = []
total_correct = 0

reward = 0
for i in range(41):

    #calculate % correct every 10 trials
    if i !=0 and i % 10 == 0:
        percent_correct.append((total_correct/10)*100)
        total_correct = 0

    #run the trial
    random_percept = random.choice([["happy"], ["sad"]])
    action = agent.observe_environment(observations=random_percept, reward=reward)
    if random_percept == ["happy"] and action == "+":
        reward = 1
        total_correct += 1
    elif random_percept == ["sad"] and action == "-":
        reward = 1
        total_correct += 1
    else:
        reward = 0
        
print(percent_correct)