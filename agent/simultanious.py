from agent import PSAgent
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

agent = PSAgent(actions=["+", "-"], deliberation=0, reflection=0, k=.35)
agent.add_clip_to_memory(clip=["happy"])
agent.add_clip_to_memory(clip=["sad"])
agent.add_clip_to_memory(clip=["good"])
agent.add_clip_to_memory(clip=["bad"])
agent.add_clip_to_memory(clip=[":)"])
agent.add_clip_to_memory(clip=[":("])

goods = ["good", "happy", ":)"]
bads = ["bad", "sad", ":("]

agent.clear_log()

percent_correct = []
intervals = []
rolling_average = []
total_correct = 0
aggregate_interval = 5

reward = 0

for i in range(201):

    #calculate % correct every 10 trials
    if i !=0 and i % aggregate_interval == 0:
        percent_correct.append((total_correct/aggregate_interval)*100)
        intervals.append(i)
        rolling_average.append(np.mean(percent_correct[-aggregate_interval:]))
        total_correct = 0

    #run the trial
    random_percept = random.choice(["happy", "sad", "good", "bad", ":)", ":("])
    action = agent.observe_environment(observations=random_percept, reward=reward)
    if random_percept in goods and action == "+":
        reward = 1
        total_correct += 1
    elif random_percept in bads and action == "-":
        reward = 1
        total_correct += 1
    else:
        reward = 0

def graph_results(intervals, percent_correct, rolling_averages):       
    fig, ax = plt.subplots()
    ax.plot(intervals, percent_correct, label="% Correct", color="blue", linestyle="dotted")
    ax.plot(intervals, rolling_average, label="Rolling Average", color="green", linewidth=2)
    ax.axes.set_xlabel("Trials")
    ax.axes.set_ylabel("% Correct")
    plt.show()

graph_results(intervals, percent_correct, rolling_average)