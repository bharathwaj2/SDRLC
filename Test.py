import numpy as np
from RL_Agent import Agent
import torch
import imageio
import datetime
import matplotlib.pyplot as plt
import os

torch.cuda.is_available()


'================                     Available environments                       ================'
environments = ['Particle mass', 'Particle mass multiple obs', 'Unicycle', 'Unicycle multiple obs']
name = environments[0]


'================               Choose the method RL-CBF or just RL                ================'
meth = 'RL-CBF'


'===========  Variable to denote if the environment has single (or) multiple obstacles  ==========='
MO = 0       # don't need to change, initialization is enough


'================     Choose the shape of single obstacles - circle (or) uneven     ================'
shape = 'uneven'


'================    Choose the environment you want to train/test the algorithm    ================'
if meth == 'RL-CBF':
    
    if name == 'Particle mass':
        from PMenv import ParticleMass
        env_name = 'Particle_mass_RL_CBF'
        env = ParticleMass(render_mode="rgb_array", method=meth)

    elif name == 'Particle mass multiple obs':
        from PM_MO import ParticleMass
        env_name = 'Particle_mass_MO_RL_CBF'
        env = ParticleMass(render_mode="rgb_array", method=meth)
        MO = 1

    elif name == 'Unicycle':
        from Uni_env import Unicycle
        env_name = 'Unicycle_RL_CBF'
        env = Unicycle(render_mode="rgb_array", method=meth)

    else:
        from Uni_MO import Unicycle
        env_name = 'Unicycle_MO_RL_CBF'
        env = Unicycle(render_mode="rgb_array", method=meth)
        MO = 1

    actions = env.action_space.shape[0]
    states = env.observation_space.shape

    directory = f'SDRLC - IEEE Space/trained_agents/DDPG/{env_name}'
    agent = Agent(alpha=0.0002, beta=0.001,
                input_dims=env.observation_space.shape, tau=0.02,
                batch_size=128, fc1_dims=128, fc2_dims=64,
                n_actions=env.action_space.shape[0], env_name=env_name, iter=1,k=5,l=1, directory=directory)

else:
    if name == 'Particle mass':
        from PMenv import ParticleMass
        env_name = 'Particle_mass_RL'
        env = ParticleMass(render_mode="rgb_array", method=meth)
    
    elif name == 'Particle mass multiple obs':
        from PM_MO import ParticleMass
        env_name = 'Particle_mass_MO_RL'
        env = ParticleMass(render_mode="rgb_array", method=meth)
        MO = 1

    elif name == 'Unicycle':
        from Uni_env import Unicycle
        env_name = 'Unicycle_RL'
        env = Unicycle(render_mode="rgb_array", method=meth)

    else:
        from Uni_MO import Unicycle
        env_name = 'Unicycle_MO_RL'
        env = Unicycle(render_mode="rgb_array", method=meth)
        MO = 1


    actions = env.action_space.shape[0]
    states = env.observation_space.shape

    directory = f'SDRLC - IEEE Space/trained_agents/DDPG/{env_name}'
    agent = Agent(alpha=0.0002, beta=0.001,
                input_dims=env.observation_space.shape, tau=0.02,
                batch_size=128, fc1_dims=128, fc2_dims=64,
                n_actions=env.action_space.shape[0], env_name=env_name, iter=1, directory=directory)



'===========  Create directory for mking GIF image of simulation  ==========='
gif = False
frames = []

gif_directory = f'SDRLC - IEEE Space/GIF/{env_name}'
if not os.path.exists(gif_directory):
    os.makedirs(gif_directory)

# Load the saved trained weights
agent.load_models()

# Number of test runs
iterations = 1

score_history = []
trajectory = []
elapsed = []
time = 0


for i in range(iterations):
    start = datetime.datetime.now()

    obs = env.reset()
    if not gif:
        xr, yr, xg, yg, xo, yo = env.plot_trajectory()
        trajectory.append([xr, yr])

    done = False
    score = 0

    while not done:
        action = agent.choose_action(obs)
        observation_, reward, done, steps, reason = env.step(action)
        
        score += reward
        obs = observation_
        
        if gif:
            frame = env.render()  
            frames.append(frame)
        else:
            x, y, xg, yg, xo, yo = env.plot_trajectory()
            trajectory.append([x, y])

    end = datetime.datetime.now()
    delta_time = end - start
    elapsed.append(delta_time.total_seconds())

    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if i >= 0:
        print('Episode ', i + 1, 'score: %.1f     ' % score, 'average score: %.1f   ' % avg_score, '\t  steps: ', steps, '\t  ', reason)

for n in range(0,iterations):
    time += elapsed[n]
avg = time/iterations

print(f'Average time for {iterations} runs : {round(avg,3)}s')


# Save frames to GIF
if gif:
    imageio.mimsave(f'{gif_directory}/{env_name}.gif', frames, fps=30)

# Plot the trajectory
else:
    fig, ax = plt.subplots(figsize=(6,6))
    x_lim = [xr - 1, xg + 1]
    y_lim = [yr - 1, yg + 1]

    plt.xlim((x_lim[0], x_lim[1]))
    plt.ylim((y_lim[0], y_lim[1]))

    plt.xlabel("X-axis (m)")
    plt.ylabel("Y-axis (m)")
    plt.title(f'{name}')

    plt.grid("true")

    ax.scatter(xr, yr, linewidth=5, color='blue', label='Start')

    lh = 0.4
    k1 = [xg - lh, xg + lh, xg + lh, xg - lh, xg - lh]
    k2 = [yg - lh, yg - lh, yg + lh, yg + lh, yg - lh]

    ax.fill(k1, k2, color='green', label='Goal', alpha=0.7)

    if not MO:
        if shape == 'uneven':
            obstacle = np.load('PINN - IEEE Oceans/Trajectories/Uneven_shape_points_RL.npy')
            xo = 3 + (obstacle[:, 0:1]).reshape((len(obstacle[:, 0:1]),))
            yo = 3 + (obstacle[:, 1:2]).reshape((len(obstacle[:, 0:1]),))
            plt.fill_between(xo, yo, color='grey', label='Obstacle')
        else:
            circle = plt.Circle([xo, yo], 0.5, color='grey', label='Obstacle')
            ax.add_artist(circle)
    else:
        for i in range(0, len(xo)):
            if i == 0:
                circle = plt.Circle([xo[i], yo[i]], 1, color='grey', label='Obstacle')
            else:
                circle = plt.Circle([xo[i], yo[i]], 1, color='grey')
            ax.add_artist(circle)

    xc, yc = zip(*trajectory)
    ax.plot(xc, yc, color='black', linewidth=3.5, label=f'{meth} trajectory')

    plt.legend(loc='upper left')
    plt.show()
