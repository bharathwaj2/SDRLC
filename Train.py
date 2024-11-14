import numpy as np
from RL_Agent import Agent
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


'================                     Available environments                       ================'
environments = ['Particle mass', 'Particle mass multiple obs', 'Unicycle', 'Unicycle multiple obs']
name = environments[0]


'================               Choose the method RL-CBF or just RL                ================'
meth = 'RL-CBF'


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


save_dir = f"trained_agents/DDPG/{env_name}"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Number of episodes you want to train the agent for
iterations = 6000


score_history = []
avg_score_history = []

n = 0
success = 0.0

# Initialize TensorBoard writer
writer = SummaryWriter(f'runs/DDPG/{env_name}')

for i in range(iterations):
    
    obs = env.reset()
    done = False 

    score = 0

    while not done:
        action = agent.choose_action(obs)
        observation_, reward, done, steps, reason = env.step(action)
        agent.remember(obs, action, reward, observation_, done)
        agent.learn()
        score += reward
        obs = observation_

    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    avg_score_history.append(avg_score)

    agent.save_models()

    # Log scores, average scores, and steps to TensorBoard
    writer.add_scalar('Reward/Score', score, i)
    writer.add_scalar('Reward/Average_Score', avg_score, i)
    writer.add_scalar('Steps/Steps_Per_Episode', steps, i)
    writer.add_scalar('Performance/Success_Rate', success, i)

    if i >= 0:
        print('Episode ', i+1, '\t  score: %.1f' % score, '\t  average score: %.1f ' % avg_score, '\t  steps: ', steps, '\t', reason, '\t', success, '%')

    if reason == 'goal reached':
        n += 1

    success = round(((n/i) * 100), 2)

    # If success percentage is above desired threshold then cut off training
    if success > 35:
        break

# Close the TensorBoard writer
writer.close()


# Plot the Average reward v/s Episode graph
x = [i+1 for i in range(0, len(avg_score_history))]
plt.plot(x, avg_score_history)
plt.title('Running average')
plt.show()





