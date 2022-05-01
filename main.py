from agent import Agent
import itertools
import gym

env = gym.make('LunarLander-v2')

obs_shape = env.observation_space.sample().shape[0]
act_shape = env.action_space.n

agent = Agent(alpha=0.1e-5, beta=0.5e-3, input_dims=obs_shape, n_actions=act_shape)

for episode in itertools.count():
    observation = env.reset()
    done = False
    for timestep in itertools.count():
        if episode % 50 == 0:
            env.render()
        action = agent.choose_action(observation)
        observation_next, reward, done, info = env.step(action)
        agent.learn(observation, action, reward, observation_next, done)

        observation = observation_next

        if done:
            print(agent.report.report_episode())
            break
