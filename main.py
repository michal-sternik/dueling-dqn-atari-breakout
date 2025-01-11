
import ale_py
from breakout import *

from agent import *
from model import *
import torch


gym.register_envs(ale_py)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

environment = DQNBreakout(device=device)

model = AtariNet(nb_actions=4)

# model.load_the_model()

agent = Agent(model=model,
              device=device,
              epsilon=1.0,
              min_epsilon=0.1,
              nb_warmup=5000, # was 5,000
              nb_actions=4,
              learning_rate=0.00002,
              memory_capacity=50000,
              batch_size=64)


agent.train(env=environment, epochs=100000, batch_identifier=0)

test_environment = DQNBreakout(device=device, render_mode='human')

agent.test(env=test_environment)







# Display the image
# display_observation_image(state)

