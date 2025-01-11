
import ale_py

from breakout import *
from agent import *
from model import *
import torch



input_file = '(best_model)model_iter2_8000_14.07.pth'

gym.register_envs(ale_py)

print(f'Using file: {input_file}')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = AtariNet(nb_actions=4)

model.load_the_model(weights_filename=input_file)

agent = Agent(model=model,
              device=device,
              epsilon=0.00,
              min_epsilon=0.00,
              nb_warmup=50, # originally 10000
              nb_actions=4,
              memory_capacity=20000,
              batch_size=32)

test_environment = DQNBreakout(device=device, render_mode='human')

agent.test(env=test_environment)



