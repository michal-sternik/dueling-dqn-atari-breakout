import torch
import torch.nn as nn


class AtariNet(nn.Module):

    def __init__(self, nb_actions=4):

        super(AtariNet, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(p=0.2)


        # rozmiar wyjścia z konwolucji dla inputu 84x84:
        # conv1: (84-8)/4+1=20 -> 20x20x32 = 12800
        # conv2: (20-4)/2+1=9 -> 9x9x64 = 5184
        # conv3: (9-3)/1+1=7 -> 7x7x64 = 3136

        self.action_value1 = nn.Linear(3136, 1024)

        self.action_value2 = nn.Linear(1024, 1024)

        self.action_value3 = nn.Linear(1024, nb_actions)

        self.state_value1 = nn.Linear(3136, 1024)

        self.state_value2 = nn.Linear(1024, 1024)

        self.state_value3 = nn.Linear(1024, 1)


    def forward(self, x):
        x = torch.Tensor(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        #state value
        state_value = self.relu(self.state_value1(x))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value2(state_value))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value3(state_value))

        #action advantage
        action_value = self.relu(self.action_value1(x))
        action_value = self.dropout(action_value)
        action_value = self.relu(self.action_value2(action_value))
        action_value = self.dropout(action_value)
        action_value = self.relu(self.action_value3(action_value))
        #Q = V(s) + A(s,a) - mean(A(s,a))
        output = state_value + (action_value - action_value.mean())

        return output

    def save_the_model(self, weights_filename='models/latest.pt'):
        torch.save(self.state_dict(), weights_filename)


    def load_the_model(self, weights_filename='models/latest.pt'):
        try:
            self.load_state_dict(torch.load(weights_filename))
            print(f"Successfully loaded weights file {weights_filename}")
        except:
            print(f"No weights file available at {weights_filename}")



