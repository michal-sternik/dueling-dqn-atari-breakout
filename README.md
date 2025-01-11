Project focuses on training a **Dueling Deep Q-Learning (DQN)** agent to play Atari's Breakout (and others) game. The agent utilizes a replay memory buffer for sampling past experiences and implements an epsilon decay strategy for balancing exploration and exploitation during training. It employs max-pooling mechanism to process visual input and stabilize learning. The model architecture was based on:  
[Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas. “Dueling Network Architectures for Deep Reinforcement Learning” 2016] https://arxiv.org/pdf/1511.06581.  
The project includes both training and testing pipelines, and the results are placed below in form of mp4 videos. You can download pre-trained models.

**Kaboom:**

https://github.com/user-attachments/assets/f660b868-29bd-4168-8f1f-1928aa713698

**Breakout:**

https://github.com/user-attachments/assets/a30c8468-d231-4a53-b63f-45856bbdf0cc


To test Kaboom just rename the code in _breakout.py_ from this:

```python
env = gym.make('ALE/Breakout-v5', render_mode=render_mode, frameskip=1, repeat_action_probability=0.0)
```

To this:

```python
env = gym.make('ALE/Kaboom-v5', render_mode=render_mode, frameskip=1, repeat_action_probability=0.0)
```

And change model name to load in _test.py_
