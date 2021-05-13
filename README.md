# Project 3: Collaboration and Competition 

This repository contains the third and final project of the [Deep Reinforcement Learning Nanodegree Program](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893), by Udacity.

## Introduction

In this [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

![Training an agent to maintain its position at the target location for as many time steps as possible.](tennis.png)

### The environment

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. 
Each agent receives its own, local observation. 

Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, 
after taking the maximum over both agents). Specifically:

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. 
- This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. 
- This yields a single score for each episode. The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

In this repository, the tennis environment has been solved with a DDPG algorithm.

## Getting started

### Installation requirements

- Python 3.6 / PyTorch 0.4.0 environment creation: follow the requirements described in the [Udacity repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)
- Clone this repository and have the files accessible in the previously set up Python environment

For this project, you will not need to install Unity. This is because Udacity has already built the environment for you, and you can download it from one of the links below. You need to only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)


    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

- Unzip the environment archive in the project's environment directory and set the path to the UnityEnvironment in the code.

## Instructions

### Files

1. Tennis.ipynb: the Jupyter notebook with the DDPG agent algorithm training
2. model_ok.py: the actor and critic neural networks
3. ddpg_agent_ok.py: the learning agent based on DDPG
4. `.pth` files: they contain the weights of previously trained agents.

### Training an agent
    
Run the `Tennis.ipynb` notebook and follow the steps in the code.

### Adjusting the Hyperparameters
Here is the list of all the hyperparameters with which you can play and see how the learning change based on them:

* n_episodes: Maximum number of training episodes
* max_t: Maximum number of time steps per episode
* random_seed: The number used to initialize the pseudorandom number generator
* gamma: Discount factor for expected rewards
* gamma_final: Final gamma discount factor
* gammma_rate: A rate (0 to 1) for increasing gamma
* tau: Multiplicative factor for the soft update of target weights
* tau_final: Final value of tau
* tau_rate: A rate (0 to 1) for decreasing tau
* update_every: The number of time steps between each updating of the neural networks
* num_updates: The number of times to update the networks at every update_every interval
* buffer_size: Replay buffer size
* batch_size: Minibatch size
* actor_fc1 and actor_fc2: sizes of the actor network's layers
* critic_fc1 and critic_fc2: sizes of the critic network's layers 
* lr_actor: Learning rate for the local actor's network
* lr_critic: Learning rate for the local critic's network
* noise_theta: theta for Ornstein-Uhlenbeck noise process
* noise_sigma: sigma for Ornstein-Uhlenbeck noise process
* noise_scale: scaling factor for the applied OU noise
  
