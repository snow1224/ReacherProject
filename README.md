# DRL final project

## 介紹(Introduction)
我們使用unity環境，實現機器手臂移動物體的任務。
In a [Unity ML](https://github.com/Unity-Technologies/ml-agents) environment, we train a double-jointed arm `agent` to reach moving objects.  A reward of `+0.1` is provided for each step that the agent's hand is in the goal location.  The goal is to maintain its position at  the target location for as many time steps as possible. The task is episodic and the environment is considered solved when the agent manages to score `+30` on average over `100` consecutive episodes.

The observation space consists of `33` variables  corresponding to position, rotation, velocity, and angular velocities of the arm.  Each `action` is a vector with four numbers, corresponding to  torque applicable to two joints.  Every entry in the action vector must  be a number between `-1` and `1`.

We train the agent using [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) algorithm and the training process is documented [here](.\Report.md). This is how the trained agent looks like:

![trained_agent](outputs/agent_after.gif)

## 安裝環境
To run the code, you need a Python 3.6 environment with required dependencies installed.

1. 創建環境

```
conda create --name reacherproject python=3.6
source activate reacherproject
```


2. 複製github repository和安裝requirements

```
git clone https://github.com/snow1224/ReacherProject.git
cd ReacherProject
pip install -r requirements.txt
conda install pytorch=0.4.1 cuda92 -c pytorch
```
3. 下載unity環境，選擇對應的OS版本

- **_Version 1: One (1) Agent_**
  - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
  - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
  - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
  - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

- **_Version 2: Twenty (20) Agents_**
  - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
  - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
  - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
  - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

## DDPG

1. Hyperparameter settings can be tweaked  in `parameters.py`.

2. Train the agent by running `train_agent.py` 

```
python train_agent.py
```

3. if you want to watch trained agent, follow the instructions in  `train_agent.py`. Saved weights for trained agent can be found in the `output\` folder.

### DDPG介紹

## DQN
1. Hyperparameter settings can be tweaked  in `train_agent_dqn.py`.

### DQN介紹
*	DQN architecture settings can be tweaked in dqn_agent1.py.
*	Max_EPISODES = 300, score>=30
*	Result 
![dqn_result](outputs/dqn_result.png)
![dqn_score](outputs/dqn_score.png)