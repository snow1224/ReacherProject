# DRL final project

## 介紹(Introduction)
* 我們使用unity環境，實現機器2個關節手臂接觸移動物體。
* 代理的手在目標位置的每一步都會提供 +0.1 的獎勵。
* 目標: 在盡可能多的時間步長內保持其在目標位置的位置。
* 當代理在 100 個連續episodes中平均得分 +30 時，環境被認為已解決。
* 觀察空間由 33 個變量組成，對應於手臂的位置、旋轉、速度和角速度。每個“動作”是一個有4個數字的向量，對應於適用於2個關節的扭矩。動作向量中的每個條目都必須是 `-1` 和 `1` 之間的數字。

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

1. H參數設定可以在`parameters.py`調整

2. 使用 `train_agent.py` 訓練

```
python train_agent.py
```

3. if you want to watch trained agent, follow the instructions in  `train_agent.py`. Saved weights for trained agent can be found in the `output\` folder.

### DDPG介紹

## DQN
### 設定與執行程式
1. 參數設定可以在 `train_agent_dqn.py` 調整

2. 使用 `train_agent_dqn.py` 訓練 

```
python train_agent_dqn.py
```

*	Max_EPISODES = 300, score>=30
*	Result 

![dqn_result](outputs/dqn_result.png)
![dqn_score](outputs/dqn_score.png)

### DQN介紹
RL的任務基本是低維度輸入、低維度輸出，這是因為高維的問題難度實在太大了，很難收斂。因此，有人提出了DQN這個專門處理離散Action演算法，使得輸入維度可以擴展到高維空間。

那我們該怎麼建立DQN，讓模型可以輸出Q值，又能輸出與最大Ｑ值對應的Aciton，達到在機械手臂的連續控制的目的呢？圖1為我們針對這個問題所提出的DQN架構。
![dqn_result](dqn_arch.png)
|:------:|
|圖1-DQN架構示意圖|

一般的DQN在128 relu的隱藏層後，就直接輸出Q，然後找與最大Q對應的action。但我們為了連續控制，我們引入了Advantage概念，也就是判斷每個動作在特定狀況下的優劣，而輸出action也其實就是Advantage最大的動作。關係式如下:

$Q(s,q)=A(s,a)+V(s)$

那我們該如何建立符合我們場域的A的矩陣呢，我們可以利用這個關係式完成：

$A(x,u|\theta^A)=-\dfrac{1}{2}(u-u(x|\theta^u))^TP(x|\theta^P)(u-u(x|\theta^u))$

整體的演算法的流程如下所示：

![dqn_algorithm](dqn_algorithm.png)
 
## 參考資料
* [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971)
* [DQN从入门到放弃7 连续控制DQN算法-NAF](https://zhuanlan.zhihu.com/p/21609472)
* [Deep Cue Learning: A Reinforcement Learning Agent for Pool](https://github.com/pyliaorachel/CS229-pool)
* [pytorch-madrl](https://github.com/ChenglongChen/pytorch-DRL)



