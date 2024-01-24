# CLVR Implementation Project - Progress Report

## Reward Induced Representation Learning

Starter Code Repo : <https://github.com/kpertsch/clvr_impl_starter>

![](/Results/model.png)

## 1. Implement the reward-indcued representation learning model
I contructed the model in [models.py](https://github.com/jellyho/CLVR_Impl_RIRL/blob/7906edb3949ef03c944951e9077b74523887ec1a/models.py#L91)

ANd I trained the model using the provided dataset with 6 given rewards.


[train_encoder_all_task.py](https://github.com/jellyho/CLVR_Impl_RIRL/blob/8cd4db4101ef8b9c0694cba546f904e20a1daf6f/train_encoder_all_task.py#L1)

```
python train_encoder_all_task.py
```


#### Loss Graph
![](/Results/encoder/encoder%20pretraining.png)

## 2. Visualizing the results

### 1) Training Encoders
I trained the model using only 1 reward (horizontal_position, vertical_position) each.

[train_encoder_single_task.py](https://github.com/jellyho/CLVR_Impl_RIRL/blob/8cd4db4101ef8b9c0694cba546f904e20a1daf6f/train_encoder_single_task.py#L1)

```
python train_encoder_single_task.py -r horizontal_position
python train_encoder_single_task.py -r vertical_position
```

#### Loss Graph - Encoder(horizontal_position)
![](/Results/encoder/encoder_horizontal_position%20pretraining.png)

#### Loss Graph - Encoder(vertical_position)
![](/Results/encoder/encoder_vertical_position%20pretraining.png)

### 2) Training Decoders
Using these pretrained encoder, I trained decoder for each encoder to see what happens.

[train_decoder.py](https://github.com/jellyho/CLVR_Impl_RIRL/blob/8cd4db4101ef8b9c0694cba546f904e20a1daf6f/train_decoder.py#L1)

```
python train_decoder.py -r horizontal_position
python train_decoder.py -r vertical_position
```
#### Loss Graph - Decoder(horizontal_position)
![](/Results/decoder/decoder_horizontal_position%20pretraining.png)

#### Loss Graph - Decoder(vertical_position)
![](/Results/decoder/decoder_vertical_position%20pretraining.png)

#### Results
The First row is **ground truth** of current state, Second row is decoded image by encoder-decoder only trained using **vertical reward**, and the Thrid row is decoded image by encoder-decoder only trained using **horizontal_reward**.
![](/images/encdec1.png)
![](/images/encdec2.png)

Circle Shape is the agent. And encoder-decoders are trained on the rewards based on target's position.

As you see, the decoded images contain information about their rewards. For example, see the third row of the image, the white part *contains* information on the horziontal coordinates of the agent, but on the ohter coordinates the information has *faded*.

Therefore, using this model structure, it could be seen that the representation learning containing information about rewards progressed well.

## 3. Implement RL Algorithm

I implemented SAC(Soft Acotr Critic) to compare the performance of image-scratch baseline and pre-trained encoder, and also oracle.

[sac.py](https://github.com/jellyho/CLVR_Impl_RIRL/blob/1ee4b380739a913e6b2b7eb7612015ceab1c7dad/sac.py#L215)

I first trained oracle version to see my implementation is correct.

Trianing code is [train_agent.py](https://github.com/jellyho/CLVR_Impl_RIRL/blob/1ee4b380739a913e6b2b7eb7612015ceab1c7dad/train_agent.py#L1)

```
python train_agent.py -t SpritesState-v0 -r . -m oracle
```

The result is shown below.
![](./images/Training_Results_oracle.png)

It seems like working well. 

The reason why agent not following well and keep staying at center more is the environment's time horizon is too short and target is keep moving around randomly. 

So for agent, it is efficient to stay at center to get high reward consistently.

The result of trained agent is shown below.

Testing code is [test_agent.py](https://github.com/jellyho/CLVR_Impl_RIRL/blob/1ee4b380739a913e6b2b7eb7612015ceab1c7dad/test_agent.py#L1)

```
python test_agent.py -m oracle -t SpritesState-v0 -d ./Results/agents -e 5000
```

<img src="./images/oracle.gif" alt="image" width="300" height="auto">

## 4. Train SAC with image-scratch baseline and pre-trained encoder.

Encoder for image-scratch version(CNN) is defined in [model.py](https://github.com/jellyho/CLVR_Impl_RIRL/blob/1ee4b380739a913e6b2b7eb7612015ceab1c7dad/models.py#L136)

SAC using CNN and Encoder version is defined in [sac.py(CNN)](https://github.com/jellyho/CLVR_Impl_RIRL/blob/1ee4b380739a913e6b2b7eb7612015ceab1c7dad/sac.py#L340) [sac.py(Encoder)](https://github.com/jellyho/CLVR_Impl_RIRL/blob/1ee4b380739a913e6b2b7eb7612015ceab1c7dad/sac.py#L353)

I trained three versions (oracle, cnn, encoder) in three environments(number of distractor 0, 1, 2)

```
python train_agent.py -m encoder -t Sprites-v0 -d ./Results/agents
python train_agent.py -m encoder -t Sprites-v1 -d ./Results/agents
python train_agent.py -m encoder -t Sprites-v2 -d ./Results/agents

python train_agent.py -m cnn -t Sprites-v0 -d ./Results/agents
python train_agent.py -m cnn -t Sprites-v1 -d ./Results/agents
python train_agent.py -m cnn -t Sprites-v2 -d ./Results/agents

python train_agent.py -m oracle -t SpritesState-v0 -d ./Results/agents
python train_agent.py -m oracle -t SpritesState-v1 -d ./Results/agents
python train_agent.py -m oracle -t SpritesState-v2 -d ./Results/agents
```

## 5. Results & Discussion

The results is shown below.

![](./Results/agents/Sprites-v0.png)
![](/images/Sprites-v0.gif)

As you see, oracle and encoder is trained well(slightly good for oracle). But, cnn is not quite well trained.

![](./Results/agents/Sprites-v1.png)
![](/images/Sprites-v1.gif)

Also orale and encoder trained well. But we can't see any progress of cnn.

![](./Results/agents/Sprites-v2.png)
![](/images/Sprites-v2.gif)

Encoder is slightly less performance than oracle.

So, we can see that pre-trained encoder helps RL algorithms to learn efficient, high performance(Almost same as oracle)

But, cnn version didn't seem to be learning.

So that, Reward Induced Representation Learing helps RL Agent to train efficiently because they have some information about ground truth state induced by meta tasks.