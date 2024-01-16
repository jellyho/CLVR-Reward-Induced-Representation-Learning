import torch
import torch.nn as nn
from models import *
from sprites_datagen.moving_sprites import MovingSpriteDataset
from general_utils import AttrDict
from sprites_datagen.rewards import *
from plotter import *
import argparse
import random

parser = argparse.ArgumentParser(description='Reward')
parser.add_argument('-r', '--reward', help='Specify the reward')
args = parser.parse_args()

spec = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        shapes_per_traj=3,      # number of shapes per trajectory
        rewards=[VertPosReward, HorPosReward, AgentXReward, AgentYReward,
                 TargetXReward, TargetYReward], # total 6 tasks here
    )

# constants
N = 5
T = 30 - N
lr = 0.001

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'current device is {device}')
# dataset
ds = MovingSpriteDataset(spec=spec)
dataset_size = 1000
print('preparing dataset')
buffer = [ds[0] for _ in range(dataset_size)]
print('dataset prepared.')
batch_size = 16

####### model definition ######
encoder = Encoder(64) # shared
hidden = HiddenStateEncoder(encoder=encoder)  #shared
fre = FutureRewardsEstimator(hidden, N=N, T=T, Heads=1)

print(fre)

for m in fre.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(m.bias, 0.1)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)

optim = torch.optim.RAdam(params=fre.parameters(), lr=lr, betas=(0.9, 0.999))
        
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda epoch: 0.9999 ** epoch, last_epoch=-1, verbose=False)

####### end of the model definition ######

Epochs = 1000
losses = []
# task_name = 'vertical_position'
task_name = args.reward

for e in range(Epochs):
    loss_epoch = 0
    batch = random.sample(buffer, batch_size)
    for data in batch:
        # preprocessing
        optim.zero_grad()
        task = torch.from_numpy(data['rewards'][task_name][N:N+T])
        input_images = data['images'][:, 0, :, :]
        input_images = input_images[:, np.newaxis, :, :]
        input_images = torch.from_numpy(input_images)

        # learn for each tasks
        estimated_reward = fre(input_images)
        loss = nn.MSELoss(reduction='sum')(estimated_reward, task)
        loss.backward()
        optim.step()
    scheduler.step()

    loss_epoch += loss.item()
    losses.append(loss_epoch)
    
    print(f"epoch:{e} - loss:{np.mean(losses[-30:]):.5f} - lr:{optim.param_groups[0]['lr']:.8f}      ")

torch.save(encoder.state_dict(), f'Results/encoder/encoder_{task_name}.pth')
torch.save(fre.state_dict(), f'Results/encoder/fre_{task_name}.pth')
plot_and_save_loss_per_epoch_1(losses, f'encoder_{task_name} pretraining', 'encoder')