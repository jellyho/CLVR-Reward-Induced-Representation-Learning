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
T = 5
lr = 0.001

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'current device is {device}')
# dataset
ds = MovingSpriteDataset(spec=spec)

####### model definition ######
encoder = Encoder(64) # shared
decoder  = Decoder(64)

for m in decoder.modules():
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')

optim_d = torch.optim.RAdam(params=decoder.parameters(), lr=lr, betas=(0.9, 0.999))
        
scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer=optim_d, lr_lambda=lambda epoch: 0.9999 ** epoch, last_epoch=-1, verbose=False)

####### end of the model definition ######

Epochs = 20000
losses_d = []
task_name = args.reward

encoder.load_state_dict(torch.load(f'./Results/encoder/encoder_{task_name}.pth'))

for e in range(Epochs):
    loss_d_epoch = 0
    # preprocessing

    data = ds[0]
    task = torch.from_numpy(data['rewards'][task_name][:])

    input_images = data['images'][:, 0, :, :]
    input_images = input_images[:, np.newaxis, :, :]
    input_images = torch.from_numpy(input_images)
    
    z = encoder(input_images)
        
    decoded = decoder(z.reshape(-1, 64, 1, 1).detach())
    loss_d = nn.MSELoss(reduction='mean')(input_images, decoded)

    optim_d.zero_grad()
    loss_d.backward()    
    optim_d.step()
    scheduler_d.step()
    
    loss_d_epoch += loss_d.item()
    losses_d.append(loss_d_epoch)
    
    print(f"epoch:{e} - loss_d:{np.mean(losses_d[-10:]):.5f} - lr:{optim_d.param_groups[0]['lr']:.8f}        ")

torch.save(decoder.state_dict(), f'Results/decoder/decoder_{task_name}.pth')
plot_and_save_loss_per_epoch_1(losses_d, f'decoder_{task_name} pretraining', 'decoder')