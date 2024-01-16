import torch
from models import *
from sprites_datagen.moving_sprites import MovingSpriteDataset
from general_utils import AttrDict
from sprites_datagen.rewards import *
from plotter import *

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'current device is {device}')
# dataset
ds = MovingSpriteDataset(spec=spec)

####### model definition ######
encoder1 = Encoder(64)
decoder1  = Decoder(64)

encoder2 = Encoder(64)
decoder2  = Decoder(64)
# reward estimator for a task
####### end of the model definition ######

task_name = ['horizontal_position', 'vertical_position']

encoder1.load_state_dict(torch.load(f'./Results/encoder/encoder_{task_name[0]}.pth'))
decoder1.load_state_dict(torch.load(f'./Results/decoder/decoder_{task_name[0]}.pth'))

encoder2.load_state_dict(torch.load(f'./Results/encoder/encoder_{task_name[1]}.pth'))
decoder2.load_state_dict(torch.load(f'./Results/decoder/decoder_{task_name[1]}.pth'))

while True:
    with torch.no_grad():
        data = ds[0]

        input_images = data['images'][:, 0, :, :]
        input_images = input_images[:, np.newaxis, :, :]
        input_images = torch.from_numpy(input_images)

        # learn for each tasks
        
        z1 = encoder1(input_images)
        decoded1 = decoder1(z1.reshape(-1, 64, 1, 1))

        z2 = encoder2(input_images)
        decoded2 = decoder2(z2.reshape(-1, 64, 1, 1))

        select_idx = [1, 3, 6, 10, 15, 21]
        plot = torch.cat([input_images[select_idx, :, :, :], decoded1[select_idx, :, :, :], decoded2[select_idx, :, :, :]], dim=0)
        plot = torch.squeeze(plot, dim=1)
        plot_imgs(plot, 6, 3, select_idx)