import torch
import torch.nn as nn
from models import *
from sprites_datagen.moving_sprites import MovingSpriteDataset
from general_utils import AttrDict
from sprites_datagen.rewards import *
from plotter import *
from general_utils import *

spec = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        shapes_per_traj=4,      # number of shapes per trajectory
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

dataset_size = 3000
print('preparing dataset')
buffer = [parse_dataset(ds[0], N, T) for _ in range(dataset_size)]
print('dataset prepared.')
batch_size = 16

####### model definition ######
encoder = Encoder(64) # shared
hidden = HiddenStateEncoder(encoder=encoder) #shared
# reward estimator for six tasks
fre = FutureRewardsEstimator(hidden, N=N, T=T, Heads=6) 

for m in fre.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')

optim = torch.optim.RAdam(params=fre.parameters(), lr=lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda epoch: 0.9999 ** epoch, last_epoch=-1, verbose=False)

####### end of the model definition ######

Epochs = 3000
losses = []

for e in range(Epochs):
    loss_epoch = 0
    batch = random.sample(buffer, batch_size)
    for data in batch:
        tasks, input_images = data

        # learn for each tasks
        optim.zero_grad()
        estimated_reward = fre(input_images)
        loss = nn.MSELoss(reduction='sum')(estimated_reward, tasks)
        loss.backward()
        optim.step()

        loss_epoch += loss
    losses.append(loss_epoch.item())
    scheduler.step()

    print(f"epoch:{e} - loss:{np.mean(losses[-30:]):.5f} - lr:{optim.param_groups[0]['lr']:.8f}")

torch.save(encoder.state_dict(), 'encoder_six.pth')
torch.save(fre.state_dict(), f'fre_six.pth')
plot_and_save_loss_per_epoch_1(losses, 'encoder pretraining', 'encoder')