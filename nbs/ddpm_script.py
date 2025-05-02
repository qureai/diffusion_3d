# %%
import segmentation_models_pytorch as smp
import torch
import torchvision
from einops import rearrange
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from tqdm.auto import tqdm
from vision_architectures.schedulers.noise import LinearNoiseScheduler

# %% [markdown]
# Load the datasets

# %%
transforms = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]
)

# %%
train_dataset = torchvision.datasets.MNIST(r"/raid/arjun/MNIST/", train=True, transform=transforms)
train_dataset

# %%
val_dataset = torchvision.datasets.MNIST(r"/raid/arjun/MNIST/", train=False, transform=transforms)
val_dataset

# %%
dataset = train_dataset + val_dataset
dataset, len(dataset)

# %% [markdown]
# Visualize the dataset

# %%
img, gt = dataset[4]

plt.imshow(rearrange(img, "c h w -> h w c"), cmap="gray")
plt.show()
gt

# %%
img, gt = dataset[60004]

plt.imshow(rearrange(img, "c h w -> h w c"), cmap="gray")
plt.show()
gt

# %% [markdown]
# Visualize some outputs

# %%
T = 200
num_timesteps = 11
timesteps = torch.linspace(1, T, num_timesteps)

x0, gt = dataset[0]
x0.unsqueeze_(0)
scheduler = LinearNoiseScheduler(T)

for i in range(num_timesteps):
    t = timesteps[i : i + 1].int()

    xt, noise = scheduler.add_noise(x0, t)

    x0_bar, xt_minus_1 = scheduler.remove_noise(xt, noise, t)

    fig, ax = plt.subplots(1, 3)

    print(f"t = {t}")
    # print(f'denoising mean = {mean}')

    ax[0].imshow(rearrange(x0, "1 c h w -> h w c"), cmap="gray")
    ax[0].set_title("t = 0")
    ax[1].imshow(rearrange(xt, "1 c h w -> h w c"), cmap="gray")
    ax[1].set_title(f"t = {int(t)}")
    ax[2].imshow(rearrange(xt_minus_1, "1 c h w -> h w c"), cmap="gray")
    ax[2].set_title(f"t = {int(t - 1)} (denoised)")
    plt.show()

    print()

# %% [markdown]
# Train model


# %%
class SimpleArchitecture(nn.Module):
    # Does not perform any time embedding

    def __init__(self):
        super().__init__()

        self.model = smp.Unet(
            encoder_name="resnet18",
            in_channels=1,
            classes=1,
            # encoder_depth=3,
            # decoder_channels=(64, 32, 16),
            # activation="tanh",
        )

    def forward(self, xt):
        return self.model(xt)


# %%
device = torch.device("cuda:0")

# %%
model = SimpleArchitecture()
model.to(device)
sum([parameter.numel() for parameter in model.parameters()])

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

# %%
T = 200
noise_scheduler = LinearNoiseScheduler(T)
noise_scheduler.to(device)

# %%
dataloader = torch.utils.data.DataLoader(dataset, batch_size=15000, shuffle=True, pin_memory=True)
len(dataloader)

# %%
num_epochs = 100
model.train()

losses = []
for _ in tqdm(range(num_epochs), "Epoch"):
    for batch in tqdm(dataloader, "Batch", leave=False):
        x0, gts = batch
        x0 = x0.to(device)
        x0 = x0 * 2 - 1  # rescale to be between -1 and 1

        t = torch.randint(1, T, (x0.shape[0],), device=device)

        xt, noise_gt = noise_scheduler.add_noise(x0, t)

        optimizer.zero_grad()
        noise_pred = model(xt)

        loss = criterion(noise_pred, noise_gt)
        loss.backward()
        optimizer.step()

    losses.append(loss.detach().cpu())

plt.plot(losses)
plt.show()

# %% [markdown]
# Generate images

# %%
model.eval()

xt = torch.randn((5, 1, 32, 32), device=device)

for t in reversed(range(1, T + 1)):
    t = torch.tensor([t], device=device)

    pred = model(xt)

    x0_bar, xt_minus_1 = noise_scheduler.remove_noise(xt, pred, t)
    xt = xt_minus_1  # This will be the new xt

    x0_bar = x0_bar.detach().cpu()
    xt_minus_1 = xt_minus_1.detach().cpu()

    if t % 40 == 0 or t == 1:
        print(f"t = {t}")
        fig, ax = plt.subplots(1, xt.shape[0])
        for i in range(xt.shape[0]):
            ax[i].imshow(rearrange(xt_minus_1[i], "c h w -> h w c"), cmap="gray")
        plt.show()

# %%


# %%
