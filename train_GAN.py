import os
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
from models.unet_generator import UNetGenerator
from models.patch_discriminator import PatchDiscriminator
from dataset import FabricDataset


# Configuracion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 256
batch_size = 8
lr = 1e-4
epochs = 30
data_dir = r"C:\Users\VitaliiPavlov\PycharmProjects\Pet_Project_GAN\nondefects.7z"  # ruta a la carpeta con imágenes de tejido normal.


# Transformaciones
transform = Compose([
    Resize((image_size, image_size)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0,1] -> [-1,1] (para Tanh)
])

# Dataset y DataLoader
train_dataset = FabricDataset(data_dir, transform=transform)
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Los modelos
model_gen = UNetGenerator().to(device)
model_dis = PatchDiscriminator().to(device)

# Optimizador y funcion de perdida
optimizer_gen = optim.Adam(model_gen.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_dis = optim.Adam(model_dis.parameters(), lr=lr, betas=(0.5, 0.999))
loss_func = nn.BCEWithLogitsLoss()
loss_recon = nn.L1Loss()

loss_gen_lst = []
loss_dis_lst = []


for _e in range(epochs):     # Ciclo de aprendizaje
    loss_mean_gen = 0
    loss_mean_dis = 0
    lm_count = 0


    train_tqdm = tqdm(train_data, leave=True)
    for real_imgs_batch in train_tqdm:

        real_imgs_batch = real_imgs_batch.to(device)
        img_gen = model_gen(real_imgs_batch)
        fake_out = model_dis(img_gen)

        targets_1 = torch.full_like(fake_out, 0.9)

        loss_gen_gan = loss_func(fake_out, targets_1)

        # Pérdida de L1 entre la imagen de entrada y la generada (para preservar la estructura)
        loss_gen_l1 = loss_recon(img_gen, real_imgs_batch)

        # Pérdida completa del generador con peso para L1 (por ejemplo 100)
        loss_gen = loss_gen_gan + 100 * loss_gen_l1

        optimizer_gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()


        img_gen_detached = img_gen.detach()
        real_out = model_dis(real_imgs_batch)
        fake_out = model_dis(img_gen_detached)

        targets_1 = torch.full_like(real_out, 0.9)
        targets_0 = torch.full_like(fake_out, 0.1)

        outputs = torch.cat([real_out, fake_out], dim=0)
        targets = torch.cat([targets_1, targets_0], dim=0)

        loss_dis = loss_func(outputs, targets)

        optimizer_dis.zero_grad()
        loss_dis.backward()
        optimizer_dis.step()

        lm_count += 1
        loss_mean_gen = 1 / lm_count * loss_gen.item() + (1 - 1 / lm_count) * loss_mean_gen
        loss_mean_dis = 1 / lm_count * loss_dis.item() + (1 - 1 / lm_count) * loss_mean_dis

        train_tqdm.set_description(f"Epoch [{_e + 1}/{epochs}], loss_mean_gen={loss_mean_gen:.3f}, loss_mean_dis={loss_mean_dis:.3f}")


    loss_gen_lst.append(loss_mean_gen)
    loss_dis_lst.append(loss_mean_dis)

    if (_e + 1) % 5 == 0:  # Cada 5 epochs
        torch.save({
            'epoch': _e + 1,
            'model_gen_state_dict': model_gen.state_dict(),
            'optimizer_gen_state_dict': optimizer_gen.state_dict(),
            'model_dis_state_dict': model_dis.state_dict(),
            'optimizer_dis_state_dict': optimizer_dis.state_dict(),
            'loss_gen': loss_mean_gen,
            'loss_dis': loss_mean_dis,
        }, f'train_saving_data_epoch_{_e + 1}.tar')


st = model_gen.state_dict()
torch.save(st, 'model_gen.tar')

st = model_dis.state_dict()
torch.save(st, 'model_dis.tar')

st = {'loss_gen': loss_gen_lst, 'loss_dis': loss_dis_lst}
torch.save(st, 'model_gan_losses.tar')


plt.plot(loss_gen_lst, label="Generator loss")
plt.plot(loss_dis_lst, label="Discriminator loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Losses")
plt.savefig("loss_plot.png")
plt.close()
