import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image,make_grid
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import random

torch.cuda.empty_cache()

# Set random seed for reproducibility.
seed = 369
random.seed(seed)
torch.manual_seed(seed)


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
gen_lr = 0.0002
dis_lr = 0.0002
batch_size = 128
epochs = 6
image_channels = 3
noise_vector = 100
beta1 = 0.5
beta2 = 0.999
image_size = 128
gen_filter_size = 64
dis_filter_size = 32

# Load data
transforms = transforms.Compose([transforms.Resize(image_size),transforms.CenterCrop(image_size),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
train = datasets.ImageFolder('bitmoji', transform=transforms)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)


# plot the first 10 images in train
print("Sample images from the dataset\n")
fig = plt.figure(figsize=(25, 4))
for i in range(10):
    ax = fig.add_subplot(2, 20//2, i+1, xticks=[], yticks=[])
    plt.imshow(((train[i][0].permute(1, 2, 0)).numpy()*255).astype(np.uint8))
plt.show()


# Define the generator
class Generator(nn.Module):
    def __init__(self,num_in_channels,num_out_channels,filter_size):
        super(Generator,self).__init__()
        # Input: Batch_size x num_in_channels x 1 x 1
        self.conv0 = nn.Sequential(nn.ConvTranspose2d(in_channels=num_in_channels,out_channels=filter_size*16,kernel_size=4,stride=1,padding=0,bias=False),nn.BatchNorm2d(filter_size*16))
        # Input: Batch_size x filter_size*8 x 4 x 4
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(filter_size*16,filter_size*8,4,stride=2,padding=1,bias=False),nn.BatchNorm2d(filter_size*8))
        # Input: Batch_size x filter_size*4 x 8 x 8
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(filter_size*8,filter_size*4,4,stride=2,padding=1,bias=False),nn.BatchNorm2d(filter_size*4))
        # Input: Batch_size x filter_size*2 x 16 x 16
        self.conv3 = nn.Sequential(nn.ConvTranspose2d(filter_size*4,filter_size*2,4,stride=2,padding=1,bias=False),nn.BatchNorm2d(filter_size*2))
        # Input: Batch_size x filter_size x 32 x 32
        self.conv4 = nn.Sequential(nn.ConvTranspose2d(filter_size*2,filter_size,4,stride=2,padding=1,bias=False),nn.BatchNorm2d(filter_size))
        # output: Batch_size x num_out_channels x 64 x 64
        self.conv5 = nn.Sequential(nn.ConvTranspose2d(filter_size,num_out_channels,4,stride=2,padding=1,bias=False))
        # output: Batch_size x num_out_channels x 128 x 128

    def forward(self,x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.tanh(self.conv5(x))
        return x


class Discriminator(nn.Module):

    def __init__(self,num_in_channels,filters):
        super(Discriminator,self).__init__()
        # Input: Batch_size x num_in_channels x 128 x 128
        self.conv1 = nn.Sequential(nn.Conv2d(num_in_channels,filters,4,stride=2,padding=1,bias=False),nn.BatchNorm2d(filters))
        # Input: Batch_size x filters x 64 x 64
        self.conv2 = nn.Sequential(nn.Conv2d(filters,filters*2,4,stride=2,padding=1,bias=False),nn.BatchNorm2d(filters*2))
        # Input: Batch_size x filters*2 x 32 x 32
        self.conv3 = nn.Sequential(nn.Conv2d(filters*2,filters*4,4,stride=2,padding=1,bias=False),nn.BatchNorm2d(filters*4))
        # Input: Batch_size x filters*4 x 16 x 16
        self.conv4 = nn.Sequential(nn.Conv2d(filters*4,filters*8,4,stride=2,padding=1,bias=False),nn.BatchNorm2d(filters*8))
        # Input: Batch_size x filters*8 x 8 x 8
        self.conv5 = nn.Sequential(nn.Conv2d(filters*8,filters*16,4,stride=2,padding=1,bias=False),nn.BatchNorm2d(filters*16))
        # Input: Batch_size x filters*16 x 4 x 4
        self.conv6 = nn.Conv2d(filters*16,1,4,stride=1,padding=0,bias=False)
        # output: Batch_size x 1 x 1 x 1
        self.activation = nn.LeakyReLU(0.2,inplace=True)

    
    def forward(self,x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.activation(self.conv5(x))
        x = torch.sigmoid(self.conv6(x))
        return x


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


# # initialize generator and discriminator
gen = Generator(num_in_channels=noise_vector,num_out_channels=image_channels,filter_size=gen_filter_size).to(device)
dis = Discriminator(num_in_channels=image_channels,filters=dis_filter_size).to(device)


# initialize weights
initialize_weights(gen)
initialize_weights(dis)


# initialise loss fn and optimizer
loss_fn  = torch.nn.BCELoss()
gen_optimizer = optim.Adam(gen.parameters(),lr = gen_lr,betas=(beta1,beta2))
dis_optimizer = optim.Adam(dis.parameters(),lr = dis_lr,betas=(beta1,beta2))

fixed_noise = torch.randn(batch_size, noise_vector, 1, 1, device=device)


# additional variables 
gen_images = []
gen_losses = []
dis_losses = []

# train generator
for epoch in range(epochs):
    for i,(x,_) in enumerate(train_loader):
            
        x = x.to(device)
        z = torch.randn(x.shape[0], noise_vector, 1, 1, device=device)
        
        gen_imgs = gen(z)

        real = torch.ones(x.shape[0],requires_grad=False).to(device)
        fake = torch.zeros(x.shape[0],requires_grad=False).to(device)


        # update discriminator parameters
        dis.zero_grad()
        dis_real_loss = loss_fn(dis(x).view(-1),real)
        # dis_real_loss = (1-dis(x)).mean()
        # dis_real_loss.backward()
        # dis_optimizer.step()

        
        dis_fake_loss = loss_fn(dis(gen_imgs.detach()).view(-1),fake)
        # dis_fake_loss = dis(gen_imgs.detach()).mean()
        # dis_fake_loss.backward()
        # dis_optimizer.step()

        dis_loss = dis_real_loss + dis_fake_loss
        dis_loss.backward()
        dis_optimizer.step()


        # update generator parameters
        gen.zero_grad()
        genloss = loss_fn(dis(gen_imgs).view(-1),real)
        # genloss= (1-dis(gen_imgs)).mean()
        genloss.backward()
        gen_optimizer.step()

        
        gen_losses.append(genloss.item())
        dis_losses.append(dis_loss.item())


        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"% (epoch,epochs, i, len(train_loader), dis_loss.item(), genloss.item()))

        batches_done = epoch * len(train_loader) + i
        if i % 200 == 0:
            images = gen(fixed_noise)
            save_image(images.data[:64], "images/%d.png" % batches_done, nrow=8, normalize=True)
            gen_images.append(make_grid(images.data[:64].detach().cpu(), padding=2, normalize=True))
            

        # Save the model.
        if epoch % 2 == 0:
            torch.save({
                'generator' : gen.state_dict(),
                'discriminator' : dis.state_dict(),
                'optimizerG' : gen_optimizer.state_dict(),
                'optimizerD' : dis_optimizer.state_dict()
                }, 'model/model_epoch_{}.pth'.format(epoch))

# Save the final trained model.
torch.save({
            'generator' : gen.state_dict(),
            'discriminator' : dis.state_dict(),
            'optimizerG' : gen_optimizer.state_dict(),
            'optimizerD' : dis_optimizer.state_dict()
            }, 'model/model_epoch_{}.pth'.format(epoch))
   

# Plot the training losses.
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(gen_losses,label="G")
plt.plot(dis_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig('dcgan_loss.png')

# Animation showing the improvements of the generator.
fig = plt.figure(figsize=(8,8))
# plt.axis("off")
images = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in gen_images]
anim = animation.ArtistAnimation(fig, images, interval=1000, repeat_delay=1000, blit=True)
# plt.show()
anim.save('bitmoji_generated_images.gif', dpi=80, writer='imagemagick')

    # # print(f"epoch{epoch} is done.")
    # print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"% (epoch, epochs, i, len(train_loader), dis_loss.item(), genloss.item()))
