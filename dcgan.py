import torch
from torch import nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from generator import Generator
from discrimenator import Discrimenator
import matplotlib.pyplot as plt
import numpy as np


class DCGAN:
    def __init__(
        self,
        channels_img=1,
        channels_noise=100,
        feature_g=16,
        feature_d=16,
        epochs=5,
        real_label=1,
        fake_label=0,
        batch_size=128,
        image_size=64):

        self.channels_img = channels_img
        self.channels_noise = channels_noise
        self.feature_g = feature_g
        self.feature_d = feature_d
        self.image_size = image_size
        self.real_label = real_label
        self.fake_label = fake_label
        self.criterion = nn.BCELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = Generator(self.channels_noise, self.channels_img, self.feature_g).to(self.device)
        self.discrimenator = Discrimenator(self.channels_img, self.feature_d).to(self.device)
    
    def train(
        self,
        train_loader,
        epochs=10,
        lr=0.001,
        beta1=0.50,
        beta2=0.999):

        fixed_noise = torch.randn(64, self.channels_noise, 1, 1).to(self.device)
        optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(beta1, beta2)
        )
        optimizer_d = optim.Adam(
            self.discrimenator.parameters(),
            lr=lr,
            betas=(beta1, beta2)
        )
        criterion = nn.BCELoss()

        self.generator.train()
        self.discrimenator.train()

        self.gen_train_loss = []
        self.dis_train_loss = []
        self.fake_imgs = []
        self.real_imgs = []

        for epoch in range(epochs):
            print(f'Training Epoch {epoch+1}')
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.device)
                batch_size = data.shape[0]
                
                self.discrimenator.zero_grad()
                label = (torch.ones(batch_size)*0.9).to(self.device)
                output = self.discrimenator(data).reshape(-1)
                dis_loss_real = criterion(output, label)
                dis_x = output.mean().item()
                
                noise = torch.rand(batch_size, self.channels_noise, 1, 1).to(self.device)
                fake = self.generator(noise)
                label = (torch.ones(batch_size)*0.1).to(self.device)
                
                output = self.discrimenator(fake.detach()).reshape(-1)
                dis_loss_fake = criterion(output, label)
                
                dis_loss = dis_loss_real+dis_loss_fake
                dis_loss.backward()
                optimizer_d.step()
                
                self.generator.zero_grad()
                label = torch.ones(batch_size).to(self.device)
                output = self.discrimenator(fake).reshape(-1)
                gen_loss = criterion(output, label)
                gen_loss.backward()
                optimizer_g.step()
                
                if batch_idx % 100 == 0:
                    print(f'Batch Index {batch_idx}, Generator Loss: {gen_loss}, Discrimenator Loss: {dis_loss}')
                    with torch.no_grad():
                        fake = self.generator(fixed_noise)
                        img_grid_fake = make_grid(fake[:64], normalize=True)
                        img_grid_real = make_grid(data[:64], normalize=True)
                        
                        self.fake_imgs.append(img_grid_fake)
                        self.real_imgs.append(img_grid_real)
                        #writer_fake.add_image('MNIST Fake Images', img_grid_fake)
                        #writer_real.add_image('MNIST Real Images', img_grid_real)
                
                self.gen_train_loss.append(gen_loss)
                self.dis_train_loss.append(dis_loss)
        
        # Save model checkpoint
        torch.save(self.generator.state_dict(), "generator_model.pth")
        torch.save(self.discrimenator.state_dict(), "discrimenator_model.pth")

    def predict(self, x):
        with torch.no_grad():
            test_fake = self.generator(x)
            test_img_grid_fake = make_grid(test_fake[:64], normalize=True)

        # Plot the real images
        plt.figure(figsize=(15, 8))
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Test imgage grid fake")
        plt.imshow(np.transpose(test_img_grid_fake.to(self.device).cpu(), (1,2,0)))
        plt.show()
    
    def plot_loss(self, figure=(15, 8)):
        plt.figure(figsize=figure)
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.gen_train_loss, label="Generator")
        plt.plot(self.dis_train_loss, label="Discriminator")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend(['Generator', 'Discriminator'])
        plt.show()
    
    def plot_fake_real(self, figure=(15, 8)):
        # Plot the real images
        plt.figure(figsize=figure)
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Real Images")
        for real_img in self.real_imgs:
            plt.imshow(np.transpose(real_img.to(self.device).cpu(), (1,2,0)))
            
        # Plot the fake images
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Fake Images")
        for fake_img in self.fake_imgs:
            plt.imshow(np.transpose(fake_img.to(self.device).cpu(), (1,2,0)))
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', required=True, default=128)
    parser.add_argument('--color', required=True, default='gray', help='gray | rgb')
    arg = parser.parse_args()

    if arg.color == 'rgb':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        print('Oops! Did not implemented yet.')

    elif arg.color == 'gray':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        # trainset = datasets.ImageFolder(root=data_path,
        #                                 transform=transform)
        trainset = datasets.MNIST('data/', download=True, train=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=arg.batch_size, shuffle=True)

        dcgan = DCGAN()
        dcgan.train(train_loader)
