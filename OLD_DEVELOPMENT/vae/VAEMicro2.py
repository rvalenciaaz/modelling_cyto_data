import numpy as np
import sys
import argparse
import os

from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
#from torchvision import datasets
#from torchvision.transforms import v2
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split

batch_size = 128
learning_rate = 1e-3
weight_decay = 1e-2
num_epochs = 200


def read_profile_array(profileFile):

    N = 0
    S = 0
    
    with open(profileFile, 'r') as f:
        
        header = next(f)
        header = header.rstrip()
        toks = header.split(',')
        toks.pop(0)
        
        species = toks
        
        S = len(species)
        
        for line in f:
            N = N+1
        f.close()
    
    profile_array = np.zeros((N,S))
    
    N = 0
    samples = []
    with open(profileFile, 'r') as f:
        
        next(f)
        
        for line in f:
            line = line.rstrip()
            toks = line.split(',')
            samples.append(toks.pop(0))
            profile_array[N,:] = np.asarray([float(x) for x in toks])
            N = N+1
        f.close()

    return (profile_array,samples,species)

@dataclass
class VAEOutput:
    """
    Dataclass for VAE output.
    
    Attributes:
        z_dist (torch.distributions.Distribution): The distribution of the latent variable z.
        z_sample (torch.Tensor): The sampled value of the latent variable z.
        x_recon (torch.Tensor): The reconstructed output from the VAE.
        loss (torch.Tensor): The overall loss of the VAE.
        loss_recon (torch.Tensor): The reconstruction loss component of the VAE loss.
        loss_kl (torch.Tensor): The KL divergence component of the VAE loss.
    """
    z_dist: torch.distributions.Distribution
    z_sample: torch.Tensor
    x_recon: torch.Tensor
    
    loss: torch.Tensor
    loss_recon: torch.Tensor
    loss_kl: torch.Tensor
    
class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class.
    
    Args:
        input_dim (int): Dimensionality of the input data.
        hidden_dim (int): Dimensionality of the hidden layer.
        latent_dim (int): Dimensionality of the latent space.
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
                
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 8, 2 * latent_dim), # 2 for mean and variance.
        )
        self.softplus = nn.Softplus()
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 8),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 8, hidden_dim // 4),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim, input_dim),
        )

   #     self.double()
        
    def encode(self, x, eps: float = 1e-8):
        """
        Encodes the input data into the latent space.
        
        Args:
            x (torch.Tensor): Input data.
            eps (float): Small value to avoid numerical instability.
        
        Returns:
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data.
        """
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        
        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
        
    def reparameterize(self, dist):
        """
        Reparameterizes the encoded data to sample from the latent space.
        
        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.

        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        return dist.rsample()
    
    def decode(self, z):
        """
        Decodes the data from the latent space to the original input space.
        
        Args:
            z (torch.Tensor): Data in the latent space.
        
        Returns:
            torch.Tensor: Reconstructed data in the original input space.
        """
        return self.decoder(z)
    
    def forward(self, x, compute_loss: bool = True):
        """
        Performs a forward pass of the VAE.
        
        Args:
            x (torch.Tensor): Input data.
            compute_loss (bool): Whether to compute the loss or not.
        
        Returns:
            VAEOutput: VAE output dataclass.
        """
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_x = self.decode(z)
        
        if not compute_loss:
            return VAEOutput(
                z_dist=dist,
                z_sample=z,
                x_recon=recon_x,
                loss=None,
                loss_recon=None,
                loss_kl=None,
            )
        
        # compute loss terms 
        loss_recon = ((x - recon_x)**2).sum(-1).mean()

        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        )
        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()
                
        loss = loss_recon + loss_kl
        
        return VAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=recon_x,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )

def train(model, dataloader, optimizer, prev_updates, device, writer=None):
    """
    Trains the model on the given data.
    
    Args:
        model (nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The data loader.
        loss_fn: The loss function.
        optimizer: The optimizer.
    """
    model.train()  # Set the model to training mode
    
    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
        n_upd = prev_updates + batch_idx
        
        data = data.to(device)
        
        optimizer.zero_grad()  # Zero the gradients
        
        output = model(data)  # Forward pass
        loss = output.loss
        
        loss.backward()
        
        if n_upd % 100 == 0:
            # Calculate and log gradient norms
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
        
            print(f'Step {n_upd:,} (N samples: {n_upd*batch_size:,}), Loss: {loss.item():.4f} (Recon: {output.loss_recon.item():.4f}, KL: {output.loss_kl.item():.4f}) Grad: {total_norm:.4f}')

            if writer is not None:
                global_step = n_upd
                writer.add_scalar('Loss/Train', loss.item(), global_step)
                writer.add_scalar('Loss/Train/MSE', output.loss_recon.item(), global_step)
                writer.add_scalar('Loss/Train/KLD', output.loss_kl.item(), global_step)
                writer.add_scalar('GradNorm/Train', total_norm, global_step)
            
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    
        
        optimizer.step()  # Update the model parameters
        
    return prev_updates + len(dataloader)

def test(model, dataloader, cur_step, latent_dim, device, writer=None):
    """
    Tests the model on the given data.
    
    Args:
        model (nn.Module): The model to test.
        dataloader (torch.utils.data.DataLoader): The data loader.
        cur_step (int): The current step.
        writer: The TensorBoard writer.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc='Testing'):
            data = data.to(device)
            data = data.view(data.size(0), -1)  # Flatten the data
            
            output = model(data, compute_loss=True)  # Forward pass
            
            test_loss += output.loss.item()
            test_recon_loss += output.loss_recon.item()
            test_kl_loss += output.loss_kl.item()
            
    test_loss /= len(dataloader)
    test_recon_loss /= len(dataloader)
    test_kl_loss /= len(dataloader)
    print(f'====> Test set loss: {test_loss:.4f} (BCE: {test_recon_loss:.4f}, KLD: {test_kl_loss:.4f})')
    
    if writer is not None:
        writer.add_scalar('Loss/Test', test_loss, global_step=cur_step)
        writer.add_scalar('Loss/Test/MSE', output.loss_recon.item(), global_step=cur_step)
        writer.add_scalar('Loss/Test/KLD', output.loss_kl.item(), global_step=cur_step)
        
        # Log reconstructions
       # writer.add_images('Test/Reconstructions', output.x_recon.view(-1, 1, 28, 28), global_step=cur_step)
      #  writer.add_images('Test/Originals', data.view(-1, 1, 28, 28), global_step=cur_step)
        
        # Log random samples from the latent space
        z = torch.randn(16, latent_dim).to(device)
        samples = model.decode(z)
     #   writer.add_images('Test/Samples', samples.view(-1, 1, 28, 28), global_step=cur_step)


def write_array_to_csv(X,sample_names,fileName):

    with open(fileName,'w') as f:
    
        D = X.shape[1]
        N = X.shape[0]
        
        hList = ['dim_' + str(d) for d in range(D)]
        
        print('Sample,%s' % (','.join(hList)), file=f)
        
        for n in range(N):
        
            fList = [str(x) for x in X[n,:].tolist()]
            
            print('%s,%s' % (sample_names[n],','.join(fList)), file=f)

def main(argv):

    parser = argparse.ArgumentParser()   

    parser.add_argument("profileFile", help="csv file")
    
    parser.add_argument("targetFile", help="csv file")

    parser.add_argument("outStub", help="csv file")

    args = parser.parse_args()
    
    #import ipdb; ipdb.set_trace()
    
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')

    # Hyperparameters
    
    (profile_array,samples,species) = read_profile_array(args.profileFile)
    
    (target_array,samples2,targets) = read_profile_array(args.targetFile)
    
    profile_array = np.float32(profile_array)
    target_array = np.float32(target_array)
    
    log_profile_array = np.log10(profile_array + 1.0e-10)
    N = profile_array.shape[0]
    idx = list(range(N))
    
    (profile_train, profile_test, target_train, target_test,idx_train,idx_test) = train_test_split(log_profile_array, target_array, idx, 
                                                                                                    test_size=0.25,random_state=42)

   
    train_data = TensorDataset(torch.from_numpy(profile_train),torch.from_numpy(target_train))
    test_data = TensorDataset(torch.from_numpy(profile_test),torch.from_numpy(target_test))
    
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False,
    )

    latent_dim_max = 10
    hidden_dim = 512
    S = log_profile_array.shape[1]

    currDir = os.getcwd()
   
    outputDir = currDir + '/' + args.outStub + '/'
   
    os.mkdir(outputDir) 
    
    for latent_dim in range(1,latent_dim_max):
    
        subDir = outputDir + 'latent_' + str(latent_dim)
    
        os.mkdir(subDir)
    
        writerFile = f'{subDir}/vae_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    
        writer = SummaryWriter(writerFile)
    
        model = VAE(input_dim=S, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Number of parameters: {num_params:,}')

# create an optimizer object
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
        prev_updates = 0
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            prev_updates = train(model, train_loader, optimizer, prev_updates, device, writer=writer)
            test(model, test_loader, prev_updates, latent_dim, device, writer=writer)

        profile_encode = model.encode(torch.from_numpy(log_profile_array).to(device))

        profile_variance = profile_encode.variance.detach().cpu().numpy()
        profile_mean = profile_encode.mean.detach().cpu().numpy()
    
        varFile   = f'{subDir}/profile_var.csv'
        meanFile  = f'{subDir}/profile_mean.csv'
        modelFile = f'{subDir}/model.th'
    
        write_array_to_csv(profile_variance,samples,varFile)
        write_array_to_csv(profile_mean,samples,meanFile)
       
        torch.save(model.state_dict(), modelFile)
if __name__ == "__main__":
    main(sys.argv[1:])
    
