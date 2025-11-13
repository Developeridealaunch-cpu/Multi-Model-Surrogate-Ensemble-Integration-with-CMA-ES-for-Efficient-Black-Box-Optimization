# models/vae.py
# Minimal VAE stub for landscape embeddings (PyTorch required at runtime).
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim//2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, input_dim),
        )

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

def train_vae(model, dataloader, epochs=10, lr=1e-3, device='cpu'):
    """Simple training loop (requires torch)."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    recon_loss = nn.MSELoss(reduction='mean')
    for e in range(epochs):
        model.train()
        total = 0.0
        for xb in dataloader:
            xb = xb.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            mse = recon_loss(recon, xb)
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse + 1e-3 * kld
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
        print(f"Epoch {e+1}/{epochs} loss={total/len(dataloader.dataset):.6f}")