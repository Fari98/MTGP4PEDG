import torch
from datasets.data_loader import load_concrete_strength, load_bioav, load_airfoil, load_boston
from utils.utils import non_zero_floor_division, smape_score, bounded_r2_score, create_dataset

loader = load_boston
real_space = torch.from_numpy(loader(X_y=False).values)

exp_raw = torch.distributions.Exponential(rate=1.0).sample(
    (real_space.shape[0], non_zero_floor_division(real_space.shape[1], 8)))
exp_flat = exp_raw.flatten()

# Min-max scaling to [0, 1]
exp_min = exp_raw.min()
exp_max = exp_raw.max()
exp_scaled = (exp_raw - exp_min) / (exp_max - exp_min)

# Bimodal Gaussian distribution: half N(-0.5, 0.1), half N(0.5, 0.1)
bimodal_1 = torch.randn((real_space.shape[0] // 2, non_zero_floor_division(real_space.shape[1], 8))) * 0.1 - 0.5
bimodal_2 = torch.randn((real_space.shape[0] // 2, non_zero_floor_division(real_space.shape[1], 8))) * 0.1 + 0.5
bimodal = torch.cat([bimodal_1, bimodal_2], dim=0)

# Final latent space
latent_space = torch.cat([exp_scaled, bimodal], dim=1)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))

# Exponential
plt.subplot(1, 2, 1)
plt.hist(exp_scaled.flatten().numpy(), bins=100, color='skyblue', edgecolor='black')
plt.title("Exponential Distribution (Truncated [0, 1])")
plt.xlabel("Value")
plt.ylabel("Frequency")

# Bimodal Gaussian
plt.subplot(1, 2, 2)
plt.hist(bimodal.flatten().numpy(), bins=100, color='salmon', edgecolor='black')
plt.title("Bimodal Gaussian (-0.5 & 0.5)")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.show()