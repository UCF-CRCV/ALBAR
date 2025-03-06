import numpy as np
import math

# Job parameters.
run_id = 'hmdb_swin_albar'
eval = False
arch = 'swin_t'
saved_model = None

# Dataset parameters.
dataset = 'hmdb51'
num_classes = 51
num_frames = 32
fix_skip = 2
num_modes = 1
data_percentage = 1.0
stillmix = False
prob_aug = 0.75
frame_choice = 16

# Dataloader parameters
batch_size = 6
v_batch_size = 12
num_workers = 8

# Training parameters
learning_rate = 1e-5
num_epochs = 50
temporal_weight = 1.0
spatial_weight = 1.0
entropy_weight = 4.25
gradpen_weight = 10.0

# Optimizer parameters
momentum = 0.9
weight_decay = 5e-2
opt_type = 'adamw'

# Learning rate schedule
lr_scheduler = 'cosine'
warmup_array = list(np.linspace(0.01, 1, 5) + 1e-9)
warmup = len(warmup_array)
cosine_lr_array = list(np.linspace(0.01, 1, 5)) + [(math.cos(x) + 1)/2 for x in np.linspace(0, math.pi/0.99, num_epochs)]

# Validation and saving frequency
val_freq = 2
save_every = 5

# Training augmentation params.
reso_h = 224
reso_w = 224

# Tracking params.
wandb = False
