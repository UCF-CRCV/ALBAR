import cv2
import glob
import json
import numpy as np
import os, traceback
import pandas as pd
import random
import time
import torch
import decord
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as trans
from torchvision.transforms import v2
import zipfile

import config as cfg


decord.bridge.set_bridge('torch')


class VideoDataset(Dataset):
    """Base video dataset class with common functionality"""
    
    DEFAULT_TRANSFORMS = {
        'train': lambda: v2.Compose([
            v2.Resize(size=256, antialias=True),
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': lambda: v2.Compose([
            v2.Resize(size=256, antialias=True),
            v2.CenterCrop(size=(224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': lambda: v2.Compose([
            v2.Resize(size=256, antialias=True),
            v2.CenterCrop(size=(224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    DATASET_CONFIGS = {
        'ucf101': {
            'path_handler': lambda split: open(os.path.join(
                cfg.ucf101_path, 'ucfTrainTestlist', 
                f'{"train" if split=="train" else "test"}list01.txt')).read().splitlines(),
            'class_mapping': lambda: json.load(open(cfg.ucf101_class_mapping))['classes'],
            'label_handler': lambda path, classes: int(classes[path.split(os.sep)[-2]]) - 1
        },
        'hmdb51': {
            'path_handler': lambda split: pd.read_csv(
                os.path.join(cfg.hmdb_path, f'hmdb51_{split}_labels.csv')),
            'label_handler': lambda row, _: int(row['label'])
        },
        'k400': {
            'path_handler': lambda split: open(os.path.join(
                cfg.kinetics_path, f'annotation_{split}_fullpath_resizedvids.txt')).read().splitlines(),
            'label_handler': lambda path, _: int(path.split(' ')[1]) - 1
        }
    }

    def __init__(self, params, dataset='ucf101', split='train', shuffle=True, data_percentage=1.0, 
                 spatial_training=False, mode=0, total_modes=5):
        """
        Initialize dataset
        Args:
            params: Parameter object containing model configuration
            dataset: Dataset name (ucf101, hmdb51, k400, or their SCUBA variants)
            split: Dataset split (train, test, val)
            shuffle: Whether to shuffle the dataset
            data_percentage: Percentage of dataset to use
            spatial_training: Whether to return an additional spatial clip
            mode: Validation/Test mode
            total_modes: Total number of validation/test modes
        """
        self.params = params
        self.dataset = dataset.lower()
        self.split = split
        self.base_dataset = self.dataset.split('_')[0]
        self.spatial_training = spatial_training
        self.mode = mode
        self.total_modes = total_modes
        
        # Set up transforms - use val transforms for test split
        if split == 'test':
            self.transform = self.DEFAULT_TRANSFORMS['test']()
        else:
            self.transform = self.DEFAULT_TRANSFORMS[split]()
        
        # Initialize paths and classes
        self.initialize_dataset()
        
        # Shuffle and subset data if needed
        if shuffle:
            random.shuffle(self.all_paths)
        self.data = self.all_paths[:int(len(self.all_paths) * params.data_percentage)]

    # Initialize dataset paths and classes.
    def initialize_dataset(self):
        """Initialize dataset paths and classes"""
        if 'scuba' in self.dataset or 'conflfg' in self.dataset or 'scufo' in self.dataset:
            self.initialize_scuba_dataset()
        else:
            self.initialize_standard_dataset()


    # IID datasets.
    def initialize_standard_dataset(self):
        """Initialize standard dataset paths"""
        if self.base_dataset == 'ucf101':
            self.classes = self.DATASET_CONFIGS['ucf101']['class_mapping']()
            self.all_paths = [x.replace('/', os.sep) for x in 
                            self.DATASET_CONFIGS['ucf101']['path_handler'](self.split)]
        elif self.base_dataset == 'hmdb51':
            anno_file = self.DATASET_CONFIGS['hmdb51']['path_handler'](self.split)
            self.all_paths = [f'{os.path.join(cfg.hmdb_path, "Videos", c, f)} {l}' 
                            for c, f, l in zip(anno_file['class'], anno_file['filename'], anno_file['label'])]
        elif self.base_dataset == 'k400':
            self.all_paths = self.DATASET_CONFIGS['k400']['path_handler'](self.split)

    # SCUBA/SCUFO datasets.
    def initialize_scuba_dataset(self):
        """Initialize SCUBA dataset paths"""
        dataset = self.dataset.upper().replace('PLACES365', 'Places365').replace('CONFLFG', 'ConflFG').replace('STRIPE', 'Stripe')
        if 'ucf101' in self.dataset:
            self.classes = json.load(open(cfg.ucf101_class_mapping))['classes']
            self.all_paths = sorted(glob.glob(os.path.join(cfg.bias_path, dataset, '*')))
        elif 'hmdb51' in self.dataset:
            anno_file = pd.read_csv(os.path.join(cfg.hmdb_path, 'hmdb51_labels.csv'))
            self.classes = {os.path.basename(k): v for k, v in 
                          zip(anno_file['filename'].to_list(), anno_file['label'].to_list())}
            self.all_paths = sorted(glob.glob(os.path.join(cfg.bias_path, dataset, '*')))
        elif 'k400' in self.dataset:
            anno_file = pd.read_csv(os.path.join(cfg.kinetics_path, 'k400_val_labels.csv'))
            self.classes = {os.path.basename(k): v for k, v in 
                          zip(anno_file['filename'].to_list(), anno_file['label'].to_list())}
            self.all_paths = sorted(glob.glob(os.path.join(cfg.bias_path, dataset, '*')))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.process_data(idx)


    def process_data(self, idx):
        """Process a single data item"""
        path_info = self.data[idx]
        
        if 'scuba' in self.dataset or 'conflfg' in self.dataset or 'scufo' in self.dataset:
            clip, label, frame_indices = self.process_scuba_data(path_info)
        else:
            clip, label, frame_indices = self.process_standard_data(path_info)

        if clip is None:
            return (None,) * (3 if self.spatial_training else 3)

        if self.spatial_training:
            # Select frame index based on params.frame_choice
            if hasattr(self.params, 'frame_choice'):
                if self.params.frame_choice == 'random':
                    frame_idx = np.random.randint(0, self.params.num_frames)
                elif isinstance(self.params.frame_choice, int):
                    frame_idx = min(self.params.frame_choice, self.params.num_frames - 1)
                else:
                    frame_idx = np.random.randint(0, self.params.num_frames)  # default to random if invalid
            else:
                frame_idx = np.random.randint(0, self.params.num_frames)  # maintain backward compatibility
                
            spat_clip = torch.tile(clip[frame_idx], (self.params.num_frames, 1, 1, 1))
            return clip, spat_clip, label

        return clip, label, idx


    def process_standard_data(self, path_info):
        """Process standard dataset video"""
        if self.base_dataset == 'ucf101':
            vid_path = os.path.join(cfg.ucf101_path, 'Videos', path_info.split(' ')[0])
            label = self.DATASET_CONFIGS['ucf101']['label_handler'](vid_path, self.classes)
        elif self.base_dataset == 'hmdb51':
            vid_path, label = path_info.split(' ')
            label = int(label)
        elif self.base_dataset == 'k400':
            vid_path = path_info.split(' ')[0]
            label = self.DATASET_CONFIGS['k400']['label_handler'](path_info, None)

        label = torch.tensor(label)

        clip, frame_list = self.build_clip(vid_path)
        return clip, label, frame_list


    def process_scuba_data(self, vid_path):
        """Process SCUBA dataset video"""
        if 'ucf101' in self.dataset:
            label = int(self.classes[vid_path.split(f'{os.sep}v_')[-1].split('_g0')[0]]) - 1
        elif 'hmdb51' in self.dataset:
            # key = os.path.basename(vid_path)[:-3] if 'scufo' in self.dataset else os.path.basename(vid_path)[:-7]
            key = os.path.basename(vid_path)[:-7]
            label = self.classes[key + '.avi']
        elif 'k400' in self.dataset:
            key = os.path.basename(vid_path) if 'scufo' in self.dataset else os.path.basename(vid_path)[:-4]
            label = self.classes[key]

        label = torch.tensor(label)

        clip, frame_list = self.build_scuba_clip(vid_path)
        return clip, label, frame_list


    def build_clip(self, vid_path):
        """Build a standard video clip"""
        try:
            # Get video dimensions and initialize reader
            cap = cv2.VideoCapture(vid_path)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Calculate resize dimensions maintaining aspect ratio
            if h < w:
                new_h, new_w = 256, int(256 * (w/h))
            else:
                new_w, new_h = 256, int(256 * (h/w))

            vr = decord.VideoReader(vid_path, width=new_w, height=new_h, ctx=decord.cpu())
            frame_count = len(vr)

            # Calculate frame indices
            skip_frames = self.params.fix_skip
            if skip_frames * self.params.num_frames > frame_count:
                skip_frames /= 2

            if self.split == 'train':
                left_over = frame_count - skip_frames * self.params.num_frames
                start_frame = np.random.randint(0, max(1, int(left_over)))
            else:
                F = frame_count - skip_frames * self.params.num_frames
                start_frame = int(np.linspace(0, max(0, F-10), self.total_modes)[self.mode])
            
            frame_indices = start_frame + np.asarray([int(skip_frames * f) for f in range(self.params.num_frames)])
            frame_indices = np.minimum(frame_indices, frame_count - 1)

            # Get frames
            frames = vr.get_batch(frame_indices)
            clip = [frame.permute(2, 0, 1) for frame in frames]
            
            # Pad if necessary
            if len(frames) < self.params.num_frames:
                clip.extend([clip[-1] for _ in range(self.params.num_frames - len(frames))])

            # Apply transforms
            clip = self.transform(torch.stack(clip, dim=0))
            
            return clip, torch.from_numpy(frame_indices)
        except:
            # print(f'Error processing video: {vid_path}')
            # traceback.print_exc()
            return None, None


    def build_scuba_clip(self, vid_path):
        """Build a SCUBA video clip"""
        try:
            if 'scufo' in self.dataset:
                # frame_list = sorted(glob.glob(os.path.join(vid_path, '*.jpg')))
                frame = torchvision.io.read_image(vid_path)
                frame = self.transform(frame)
                frame_indices = [0] * self.params.num_frames
                full_clip = [frame] * self.params.num_frames
            else:
                with zipfile.ZipFile(vid_path, 'r') as z:
                    frame_list = sorted(z.namelist())
                    frame_count = len(frame_list)
                    skip_frames = self.params.fix_skip

                    if skip_frames * self.params.num_frames > frame_count:
                        skip_frames /= 2

                    if self.split == 'train':
                        left_over = frame_count - skip_frames * self.params.num_frames
                        start_frame = np.random.randint(0, max(1, int(left_over)))
                    else:
                        F = frame_count - skip_frames * self.params.num_frames
                        start_frame = int(np.linspace(0, max(0, F-10), self.total_modes)[self.mode])

                    frame_indices = start_frame + np.asarray([int(skip_frames * f) for f in range(self.params.num_frames)])
                    frame_indices = np.minimum(frame_indices, frame_count - 1)

                    full_clip = []
                    for frame_idx in frame_indices:
                        frame = torchvision.io.decode_image(
                            torch.frombuffer(bytearray(z.read(frame_list[frame_idx])), dtype=torch.uint8))
                        full_clip.append(self.transform(frame))

            return torch.stack(full_clip, dim=0), torch.tensor(frame_indices)
        except:
            # print(f'Error processing video: {vid_path}')
            # traceback.print_exc()
            return None, None


def collate_fn(batch):
    """Remove None values and stack remaining samples"""
    valid_samples = [x for x in batch if not any(i is None for i in x)]
    if not valid_samples:
        return tuple([None] * len(batch[0]))
    return tuple(torch.stack([s[i] for s in valid_samples], dim=0) if isinstance(valid_samples[0][i], torch.Tensor)
                else [s[i] for s in valid_samples] for i in range(len(valid_samples[0])))


def create_dataloader(dataset_params):
    """Factory function to create dataloaders"""
    dataset = VideoDataset(**dataset_params)
    return DataLoader(
        dataset,
        batch_size=dataset_params['params'].batch_size,
        shuffle=dataset_params.get('shuffle', True),
        collate_fn=collate_fn,
        num_workers=dataset_params['params'].num_workers
    )


if __name__ == '__main__':
    import params_debias as params
    import time

    # Example usage
    train_loader = create_dataloader({
        'params': params,
        'dataset': 'hmdb51',
        'split': 'train',
        'spatial_training': True  # Enable spatial training
    })

    val_loader = create_dataloader({
        'params': params,
        'dataset': 'hmdb51',
        'split': 'val',
        'shuffle': False
    })

    test_loader = create_dataloader({
        'params': params,
        'dataset': 'hmdb51_scuba_places365',
        'split': 'test',
        'shuffle': False
    })

    # print(f'Training samples: {len(train_loader.dataset)}')
    # print(f'Steps per epoch: {len(train_loader)}')

    # t0 = time.time()
    # for i, batch in enumerate(train_loader):
    #     if i % 10 == 0:
    #         print(f'Batch {i}: {[x.shape if isinstance(x, torch.Tensor) else len(x) for x in batch]}')

    # print(f'Time elapsed: {time.time() - t0:.2f}s')

    # print(f'Validation samples: {len(val_loader.dataset)}')
    # print(f'Steps per epoch: {len(val_loader)}')
    # print(f'Batch size: {val_loader.batch_size}')

    # t0 = time.time()
    # for i, batch in enumerate(val_loader):
    #     if i % 10 == 0:
    #         print(f'Batch {i}: {[x.shape if isinstance(x, torch.Tensor) else len(x) for x in batch]}')

    # print(f'Time elapsed: {time.time() - t0:.2f}s')

    print(f'Test samples: {len(test_loader.dataset)}')
    print(f'Steps per epoch: {len(test_loader)}')

    t0 = time.time()
    for i, batch in enumerate(test_loader):
        if i % 10 == 0:
            print(f'Batch {i}: {[x.shape if isinstance(x, torch.Tensor) else len(x) for x in batch]}')

    print(f'Time elapsed: {time.time() - t0:.2f}s')
