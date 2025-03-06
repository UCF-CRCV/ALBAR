from accelerate import Accelerator
from accelerate.utils import set_seed
import importlib
import numpy as np
import os
import pandas as pd
from tensorboardX import SummaryWriter
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from entropy_loss import entropy_loss, compute_gradient_penalty
from model_loaders import load_model
from multi_dataset_dl import VideoDataset, collate_fn
from stillmix import StillMixRandomBlending


# Make training reproducible.
set_seed(1)

# Trainer object to handle training loop.
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        accelerator: Accelerator,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        snapshot_path: str,
        writer: SummaryWriter,
        params: dict,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.train_data = train_data
        self.val_data = None
        self.save_every = params.save_every
        self.learning_rate = params.learning_rate
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs_run = 1
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            self.accelerator.print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        self.writer = writer
        self.params = params
        self.data_percentage = params.data_percentage
        self.val_freq = params.val_freq
        self.blender = StillMixRandomBlending(prob_aug=params.prob_aug)
        
    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path, map_location=self.accelerator.device)
        # Unwrap model, load state dict, and rewrap model.
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(snapshot['model_state_dict'])
        self.model = self.accelerator.prepare(self.model)
        self.optimizer.load_state_dict(snapshot['optimizer'])
        self.epochs_run = snapshot['epoch']
        # if self.params.lr_scheduler == 'cosine':
        #     self.scheduler = self.accelerator.prepare(torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-9, last_epoch=self.epochs_run))
        self.accelerator.print(f'Resuming training from snapshot at Epoch {self.epochs_run}')

    def _save_snapshot(self, epoch):
        model = self.accelerator.unwrap_model(self.model)

        snapshot = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        self.accelerator.save(snapshot, self.snapshot_path)
        print(f'Epoch {epoch} | Training snapshot saved at {self.snapshot_path}')
    
    def _save_model(self, epoch, acc=0):
        # Remove all old models.
        for file in os.listdir(os.path.dirname(self.snapshot_path)):
            if 'model' in file and f'{acc:.4f}' in file:
                return
        # Save new best model.
        model_name = f'model_{epoch}_bestAcc_{acc:.4f}.pth' if acc > 0 else f'model_{epoch}.pth'
        model_path = os.path.join(os.path.dirname(self.snapshot_path), model_name)
        model = self.accelerator.unwrap_model(self.model)
        self.accelerator.save({'model_state_dict': model.state_dict()}, model_path)
        print(f'Epoch {epoch} | Model saved at {model_path}')

    def _train_epoch(self, epoch):
        # print(f'Train at epoch {epoch}')
        if self.accelerator.is_main_process:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                self.writer.add_scalar('Learning Rate', param_group['lr'], epoch)  
                print(f'Learning rate is: {param_group["lr"]}')

        # Set model to train.
        self.model.train()

        print(f'Training: [GPU{self.accelerator.process_index}] Epoch {epoch} | Steps: {len(self.train_data)}')

        losses = []
        gt, predictions = [], []

        for i, (temp_clips, spat_clips, labels) in enumerate(tqdm(self.train_data, ncols=100)):
            if self.params.stillmix:
                temp_clips, spat_clips, order = self.blender(temp_clips, spat_clips)
                spat_clips = spat_clips[order]

            temp_clips = temp_clips.permute(0,2,1,3,4)
            spat_clips = spat_clips.permute(0,2,1,3,4)
            gt.extend(labels.cpu())
            self.optimizer.zero_grad()

            temporal_output = self.model(temp_clips)
            temporal_loss = self.loss_fn(temporal_output, labels)

            if self.params.gradpen_weight > 0:
                spat_clips.requires_grad = True
                spatial_output = self.model(spat_clips)
                gradpen = compute_gradient_penalty(spatial_output, spat_clips)
                spat_clips.requires_grad = False
            else:
                gradpen = torch.tensor(0.0, device=self.accelerator.local_process_index)

            if self.params.entropy_weight > 0:
                if self.params.gradpen_weight == 0:
                    spatial_output = self.model(spat_clips)
                spatial_entropy = entropy_loss(spatial_output)
            else:
                spatial_entropy = torch.tensor(0.0, device=self.accelerator.local_process_index)

            if self.params.spatial_weight > 0:
                if self.params.entropy_weight == 0 and self.params.gradpen_weight == 0:
                    spatial_output = self.model(spat_clips)
                spatial_loss = self.loss_fn(spatial_output, labels)
            else:
                spatial_loss = torch.tensor(0.0, device=self.accelerator.local_process_index)

            loss = self.params.temporal_weight*temporal_loss - self.params.spatial_weight*spatial_loss + self.params.entropy_weight*spatial_entropy + self.params.gradpen_weight*gradpen

            self.accelerator.backward(loss)
            self.optimizer.step()
            losses.append(loss.item())
            predictions.extend(torch.max(temporal_output, axis=1).indices.cpu())
            # if i % (len(self.train_data) // 5) == 0:
            #     print(f'Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses):.5f}', flush=True)

        loss = np.mean(losses)
        loss = torch.as_tensor(loss, device=self.accelerator.local_process_index)
        loss = self.accelerator.reduce(loss, 'mean')

        predictions = np.asarray(predictions)
        gt = np.asarray(gt)
        accuracy = ((predictions==gt).sum())/np.size(predictions)
        accuracy = torch.as_tensor(accuracy, device=self.accelerator.local_process_index)
        accuracy = self.accelerator.reduce(accuracy, 'mean')

        if self.accelerator.is_main_process:
            print(f'Training Epoch: {epoch}, Loss: {loss}')
            self.writer.add_scalar('Training Loss', loss, epoch)
            print(f'Training Accuracy at Epoch {epoch} is {accuracy*100:0.3f}%')
            if self.params.wandb:
                self.accelerator.log({
                    'train_loss': loss,
                    'train_acc': accuracy*100,
                    'loss': loss},
                    step=epoch)

        del losses, temporal_output, spatial_output, temp_clips, spat_clips, labels, accuracy
        return loss

    def _val_epoch(self, epoch, mode, pred_dict, label_dict, scufo_pred_dict=None, scufo_label_dict=None):
        # Set model to eval.
        self.model.eval()
        print(f'Validation: [GPU{self.accelerator.process_index}] Epoch {epoch} | Mode: {mode}')
        
        losses = []
        gt, predictions = [], []
        vid_paths = []

        for i, (inputs, labels, idx) in enumerate(self.val_data):
            inputs = inputs.permute(0,2,1,3,4)
            gt.extend(labels.cpu())
            with torch.no_grad():
                output = self.model(inputs)
                loss = self.loss_fn(output, labels)
                predictions.extend(torch.nn.functional.softmax(output, dim=1).cpu().data.numpy())
                losses.append(loss.item())
                vid_paths.extend(idx)
        loss = np.mean(losses)
        loss = torch.as_tensor(loss, device=self.accelerator.local_process_index)
        loss = self.accelerator.reduce(loss, 'mean')

        # predictions = np.asarray(predictions)
        # gt = np.asarray(gt)

        # accuracy = ((predictions==gt).sum())/np.size(predictions)
        # accuracy = torch.as_tensor(accuracy, device=self.accelerator.local_process_index)
        # accuracy = self.accelerator.reduce(accuracy, 'mean')

        predictions = torch.as_tensor(np.asarray(predictions), device=self.accelerator.local_process_index)
        predictions = self.accelerator.gather(predictions).cpu().numpy()

        gt = torch.as_tensor(np.asarray(gt), device=self.accelerator.local_process_index)
        gt = self.accelerator.gather(gt).cpu().numpy()

        vid_paths = torch.as_tensor(np.asarray(vid_paths), device=self.accelerator.local_process_index)
        vid_paths = self.accelerator.gather(vid_paths).cpu().numpy()

        pred_array = np.flip(np.argsort(predictions, axis=1), axis=1)
        c_pred = pred_array[:, 0]

        for entry in range(len(vid_paths)):
            if vid_paths[entry] not in pred_dict.keys():
                pred_dict[vid_paths[entry]] = []
                pred_dict[vid_paths[entry]].append(predictions[entry])
            else:
                pred_dict[vid_paths[entry]].append(predictions[entry])

        for entry in range(len(vid_paths)):
            if vid_paths[entry] not in label_dict.keys():
                label_dict[vid_paths[entry]] = gt[entry]  

        correct_count = np.sum(c_pred==gt)
        accuracy = float(correct_count)/len(c_pred)
        
        if self.accelerator.is_main_process:
            print(f'Validation Epoch: {epoch}, Loss: {loss}')
            print(f'Validation Accuracy at Epoch {epoch}, mode {mode} is {accuracy*100:0.3f}%')
            self.writer.add_scalar('Validation Loss', loss, epoch)
            self.writer.add_scalar('Validation Accuracy', accuracy, epoch)

        del inputs, output, labels, accuracy

        # To compute contrasted accuracy for SCUBA/SCUFO pairs.
        if isinstance(scufo_pred_dict, dict) and isinstance(scufo_label_dict, dict):
            losses = []
            gt, predictions = [], []
            vid_paths = []

            for inputs, labels, idx in self.scufo_data:
                inputs = inputs.permute(0,2,1,3,4)
                gt.extend(labels.cpu())
                with torch.no_grad():
                    output = self.model(inputs)
                    predictions.extend(torch.nn.functional.softmax(output, dim=1).cpu().data.numpy())
                    vid_paths.extend(idx)

            predictions = torch.as_tensor(np.asarray(predictions), device=self.accelerator.local_process_index)
            predictions = self.accelerator.gather(predictions).cpu().numpy()

            gt = torch.as_tensor(np.asarray(gt), device=self.accelerator.local_process_index)
            gt = self.accelerator.gather(gt).cpu().numpy()

            vid_paths = torch.as_tensor(np.asarray(vid_paths), device=self.accelerator.local_process_index)
            vid_paths = self.accelerator.gather(vid_paths).cpu().numpy()

            pred_array = np.flip(np.argsort(predictions, axis=1), axis=1)
            c_pred = pred_array[:, 0]

            for entry in range(len(vid_paths)):
                if vid_paths[entry] not in scufo_pred_dict.keys():
                    scufo_pred_dict[vid_paths[entry]] = []
                    scufo_pred_dict[vid_paths[entry]].append(predictions[entry])
                else:
                    scufo_pred_dict[vid_paths[entry]].append(predictions[entry])

            for entry in range(len(vid_paths)):
                if vid_paths[entry] not in scufo_label_dict.keys():
                    scufo_label_dict[vid_paths[entry]] = gt[entry]
            return pred_dict, label_dict, scufo_pred_dict, scufo_label_dict, loss

        return pred_dict, label_dict, loss
    

    # Full training loop.
    def train(self, max_epochs: int, params: list):
        # self.train_data.dataset.update_device(self.accelerator.local_process_index)
        best_train_loss = 100000
        best_val_acc = 0
        val_acc, train_loss = 0, 0
        scheduler_epoch = 0
        scheduler_step = 1
        for epoch in range(self.epochs_run, max_epochs + 1):
            start = time.time()
            self.accelerator.print('-------------------------------------------')
            # If data percentage < 1, shuffle dataset each epoch.
            if not self.params.eval:
                if epoch > self.epochs_run and params.data_percentage != 1.0:
                    train_dataset = VideoDataset(
                        params=params,
                        dataset=params.dataset,
                        split='train',
                        spatial_training=True
                    )
                    self.train_data = self.accelerator.prepare(DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn))
                
                # Call training loop.
                train_loss = self._train_epoch(epoch)

                # Learning rate scheduler.
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    scheduler_epoch = 0
                else:
                    scheduler_epoch += 1

                if params.lr_scheduler == 'cosine':
                    self.learning_rate = params.cosine_lr_array[epoch-1]*params.learning_rate
                if params.warmup and epoch-1 < len(params.warmup_array):
                    self.learning_rate = self.learning_rate
                    # self.optimizer.param_groups[0]['lr'] = params.warmup_array[epoch-1]*self.optimizer.param_groups[0]['lr']
                
                self.accelerator.print('-------------------------------------------')

            # Validation.
            if epoch % self.val_freq == 0:
                pred_dict, label_dict = {}, {}
                val_losses = []
                for mode in list(range(params.num_modes)):
                    # Create new dataset/dataloader for each validation mode.
                    validation_dataset = VideoDataset(
                        params=params,
                        dataset=params.dataset,
                        split='test',
                        shuffle=True,
                        mode=mode,
                        total_modes=params.num_modes
                    )
                    validation_dataloader = DataLoader(validation_dataset, batch_size=params.v_batch_size, shuffle=False, num_workers=params.num_workers, collate_fn=collate_fn)
                    self.val_data = self.accelerator.prepare(validation_dataloader)
                    
                    # Print dataset length on first mode
                    if mode == 0:
                        self.accelerator.print(f'Validation dataset length: {len(validation_dataset)}')
                        self.accelerator.print(f'Validation dataset steps per epoch: {len(validation_dataset)/params.v_batch_size}')

                    # Call validation function.
                    pred_dict, label_dict, val_loss = self._val_epoch(epoch, mode, pred_dict, label_dict)
                    val_losses.append(val_loss.cpu())

                    # Evaluate using pred/gt dicts
                    predictions = np.zeros((len(list(pred_dict.keys())), params.num_classes))
                    ground_truth = []
                    for entry, key in enumerate(pred_dict.keys()):
                        predictions[entry] = np.mean(pred_dict[key], axis=0)

                    for key in label_dict.keys():
                        ground_truth.append(label_dict[key])

                    pred_array = np.flip(np.argsort(predictions, axis=1), axis=1)  # Prediction with the most confidence is the first element here.
                    c_pred = pred_array[:, 0]

                    correct_count = np.sum(c_pred==ground_truth)
                    val_acc = float(correct_count)/len(c_pred)
                    self.accelerator.print(f'Running Avg Accuracy for epoch {epoch}, mode {mode}, is {val_acc*100:.3f}%')
                    if self.params.wandb:
                        self.accelerator.log({'val_acc': val_acc*100}, step=epoch)
                    self.accelerator.wait_for_everyone()
            
                val_loss = np.mean(val_losses)
                # Display/save metrics.
                if self.accelerator.is_main_process:
                    print(f'Val loss for epoch {epoch} is {val_loss}')
                    print(f'Correct Count is {correct_count} out of {len(c_pred)}')
                    print(f'Overall Accuracy is for epoch {epoch} is {val_acc*100:.3f}%')
                    self.writer.add_scalar('Validation Loss', val_loss, epoch)
                    self.writer.add_scalar('Validation Accuracy', val_acc, epoch)
                self.accelerator.wait_for_everyone()
            scubby = False
            # Save model on best validation score.
            if val_acc > best_val_acc:
                scubby = True
                cur_ds = params.dataset
                run_eval(self, params, epoch)
                params.dataset = cur_ds
                if self.accelerator.is_main_process:
                    print('++++++++++++++++++++++++++++++')
                    print(f'Epoch {epoch} is the best model till now for {params.run_id}!')
                    print('++++++++++++++++++++++++++++++')
                    self._save_model(epoch, val_acc)
                self.accelerator.wait_for_everyone()
                best_val_acc = val_acc

            # Save snapshot.
            if epoch % self.save_every == 0:
                if not scubby:
                    cur_ds = params.dataset
                    run_eval(self, params, epoch)
                    params.dataset = cur_ds
                if self.accelerator.is_main_process:
                    self._save_snapshot(epoch)
                    self._save_model(epoch, val_acc)
                self.accelerator.wait_for_everyone()
            
            # Calculate time per epoch.
            taken = time.time() - start
            self.accelerator.print(f'Epoch {epoch} took {taken/60:.3f} minutes.')

    # Full evaluation loop.
    def eval(self, params: list):
        start = time.time()
        epoch = 1
        pred_dict, label_dict = {}, {}
        scufo_pred_dict, scufo_label_dict = {}, {}
        # val_losses = []
        for mode in list(range(params.num_modes)):
            # Create new dataset/dataloader for each validation mode.
            validation_dataset = VideoDataset(
                params=params,
                dataset=params.dataset,
                split='test',
                shuffle=False,
                mode=mode,
                total_modes=params.num_modes
            )
            validation_dataloader = DataLoader(validation_dataset, batch_size=params.v_batch_size, shuffle=False, num_workers=params.num_workers, collate_fn=collate_fn)
            if 'scuba' in params.dataset:
                scufo_dataset = VideoDataset(
                    params=params,
                    dataset=params.dataset.replace('scuba', 'scufo'),
                    split='test',
                    shuffle=False,
                    mode=mode,
                    total_modes=params.num_modes
                )
                scufo_dataloader = DataLoader(scufo_dataset, batch_size=params.v_batch_size, shuffle=False, num_workers=params.num_workers, collate_fn=collate_fn)
                self.scufo_data = self.accelerator.prepare(scufo_dataloader)

            self.val_data = self.accelerator.prepare(validation_dataloader)

            # Print dataset length on first mode.
            if mode == 0:
                self.accelerator.print(f'Validation dataset length: {len(validation_dataset)}')
                self.accelerator.print(f'Validation dataset steps per epoch: {len(validation_dataset)/params.v_batch_size}')

            # Call validation function.
            if 'scuba' in params.dataset:
                pred_dict, label_dict, scufo_pred_dict, scufo_label_dict, _ = self._val_epoch(epoch, mode, pred_dict, label_dict, scufo_pred_dict, scufo_label_dict)
            else:
                pred_dict, label_dict, _ = self._val_epoch(epoch, mode, pred_dict, label_dict)
            # val_losses.append(val_loss.cpu())

            # Evaluate using pred/gt dicts.
            predictions = np.zeros((len(list(pred_dict.keys())), params.num_classes))
            ground_truth = []
            for entry, key in enumerate(pred_dict.keys()):
                predictions[entry] = np.mean(pred_dict[key], axis=0)

            for key in label_dict.keys():
                ground_truth.append(label_dict[key])

            pred_array = np.flip(np.argsort(predictions, axis=1), axis=1)  # Prediction with the most confidence is the first element here.
            c_pred = pred_array[:, 0]

            correct_count = np.sum(c_pred==ground_truth)
            val_acc = float(correct_count)/len(c_pred)
            self.accelerator.print(f'Running Avg Accuracy for epoch {epoch}, mode {mode}, is {val_acc*100:.3f}%')
            self.accelerator.wait_for_everyone()

            if 'scuba' in params.dataset:
                s_predictions = np.zeros((len(list(scufo_pred_dict.keys())), params.num_classes))
                s_ground_truth = []
                for entry, key in enumerate(scufo_pred_dict.keys()):
                    s_predictions[entry] = np.mean(scufo_pred_dict[key], axis=0)

                for key in scufo_label_dict.keys():
                    s_ground_truth.append(scufo_label_dict[key])

                s_pred_array = np.flip(np.argsort(s_predictions, axis=1), axis=1)  # Prediction with the most confidence is the first element here.
                s_c_pred = s_pred_array[:, 0]

                s_correct_count = np.sum(s_c_pred==s_ground_truth)
                s_val_acc = float(s_correct_count)/len(s_c_pred)

                contra_pred = np.where(c_pred != s_c_pred, c_pred, -1)
                contra_count = np.sum(contra_pred==ground_truth)
                contral_val_acc = float(contra_count)/len(contra_pred)
                self.accelerator.wait_for_everyone()
    
        # val_loss = np.mean(val_losses)
        # Display/save metrics.
        if self.accelerator.is_main_process:
            # print(f'Val loss for epoch {epoch} is {val_loss}') 
            if 'scuba' in params.dataset:
                print(f'SCUBA Correct Count is {correct_count} out of {len(c_pred)}')
                print(f'SCUBA Overall Accuracy is for epoch {epoch} is {val_acc*100:.3f}%')
                print(f'SCUFO Correct Count is {s_correct_count} out of {len(s_c_pred)}')
                print(f'SCUFO Accuracy is for epoch {epoch} is {s_val_acc*100:.3f}%')
                print(f'Contrastive Correct Count is {contra_count} out of {len(contra_pred)}')
                print(f'Contrastive Accuracy is for epoch {epoch} is {contral_val_acc*100:.3f}%')
            else:
                print(f'Correct Count is {correct_count} out of {len(c_pred)}')
                print(f'Overall Accuracy is for epoch {epoch} is {val_acc*100:.3f}%')

        self.accelerator.wait_for_everyone()

        if 'scuba' in params.dataset:
            # Calculate time per epoch.
            taken = time.time() - start
            self.accelerator.print(f'Validation for {self.params.dataset} and {self.params.dataset.replace("scuba", "scufo")} took {taken/60:.3f} minutes.')
            return [self.params.dataset, val_acc*100, s_val_acc*100, contral_val_acc*100]
        # Calculate time per epoch.
        taken = time.time() - start
        self.accelerator.print(f'Validation for {self.params.dataset} took {taken/60:.3f} minutes.')
        return [self.params.dataset, val_acc*100]
    

# Load all training objectives.
def setup_training(params):
    # Create datasets
    train_dataset = VideoDataset(
        params=params,
        dataset=params.dataset,
        split='train',
        spatial_training=True
    )
        
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        collate_fn=collate_fn
    )
    
    # Prepare model and optimizer
    model = load_model(arch=params.arch, num_classes=params.num_classes, kin_pretrained=True)
    optimizer = {
        'adam': torch.optim.Adam(model.parameters(), lr=params.learning_rate),
        'adamw': torch.optim.AdamW(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay),
        'sgd': torch.optim.SGD(model.parameters(), lr=params.learning_rate, momentum=params.momentum, weight_decay=params.weight_decay)
    }[params.opt_type]
    
    # Init loss function.
    loss_fn = torch.nn.CrossEntropyLoss()
    
    return train_dataloader, model, optimizer, loss_fn


# Full debiasing evaluation loop.
def run_eval(trainer, params, epoch):

    results = []
    params.num_modes = 1
    params.eval = True

    if params.dataset == 'ucf101':
        for dataset in ['ucf101', 'ucf101_scuba_places365', 'ucf101_scuba_vqgan', 'ucf101_scuba_stripe', 'ucf101_conflfg_stripe']:
            params.dataset = dataset
            result = trainer.eval(params)
            if 'scuba' in params.dataset:
                results.append([result[0], result[1]])
                results.append([result[0].replace('scuba', 'scufo'), result[2]])
                results.append([result[0].replace('scuba', 'contra'), result[3]])
            else:
                results.append(result)
    elif params.dataset == 'hmdb51':
        for dataset in ['hmdb51', 'hmdb51_scuba_places365', 'hmdb51_scuba_vqgan', 'hmdb51_scuba_stripe', 'hmdb51_conflfg_stripe']:
            params.dataset = dataset
            result = trainer.eval(params)
            if 'scuba' in params.dataset:
                results.append([result[0], result[1]])
                results.append([result[0].replace('scuba', 'scufo'), result[2]])
                results.append([result[0].replace('scuba', 'contra'), result[3]])
            else:
                results.append(result)
    else:
        result = trainer.eval(params)
        if 'scuba' in params.dataset:
            results.append([result[0], result[1]])
            results.append([result[0].replace('scuba', 'scufo'), result[2]])
            results.append([result[0].replace('scuba', 'contra'), result[3]])
        else:
            results.append(result)

    # Create/update results DataFrame
    # Extract dataset names and accuracies from results
    datasets = [x[0] for x in results]
    accuracies = [float(str(x[1])[:7]) for x in results]
    
    # Create DataFrame for this epoch
    epoch_data = {'epoch': epoch}
    for dataset, accuracy in zip(datasets, accuracies):
        epoch_data[dataset] = accuracy
    
    epoch_results = pd.DataFrame([epoch_data])  # Note: wrapped in list to create single row
    
    # Load existing results if available, otherwise create new DataFrame
    results_path = f'eval_results/{params.run_id}.csv'
    if os.path.exists(results_path):
        all_results = pd.read_csv(results_path)
        all_results = pd.concat([all_results, epoch_results], ignore_index=True)
    else:
        all_results = epoch_results
    
    # Save updated results
    all_results.to_csv(results_path, index=False)
    
    # Format results for printing
    results = [float(str(x[1])[:7]) for x in results]
    # Prepend unique printout to results.
    results = ['ALL RESULTS:'] + results
    print(*results, sep='\t')
    params.eval = False

# Starting main function.
def main(params):
    if params.wandb:
        accelerator = Accelerator(log_with='wandb')
        accelerator.init_trackers(
            'debiasing_action_recognition',
            config={
                'learning_rate': params.learning_rate,
                'batch_size': params.batch_size,
                'epochs': params.num_epochs,
            },
            init_kwargs={
                "wandb": {
                    "name": params.run_id,
                }
            },
        )
    else:
        accelerator = Accelerator()

    save_dir = os.path.join(cfg.saved_models_dir, params.run_id)

    if accelerator.is_main_process:
        # Print relevant parameters.
        for k, v in params.__dict__.items():
            if '__' not in k:
                print(f'{k} : {v}')
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(cfg.logs, str(params.run_id)))
    else:
        writer = None

    # Load training objects.
    train_dataloader, model, optimizer, loss_fn = setup_training(params)

    # Path to save training checkpoint.
    snapshot_path = save_dir + '/snapshot.pt'

    # Prepare accelerator object.
    train_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, model, optimizer
    )

    # Init trainer.
    trainer = Trainer(
        model,
        accelerator,
        train_dataloader,
        optimizer,
        loss_fn,
        snapshot_path,
        writer,
        params
    )
    
    # Run full training loop.
    trainer.train(params.num_epochs, params)

    if params.wandb:
        accelerator.end_training()

    run_eval(trainer, params, params.num_epochs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument("--params", dest='params', type=str, required=False, default='params_debias.py', help='params')
    args = parser.parse_args()
    if os.path.exists(args.params):
        params = importlib.import_module(args.params.replace('.py', '').replace('/', '.'))
        print(f'{args.params} is loaded as parameter file.')
    else:
        print(f'{args.params} does not exist, change to valid filename.')

    main(params)
