import logging
import logging.config
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import prodict
import torch
import torchvision.utils
import wandb
from omegaconf import DictConfig, ListConfig, OmegaConf
from prodict import Prodict
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib import logger, visutils
from lib.data_utils import compute_false_color, extract_sample, to_device
from lib.logger import AverageMeter
from lib.loss import TrainLoss
from lib.metrics import EvalMetrics

OBJECTIVE = {'l1': 'min',
             'l1_occluded_input_pixels': 'min',
             'l1_observed_input_pixels': 'min',
             'masked_l1_loss': 'min',
             'mae': 'min',
             'masked_mae': 'min',
             'mse': 'min',
             'masked_mse': 'min',
             'rmse': 'min',
             'masked_rmse': 'min',
             'psnr': 'max',
             'sam': 'min',
             'ssim': 'max',
             'total_loss': 'min',
             'mad': 'max',
             'ols': 'max',
             'emd': 'max',
             'ens': 'max'
             }


def seconds_to_dd_hh_mm_ss(seconds_elapsed: int) -> Tuple[int, int, int, int]:
    days = seconds_elapsed // (24 * 3600)
    seconds_remainder = seconds_elapsed % (24 * 3600)
    hours = seconds_remainder // 3600
    seconds_remainder %= 3600
    minutes = seconds_remainder // 60
    seconds_remainder %= 60
    seconds = seconds_remainder

    return days, hours, minutes, seconds


class Trainer:
    def __init__(
            self,
            args: DictConfig,
            train_loader: torch.utils.data.dataloader.DataLoader,
            val_loader: torch.utils.data.dataloader.DataLoader,
            model,
            optimizer,
            scheduler
    ):
        self.args = args
        self.use_wandb = bool('wandb' in args)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.dataloader = {'train': train_loader, 'val': val_loader}
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model.to(self.device)
        self.args.accum_iter = self.args.get('accum_iter', 1)  # accumulate gradients for `accum_iter` iterations

        self.compute_losses = TrainLoss(self.args.loss)
        self.compute_metrics = EvalMetrics(self.args.metrics)

        # Losses: Initialize statistics
        self.train_stats = self._stats_meter(stats_type='loss')
        self.val_stats = self._stats_meter(stats_type='loss')

        # Losses: Initialize metrics
        self.train_metrics = self._stats_meter(stats_type='metrics')
        self.val_metrics = self._stats_meter(stats_type='metrics')

        self.best_loss = np.inf
        self.epoch_best_loss = np.nan

        os.makedirs(self.args.save_dir, exist_ok=True)
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        self.args.path_model_best = os.path.join(self.args.checkpoint_dir, 'Model_best.pth')
        self.args.path_model_last = os.path.join(self.args.checkpoint_dir, 'Model_last.pth')
        self.logger = logger.prepare_logger('train_logger', level=logging.INFO, log_to_console=True,
                                            log_file=os.path.join(args.save_dir, 'training.log'))

        # Set up wandb
        if self.use_wandb:
            os.makedirs(self.args.wandb.dir, exist_ok=True)
            wandb.init(**self.args.wandb, settings=wandb.Settings(start_method="fork"))
            wandb.config.update(OmegaConf.to_container(self.args))
            self.writer = None

            # Define the wandb summary metrics
            for key, value in self.args.metrics.items():
                if key == 'masked_metrics':
                    pass
                elif value:
                    wandb.define_metric(f"train_metrics/{key}", summary=OBJECTIVE[key])
                    wandb.define_metric(f"val_metrics/{key}", summary=OBJECTIVE[key])

            wandb.define_metric('train/total_loss', summary=OBJECTIVE['total_loss'])
            wandb.define_metric('val/total_loss', summary=OBJECTIVE['total_loss'])
        else:
            os.makedirs(os.path.join(self.args.save_dir, 'tb'), exist_ok=True)
            self.writer = SummaryWriter(log_dir=os.path.join(self.args.save_dir, 'tb'))

        # Resume training
        if self.args.resume and self.args.pretrained_path:
            self._resume(path=self.args.pretrained_path)
        else:
            self.logger.info('\nTraining from scratch.\n')
            self.epoch = 0
            self.iter = 0

    def _get_lr(self, group: int = 0) -> float:
        return self.optimizer.param_groups[group]['lr']

    def _resume(self, path: str) -> None:
        """
        Resumes training.

        Args:
            path:  str, path of the pretrained model weights.
        """

        if not os.path.isfile(path):
            raise FileNotFoundError(f'No checkpoint found at {path}\n')

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.args.get('load_scheduler_state_dict', True) and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Extract the last training epoch
        self.epoch = checkpoint['epoch'] + 1
        self.iter = checkpoint['iter']
        self.args.num_epochs += self.epoch

        # Best validation loss so far
        self.best_loss = checkpoint['best_loss']
        self.epoch_best_loss = checkpoint['epoch']

        self.logger.info('\n\nRestoring the pretrained model from epoch %d.', self.epoch - 1)
        self.logger.info('Successfully loaded pretrained model weights from %s.\n', path)
        self.logger.info('Current best loss %.4f\n', self.best_loss)

    def _save_checkpoint(self, filepath: str) -> None:
        state = {
            'epoch': self.epoch,
            'iter': self.iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'best_epoch': self.epoch_best_loss
        }

        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(state, filepath)

    def _log_stats_meter(self, phase: str) -> None:
        if self.use_wandb:
            if phase == 'train':
                wandb.log({
                    'train_losses/' + k: v.avg for k, v in self.train_stats.items()
                }, step=self.iter)
                wandb.log({
                    'train_metrics/' + k: v.avg for k, v in self.train_metrics.items()
                }, step=self.iter)
            else:
                stats = {'val_losses/' + k: v.avg for k, v in self.val_stats.items()}
                stats['epoch'] = self.epoch
                wandb.log(stats, step=self.iter)

                stats = {'val_metrics/' + k: v.avg for k, v in self.val_metrics.items()}
                wandb.log(stats, step=self.iter)
        else:
            if phase == 'train':
                for k, v in self.train_stats.items():
                    self.writer.add_scalar('train_losses/' + k, v.avg, self.iter)
                for k, v in self.train_metrics.items():
                    self.writer.add_scalar('train_metrics/' + k, v.avg, self.iter)
            else:
                for k, v in self.val_stats.items():
                    self.writer.add_scalar('val_losses/' + k, v.avg, self.iter)
                for k, v in self.val_metrics.items():
                    self.writer.add_scalar('val_metrics/' + k, v.avg, self.iter)

        # Write validation stats and metrics to the log file
        if phase == 'val':
            self.logger.info((f'val:\tEpoch: {self.epoch}\t' +
                              ''.join([f'{k}: {v.avg:.5f}\t' for k, v in self.val_stats.items()]) +
                              ''.join([f'{k}: {v.avg:.5f}\t' for k, v in self.val_metrics.items()])))

    def _log_iter_epoch(self) -> None:
        if self.use_wandb:
            wandb.log({'epoch': self.epoch}, step=self.iter)
        else:
            self.writer.add_scalar('epoch', self.epoch, self.iter)

    def _log_learning_rate(self) -> None:
        if self.use_wandb:
            wandb.log({'log_lr': np.log10(self._get_lr()), 'epoch': self.epoch}, step=self.iter)
        else:
            self.writer.add_scalar('log_lr', np.log10(self._get_lr()), self.epoch)

    def _stats_dict(self, stats_type: str) -> prodict.Prodict:
        stats = Prodict()

        if stats_type == 'metrics':
            masked_metrics = self.args.metrics.masked_metrics
            for key, value in self.args.metrics.items():
                if key == 'masked_metrics':
                    pass
                elif value:
                    if masked_metrics and key != 'ssim':
                        stats[f'masked_{key}'] = np.inf
                    else:
                        stats[key] = np.inf

        elif stats_type == 'loss':
            for key, value in self.args.loss.items():
                # Exclude weight keys
                if value and isinstance(value, bool):
                    stats[key] = np.inf
            stats.total_loss = np.inf

        return stats

    def _stats_meter(self, stats_type: str) -> prodict.Prodict:
        meters = Prodict()
        stats = self._stats_dict(stats_type)
        for key, _ in stats.items():
            meters[key] = AverageMeter()

        return meters

    def _visualize_sample_wandb(self, sample_index: Optional[int] = None) -> None:
        if sample_index is None:
            # Get one random batch
            batch = next(iter(self.dataloader['val']))

        else:
            # Get specific sample and introduce batch dimension
            batch = self.dataloader['val'].dataset.__getitem__(sample_index)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.unsqueeze(0)
                elif isinstance(v, int):
                    batch[k] = [v]

        batch = to_device(batch, self.device)
        x, y, _, mask_valid, _, indices_rgb, index_nir = extract_sample(batch)

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x, batch_positions=batch['position_days'])

        # Visualize the valid frames of the first sample in the batch, T x C x H x W
        valid = mask_valid[0] if mask_valid is not None else torch.ones((x.shape[1],))
        x = x[0, valid == 1].cpu()
        y_pred = y_pred[0, valid == 1].cpu()
        y = y[0, valid == 1].cpu()
        ncols = x.shape[0] if x.shape[0] <= 15 else 10
        indices_rgb = indices_rgb.cpu()

        title = 'examples_val' if sample_index is None else f'sample_{sample_index}'

        wandb.log({
            f'{title}_true_color_RGB': wandb.Image(torchvision.utils.make_grid([
                # gallery (grid of frames) reshaped from (H x W x C) to (C x H x W)
                visutils.gallery(x[:, indices_rgb, :, :], ncols=ncols).permute(2, 0, 1),
                visutils.gallery(y_pred[:, indices_rgb, :, :], ncols=ncols).permute(2, 0, 1),
                visutils.gallery(y[:, indices_rgb, :, :], ncols=ncols).permute(2, 0, 1)
            ], nrow=1),
                caption=f"Epoch {self.epoch}, validation Sample {batch['sample_index'][0]}\n"
                        "top: input, middle: prediction, bottom: observed")
        }, step=self.iter)

        if not np.isnan(index_nir):
            wandb.log({
                f'{title}_false_color_NIRRG': wandb.Image(torchvision.utils.make_grid([
                    # gallery reshaped from (H x W x C) to (C x H x W)
                    visutils.gallery(compute_false_color(x, index_rgb=indices_rgb, index_nir=index_nir),
                                     ncols=ncols).permute(2, 0, 1),
                    visutils.gallery(compute_false_color(y_pred, index_rgb=indices_rgb, index_nir=index_nir),
                                     ncols=ncols).permute(2, 0, 1),
                    visutils.gallery(compute_false_color(y, index_rgb=indices_rgb, index_nir=index_nir),
                                     ncols=ncols).permute(2, 0, 1)
                ], nrow=1),
                    caption=f"Epoch {self.epoch}, validation Sample {batch['sample_index'][0]}\n"
                            "top: input, middle: prediction, bottom: observed")
            }, step=self.iter)

    def train(self) -> None:
        # Log gradients and model parameters
        if self.use_wandb and self.args.get('log_gradients', False):
            wandb.watch(self.model, log='all')

        # Log validation metrics before training starts (to log initial improvement)
        if not (self.args.resume and self.args.pretrained_path):
            self.validate_epoch()
            self._log_stats_meter(phase='val')

        self.logger.info('\nStart training...\n')
        start_time = time.time()

        with tqdm(range(self.epoch, self.args.num_epochs), leave=True) as tnr:
            tnr.set_description("Epoch")
            tnr.set_postfix(epoch=self.epoch, training_loss=np.nan, validation_loss=np.nan,
                            best_validation_loss=self.best_loss)
            for _ in tnr:
                if self.scheduler is not None:
                    self._log_learning_rate()

                # -------------------------------- TRAINING -------------------------------- #
                self.train_epoch(tnr)

                # -------------------------------- VALIDATION -------------------------------- #
                if (self.epoch + 1) % self.args.val_every_n_epochs == 0:
                    self.validate_epoch(tnr)
                    self._log_stats_meter(phase='val')

                    # Save the best model
                    if self.val_stats.total_loss.avg < self.best_loss:
                        self.best_loss = self.val_stats.total_loss.avg
                        self.epoch_best_loss = self.epoch
                        self._save_checkpoint(self.args.path_model_best)
                        if self.use_wandb:
                            wandb.run.summary["best_loss"] = self.val_stats.total_loss.avg
                            wandb.run.summary["epoch_best_loss"] = self.epoch

                    # Plot inference
                    if (self.epoch + 1) % self.args.plot_every_n_epochs == 0 and self.use_wandb:
                        self._visualize_sample_wandb()  # Plot a random validation sample
                        if self.args.get('plot_val_sample', None) is not None:
                            # Plot specific validation sample(s)
                            if isinstance(self.args.plot_val_sample, int):
                                self._visualize_sample_wandb(sample_index=self.args.plot_val_sample)
                            elif isinstance(self.args.plot_val_sample, (list, ListConfig)):
                                for idx in self.args.plot_val_sample:
                                    self._visualize_sample_wandb(sample_index=idx)

                # After the epoch if finished, update the learning rate scheduler
                if self.scheduler is not None:
                    self._log_learning_rate()

                    if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                        self.scheduler.step(self.val_stats.total_loss.avg)
                    else:
                        self.scheduler.step()

                # Save the model at the selected interval
                if (self.epoch + 1) % self.args.checkpoint_every_n_epochs == 0:
                    name = 'Model_after_' + str(self.epoch + 1) + '_epochs.pth'
                    self._save_checkpoint(os.path.join(self.args.checkpoint_dir, name))

                self.epoch += 1

        time_elapsed = int(time.time() - start_time)
        self.logger.info('\n\nTraining finished!\nTraining time: %dd %dh %dm %ds' % seconds_to_dd_hh_mm_ss(time_elapsed))
        self.logger.info('\nBest model at epoch: %d', self.epoch_best_loss)
        self.logger.info(f'Validation loss of the best model: {self.best_loss:.4f}')

        # Save the last model
        self._save_checkpoint(self.args.path_model_last)

        if self.use_wandb:
            wandb.finish()

    def train_epoch(self, tnr=None) -> None:
        # Initialize stats meter
        self.train_stats = self._stats_meter(stats_type='loss')
        self.train_metrics = self._stats_meter(stats_type='metrics')
        self.model.train()

        # Clear gradients
        for param in self.model.parameters():
            param.grad = None

        with tqdm(self.dataloader['train'], leave=False) as tnr_train:
            tnr_train.set_description("Training")
            tnr_train.set_postfix(epoch=self.epoch, training_loss=np.nan,
                                  **{k: v.avg for (k, v) in self.train_metrics.items()})

            for i, batch in enumerate(tnr_train):
                self._log_iter_epoch()
                loss_dict, metrics, loss = self.inference_one_batch(batch, phase='train')

                # Update to stats_meter
                for key, value in loss_dict.items():
                    self.train_stats[key].update(value)
                for key, value in metrics.items():
                    self.train_metrics[key].update(value)

                loss = loss / self.args.accum_iter
                loss.backward()

                if ((i + 1) % self.args.accum_iter == 0) or (i + 1 == len(self.dataloader['train'])):
                    # Gradient clipping
                    if getattr(self.args, 'gradient_clip_norm', False) and self.args.gradient_clip_norm > 0.:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip_norm)

                    elif getattr(self.args, 'gradient_clip_value', False) and self.args.gradient_clip_value > 0.:
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.gradient_clip_value)

                    self.optimizer.step()

                    # Clear gradients
                    for param in self.model.parameters():
                        param.grad = None

                if (i + 1) % min(self.args.logstep_train, len(self.dataloader['train'])) == 0:
                    self._log_stats_meter(phase='train')

                    tnr_train.set_postfix(epoch=self.epoch, training_loss=self.train_stats.total_loss.avg,
                                          **{k: v.avg for (k, v) in self.train_metrics.items()})
                    if tnr is not None:
                        tnr.set_postfix(epoch=self.epoch,
                                        training_loss=self.train_stats.total_loss.avg,
                                        validation_loss=self.val_stats.total_loss.avg,
                                        best_validation_loss=self.best_loss)

                    # Reset stats and metrics
                    for key in self.train_stats:
                        self.train_stats[key].reset()
                    for key in self.train_metrics:
                        self.train_metrics[key].reset()

                self.iter += 1

    def validate_epoch(self, tnr=None) -> None:
        # Initialize stats meter
        self.val_stats = self._stats_meter(stats_type='loss')
        self.val_metrics = self._stats_meter(stats_type='metrics')
        self.model.eval()

        with tqdm(self.dataloader['val'], leave=False) as tnr_val:
            tnr_val.set_description("Validation")
            tnr_val.set_postfix(epoch=self.epoch)

            for _, batch in enumerate(tnr_val):
                loss_dict, metrics = self.inference_one_batch(batch, phase='val')

                # Update to stats_meter
                for key, value in loss_dict.items():
                    self.val_stats[key].update(value)
                for key, value in metrics.items():
                    self.val_metrics[key].update(value)

        if tnr is not None:
            tnr.set_postfix(epoch=self.epoch,
                            training_loss=self.train_stats.total_loss.avg,
                            validation_loss=self.val_stats.total_loss.avg,
                            best_validation_loss=self.best_loss)

    def inference_one_batch(
            self, batch: Dict[str, Any], phase: str
    ) -> Tuple[Dict[str, float], Dict[str, float], Tensor] | Tuple[Dict[str, float], Dict[str, float]]:
        assert phase in ['train', 'val', 'test']

        batch = to_device(batch, self.device)

        if phase == 'train':
            # with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # casts operations to mixed precision
            y_pred = self.model(batch['x'], batch_positions=batch['position_days'])

            # Compute losses and evaluation metrics
            loss_dict, loss = self.compute_losses(batch, y_pred)
            metrics = self.compute_metrics(batch, y_pred)

            return loss_dict, metrics, loss

        # with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # casts operations to mixed precision
        with torch.no_grad():
            y_pred = self.model(batch['x'], batch_positions=batch['position_days'])

            # Compute  losses and evaluation metrics
            loss_dict, _ = self.compute_losses(batch, y_pred)
            metrics = self.compute_metrics(batch, y_pred)

            return loss_dict, metrics
