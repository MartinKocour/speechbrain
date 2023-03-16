#!/usr/bin/env python
"""
Recipe for training a neral speech separation on 
"""

import sys
import logging

import torch
import torchaudio
import torch.nn.functional as F

import speechbrain as sb
from speechbrain.nnet import schedulers
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml

logger = logging.getLogger(__name__)

# Define training procedure
class Separation(sb.Brain):
    def compute_forward(self, batch, stage):
        """
        Forward computations from the mixture to separated source.
        """
        batch = batch.to(self.device)
    
        mix, mix_lens = batch.mix_sig

        # Separation
        mix_w = self.modules.encoder(mix)
        est_mask = self.modules.masknet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.modules.decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fit it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source
    
    def compute_objectives(self, est_source, batch, stage=sb.Stage.TRAIN):
        """Computes the si-snr loss"""
        # Convert targets to tensor
        targets = [batch.s1_sig, batch.s2_sig]
        targets = torch.cat(
            [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        ).to(self.device)

        return self.hparams.si_snr(targets, est_source)
    
    def on_stage_end(self, stage, stage_loss, epoch):
        # Compute/store loss
        stage_stats = {"si-snr": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        if stage == sb.Stage.VALID:
            # update lr
            lr_scheduler = self.hparams.lr_scheduler
            if isinstance(lr_scheduler, schedulers.ReduceLROnPlateau):
                current_lr, next_lr = lr_scheduler([self.optimizer], epoch, stage_loss)
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"si-snr": stage_loss}, min_keys=["si-snr"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta = {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_loss,
            )
    

if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Brain class initialization
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # re-initialize the parameters if we don't use a pretrained model
    if "pretrained_separator" not in hparams:
        for module in separator.modules.values():
            separator.reset_layer_recursively(module)

    if not hparams["test_only"]:
        # Training
        separator.fit(
            separator.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_dataloader_opts"],
            valid_loader_kwargs=hparams["valid_dataloader_opts"],
        )
    
    # Eval
    separator.evaluate(test_data, min_key="si-snr")

