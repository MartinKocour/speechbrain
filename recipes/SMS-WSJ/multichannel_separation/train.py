#!/usr/bin/env python
"""
Recipe for training a neral speech separation on
"""

import sys
import logging
import random

import torch
import torchaudio
import torch.nn.functional as F

import speechbrain as sb
from speechbrain.nnet import schedulers
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml

from safe_gpu import safe_gpu

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
        mix_w = mix_w[:, self.hparams.ref_microphone, ...]
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
        T_origin = mix.size(-1)
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
        targets = torch.cat([t[0].unsqueeze(-1) for t in targets], dim=-1).to(self.device)

        loss, _ = self.hparams.pit_si_snr(targets, est_source)
        return loss

    def on_stage_end(self, stage, stage_loss, epoch):
        # Compute/store loss
        stage_stats = {"si-snr": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        if stage == sb.Stage.VALID:
            # update lr
            lr_scheduler = self.hparams.lr_scheduler
            if isinstance(lr_scheduler, schedulers.ReduceLROnPlateau):
                current_lr, next_lr = lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
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
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_loss,
            )

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)


def dataio_prep(hparams):
    """Creates data processing pipeline"""

    # 1. Define datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Provide audio pipelines
    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides("mix_sig")
    def audio_pipeline_mix(mix_wav):
        mix_sig, fs = torchaudio.load(mix_wav)
        assert fs == hparams["sample_rate"]

        if "mix_microphones" in hparams:
            microphones = hparams["mix_microphones"]
            mix_sig = mix_sig[microphones]

        return mix_sig

    @sb.utils.data_pipeline.takes("s1_clean_wav", "s1_rvbearly_wav", "s1_rvbtail_wav")
    @sb.utils.data_pipeline.provides("s1_sig")
    def audio_pipeline_s1(s1_clean_wav, s1_early_wav, s1_tail_wav):
        if hparams["reverb_source"]:
            s1_early_sig, fs = torchaudio.load(s1_early_wav)
            s1_tail_sig, _ = torchaudio.load(s1_tail_wav)
            s1_sig = s1_early_sig + s1_tail_sig
        else:
            s1_sig, fs = torchaudio.load(s1_clean_wav)

        assert fs == hparams["sample_rate"]

        if "ref_microphone" in hparams:
            s1_sig = s1_sig[hparams["ref_microphone"]]
        return s1_sig

    @sb.utils.data_pipeline.takes("s2_clean_wav", "s2_rvbearly_wav", "s2_rvbtail_wav")
    @sb.utils.data_pipeline.provides("s2_sig")
    def audio_pipeline_s2(s2_clean_wav, s2_early_wav, s2_tail_wav):
        if hparams["reverb_source"]:
            s2_early_sig, fs = torchaudio.load(s2_early_wav)
            s2_tail_sig, _ = torchaudio.load(s2_tail_wav)
            s2_sig = s2_early_sig + s2_tail_sig
        else:
            s2_sig, fs = torchaudio.load(s2_clean_wav)

        assert fs == hparams["sample_rate"]

        if "ref_microphone" in hparams:
            s2_sig = s2_sig[hparams["ref_microphone"]]
        return s2_sig

    @sb.utils.data_pipeline.takes("mix_sig", "s1_sig", "s2_sig")
    @sb.utils.data_pipeline.provides("mix_sig", "s1_sig", "s2_sig")
    def cut_audio_pipeline(mix_sig, s1_sig, s2_sig):
        if s1_sig.ndim == 1:
            tmp = torch.cat([mix_sig, s1_sig.unsqueeze(0), s2_sig.unsqueeze(0)], dim=0)
        else:
            tmp = torch.cat([mix_sig, s1_sig, s2_sig], dim=0)

        # tmp, gap = hparams["MaskNet"]._Segmentation(tmp.unsqueeze(0), chunk_length)
        # tmp = tmp.squeeze(0)
        # chunk_idx = random.randint(0, tmp.size(-1) - 1)
        # tmp = tmp[..., chunk_idx]
        chunk_length = random.randint(hparams["sample_rate"], hparams["training_signal_len"])
        assert chunk_length > 0

        offset = random.randint(0, max(tmp.size(-1) - chunk_length, 0))
        tmp = tmp[..., offset:offset + chunk_length]

        if s1_sig.ndim == 1:
            s1_sig = tmp[mix_sig.size(0)]
            s2_sig = tmp[-1]
        else:
            s1_sig = tmp[mix_sig.size(0):s1_sig.size(0)]
            s2_sig = tmp[-s2_sig.size(0):]

        mix_sig = tmp[:mix_sig.size(0)]
        return mix_sig, s1_sig, s2_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_mix)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s1)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s2)
    if hparams["limit_training_signal_len"]:
        train_data.add_dynamic_item(cut_audio_pipeline)

    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "mix_sig", "s1_sig", "s2_sig"]
    )

    return train_data, valid_data, test_data


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    if "cuda" in run_opts["device"]:
        # acquire GPU
        gpu_owner = safe_gpu.GPUOwner()
        gpu_nb = gpu_owner.devices_taken[0]
        run_opts["device"] = "cuda:" + str(gpu_nb)
        run_opts["device"] = "cuda"
        logger.info("Acquired Device: " + run_opts["device"])

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation
    from smswsj_prepare import prepare_smswsj

    run_on_main(
        prepare_smswsj,
        kwargs={
            "datapath": hparams["data_folder"],
            "savepath": hparams["save_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Create dataset objects
    train_data, valid_data, test_data = dataio_prep(hparams)

    # Load pretrained model if pretrained_separator is present in the yaml
    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected()

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
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts"],
        )

    # Eval
    separator.evaluate(test_data, min_key="si-snr")
