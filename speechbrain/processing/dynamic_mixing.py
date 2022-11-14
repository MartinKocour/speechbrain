"""Implementation of dynamic mixing for speech separation

Authors
    * Samuele Cornell 2021
    * Cem Subakan 2021
    * Martin Kocour 2022
"""

from speechbrain.processing.signal_processing import reverberate

import torch
import torchaudio
import numpy as np
import numbers
import random
import warnings
import uuid
import pyloudnorm  # WARNING: External dependency

from dataclasses import dataclass, fields
from typing import Optional, Union


@dataclass
class DynamicMixingConfig:
    num_spkrs: Union[int, list] = 2
    overlap_ratio: Union[int, list] = 1.0
    audio_norm: bool = True  # normalize loudness of sources
    audio_min_loudness: float = -33.0  # dB
    audio_max_loudness: float = -25.0  # dB
    audio_max_amp: float = 0.9  # max amplitude in mixture and sources
    noise_add: bool = False
    noise_prob: float = 1.0
    noise_files: Optional[list] = None
    # noise_snr: float = 20.0 # dB TODO
    noise_min_loudness: float = -33.0 - 5
    noise_max_loudness: float = -25.0 - 5
    white_noise_add: bool = True
    white_noise_mu: float = 0.0
    white_noise_var: float = 1e-7
    rir_add: bool = False
    rir_prob: float = 1.0
    rir_files: Optional[list] = None  # RIR waveforms
    min_source_len: int = 32000
    max_source_len: int = 64000

    @classmethod
    def from_hparams(cls, hparams):
        config = {}
        for fld in fields(cls):
            config[fld.name] = hparams.get(fld.name, fld.default)
        return cls(**config)

    def __post_init__(self):
        if isinstance(self.num_spkrs, int):
            self.num_spkrs = [self.num_spkrs]

        if isinstance(self.overlap_ratio, numbers.Real):
            self.overlap_ratio = [self.overlap_ratio]


class DynamicMixingDataset(torch.utils.data.Dataset):
    """Dataset which creates mixtures from single-talker dataset

    Example
    -------
    >>> data = DynamicItemDataset.from_csv(csv_path)
    ... data = [
    ...     {
    ...         'wav_file': '/example/path/src1.wav',
    ...         'spkr': 'Gandalf',
    ...     },
    ...     {   'wav_file': '/example/path/src2.wav',
    ...         'spkr': 'Frodo',
    ...     }
    ... ]
    >>> config = DynamicMixingConfig.from_hparams(hparams)
    >>> dm_dataset = DynamicMixixingDataset.from_didataset(data, config, "wav_file", "spkr")
    >>> mixture, spkrs, ratios, sources = dm_dataset.generate()

    Arguments
    ---------
    spkr_files : dict
    config: DynamicMixingConfig
    """

    def __init__(self, spkr_files, config):
        if len(spkr_files.keys()) < max(config.num_spkrs):
            raise ValueError(
                f"Expected at least {config.num_spkrs} spkrs in spkr_files"
            )

        self.num_spkrs = config.num_spkrs
        self.overlap_ratio = config.overlap_ratio
        self.normalize_audio = config.audio_norm
        self.spkr_files = spkr_files

        tmp_file, _ = next(iter(spkr_files.values()))[0]
        self.sampling_rate = torchaudio.info(tmp_file).sample_rate

        self.meter = None
        if self.normalize_audio:
            self.meter = pyloudnorm.Meter(self.sampling_rate)

        self.config = config
        self.dataset = None  # used for inner database

    @classmethod
    def from_didataset(cls, dataset, config, wav_key=None, spkr_key=None):
        if wav_key is None:
            raise ValueError("Provide valid wav_key for dataset item")

        if spkr_key is None:
            files = [(d[wav_key], idx) for idx, d in enumerate(dataset)]
            dmdataset = cls.from_wavs(files, config)
        else:
            spkr_files = {}
            for idx, d in enumerate(dataset):
                spkr_files[d[spkr_key]] = spkr_files.get(d[spkr_key], [])
                spkr_files[d[spkr_key]].append((d[wav_key], idx))
            dmdataset = cls(spkr_files, config)
        dmdataset.set_dataset(dataset)
        return dmdataset

    @classmethod
    def from_wavs(cls, wav_file_list, config):
        spkr_files = {}
        spkr = 0
        # we assume that each wav is coming from different spkr
        for wavfile in wav_file_list:
            spkr_files[f"spkr{spkr}"] = [(wavfile, None)]
            spkr += 1

        return cls(spkr_files, config)

    def set_dataset(self, dataset):
        self.dataset = dataset

    def generate(self):
        """Generate new audio mixture

        Returns:
          - mixture
          - mixed spkrs
          - used overlap ratios
          - padded sources
          - noise
          - data
        """
        # TODO: Refactor completly, add Mixture class
        n_spkrs = np.random.choice(self.config.num_spkrs)
        if n_spkrs <= 0:
            length = random.randint(
                self.config.min_source_len, self.config.max_source_len
            )
            sources = [torch.zeros(length)]
            mixture, sources, noise = self.__postprocess__(sources[0], sources)
            return mixture, [], [], sources, noise, []

        mix_spkrs = np.random.choice(list(self.spkr_files.keys()), n_spkrs)
        rir = None
        if self.config.rir_add:
            rir_file = np.random.choice(self.config.rir_files)
            rir, fs = torchaudio.load(rir_file)
            assert (
                fs == self.sampling_rate
            ), f"{self.sampling_rate} Hz sampling rate expected, but found {fs}"
            rir = rir[0]

        sources = []
        source_idxs = []
        fs = None
        for spkr in mix_spkrs:
            spkr_idx = random.randint(0, len(self.spkr_files[spkr]) - 1)
            src_file, src_idx = self.spkr_files[spkr][spkr_idx]
            src_audio, fs = torchaudio.load(src_file)
            assert (
                fs == self.sampling_rate
            ), f"{self.sampling_rate} Hz sampling rate expected, but found {fs}"
            src_audio = src_audio[0]  # Support only single channel
            # use same RIR for all sources
            src_audio = self.__prepare_source__(src_audio, rir)
            sources.append(src_audio)
            source_idxs.append(src_idx)

        sources, source_idxs = zip(
            *sorted(
                zip(sources, source_idxs),
                key=lambda x: x[0].size(0),
                reverse=True,
            )
        )
        mixture = sources[0]  # longest audio
        padded_sources = [sources[0]]
        overlap_ratios = []
        for i in range(1, len(sources)):
            src = sources[i]
            ratio = np.random.choice(self.config.overlap_ratio)
            overlap_samples = int(src.size(0) * ratio)

            mixture, padded_tmp, paddings = mix(src, mixture, overlap_samples)
            # padded sources are returned in same order
            overlap_ratios.append((ratio, paddings))

            # previous padded_sources are shorter
            padded_sources = __pad_sources__(
                padded_sources,
                [paddings[1] for _ in range(len(padded_sources))],
            )
            padded_sources.append(padded_tmp[0])
        mixture, padded_source, noise = self.__postprocess__(
            mixture, padded_sources
        )

        data = None
        if self.dataset is not None:
            data = [self.dataset[idx] for idx in source_idxs]
        return mixture, mix_spkrs, overlap_ratios, padded_sources, noise, data

    def __prepare_source__(self, source, rir, is_noise=False):

        # cut the source to a random length
        length = random.randint(
            self.config.min_source_len, self.config.max_source_len
        )
        source = source[:length]

        if self.normalize_audio:
            # normalize loudness and apply random gain
            source = normalize(
                source,
                self.meter,
                self.config.audio_min_loudness
                if not is_noise
                else self.config.noise_min_loudness,
                self.config.audio_max_loudness
                if not is_noise
                else self.config.noise_max_loudness,
                self.config.audio_max_amp,
            )

        # add reverb
        if (
            not is_noise
            and self.config.rir_add
            and random.uniform(0, 1) < self.config.rir_prob
        ):
            # noise is not reverberated
            reverberate(source, rir)
        return source

    def __postprocess__(self, mixture, sources):
        # add noise
        noise = None
        if (
            self.config.noise_add
            and random.uniform(0, 1) < self.config.noise_prob
        ):
            noise_f = np.random.choice(self.config.noise_files)
            noise, fs = torchaudio.load(noise_f)
            assert (
                fs == self.sampling_rate
            ), f"{self.sampling_rate} Hz sampling rate expected, but found {fs}"
            noise = self.__prepare_source__(noise[0], is_noise=True)
            noise = noise.repeat(
                mixture.size(0) // noise.size(0) + 1
            )  # extend the noise
            mixture += noise[: mixture.size(0)]

        # replace zeros with small gaussian noise
        if self.config.white_noise_add:
            white_noise = np.random.normal(
                self.config.white_noise_mu,
                self.config.white_noise_var,
                size=mixture.size(0),
            )
            white_noise = torch.from_numpy(white_noise)
            mixture += white_noise

        # normalize gain
        # this should be the final step
        mix_max_amp = mixture.abs().max().item()
        gain = 1.0
        if mix_max_amp > self.config.audio_max_amp:
            gain = self.config.audio_max_amp / mix_max_amp

        mixture = gain * mixture
        sources = map(lambda src: gain * src, sources)
        return mixture, sources, noise

    def __len__(self):
        return sum(map(len, self.spkr_files.values()))  # dict of lists

    def __getitem__(self, idx):
        # TODO: Refactor completly
        mix, spkrs, ratios, srcs, noise, data = self.generate()
        if len(srcs) != 2:
            raise NotImplementedError("getitem supports exactly 2 sources")

        if idx is None:
            idx = uuid.uuid4()
        mix_id = (
            str(idx)
            + "_"
            + "-".join(spkrs)
            + "_overlap"
            + "-".join(map(lambda x: f"{x[0]:.2f}", ratios))
        )
        # "id", "mix_sig", "s1_sig", "s2_sig", "s3_sig", "noise_sig"
        dct = {
            "mix_id": mix_id,
            "mix_sig": mix,
            "s1_sig": srcs[0],
            "s2_sig": srcs[1],
            "s3_sig": torch.zeros(mix.size(0)),
            "noise_sig": noise if noise else torch.zeros(mix.size(0)),
            "data": data,
        }

        return dct


def normalize(audio, meter, min_loudness=-33, max_loudness=-25, max_amp=0.9):
    """This function normalizes the loudness of audio signal"""
    audio = audio.numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_loudness = meter.integrated_loudness(audio)
        target_loudness = random.uniform(min_loudness, max_loudness)
        # TODO: pyloudnorm.normalize.loudness could be replaced by rescale from SB
        signal = pyloudnorm.normalize.loudness(
            audio, c_loudness, target_loudness
        )

        # check for clipping
        if np.max(np.abs(signal)) >= 1:
            signal = signal * max_amp / np.max(np.abs(signal))

    return torch.from_numpy(signal)


def mix(src1, src2, overlap_samples):
    """Mix two audio samples"""
    n_diff = len(src1) - len(src2)
    swapped = False
    if n_diff >= 0:
        longer_src = src1
        shorter_src = src2
        swapped = True
    else:
        longer_src = src2
        shorter_src = src1
        n_diff = abs(n_diff)
    n_long = len(longer_src)
    n_short = len(shorter_src)

    paddings = []
    if overlap_samples >= n_short:
        # full overlap
        lpad = np.random.choice(range(n_diff)) if n_diff > 0 else 0
        rpad = n_diff - lpad
        paddings = [(lpad, rpad), (0, 0)]
    elif overlap_samples > 0:
        # partial overlap
        start_short = np.random.choice([True, False])  # start with short
        n_total = n_long + n_short - overlap_samples
        if start_short:
            paddings = [(0, n_total - n_short), (n_total - n_long, 0)]
        else:
            paddings = [(n_total - n_short, 0), (0, n_total - n_long)]
    else:
        # no-overlap
        sil_between = abs(overlap_samples)
        start_short = np.random.choice([True, False])  # start with short
        if start_short:
            paddings = [(0, sil_between + n_long), (sil_between + n_short, 0)]
        else:
            paddings = [(sil_between + n_long, 0), (0, sil_between + n_short)]

    assert len(paddings) == 2
    src1, src2 = __pad_sources__([shorter_src, longer_src], paddings)
    sources = (
        torch.stack((src2, src1)) if swapped else torch.stack((src1, src2))
    )
    if swapped:
        paddings.reverse()

    mixture = torch.sum(sources, dim=0)
    return mixture, sources, paddings


def __pad_sources__(sources, paddings):
    result = []
    for src, (lpad, rpad) in zip(sources, paddings):
        nsrc = torch.cat((torch.zeros(lpad), src, torch.zeros(rpad)))
        result.append(nsrc)
    return result
