"""
SMS-WSJ data preparation.

Download: https://github.com/fgnt/sms_wsj

Author
------
2023 Martin Kocour (Brno University of Technology)
"""


import csv
import os
import torchaudio
import logging

logger = logging.getLogger(__name__)


def prepare_smswsj(datapath, savepath, skip_prep=False):
    """
    Prepare SMS-WSJ data

    Arguments:
    ----------
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file.
    """
    if skip_prep:
        return
    _create_smswsj_csv(datapath, savepath)


def _create_smswsj_csv(datapath, savepath):
    folder_names = {
        "mix": "observation",
        "src": "speech_source",
        "noise": "noise",
        "reverb1": "early",
        "reverb2": "tail",
    }

    os.makedirs(savepath, exist_ok=True)

    for set_type in ["cv_dev93", "test_eval92", "train_si284"]:
        logger.info(f"Preparing csv file for: {set_type}")

        mix_path = os.path.join(datapath, folder_names["mix"], set_type)
        src_path = os.path.join(datapath, folder_names["src"], set_type)
        noise_path = os.path.join(datapath, folder_names["noise"], set_type)
        early_path = os.path.join(datapath, folder_names["reverb1"], set_type)
        tail_path = os.path.join(datapath, folder_names["reverb2"], set_type)

        files = os.listdir(mix_path)
        mix_utts = list(
            map(lambda path: path.split(".")[0], files)
        )  # remove suffix

        mix_fl_paths = [os.path.join(mix_path, fl + ".wav") for fl in mix_utts]
        noise_fl_paths = [
            os.path.join(noise_path, fl + ".wav") for fl in mix_utts
        ]

        src0_clean_fl_paths = [
            os.path.join(src_path, fl + "_0.wav") for fl in mix_utts
        ]
        src0_rvbearly_fl_paths = [
            os.path.join(early_path, fl + "_0.wav") for fl in mix_utts
        ]
        src0_rvblate_fl_paths = [
            os.path.join(tail_path, fl + "_0.wav") for fl in mix_utts
        ]

        src1_clean_fl_paths = [
            os.path.join(src_path, fl + "_1.wav") for fl in mix_utts
        ]
        src1_rvbearly_fl_paths = [
            os.path.join(early_path, fl + "_1.wav") for fl in mix_utts
        ]
        src1_rvbtail_fl_paths = [
            os.path.join(tail_path, fl + "_1.wav") for fl in mix_utts
        ]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav",
            "mix_wav_format",
            "mix_wav_opts",
            "s1_clean_wav",
            "s1_clean_wav_format",
            "s1_clean_wav_opts",
            "s1_rvbearly_wav",
            "s1_rvbearly_wav_format",
            "s1_rvbearly_wav_opts",
            "s1_rvbtail_wav",
            "s1_rvbtail_wav_format",
            "s1_rvbtail_wav_opts",
            "s2_clean_wav",
            "s2_clean_wav_format",
            "s2_clean_wav_opts",
            "s2_rvbearly_wav",
            "s2_rvbearly_wav_format",
            "s2_rvbearly_wav_opts",
            "s2_rvbtail_wav",
            "s2_rvbtail_wav_format",
            "s2_rvbtail_wav_opts",
            "noise_wav",
            "noise_wav_format",
            "noise_wav_opts",
        ]

        all_fl_paths = [
            mix_fl_paths,
            src0_clean_fl_paths,
            src0_rvbearly_fl_paths,
            src0_rvblate_fl_paths,
            src1_clean_fl_paths,
            src1_rvbearly_fl_paths,
            src1_rvbtail_fl_paths,
            noise_fl_paths,
        ]

        with open(
            os.path.join(savepath, "smswsj" + "_" + set_type + ".csv"), "w"
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (m, s1_c, s1_e, s1_t, s2_c, s2_e, s2_t, n) in enumerate(
                zip(*all_fl_paths)
            ):

                meta_info = torchaudio.info(m)
                row = {
                    "ID": i,
                    "duration": round(
                        meta_info.num_frames / meta_info.sample_rate, 3
                    ),
                    "mix_wav": m,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_clean_wav": s1_c,
                    "s1_clean_wav_format": "wav",
                    "s1_clean_wav_opts": None,
                    "s1_rvbearly_wav": s1_e,
                    "s1_rvbearly_wav_format": "wav",
                    "s1_rvbearly_wav_opts": None,
                    "s1_rvbtail_wav": s1_t,
                    "s1_rvbtail_wav_format": "wav",
                    "s1_rvbtail_wav_opts": None,
                    "s2_clean_wav": s2_c,
                    "s2_clean_wav_format": "wav",
                    "s2_clean_wav_opts": None,
                    "s2_rvbearly_wav": s2_e,
                    "s2_rvbearly_wav_format": "wav",
                    "s2_rvbearly_wav_opts": None,
                    "s2_rvbtail_wav": s2_t,
                    "s2_rvbtail_wav_format": "wav",
                    "s2_rvbtail_wav_opts": None,
                    "noise_wav": n,
                    "noise_wav_format": "wav",
                    "noise_wav_opts": None,
                }
                writer.writerow(row)
