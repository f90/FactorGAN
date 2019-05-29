import glob
import os
import os.path
from multiprocessing import Pool

import musdb
import numpy as np
import soundfile
import torch
from torch.utils.data.dataset import Dataset

import Utils

def preprocess_song(item):
    idx, song, out_path, sample_rate, input_width, mode, fft_size, hop_size = item

    if not os.path.exists(os.path.join(out_path, str(idx))):
        os.makedirs(os.path.join(out_path, str(idx)))

    length = np.Inf
    if mode == "paired" or mode == "mix":
        mix_audio, _ = Utils.load(song["mix"], sr=sample_rate, mono=True)
        mix, _ = Utils.compute_spectrogram(np.squeeze(mix_audio, 1), fft_size, hop_size)
        length = min(mix.shape[1], length)

    if mode == "paired" or mode == "accompaniment":
        accompaniment_audio, _ = Utils.load(song["accompaniment"], sr=sample_rate, mono=True)
        accompaniment, _ = Utils.compute_spectrogram(np.squeeze(accompaniment_audio, 1), fft_size, hop_size)
        length = min(accompaniment.shape[1], length)

    if mode == "paired" or mode == "vocals":
        vocals_audio, _ = Utils.load(song["vocals"], sr=sample_rate, mono=True)
        vocals, _ = Utils.compute_spectrogram(np.squeeze(vocals_audio, 1), fft_size, hop_size)
        length = min(vocals.shape[1], length)

    sample_num = 0
    for start_pos in range(0, length - input_width, input_width // 2):
        sample = list()
        if mode == "paired" or mode == "mix":
            sample.append(mix[:, start_pos:start_pos + input_width])

        if mode == "paired" or mode == "accompaniment":
            sample.append(accompaniment[:, start_pos:start_pos + input_width])

        if mode == "paired" or mode == "vocals":
            sample.append(vocals[:, start_pos:start_pos + input_width])

        # Write current snippet
        sample = np.stack(sample, axis=0)
        np.save(os.path.join(out_path, str(idx), str(sample_num) + ".npy"), sample)
        sample_num += 1

class MUSDBDataset(Dataset):
    def __init__(self, opt, song_idx, mode):
        self.opt = opt
        self.mode = mode
        # Load MUSDB/convert to wav
        dataset = getMUSDB(opt.musdb_path)[0]

        self.out_path = os.path.join(opt.preprocessed_dataset_path, mode)

        if not os.path.exists(self.out_path):
            # Preprocess audio into spectrogram and write into each sample into a numpy file
            p = Pool(10) #multiprocessing.cpu_count())
            p.map(preprocess_song, [(curr_song_idx, song, self.out_path, opt.sample_rate, opt.input_width, mode, opt.fft_size, opt.hop_size) for curr_song_idx, song in enumerate(dataset)])

        # Select songs to use for training
        file_list = list()
        for idx in song_idx:
            npy_files = glob.glob(os.path.join(self.out_path, str(idx), "*.npy"))
            file_list.extend(npy_files)

        self.dataset = file_list

    def __getitem__(self, index):
        return self.npy_loader(self.dataset[index])

    def __len__(self):
        return len(self.dataset)

    def npy_loader(self, path):
        sample = torch.from_numpy(np.load(path))
        return sample

def getMUSDB(database_path):
    mus = musdb.DB(root_dir=database_path, is_wav=False)

    subsets = list()

    for subset in ["train", "test"]:
        tracks = mus.load_mus_tracks(subset)
        samples = list()

        # Go through tracks
        for track in tracks:
            # Skip track if mixture is already written, assuming this track is done already
            track_path = track.path[:-4]
            mix_path = track_path + "_mix.wav"
            acc_path = track_path + "_accompaniment.wav"
            if os.path.exists(mix_path):
                print("WARNING: Skipping track " + mix_path + " since it exists already")

                # Add paths and then skip
                paths = {"mix" : mix_path, "accompaniment" : acc_path}
                paths.update({key : track_path + "_" + key + ".wav" for key in ["bass", "drums", "other", "vocals"]})

                samples.append(paths)

                continue

            rate = track.rate

            # Go through each instrument
            paths = dict()
            stem_audio = dict()
            for stem in ["bass", "drums", "other", "vocals"]:
                path = track_path + "_" + stem + ".wav"
                audio = track.targets[stem].audio
                soundfile.write(path, audio, rate, "PCM_16")
                stem_audio[stem] = audio
                paths[stem] = path

            # Add other instruments to form accompaniment
            acc_audio = np.clip(sum([stem_audio[key] for key in stem_audio.keys() if key != "vocals"]), -1.0, 1.0)
            soundfile.write(acc_path, acc_audio, rate, "PCM_16")
            paths["accompaniment"] = acc_path

            # Create mixture
            mix_audio = track.audio
            soundfile.write(mix_path, mix_audio, rate, "PCM_16")
            paths["mix"] = mix_path

            diff_signal = np.abs(mix_audio - acc_audio - stem_audio["vocals"])
            print("Maximum absolute deviation from source additivity constraint: " + str(np.max(diff_signal)))# Check if acc+vocals=mix
            print("Mean absolute deviation from source additivity constraint:    " + str(np.mean(diff_signal)))

            samples.append(paths)

        subsets.append(samples)

    return subsets