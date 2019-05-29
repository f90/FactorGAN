import os

import librosa
import numpy as np
import torch
import random
import math

def create_image_pair(x, ch1, ch2):
    '''
    Concatenates two images horizontally (that are saved in x using 3 or 1 channels for each image
    :param x: Pair of images (along channel dimension)
    :param ch1: Number of channels for image 1
    :param ch2: Number of channels for image 2
    :return: Horizontally stacked image pair
    '''
    assert(ch1 == 3 or ch1 == 1)
    assert(ch2 == 3 or ch2 == 1)
    assert(x.shape[1] == ch1 + ch2)

    repeat_left = 3 if ch1 == 1 else 1
    repeat_right = 3 if ch2 == 1 else 1
    return torch.cat([x[:, :ch1, :, :].repeat(1, repeat_left, 1, 1), x[:, ch1:, :, :].repeat(1,repeat_right,1,1)], dim=3)

def is_square(integer):
    '''
    Check if number is a square of another number
    :param integer: Number to be checked
    :return: Whether number is square of another number
    '''
    root = math.sqrt(integer)
    if int(root + 0.5) ** 2 == integer:
        return True
    else:
        return False

def shuffle_batch_dims(batch, marginal_index, dim=1):
    '''
    Shuffles groups of dimensions of a batch of samples so that groups are drawn independently of each other
    :param batch: Input batch to be shuffled
    :param marginal_index: If list: List of indices that denote the boundaries of the groups to be shuffled, excluding 0 and batch.shape[1].
     If int: Each group has this many dimensions, batch.shape[1] must be divisible by this number. If None: Input batch needs to have groups as dimensions: [Num_samples, Group1_dim, ... GroupN_dim]
    :return: Shuffled batch
    '''

    if isinstance(batch, torch.Tensor):
        out = batch.clone()
    else:
        out = batch.copy()

    if isinstance(marginal_index, int):
        assert (batch.shape[dim] % marginal_index == 0)
        marginal_index = [(x+1)*marginal_index for x in range(int(batch.shape[1] / marginal_index) - 1)]
    if isinstance(marginal_index, list):
        groups = marginal_index + [batch.shape[dim]]
        for group_idx in range(len(groups)-1): # Shuffle each group, except the first one
            dim_start = groups[group_idx]
            dim_end = groups[group_idx+1]
            ordering = np.random.permutation(batch.shape[0])
            if dim == 1:
                out[:,dim_start:dim_end] = batch[ordering, dim_start:dim_end]
            elif dim == 2:
                out[:, :, dim_start:dim_end] = batch[ordering, :, dim_start:dim_end]
            elif dim == 3:
                out[:, :, :, dim_start:dim_end] = batch[ordering, :, :, dim_start:dim_end]
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError

    return out

def load(path, sr=22050, mono=True, offset=0.0, duration=None, dtype=np.float32):
    # ALWAYS output (n_frames, n_channels) audio
    y, orig_sr = librosa.load(path, sr, mono, offset, duration, dtype)
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=0)
    return y.T, orig_sr

def shuffle_batch_image_quadrants(batch):
    '''
    Given an input batch of square images, shuffle the four quadrants independently across examples
    :param batch: Input batch of square images
    :return: Shuffled square images
    '''
    input_shape = batch.shape
    if len(batch.shape) == 2: # [batch, dim] shape means we have to reshape
        # Check if data can be shaped into square image, if it is not already
        dim = int(batch.shape[1])
        root = int(math.sqrt(dim) + 0.5)
        assert(root ** 2 == dim)
    elif len(batch.shape) > 2:
        # Check if last two dimensions are the same size N, and reshape to [-1, C, N, N]
        assert(batch.shape[-2] == batch.shape[-1])
        root = batch.shape[-1]
    else:
        raise SyntaxError
    assert(root % 2 == 0) # Image should be splittable in half
    q = root // 2 # Length/width of each quadrant

    # Change to [B, C, N, N] shape
    if isinstance(batch, torch.Tensor):
        batch_reshape = batch.view((batch.shape[0], -1, root, root))
        out = batch_reshape.clone()
    else:
        batch_reshape = np.reshape(batch, (batch.shape[0], -1, root, root))
        out = batch_reshape.copy()

    # Shuffle the four quadrants of the square image around
    for row in range(2):
        for col in range(2):
            if row == 0 and col == 0: continue # Do not need to shuffle first quadrant, if we shuffle all the others across the batch

            ordering = np.random.permutation(batch.shape[0])
            out[:, :, row*q:(row+1)*q, col*q:(col+1)*q] = batch_reshape[ordering, :, row*q:(row+1)*q, col*q:(col+1)*q]

    # Reshape to the shape of the original input
    if isinstance(batch, torch.Tensor):
        out = out.view(input_shape)
    else:
        out = np.reshape(out, input_shape)

    return out

def compute_spectrogram(audio, fft_size, hop_size):
    '''
    Compute magnitude spectrogram for audio signal
    :param audio: Audio input signal
    :param fft_size: FFT Window size (samples)
    :param hop_size: Hop size (samples) for STFT
    :return: Magnitude spectrogram
    '''
    stft = librosa.core.stft(audio, fft_size, hop_size)
    mag, ph = librosa.core.magphase(stft)

    return normalise_spectrogram(mag), ph

def normalise_spectrogram(mag, cut_last_freq=True):
    '''
    Normalise audio spectrogram with log-normalisation
    :param mag: Magnitude spectrogram to be normalised
    :param cut_last_freq: Whether to cut highest frequency bin to reach power of 2 in number of bins
    :return: Normalised spectrogram
    '''
    if cut_last_freq:
        # Throw away last freq bin to make it number of freq bins a power of 2
        out = mag[:-1,:]

    # Normalize with log1p
    out = np.log1p(out)
    return out

def normalise_spectrogram_torch(mag):
    return torch.log1p(mag)

def denormalise_spectrogram(mag, pad_freq=True):
    '''
    Reverses normalisation performed in "normalise_spectrogram" function
    :param mag: Normalised magnitudes
    :param pad_freq: Whether to append a frequency bin as highest frequency with 0 as energy
    :return: Reconstructed spectrogram
    '''
    out = np.expm1(mag)

    if pad_freq:
        out = np.pad(out, [(0,1), (0, 0)], mode="constant")

    return out

def denormalise_spectrogram_torch(mag):
    return torch.expm1(mag)

def spectrogramToAudioFile(magnitude, fftWindowSize, hopSize, phaseIterations=10, phase=None, length=None):
    '''
    Computes an audio signal from the given magnitude spectrogram, and optionally an initial phase.
    Griffin-Lim is executed to recover/refine the given the phase from the magnitude spectrogram.
    :param magnitude: Magnitudes to be converted to audio
    :param fftWindowSize: Size of FFT window used to create magnitudes
    :param hopSize: Hop size in frames used to create magnitudes
    :param phaseIterations: Number of Griffin-Lim iterations to recover phase
    :param phase: If given, starts ISTFT with this particular phase matrix
    :param length: If given, audio signal is clipped/padded to this number of frames
    :return:
    '''
    if phase is not None:
        if phaseIterations > 0:
            # Refine audio given initial phase with a number of iterations
            return reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations, phase, length)
        # reconstructing the new complex matrix
        stftMatrix = magnitude * np.exp(phase * 1j) # magnitude * e^(j*phase)
        audio = librosa.istft(stftMatrix, hop_length=hopSize, length=length)
    else:
        audio = reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations)
    return audio

def reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations=10, initPhase=None, length=None):
    '''
    Griffin-Lim algorithm for reconstructing the phase for a given magnitude spectrogram, optionally with a given
    intial phase.
    :param magnitude: Magnitudes to be converted to audio
    :param fftWindowSize: Size of FFT window used to create magnitudes
    :param hopSize: Hop size in frames used to create magnitudes
    :param phaseIterations: Number of Griffin-Lim iterations to recover phase
    :param initPhase: If given, starts reconstruction with this particular phase matrix
    :param length: If given, audio signal is clipped/padded to this number of frames
    :return:
    '''
    for i in range(phaseIterations):
        if i == 0:
            if initPhase is None:
                reconstruction = np.random.random_sample(magnitude.shape) + 1j * (2 * np.pi * np.random.random_sample(magnitude.shape) - np.pi)
            else:
                reconstruction = np.exp(initPhase * 1j) # e^(j*phase), so that angle => phase
        else:
            reconstruction = librosa.stft(audio, fftWindowSize, hopSize)
        spectrum = magnitude * np.exp(1j * np.angle(reconstruction))
        if i == phaseIterations - 1:
            audio = librosa.istft(spectrum, hopSize, length=length)
        else:
            audio = librosa.istft(spectrum, hopSize)
    return audio

def make_dirs(dirs):
    if isinstance(dirs, str):
        dirs = [dirs]
    assert(isinstance(dirs, list))
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def create_optim(parameters, opt):
    return torch.optim.Adam(parameters, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.L2)

def get_device(cuda):
    if torch.cuda.is_available() and not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda:0" if cuda else "cpu")
    return device

def set_seeds(opt):
    '''
    Set Python, numpy as and torch random seeds to a fixed number
    :param opt: Option dictionary containined .seed member value
    '''
    if opt.seed is None:
        opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)