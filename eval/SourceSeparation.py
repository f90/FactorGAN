import librosa
import musdb
import museval
import numpy as np
import torch
import glob
import os
import json

import Utils


def produce_musdb_source_estimates(model_config, model, model_noise, output_path, subsets=None):
    '''
    Predicts source estimates for MUSDB for a given model checkpoint and configuration, and evaluate them.
    :param model_config: Model configuration of the model to be evaluated
    :param load_model: Model checkpoint path
    :return:
    '''
    print("Evaluating trained model on MUSDB and saving source estimate audio to " + str(output_path))
    model.eval()

    mus = musdb.DB(root_dir=model_config.musdb_path)
    predict_fun = lambda track : predict(track, model_config, model, model_noise, output_path)
    assert(mus.test(predict_fun))
    mus.run(predict_fun, estimates_dir=output_path, subsets=subsets)


def predict(track, model_config, model, model_noise, results_dir=None):
    '''
    Function in accordance with MUSB evaluation API. Takes MUSDB track object and computes corresponding source estimates, as well as calls evlauation script.
    Model has to be saved beforehand into a pickle file containing model configuration dictionary and checkpoint path!
    :param track: Track object
    :param results_dir: Directory where SDR etc. values should be saved
    :return: Source estimates dictionary
    '''

    # Get noise once, use that for all predictions to keep consistency
    noise = model_noise.sample()

    # Determine input and output shapes, if we use U-net as separator
    sep_input_shape = [1, 1, model_config.input_height, model_config.input_width]  # [N, C, H, W]

    print("Testing...")

    mix_audio, orig_sr, mix_channels = track.audio, track.rate, track.audio.shape[1] # Audio has (n_samples, n_channels) shape
    separator_preds = predict_track(model_config, model, noise, mix_audio, orig_sr, sep_input_shape, sep_input_shape)

    # Upsample predicted source audio and convert to stereo. Make sure to resample back to the exact number of samples in the original input (with fractional orig_sr/new_sr this causes issues otherwise)
    pred_audio = {name : librosa.resample(separator_preds[name], model_config.sample_rate, orig_sr)[:len(mix_audio)] for name in separator_preds.keys()}

    if mix_channels > 1: # Convert to multichannel if mixture input was multichannel by duplicating mono estimate
        pred_audio = {name : np.repeat(np.expand_dims(pred_audio[name], 1), mix_channels, axis=1) for name in pred_audio.keys()}

    # Evaluate using museval, if we are currently evaluating MUSDB
    if results_dir is not None:
        scores = museval.eval_mus_track(track, pred_audio, output_dir=results_dir, win=15, hop=15.0)

        # print nicely formatted mean scores
        print(scores)

    return pred_audio


def predict_track(model_config, model, model_noise, mix_audio, mix_sr, sep_input_shape, sep_output_shape):
    '''
    Outputs source estimates for a given input mixture signal mix_audio [n_frames, n_channels]
    It iterates through the track, collecting segment-wise predictions to form the output.
    :param model_config: Model configuration dictionary
    :param mix_audio: [n_frames, n_channels] audio signal (numpy array). Can have higher sampling rate or channels than the model supports, will be downsampled correspondingly.
    :param mix_sr: Sampling rate of mix_audio
    :param sep_input_shape: Input shape of separator
    :param sep_output_shape: Input shape of separator
    :return:
    '''
    # Load mixture, convert to mono and downsample then
    assert(len(mix_audio.shape) == 2)

    # Prepare mixture
    mix_audio = np.mean(mix_audio, axis=1)
    mix_audio = librosa.resample(mix_audio, mix_sr, model_config.sample_rate)

    mix_audio = librosa.util.fix_length(mix_audio, len(mix_audio) + model_config.fft_size // 2)

    # Convert to spectrogram
    mix_mags, mix_ph = Utils.compute_spectrogram(mix_audio, model_config.fft_size, model_config.hop_size)

    # Preallocate source predictions (same shape as input mixture)
    source_time_frames = mix_mags.shape[1]
    source_preds = {name : np.zeros(mix_mags.shape, np.float32) for name in ["accompaniment", "vocals"]}

    input_time_frames = sep_input_shape[3]
    output_time_frames = sep_output_shape[3]

    # Iterate over mixture magnitudes, fetch network rpediction
    for source_pos in range(0, source_time_frames, output_time_frames):
        # If this output patch would reach over the end of the source spectrogram, set it so we predict the very end of the output, then stop
        if source_pos + output_time_frames > source_time_frames:
            source_pos = source_time_frames - output_time_frames

        # Prepare mixture excerpt by selecting time interval
        mix_part = mix_mags[:, source_pos:source_pos + input_time_frames]
        mix_part = np.expand_dims(np.expand_dims(mix_part, axis=0), axis=0)

        device = next(model.parameters()).device
        source_parts = model([torch.from_numpy(mix_part).to(device), model_noise.to(device)]).detach().cpu().numpy()

        # Save predictions
        source_preds["accompaniment"][:,source_pos:source_pos + output_time_frames] = source_parts[0, 1]
        if source_parts[0].shape[0] > 2:
            source_preds["vocals"][:, source_pos:source_pos + output_time_frames] = source_parts[0, 2]
        else:
            source_preds["vocals"][:, source_pos:source_pos + output_time_frames] = source_parts[0, 1] # Copy acc prediction into vocals for acc-only model

    # Convert predictions back to audio signal
    for key in source_preds.keys():
        mags = Utils.denormalise_spectrogram(source_preds[key])
        source_preds[key] = Utils.spectrogramToAudioFile(mags, model_config.fft_size, model_config.hop_size, phase=np.angle(mix_ph))

    return source_preds

def compute_mean_metrics(json_folder, compute_averages=True, metric="SDR"):
    '''
    Computes averages or collects evaluation metrics produced from MUSDB evaluation of a separator
     (see "produce_musdb_source_estimates" function), namely the mean, standard deviation, median, and median absolute
     deviation (MAD). Function is used to produce the results in the paper.
     Averaging ignores NaN values arising from parts where a source is silent
    :param json_folder: Path to the folder in which a collection of json files was written by the MUSDB evaluation library, one for each song.
    This is the output of the "produce_musdb_source_estimates" function.(By default, this is model_config["estimates_path"] + test or train)
    :param compute_averages: Whether to compute the average over all song segments (to get final evaluation measures) or to return the full list of segments
    :param metric: Which metric to evaluate (either "SDR", "SIR", "SAR" or "ISR")
    :return: IF compute_averages is True, returns a list with length equal to the number of separated sources, with each list element a tuple of (median, MAD, mean, SD).
    If it is false, also returns this list, but each element is now a numpy vector containing all segment-wise performance values
    '''
    files = glob.glob(os.path.join(json_folder, "*.json"))
    inst_list = None
    print("Found " + str(len(files)) + " JSON files to evaluate...")
    for path in files:
        #print(path)
        if path.__contains__("test.json"):
            print("Found test JSON, skipping...")
            continue

        with open(path, "r") as f:
            js = json.load(f)

        if inst_list is None:
            inst_list = [list() for _ in range(len(js["targets"]))]

        for i in range(len(js["targets"])):
            inst_list[i].extend([np.float(f['metrics'][metric]) for f in js["targets"][i]["frames"]])

    #return np.array(sdr_acc), np.array(sdr_voc)
    inst_list = [np.array(perf) for perf in inst_list]

    if compute_averages:
        return [(np.nanmedian(perf), np.nanmedian(np.abs(perf - np.nanmedian(perf))), np.nanmean(perf), np.nanstd(perf)) for perf in inst_list]
    else:
        return inst_list