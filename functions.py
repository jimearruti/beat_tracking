import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve, butter, filtfilt
from scipy.signal.windows import gaussian
from evaluate import evaluate


def butterworth_highpass(data, cutoff_freq, fs, order=4):
    """
    Apply a Butterworth high-pass filter to a signal.

    Parameters:
        data: signal to filter
        cutoff_freq: cutoff frequency in Hz
        fs: sampling rate of the data (use 1/hop_time for onset detection functions)
        order: filter order (default: 4)
    """
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normalized_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def get_onset_strength(input_file, window_time=0.032, hop_time=0.004, plot=False):
    """Calculate the onset strength function for an audio file.

    Parameters:
        input_file: path to the audio file
        window_time: time length of the analysis window in seconds (default: 0.032)
        hop_time: time interval between consecutive analysis windows in seconds (default: 0.004)
        plot: whether to plot intermediate signals (default: False)

    Returns:
        normalised_onset_strength: onset strength normalised by standard deviation
    """
    y, sr = librosa.load(input_file, sr=8000)
    # calculate spectrogram
    X_stft = librosa.stft(y, win_length=int(window_time*sr), hop_length=int(hop_time*sr))
    spect = np.square(np.abs(X_stft))
    mel_spectrogram = librosa.feature.melspectrogram(S=spect, sr=sr, n_mels=40, hop_length=int(hop_time*sr), n_fft=int(window_time*sr))
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram)
    # calculate first order diff
    mel_spectrogram_db_diff = np.diff(mel_spectrogram_db, axis=1)
    # set all negatives to 0
    rectified_diff = np.maximum(0, mel_spectrogram_db_diff)
    # sum across frequencies
    onset_strength = np.sum(rectified_diff, axis=0)
    # high-pass filter with 0.4Hz cutoff
    filtered_onset_strength = butterworth_highpass(onset_strength, cutoff_freq=0.4, fs=1/hop_time)

    # # Create 20ms Gaussian window
    window_duration = 0.020  # 20ms
    window_samples = int(window_duration / hop_time)
    gaussian_window = gaussian(window_samples, window_samples/6)
    gaussian_window /= np.sum(gaussian_window)

    # window filtered_onset_strength with gaussian window
    smoothed_onset_strength = convolve(filtered_onset_strength, gaussian_window, mode='same')
    # normalise over standard deviation
    normalised_onset_strength = smoothed_onset_strength/np.std(smoothed_onset_strength)

    if plot:
        times = np.arange(0, len(filtered_onset_strength)) * hop_time

        plt.subplot(2,1,1)
        librosa.display.specshow(spect, sr=sr, hop_length=int(hop_time*sr), x_axis="time", y_axis="linear")
        # plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram")
        plt.grid()
        plt.xlim(5,10)

        plt.subplot(2,1,2)
        plt.plot(times, onset_strength, label="onset_strength")
        plt.plot(times, filtered_onset_strength, label="filtered_onset_strength")
        plt.plot(times, smoothed_onset_strength, label="smoothed_onset_strength")
        plt.plot(times, normalised_onset_strength, label="normalised_onset_strength")
        plt.grid()
        plt.xlim(5,10)
        plt.ylim([-30, 30])
        plt.legend()

    return normalised_onset_strength


def apply_perceptual_weighting(signal, sigma_t, preferred_perceptual_interval_s=0.5, hop_time=0.004):
    """Apply log-Gaussian perceptual weighting to a signal.

    Parameters:
        signal: input signal to weight
        sigma_t: standard deviation of the log-Gaussian weighting function (recommended: ~1.4)
        preferred_perceptual_interval_s: preferred perceptual tempo interval in seconds (default: 0.5)
        hop_time: time interval between consecutive frames in seconds (default: 0.004)

    Returns:
        weighted signal of the same length as input
    """
    i = np.arange(len(signal))
    # avoid log(0)
    i[0] = 1  
    weights = np.exp(-0.5 * np.square(np.log2(i * hop_time / preferred_perceptual_interval_s) / sigma_t))
    weights[0] = 0
    return weights * signal


def get_tempo_period_strength(onset_strength, hop_time=0.004, sigma_t=1.4, min_bpm=70, max_bpm=225, plot=False):
    """Calculate the tempo period strength (TPS) from the onset strength function.

    Computes the autocorrelation of the onset strength, restricts it to the
    BPM range [min_bpm, max_bpm], and applies perceptual weighting.

    Parameters:
        onset_strength: onset strength function (1D array)
        hop_time: time interval between consecutive frames in seconds (default: 0.004)
        sigma_t: standard deviation of the log-Gaussian perceptual weighting (default: 1.4)
        min_bpm: minimum tempo in BPM (default: 70)
        max_bpm: maximum tempo in BPM (default: 225)
        plot: whether to plot the autocorrelation and weighted autocorrelation (default: False)

    Returns:
        weighted_autocorrelation: perceptually weighted tempo period strength (1D array)
    """
    samples_min_bpm = round(60 / (max_bpm * hop_time))
    samples_max_bpm = round(60 / (min_bpm * hop_time))

    autocorrelation = np.correlate(onset_strength, onset_strength, mode='full')
    # Take only positive lags (second half)
    autocorrelation = autocorrelation[len(autocorrelation)//2:]
    # Normalize (adding epsilon to avoid division by zero)
    autocorrelation = autocorrelation / (autocorrelation[0] + 1e-10)

    # Zero out lags outside the BPM range
    autocorrelation[:samples_min_bpm] = 0
    autocorrelation[samples_max_bpm:] = 0

    weighted_autocorrelation = apply_perceptual_weighting(autocorrelation, sigma_t)
    
    if plot:
        plt.figure()
        plt.plot(autocorrelation, label="autocorrelation")
        plt.plot(weighted_autocorrelation, label="weighted_autocorrelation")
        plt.legend()
    return weighted_autocorrelation


def estimate_tempo(onset_strength, sigma_t=0.9, min_bpm=70, max_bpm=225, plot=False):
    """Estimate the tempo period in frames from the onset strength function.

    Parameters:
        onset_strength: onset strength function (1D array)
        sigma_t: standard deviation of the log-Gaussian perceptual weighting (default: 0.9)
        min_bpm: minimum tempo in BPM (default: 70)
        max_bpm: maximum tempo in BPM (default: 225)
        plot: whether to plot the tempo period strength (default: False)

    Returns:
        primary_tempo_samples: estimated tempo period in frames (int)
    """
    tempo_period_strength = get_tempo_period_strength(onset_strength, min_bpm=min_bpm, max_bpm=max_bpm, sigma_t=sigma_t)
    if plot:
        plt.plot(tempo_period_strength)
        plt.xlim([0, 2000])

    primary_tempo_samples = np.argmax(tempo_period_strength)
    return primary_tempo_samples


def estimate_tempo_and_division(onset_strength, hop_time=0.004, sigma_t=0.5, min_bpm=70, max_bpm=225, plot=False):
    """Estimate the tempo period and metre (double or triple) from the onset strength function.

    Computes combined tempo period strength for duple and triple metres and
    selects the one with the higher peak.

    Parameters:
        onset_strength: onset strength function (1D array)
        hop_time: time interval between consecutive frames in seconds (default: 0.004)
        sigma_t: standard deviation of the log-Gaussian perceptual weighting (default: 0.5)
        min_bpm: minimum tempo in BPM (default: 70)
        max_bpm: maximum tempo in BPM (default: 225)
        plot: whether to plot the tempo period strength functions (default: False)

    Returns:
        metre: detected metre, either "double" (duple) or "triple"
        max_value_index: estimated beat period in frames corresponding to the strongest tempo peak
    """
    tempo_period_strength = get_tempo_period_strength(onset_strength, min_bpm=min_bpm, max_bpm=max_bpm, sigma_t=sigma_t)

    N = len(tempo_period_strength)
    tempo_period_strength_padded = np.zeros(N * 3 + 2)
    tempo_period_strength_padded[:N] = tempo_period_strength
    
    taus = np.arange(1, len(tempo_period_strength))
    taus_doubled = 2 * taus
    taus_tripled = 3 * taus

    tempo_period_strength_double = (
        tempo_period_strength_padded[taus] + 0.5 * tempo_period_strength_padded[taus_doubled] +
        0.25 * tempo_period_strength_padded[taus_doubled - 1] + 0.25 * tempo_period_strength_padded[taus_doubled + 1]
    )
    tempo_period_strength_triple = (
        tempo_period_strength_padded[taus] + 0.33 * tempo_period_strength_padded[taus_tripled] +
        0.33 * tempo_period_strength_padded[taus_tripled - 1] + 0.33 * tempo_period_strength_padded[taus_tripled + 1]
    )

    if plot:
        period_max = round(60 / (min_bpm * hop_time))
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(taus, tempo_period_strength[taus])
        plt.grid()
        plt.xlim([0, period_max])
        plt.ylabel("TPS")
        plt.xlabel("Tau (frames)")

        plt.subplot(3,1,2)
        plt.plot(taus, tempo_period_strength_double, label="TPS2")
        plt.grid()
        plt.xlim([0, period_max])
        plt.ylabel("TPS2")
        plt.xlabel("Tau (frames)")

        plt.subplot(3,1,3)
        plt.plot(taus, tempo_period_strength_triple, label="TPS3")
        plt.grid()
        plt.xlim([0, period_max])
        plt.ylabel("TPS3")
        plt.xlabel("Tau (frames)")

    double_max_value = np.max(tempo_period_strength_double)
    triple_max_value = np.max(tempo_period_strength_triple)

    if double_max_value >= triple_max_value:
        metre = "double"
        max_value_index = taus[np.argmax(tempo_period_strength_double)]
    
    else:
        metre = "triple"
        max_value_index = taus[np.argmax(tempo_period_strength_triple)]

    return metre, max_value_index


def beats_simple(input_file, window_time=0.032, hop_time=0.004, bpm=None, min_bpm=70, max_bpm=225,
                 alpha=400, plot=False, ignore_seconds=0.5, sigma_t=0.9, downbeats=True):
    """Detect beats (and optionally downbeats) in an audio file using dynamic programming.

    Parameters:
        input_file: path to the audio file
        window_time: time length of the analysis window in seconds (default: 0.032)
        hop_time: time interval between consecutive frames in seconds (default: 0.004)
        bpm: beats per minute, if known; estimated automatically if None (default: None)
        min_bpm: minimum tempo in BPM for tempo estimation (default: 70)
        max_bpm: maximum tempo in BPM for tempo estimation (default: 225)
        alpha: weight for the tempo consistency cost in dynamic programming (default: 400)
        plot: whether to plot cumulative scores and onset strength (default: False)
        ignore_seconds: seconds at the start of the file to exclude from beat selection (default: 0.5)
        sigma_t: standard deviation of the log-Gaussian weighting for tempo estimation (default: 0.9)
        downbeats: whether to also estimate downbeat positions (default: True)

    Returns:
        beats_time: array of estimated beat times in seconds
        downbeats_time: array of estimated downbeat times in seconds, or None if downbeats=False
    """
    onset_strength = get_onset_strength(input_file, window_time=window_time, hop_time=hop_time)    
    if bpm:
        period_frames = 60 / (bpm * hop_time)
    else:
        period_frames = estimate_tempo(onset_strength, sigma_t=sigma_t, min_bpm=min_bpm, max_bpm=max_bpm, plot=plot)
    
    # backllink[time] is best predecesor for that point
    # we start in all -1 and populate it later
    backlink = np.ones(len(onset_strength), dtype=int) * (-1)
    # total cumulated score to this point
    cumulative_score = np.copy(onset_strength)

    # search range for previous beat
    previous_beat_samples_range = np.arange(round(-2*period_frames), -round(period_frames/2))

    tempo_consistency_penalty = -np.square(np.log(np.abs(previous_beat_samples_range) / period_frames))

    # log gaussian window over range, multiplied by weight alpha
    tempo_consistency_cost = alpha * tempo_consistency_penalty

    for i in range(np.max(-previous_beat_samples_range + 1), len(onset_strength)):
        # iterate over possible beats, start
        periods_range = i + previous_beat_samples_range
        # compute scores all predecesors
        score_candidates = tempo_consistency_cost + cumulative_score[periods_range]
            
        # get the one with best score
        max_score_index = np.argmax(score_candidates)
        max_score_value = score_candidates[max_score_index]
        # store the values and indexes
        cumulative_score[i] = max_score_value + onset_strength[i]
        backlink[i] = periods_range[max_score_index]

    processed_cumulative_score = np.copy(cumulative_score)
    ignore_frames = int(ignore_seconds / hop_time)
    processed_cumulative_score[:ignore_frames] = 0

    # Get the best cumulated score
    max_cum_score_index = np.argmax(processed_cumulative_score)
    # assign last beat to that index
    beats = np.asarray([max_cum_score_index])

    if plot:
        plt.figure()
        plt.plot(cumulative_score)
        plt.figure()
        plt.plot(processed_cumulative_score)
        plt.figure()
        plt.plot(onset_strength)

    while backlink[beats[0]]>= 0:
        beats = np.concatenate(([backlink[beats[0]]], beats))

    beats_time = beats * hop_time
    if downbeats:
        metre, _ = estimate_tempo_and_division(onset_strength, sigma_t=0.5, min_bpm=min_bpm, max_bpm=max_bpm, plot=plot)
        if metre == "double":
            downbeat_period_frames = 4 * period_frames
        else:
            downbeat_period_frames = 6 * period_frames
        bpm_downbeats = 60 / (hop_time * downbeat_period_frames)
        downbeats_time, _ = beats_simple(input_file, bpm=bpm_downbeats, min_bpm=min_bpm, max_bpm=max_bpm, downbeats=False)
    else:
        downbeats_time = None
    return beats_time, downbeats_time


def compare_beats(onset_strength, ground_truth_beats, beats_time, file_name, hop_time=0.004):
    """Compare estimated beats against ground truth and plot the results.

    Parameters:
        onset_strength: onset strength function (1D array)
        ground_truth_beats: array of ground truth beat times in seconds
        beats_time: array of estimated beat times in seconds
        file_name: name of the audio file (used in plot title and printed output)
        hop_time: time interval between consecutive frames in seconds (default: 0.004)
    """
    result = evaluate(beats_time, ground_truth_beats)
    print('Input file: ' + file_name)
    print('C  FP  FN   F    Err')
    print('{:3.0f} {:3.0f} {:3.0f} {:5.3f} {:5.3f}'.format(*result[0]))

    plt.figure(figsize=(10, 4))
    times = np.arange(0, len(onset_strength)) * hop_time
    plt.plot(times, onset_strength)
    plt.xlabel("Time (s)") 
    plt.title('Detection (red) vs ground truth (green) for file: ' + file_name)
    for j in ground_truth_beats:
        plt.axvline(j, ymin=0.5, color='g')
    for j in beats_time:
        plt.axvline(j, ymax=0.5, color='r')