import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import h5py
import torch  # Retained for potential future use
from scipy import signal


def cal_diff(sample, max_len=130):
    """
    Calculate the differences at the crossing points of the mean of the signal.
    Only returns the first max_len differences.

    Args:
        sample (np.array): Input signal.
        max_len (int): Maximum number of differences to return.

    Returns:
        list: List of differences at crossing points.
    """
    mean_value = np.mean(sample)
    crossing_diff = []
    for i in range(1, len(sample)):
        if (sample[i - 1] < mean_value and sample[i] >= mean_value) or \
                (sample[i - 1] > mean_value and sample[i] <= mean_value):
            crossing_diff.append(sample[i] - sample[i - 1])
    return crossing_diff[:max_len]


def normalize1(signal):
    """
    Normalization method 1: subtract the mean and then divide by the maximum absolute value.

    Args:
        signal (np.array): Input signal.

    Returns:
        np.array: Normalized signal.
    """
    signal = signal.astype(np.float32)
    mean = np.mean(signal)
    shifted_signal = signal - mean
    signal_max = max(np.max(shifted_signal), abs(np.min(shifted_signal)))
    if signal_max == 0:
        return np.zeros_like(shifted_signal)
    normalized_signal = shifted_signal / signal_max
    return normalized_signal


def normalize_method2(signal):
    """
    Normalization method 2: scale the signal linearly to the range [0, 1].

    Args:
        signal (np.array): Input signal.

    Returns:
        np.array: Normalized signal.
    """
    signal_max = np.max(signal)
    signal_min = np.min(signal)
    if signal_max == signal_min:
        return np.zeros_like(signal)
    normalized_signal = (signal - signal_min) / (signal_max - signal_min)
    return normalized_signal


def normalize_method3(signal):
    """
    Normalization method 3: normalize by the root mean square (RMS) value.

    Args:
        signal (np.array): Input signal.

    Returns:
        np.array: Normalized signal.
    """
    signal = signal.astype(np.float32)
    rms = np.sqrt(np.mean(np.square(signal)))
    if rms == 0:
        return signal
    normalized_signal = signal / rms
    return normalized_signal


def normalize_method4(signal):
    """
    Normalization method 4: same as normalize_method3.
    Retained for potential future modifications.

    Args:
        signal (np.array): Input signal.

    Returns:
        np.array: Normalized signal.
    """
    signal = signal.astype(np.float32)
    rms = np.sqrt(np.mean(np.square(signal)))
    if rms == 0:
        return signal
    normalized_signal = signal / rms
    return normalized_signal


def calculate_VFDT(signal, window_size=20, stride=2, samp_rate=2.4e6):
    """
    Calculate the VFDT feature.

    Args:
        signal (np.array): Input signal array.
        window_size (int): Size of the window.
        stride (int): Stride for moving the window.
        samp_rate (float): Sampling rate.

    Returns:
        list: List of VFDT feature values.
    """
    delta_w = window_size / samp_rate
    VFDT = []
    for i in range(0, len(signal) - window_size, stride):
        window = signal[i:i + window_size]
        delta_x = np.diff(window)
        var_delta_x = np.var(delta_x)
        log_var_x = np.log(var_delta_x)
        VFDT.append(2 - log_var_x / (2 * np.log(delta_w)))
    return VFDT


def method_0(folder_path, output_folder, max_samples=300):
    """
    Method 0: Extract signal segments from WAV and JSON/JS files and save to an HDF5 file.
    For each file, based on the frame data in the JSON file, extract the sample start points
    corresponding to '04:00' and '08:B6:DD' from the WAV file, and then extract fixed-length signals.
    Currently, only the atqa_signal is saved.

    Args:
        folder_path (str): Path to the input folder.
        output_folder (str): Path to the output folder.
        max_samples (int): Maximum number of samples to process.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    js_files = [f for f in os.listdir(folder_path) if f.endswith('.js') or f.endswith('.json')]

    for wav_file in wav_files:
        base_name = re.sub(r'\.wav$', '', wav_file)
        js_file = base_name + '.js' if base_name + '.js' in js_files else base_name + '.json'
        if js_file in js_files:
            wav_path = os.path.join(folder_path, wav_file)
            json_path = os.path.join(folder_path, js_file)
            hdf5_filename = os.path.join(output_folder, base_name + '.hdf5')

            _, signal_data = wav.read(wav_path)
            signals = []

            with open(json_path, 'r', encoding='utf8') as fp:
                json_data = json.load(fp)
                frames = json_data['frames']
                atqa_starts = [frame['sampleStart'] for frame in frames if "04:00" in frame['frameData']]
                sak_starts = [frame['sampleStart'] for frame in frames if "08:B6:DD" in frame['frameData']]

            for i in range(min(len(atqa_starts), len(sak_starts))):
                if i >= max_samples:
                    break
                atqa_signal = signal_data[atqa_starts[i]:atqa_starts[i] + 410]
                sak_signal = signal_data[sak_starts[i]:sak_starts[i] + 620]
                if len(atqa_signal) < 410 or len(sak_signal) < 620:
                    continue
                # You can combine atqa_signal and sak_signal or process them further.
                # Currently, only atqa_signal is appended.
                signals.append(atqa_signal)

            print('Sample Collected:', len(signals))
            with h5py.File(hdf5_filename, 'w') as hdf5_file:
                dataset = hdf5_file.create_dataset("signals", (len(signals), 410), dtype='float32')
                for i, sig in enumerate(signals):
                    dataset[i] = sig
            print(f"Processed {base_name} to HDF5.")


def method_1(folder_path, output_folder, max_samples=300):
    """
    Method 1: Extract atqa_signal from the WAV file, normalize using normalize1,
    and save the result to an HDF5 file.

    Args:
        folder_path (str): Path to the input folder.
        output_folder (str): Path to the output folder.
        max_samples (int): Maximum number of samples to process.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    js_files = [f for f in os.listdir(folder_path) if f.endswith('.js') or f.endswith('.json')]

    for wav_file in wav_files:
        base_name = re.sub(r'\.wav$', '', wav_file)
        js_file = base_name + '.js' if base_name + '.js' in js_files else base_name + '.json'

        if js_file in js_files:
            wav_path = os.path.join(folder_path, wav_file)
            json_path = os.path.join(folder_path, js_file)
            hdf5_filename = os.path.join(output_folder, base_name + '.hdf5')

            _, signal_data = wav.read(wav_path)
            signals = []
            with open(json_path, 'r', encoding='utf8') as fp:
                json_data = json.load(fp)
                frames = json_data['frames']
                atqa_starts = [frame['sampleStart'] for frame in frames if "04:00" in frame['frameData']]
                for i in range(len(atqa_starts)):
                    if i >= max_samples:
                        break
                    atqa_signal = signal_data[atqa_starts[i]:atqa_starts[i] + 410]
                    atqa_norm = normalize1(atqa_signal)
                    signals.append(atqa_norm)
            print('Sample Collected:', len(signals))
            with h5py.File(hdf5_filename, 'w') as hdf5_file:
                dataset = hdf5_file.create_dataset("signals", (len(signals), 410), dtype='float32')
                for i, sig in enumerate(signals):
                    dataset[i] = sig
            print(f"Processed {base_name} to HDF5.")


def method_2(folder_path, output_folder, max_samples=200):
    """
    Method 2: Normalize atqa_signal using normalize_method2 and save the result to an HDF5 file.

    Args:
        folder_path (str): Path to the input folder.
        output_folder (str): Path to the output folder.
        max_samples (int): Maximum number of samples to process.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    js_files = [f for f in os.listdir(folder_path) if f.endswith('.js') or f.endswith('.json')]

    for wav_file in wav_files:
        base_name = re.sub(r'\.wav$', '', wav_file)
        js_file = base_name + '.js' if base_name + '.js' in js_files else base_name + '.json'

        if js_file in js_files:
            wav_path = os.path.join(folder_path, wav_file)
            json_path = os.path.join(folder_path, js_file)
            hdf5_filename = os.path.join(output_folder, base_name + '.hdf5')

            _, signal_data = wav.read(wav_path)
            signals = []
            with open(json_path, 'r', encoding='utf8') as fp:
                json_data = json.load(fp)
                frames = json_data['frames']
                atqa_starts = [frame['sampleStart'] for frame in frames if "04:00" in frame['frameData']]
                for i in range(len(atqa_starts)):
                    if i >= max_samples:
                        break
                    atqa_signal = signal_data[atqa_starts[i]:atqa_starts[i] + 410]
                    atqa_norm = normalize_method2(atqa_signal)
                    signals.append(atqa_norm)
            print('Sample Collected:', len(signals))
            print('Sample Length:', len(signals[0]) if signals else 0)
            with h5py.File(hdf5_filename, 'w') as hdf5_file:
                dataset = hdf5_file.create_dataset("signals", (len(signals), 410), dtype='float32')
                for i, sig in enumerate(signals):
                    dataset[i] = sig
            print(f"Processed {base_name} to HDF5.")


def method_3(folder_path, output_folder, max_samples=200):
    """
    Method 3: Normalize atqa_signal using normalize_method3 and save the result to an HDF5 file.

    Args:
        folder_path (str): Path to the input folder.
        output_folder (str): Path to the output folder.
        max_samples (int): Maximum number of samples to process.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    js_files = [f for f in os.listdir(folder_path) if f.endswith('.js') or f.endswith('.json')]

    for wav_file in wav_files:
        base_name = re.sub(r'\.wav$', '', wav_file)
        js_file = base_name + '.js' if base_name + '.js' in js_files else base_name + '.json'

        if js_file in js_files:
            wav_path = os.path.join(folder_path, wav_file)
            json_path = os.path.join(folder_path, js_file)
            hdf5_filename = os.path.join(output_folder, base_name + '.hdf5')

            _, signal_data = wav.read(wav_path)
            signals = []
            with open(json_path, 'r', encoding='utf8') as fp:
                json_data = json.load(fp)
                frames = json_data['frames']
                atqa_starts = [frame['sampleStart'] for frame in frames if "04:00" in frame['frameData']]
                for i in range(len(atqa_starts)):
                    if i >= max_samples:
                        break
                    atqa_signal = signal_data[atqa_starts[i]:atqa_starts[i] + 410]
                    atqa_signal = atqa_signal.astype(np.float32)
                    atqa_norm = normalize_method3(atqa_signal)
                    signals.append(atqa_norm)
            print('Sample Collected:', len(signals))
            print('Sample Length:', len(signals[0]) if signals else 0)
            with h5py.File(hdf5_filename, 'w') as hdf5_file:
                dataset = hdf5_file.create_dataset("signals", (len(signals), 410), dtype='float32')
                for i, sig in enumerate(signals):
                    dataset[i] = sig
            print(f"Processed {base_name} to HDF5.")


def method_4(folder_path, output_folder, max_samples=200):
    """
    Method 4: Calculate VFDT features and save them to an HDF5 file.
    For each signal, extract 410 samples and compute a 195-dimensional VFDT feature vector.

    Args:
        folder_path (str): Path to the input folder.
        output_folder (str): Path to the output folder.
        max_samples (int): Maximum number of samples to process.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    js_files = [f for f in os.listdir(folder_path) if f.endswith('.js') or f.endswith('.json')]

    for wav_file in wav_files:
        base_name = re.sub(r'\.wav$', '', wav_file)
        js_file = base_name + '.js' if base_name + '.js' in js_files else base_name + '.json'

        if js_file in js_files:
            wav_path = os.path.join(folder_path, wav_file)
            json_path = os.path.join(folder_path, js_file)
            hdf5_filename = os.path.join(output_folder, base_name + '.hdf5')

            _, signal_data = wav.read(wav_path)
            signals = []
            with open(json_path, 'r', encoding='utf8') as fp:
                json_data = json.load(fp)
                frames = json_data['frames']
                atqa_starts = [frame['sampleStart'] for frame in frames if "04:00" in frame['frameData']]
            for i in range(len(atqa_starts)):
                if i >= max_samples:
                    break
                atqa_signal = signal_data[atqa_starts[i]:atqa_starts[i] + 410]
                atqa_norm = calculate_VFDT(atqa_signal)
                signals.append(atqa_norm)
            print('Sample Collected:', len(signals))
            print('Sample Length:', len(signals[0]) if signals else 0)
            with h5py.File(hdf5_filename, 'w') as hdf5_file:
                # Fixed VFDT feature dimension to 195
                dataset = hdf5_file.create_dataset("signals", (len(signals), 195), dtype='float32')
                for i, sig in enumerate(signals):
                    dataset[i] = sig
            print(f"Processed {base_name} to HDF5.")


def auto_subtract_signals(folder_path, base_filename, dataset_name):
    """
    Automatically subtract the dataset in the base file from the dataset of all other HDF5 files in the folder.

    Args:
        folder_path (str): Path to the folder containing HDF5 files.
        base_filename (str): Filename of the base HDF5 file.
        dataset_name (str): Name of the dataset to be subtracted.
    """
    base_filepath = os.path.join(folder_path, base_filename)
    # Load the base dataset
    with h5py.File(base_filepath, 'r') as base_file:
        base_dataset = base_file[dataset_name][()]

    hdf5_files = [f for f in os.listdir(folder_path) if f.endswith('.hdf5') and f != base_filename]

    for hdf5_file in hdf5_files:
        hdf5_filepath = os.path.join(folder_path, hdf5_file)
        with h5py.File(hdf5_filepath, 'r+') as h5_file:
            if dataset_name in h5_file:
                h5_file[dataset_name][:] -= base_dataset
                print(f"Subtraction performed for {hdf5_file}")
            else:
                print(f"Dataset '{dataset_name}' not found in {hdf5_file}")


def atqa_curves(folder_path, output_folder, max_samples=10):
    """
    Plot the average of normalized atqa_signal for each dataset.

    Args:
        folder_path (str): Path to the input folder.
        output_folder (str): Path to the output folder.
        max_samples (int): Maximum number of samples to process for each file.
    """
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    js_files = [f for f in os.listdir(folder_path) if f.endswith('.js') or f.endswith('.json')]
    samples_means = []

    for wav_file in wav_files:
        base_name = re.sub(r'\.wav$', '', wav_file)
        js_file = base_name + '.js' if base_name + '.js' in js_files else base_name + '.json'
        if js_file in js_files:
            wav_path = os.path.join(folder_path, wav_file)
            json_path = os.path.join(folder_path, js_file)
            _, signal_data = wav.read(wav_path)
            signals = []
            with open(json_path, 'r', encoding='utf8') as fp:
                json_data = json.load(fp)
                frames = json_data['frames']
                atqa_starts = [frame['sampleStart'] for frame in frames if "04:00" in frame['frameData']]

                for i in range(min(len(atqa_starts), max_samples)):
                    atqa_signal = signal_data[atqa_starts[i]:atqa_starts[i] + 410]
                    atqa_signal = atqa_signal.astype(np.float32)
                    # Using normalize_method3 for normalization
                    atqa_norm = normalize_method3(atqa_signal)
                    signals.append(atqa_norm)
                samples_mean = np.mean([np.sqrt(np.mean(np.square(sig))) for sig in signals])
                samples_means.append(samples_mean)

    print("CV:", np.std(samples_means) / np.mean(samples_means))
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(samples_means) + 1), samples_means, marker='o')
    plt.title('Average of averages atqa_norm for Each Tag')
    plt.xlabel('Dataset')
    plt.ylabel('Average of averages atqa_norm')
    plt.show()


def plot_power_curve(folder_path, output_folder, max_samples=300):
    """
    Plot the RMS curve for each dataset.

    Args:
        folder_path (str): Path to the input folder.
        output_folder (str): Path to the output folder.
        max_samples (int): Maximum number of samples to process for each file.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    js_files = [f for f in os.listdir(folder_path) if f.endswith('.js') or f.endswith('.json')]
    all_RMSs = []

    for wav_file in wav_files:
        base_name = re.sub(r'\.wav$', '', wav_file)
        js_file = base_name + '.js' if base_name + '.js' in js_files else base_name + '.json'
        if js_file in js_files:
            wav_path = os.path.join(folder_path, wav_file)
            json_path = os.path.join(folder_path, js_file)
            _, signal_data = wav.read(wav_path)
            RMSs = []
            with open(json_path, 'r', encoding='utf8') as fp:
                json_data = json.load(fp)
                frames = json_data['frames']
                atqa_starts = [frame['sampleStart'] for frame in frames if "04:00" in frame['frameData']]
            for i in range(len(atqa_starts)):
                if i >= max_samples:
                    break
                atqa_signal = signal_data[atqa_starts[i]:atqa_starts[i] + 410]
                atqa_signal = atqa_signal.astype(np.float32)
                RMSs.append(np.sqrt(np.mean(np.square(atqa_signal))))
            all_RMSs.append(RMSs)

    for rms in all_RMSs:
        plt.plot(rms)
    plt.show()


def plot_dataset_sample(folder_path, output_folder, max_samples=300):
    """
    Plot the comparison curves of the raw means and normalized means.

    Args:
        folder_path (str): Path to the input folder.
        output_folder (str): Path to the output folder.
        max_samples (int): Maximum number of samples to process for each file.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    js_files = [f for f in os.listdir(folder_path) if f.endswith('.js') or f.endswith('.json')]

    for wav_file in wav_files:
        base_name = re.sub(r'\.wav$', '', wav_file)
        js_file = base_name + '.js' if base_name + '.js' in js_files else base_name + '.json'
        raw_means = []
        norm_means = []
        if js_file in js_files:
            wav_path = os.path.join(folder_path, wav_file)
            json_path = os.path.join(folder_path, js_file)
            _, signal_data = wav.read(wav_path)
            with open(json_path, 'r', encoding='utf8') as fp:
                json_data = json.load(fp)
                frames = json_data['frames']
                atqa_starts = [frame['sampleStart'] for frame in frames if "04:00" in frame['frameData']]
                sak_starts = [frame['sampleStart'] for frame in frames if "08:B6:DD" in frame['frameData']]
                for i in range(len(atqa_starts)):
                    if i >= max_samples:
                        break
                    atqa_signal = signal_data[atqa_starts[i]:atqa_starts[i] + 410]
                    # sak_signal is not used, can be modified if needed
                    sak_signal = signal_data[atqa_starts[i]:atqa_starts[i] + 410]
                    atqa_norm = atqa_signal / np.mean(atqa_signal)
                    raw_means.append(np.mean(atqa_signal))
                    norm_means.append(np.mean(atqa_norm))
        plt.plot(raw_means, label='Raw Means')
        plt.plot(np.array(norm_means) + np.mean(raw_means), label='Normalized Means')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # Example usage; please adjust folder paths as needed.
    folder = "data_folder"
    output = "output_folder"

    # Uncomment the desired function calls:
    # method_0(folder, output)
    # method_1(folder, output)
    # method_2(folder, output)
    # method_3(folder, output)
    # method_4(folder, output)
    # atqa_curves(folder, output)
    # plot_power_curve(folder, output)
    # plot_dataset_sample(folder, output)
    pass
