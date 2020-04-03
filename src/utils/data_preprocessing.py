import numpy as np
import os

from src.utils.misc import fft_psd
from PIL import Image

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 5

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

## EEG
def EEG_preprocess_task(task, batch_size=256, num_workers=4):
    files = []
    print("../data/EEG_Raw/{}/".format(task))
    for (dirpath, dirnames, filenames) in os.walk("../data/EEG_Raw/{}/".format(task)):
        for file in filenames:
            if file.endswith(".csv"):
                files.append(dirpath + "/" + file)

    assert len(files) == 2

    # Actual task data
    for i in range(2):
        data = []
        sample_n = 256

        f = files[i]
        x = np.genfromtxt(f, delimiter=',', skip_header=1, dtype=float)
        for j in range(np.shape(x)[0] - sample_n):
            pre_pro = x[j: j + sample_n]
            pre_pro = np.delete(pre_pro, 0, 1)
            pre_pro = np.delete(pre_pro, -1, 1)
            pro = fft_psd(1, sample_n, pre_pro)
            assert np.shape(pro[1]) == (sample_n, 4)
            data.append(pro[1])

        if i == 0:
            np.save("../data/EEG_Processed/{}_task".format(task), data)
        else:
            np.save("../data/EEG_Processed/{}_random".format(task), data)


def EEG_preprocess_tasks_to_binary():
    files = []
    for (dirpath, dirnames, filenames) in os.walk("../data/EEG_Raw/calm/"):
        for file in filenames:
            if file.endswith(".csv"):
                files.append(dirpath + file)

    # Actual task data
    data = []
    band_data = []
    sample_n = 256

    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) / SHIFT_LENGTH + 1))
    band_buffer = np.zeros((n_win_test, 4))
    fs = 256

    f = files[0]
    x = np.genfromtxt(f, delimiter=',', skip_header=1, dtype=float)
    for i in range(np.shape(x)[0] - sample_n):
        pre_pro = x[i: i + sample_n]
        pre_pro = np.delete(pre_pro, 0, 1)
        pre_pro = np.delete(pre_pro, -1, 1)

        # Compute band powers
        band_powers = utils.compute_band_powers(pre_pro, fs*EPOCH_LENGTH)
        band_buffer, _ = utils.update_dataset_buffer(band_buffer,
                                             np.asarray([band_powers]))
        # Compute the average band powers for all epochs in buffer
        # This helps to smooth out noise
        smooth_band_powers = np.mean(band_buffer, axis=0)

        band = np.concatenate((np.array([smooth_band_powers[Band.Alpha]]), np.array([smooth_band_powers[Band.Beta]])), axis=0)
        pro = _fft_psd(1, sample_n, pre_pro)
        # assert np.shape(pro[1]) == (sample_n, 2)
        band_data.append(band)
        data.append(pro[1])
    data = np.array(data)
    band_data = np.array(band_data)
    np.save("../data/EEG_Processed/calm_band", band_data)
    np.save("../data/EEG_Processed/calm", data)


    files = []
    for (dirpath, dirnames, filenames) in os.walk("../data/EEG_Raw/normal/"):
        for file in filenames:
            if file.endswith(".csv"):
                files.append(dirpath + "/" + file)

    data = []
    band_data = []
    sample_n = 256

    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) / SHIFT_LENGTH + 1))
    band_buffer = np.zeros((n_win_test, 4))
    fs = 256

    f = files[0]
    x = np.genfromtxt(f, delimiter=',', skip_header=1, dtype=float)
    for i in range(np.shape(x)[0] - sample_n):
        pre_pro = x[i: i + sample_n]
        pre_pro = np.delete(pre_pro, 0, 1)
        pre_pro = np.delete(pre_pro, -1, 1)

        # Compute band powers
        band_powers = utils.compute_band_powers(pre_pro, fs * EPOCH_LENGTH)
        band_buffer, _ = utils.update_dataset_buffer(band_buffer,
                                                     np.asarray([band_powers]))
        # Compute the average band powers for all epochs in buffer
        # This helps to smooth out noise
        smooth_band_powers = np.mean(band_buffer, axis=0)

        band = np.concatenate((np.array([smooth_band_powers[Band.Alpha]]), np.array([smooth_band_powers[Band.Beta]])),
                              axis=0)
        pro = _fft_psd(1, sample_n, pre_pro)
        # assert np.shape(pro[1]) == (sample_n, 2)
        band_data.append(band)
        data.append(pro[1])
    data = np.array(data)
    band_data = np.array(band_data)
    np.save("../data/EEG_Processed/normal_band", band_data)
    np.save("../data/EEG_Processed/normal", data)

### MISC
def dataset_reshaping(name, directory_path, new_size=(640, 480)):
    files = []

    for (dirpath, dirnames, filenames) in os.walk(directory_path):
        for file in filenames:
            if file.endswith(".jpg") or file.endswith(".jfif"):
                files.append(dirpath + "/" + file)

    for i, f in enumerate(files):
        img = Image.open(f)
        new_img = img.resize(new_size)
        new_img.save(directory_path + "resized/1/{}_{}.jpg".format(name, i))