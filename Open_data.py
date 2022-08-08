import gc
import os
import numpy as np
import tifffile
import tqdm

random_state = 42

FILENAME = r"filtered_HS_L5_80-80-100mW_offallRaf594Bcy3_300ms_178-180_1120FOV.tif"
PATHNAME = r"/DATA1/Data_CoCoS/Nanostring/NoBeads 050422/L5_80-80-100mW_offallRaf594Bcy3_300ms_178-180_1120FOV_1"
SAMPLENAME = "samples_NoBeads_L5_178_1234channels_subset"
FOV_NUM = 1018


def generate_data_hyperstack(input_samples,output_samples):


    print('input samples shape = ' + str(input_samples.shape))
    print('output samples shape = ' + str(output_samples.shape))

    X_samples = []
    Y_samples = []
    crop_size = 256
    stride_step = 0.5

    for sample in tqdm.tqdm(range(input_samples.shape[0])):
        # for fov in range(input_samples.shape[1]):
            for ii in range(int((input_samples.shape[2] / crop_size))):
                for mm in range(int((input_samples.shape[2] / crop_size))):
                    for ll in range(int(1/stride_step)):
                        for kk in range(int(1 / stride_step)):
                            row_start = int((ii+ll*stride_step)*crop_size)
                            row_end = int((ii+ll*stride_step+1)*crop_size)
                            col_start = int((mm+kk*stride_step)*crop_size)
                            col_end = int((mm+kk*stride_step+1)*crop_size)

                            if row_end > input_samples.shape[2] or col_end > input_samples.shape[2]:
                                continue

                            eps = 1

                            X_samples.append(input_samples[sample, row_start:row_end, col_start:col_end])
                            Y_samples.append(output_samples[sample, :, row_start:row_end, col_start:col_end])

    del input_samples, output_samples
    gc.collect(generation=2)

    X_samples = np.array(X_samples)
    Y_samples = np.array(Y_samples)

    print("X_samples shape is: " + str(X_samples.shape))
    print("Y_samples shape is: " + str(Y_samples.shape))

    return X_samples, Y_samples


def generate_data(data_array):

    input_samples = data_array[:, 1, :, :]

    output_samples = data_array[:, 2, :, :]  # Output channels


    X_samples = []
    Y_samples = []
    crop_size = 256
    stride_step = 0.5

    for jj in range(input_samples.shape[0]):
        for ii in range(int((input_samples.shape[2] / crop_size))):
            for mm in range(int((input_samples.shape[2] / crop_size))):
                for ll in range(int(1/stride_step)):
                    for kk in range(int(1 / stride_step)):
                        row_start = int((ii+ll*stride_step)*crop_size)
                        row_end = int((ii+ll*stride_step+1)*crop_size)
                        col_start = int((mm+kk*stride_step)*crop_size)
                        col_end = int((mm+kk*stride_step+1)*crop_size)

                        if row_end > input_samples.shape[2] or col_end > input_samples.shape[2]:
                            continue

                        eps = 1
                        X_samples.append(input_samples[jj, row_start:row_end, col_start:col_end])
                        Y_samples.append(output_samples[jj, :, row_start:row_end, col_start:col_end])

    X_samples = np.array(X_samples)
    Y_samples = np.array(Y_samples)

    print("X_samples shape is: " + str(X_samples.shape))
    print("Y_samples shape is: " + str(Y_samples.shape))

    return X_samples, Y_samples



if __name__ == "__main__":


    image_path = PATHNAME

    im_tif = tifffile.imread(os.path.join(image_path, FILENAME))
    read_data_total = np.array(im_tif).astype(np.float32)

    read_data_subset=read_data_total
    input_samples = read_data_subset[:, 0, 0, :, :]
    output_samples = read_data_subset[:, 1, 1:5, :, :]  # Output channels
    gc.collect(generation=2)

    print('input samples shape = ' + str(input_samples.shape))
    print('output samples shape = ' + str(output_samples.shape))

    x_samples, y_samples = generate_data_hyperstack(input_samples, output_samples)

    np.save(os.path.join(image_path, r"x_"+SAMPLENAME+"% s"%FOV_NUM+".npy"), x_samples)
    np.save(os.path.join(image_path, r"y_"+SAMPLENAME+"% s"%FOV_NUM+".npy"), y_samples)

    print("Finished generating data!")
    print("Data size is: " + str(x_samples.shape))

