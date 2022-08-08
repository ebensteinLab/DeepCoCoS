import os
import numpy as np
from Models import create_model_with_skip
import tifffile
import tqdm


OUTPUTPRENAME="L1_178_"
PREDPATH="L1_178_Prediction_accToL5"

FILENAME = r"dispL1_80-80-100mW_offallRaf594Bcy3180_300ms_178-180_1003FOV_2.tif"
PATHNAME = r"/DATA1/Data_CoCoS/Nanostring/NoBeads 050422/L1_80-80-100mW_offallRaf594Bcy3180_300ms_178-180_1003FOV_2"
SAMPLENAME = "samples_NoBeads_L5_178_1234channels.npy"
MODELSPATH=r"/DATA1/Data_CoCoS/Nanostring/NoBeads 050422/Prediction Models"
MODELNAME=r"loss_MAE_NoBeads_L5_178_"



DISPERSION_ANGLE=178

TESTSIZE=0.1
BATCH_SIZE = 16
NUM_EPOCHS = 200
random_state = 42
HIST_BINS = 20
IMG_COLS = 256
IMG_ROWS = IMG_COLS
NUM_CLASSES = 1



def predict_test_images_from_tif_file( tif_file, model_path, channel_save):
    model_path = model_path

    im_tif = tifffile.imread(tif_file)
    imarray = np.array(im_tif).astype(np.float32)

    input_samples = imarray[:, :, :]  # Input channel without gt
    # input_samples = imarray[:, 0, 0, :, :] # Input channel in case of gt hyperstack
    output_samples = imarray[:, 1, 1:5, :, :]  # Output channels
    del(im_tif, imarray)

    # Xtrain, input_samples, Ytrain, output_samples = train_test_split(input_samples, output_samples, test_size=0.1,
    #                                                                random_state=random_state)
    val_size = TESTSIZE*2

    # Xtrain, Xleft, Ytrain, Yleft = train_test_split(input_samples, output_samples, test_size=val_size,
    #                                                 random_state=random_state)
    # Xvalid, input_samples, Yvalid, output_samples = train_test_split(Xleft, Yleft, test_size=0.5,
    #                                                 random_state=random_state)

    # del(Xtrain, Xvalid, Ytrain, Yvalid)
    print('input samples shape = ' + str(input_samples.shape))
    print('output samples shape = ' + str(output_samples.shape))

    crop_size = 256
    stride_step = 1

    dft_model = create_model_with_skip(img_rows=crop_size, img_cols=crop_size, channel_out=len(channel_save))

    dft_model.load_weights(model_path)
    predicted_samples = []

    for sample in tqdm.tqdm(range(input_samples.shape[0])):
        X_samples = []
        Y_samples = []
        cur_test_in = input_samples[sample]
        cur_test_out = output_samples[sample]
        for ii in range(int((cur_test_in.shape[0] / crop_size))):
            for mm in range(int((cur_test_in.shape[0] / crop_size))):
                for ll in range(int(1 / stride_step)):
                    for kk in range(int(1 / stride_step)):
                        row_start = int((ii + ll * stride_step) * crop_size)
                        row_end = int((ii + ll * stride_step + 1) * crop_size)
                        col_start = int((mm + kk * stride_step) * crop_size)
                        col_end = int((mm + kk * stride_step + 1) * crop_size)

                        if row_end > cur_test_in.shape[0] or col_end > cur_test_in.shape[0]:
                            continue

                        eps = 1
                        X_samples.append(cur_test_in[row_start:row_end, col_start:col_end])
                        Y_samples.append(cur_test_out[:, row_start:row_end, col_start:col_end])

        X_samples = np.array(X_samples)
        Y_samples = np.array(Y_samples)

        num_samples = X_samples.shape[0]

        full_image_pred = dft_model.predict(np.expand_dims(X_samples[0: num_samples - (num_samples % BATCH_SIZE)], 3),
                                            batch_size=BATCH_SIZE)
        print("X_samples shape is: " + str(X_samples.shape))
        print("Y_samples shape is: " + str(Y_samples.shape))

        MAE_test = np.mean(np.abs(Y_samples-full_image_pred))
        MSE_test = np.mean(np.square(Y_samples - full_image_pred))
        print(f'MAE_test: {MAE_test}')
        print(f'MSE_test: {MSE_test}')

        stitched_full_image = np.zeros((cur_test_in.shape[0], cur_test_in.shape[1], 4)).astype(np.float32)
        for j in range(int((cur_test_in.shape[0] / crop_size))):
            for k in range(int((cur_test_in.shape[0] / crop_size))):

                stitched_full_image[j * 256:j * 256 + 256, k * 256:k * 256 + 256, channel_save] = \
                    (full_image_pred[k + int((cur_test_in.shape[0] / crop_size)) * j])

        predicted_samples.append(np.moveaxis(stitched_full_image, 2, 0))

    predicted_samples = np.array(predicted_samples)
    return predicted_samples

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    image_path = PATHNAME
    full_image_file = (os.path.join(image_path, FILENAME))

    prediction_path = os.path.join(image_path, PREDPATH)
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)

    dispersion_angle = DISPERSION_ANGLE

    save_folder1234 = os.path.join(prediction_path)


    if not os.path.exists(save_folder1234):
        os.makedirs(save_folder1234)

    model_path1234 = os.path.join(MODELSPATH,MODELNAME+"_1234channels.h5")
    model_path123 = os.path.join(MODELSPATH,MODELNAME+"_123channels.h5")
    model_path4 = os.path.join(MODELSPATH,MODELNAME+"_4channel.h5")


    pred_ch123 =predict_test_images_from_tif_file(save_folder1234, full_image_file, model_path123, [0,1,2],OUTPUTPRENAME+"ch123")
    pred_ch4 =predict_test_images_from_tif_file(save_folder1234, full_image_file, model_path4, [3],OUTPUTPRENAME+"ch4")
    pred_ch1234=pred_ch123
    pred_ch1234[:,3,:,:]=pred_ch4[:,3,:,:]
    tifffile.imsave(os.path.join(save_folder1234, OUTPUTPRENAME+"pred_"+"123_4"+".tif"), pred_ch1234, imagej=True)
