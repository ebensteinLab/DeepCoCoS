import time

from sklearn.model_selection import train_test_split
from Models import create_model_with_skip

from tensorflow.keras.callbacks import ModelCheckpoint
from scipy.io import savemat
import numpy as np
import glob
import os


FILENAME = r"filteredHS_L5_80-80-100mW_offallRaf594Bcy3_300ms_178-180_1120FOV.tif"
PATHNAME = r"/DATA1/Data_CoCoS/Nanostring/NoBeads 050422/L5_80-80-100mW_offallRaf594Bcy3_300ms_178-180_1120FOV_1"
SAMPLENAME = "samples_NoBeads_L5_178_1234channels"

MODELPATH= r"loss_MAE_NoBeads_L5_178"


TESTSIZE=0.1
BATCH_SIZE = 16
NUM_EPOCHS = 200
GENERATE_DATA = 0
GENERATE_HISTS = 0
LOAD_WEIGHTS = 0
random_state = 42
HIST_BINS = 20
IMG_COLS = 256
IMG_ROWS = IMG_COLS



def generate_batch(x_samples, y_samples):
    num_batches = len(x_samples) // BATCH_SIZE

    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            yield [np.array(x_samples[start:end])], [np.array(y_samples[start:end])]


def train_model(model, weight_file_path=r"", train_gen=None, test_gen=None, train_num_batches=10, test_num_batches=2):
    # Create a DLBot instance
    # bot = DLBot(token=telegram_token, user_id=telegram_user_id)
    # Create a TelegramBotCallback instance
    # telegram_callback = TelegramBotCallback(bot)
    checkpoint = ModelCheckpoint(filepath=weight_file_path, save_best_only=True)
    history = model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                  epochs=NUM_EPOCHS,
                                  verbose=1, validation_data=test_gen, validation_steps=test_num_batches,
                                  callbacks=[checkpoint])

    return history




def main():


    t0 = time.time()
    folder_path = PATHNAME
    if GENERATE_DATA == 1:
        print('already done!')
    else:

        x_samples = np.load(os.path.join(folder_path, r"x_" + SAMPLENAME  + ".npy"))
        y_samples = np.load(os.path.join(folder_path, r"y_" + SAMPLENAME  + ".npy"))


    print('Size of x_samples array' + str(x_samples.shape))
    print('Size of y_samples array' + str(y_samples.shape))

    y_samples = np.moveaxis(y_samples, 1, -1)
    val_size = TESTSIZE * 2
    Xtrain, Xleft, Ytrain, Yleft = train_test_split(x_samples, y_samples, test_size=val_size,
                                                    random_state=random_state)
    Xvalid, Xtest, Yvalid, Ytest = train_test_split(Xleft, Yleft, test_size=0.5,
                                                    random_state=random_state)
    Xtrain_subset, Xleft_subset, Ytrain_subset, Yleft_subset = Xtrain, Xleft, Ytrain, Yleft


   # ranges = chain(range(80,110,10),range(200, 700, 100))
    ranges =range(20, 21, 1)
    for fov_num in ranges:
        # fov_ind = random.sample(range(0, Ytrain.shape[0]//49), fov_num)
        # crop_ind=list([])
        # for crop in fov_ind:
        #     crop_ind=crop_ind+list(range(49*crop,49*(crop+1)))
        #
        # Xtrain_subset, Xleft_subset, Ytrain_subset, Yleft_subset = train_test_split(Xtrain[crop_ind,:,:],
        #                                                                             Ytrain[crop_ind,:,:,:],
        #                                                                             test_size=val_size,
        #                                                                             random_state=random_state)
        # Xvalid, Xtest, Yvalid, Ytest = train_test_split(Xleft_subset, Yleft_subset, test_size=0.5,
        #                                                 random_state=random_state)



        ####################################################################################
        #### Channels 123
        ####################################################################################

        # y_samples123 = y_samples[:, :, :, 0:3]
        Ytrain_subset123=Ytrain_subset[:,:,:,0:3]
        Yvalid123=Yvalid[:,:,:,0:3]
        model123 = create_model_with_skip(img_rows=x_samples.shape[1], img_cols=x_samples.shape[2], channel_out=Ytrain_subset123.shape[3])
        weight_file_path123 = os.path.join(folder_path, MODELPATH+"subset"+str(fov_num)+"_123channels_Test.h5")  # 1



        if LOAD_WEIGHTS == 1:
            model123.load_weights(weight_file_path123)


        train_gen = generate_batch(Xtrain_subset, Ytrain_subset123)
        valid_gen = generate_batch(Xvalid, Yvalid123)

        train_num_batches = len(Xtrain_subset) // BATCH_SIZE
        valid_num_batches = len(Xvalid) // BATCH_SIZE
        history = train_model(model123, weight_file_path=weight_file_path123, train_gen=train_gen, test_gen=valid_gen,
                              train_num_batches=train_num_batches, test_num_batches=valid_num_batches)

        num_samples = Xleft.shape[0]
        Yleft123=Yleft[:,:,:,0:3]
        loss_test, MAE_test, MSE_test=model123.evaluate(Xleft[0: num_samples - (num_samples % BATCH_SIZE)],Yleft123[0: num_samples - (num_samples % BATCH_SIZE)],batch_size=BATCH_SIZE)

        print("X_samples shape is: " + str(Xleft.shape))
        print("Y_samples shape is: " + str(Yleft123.shape))

        print(f'MAE_test: {MAE_test}')
        print(f'MSE_test: {MSE_test}')

        np.savez(os.path.join(PATHNAME, 'history_200Esubset'+str(fov_num)+'123c_Test.npz'), train_loss=history.history['loss'], train_mae=history.history['mae'],
                 train_mse=history.history['mse'], val_loss=history.history['val_loss'], val_mae=history.history['val_mae'],
                 val_mse=history.history['val_mse'], test_loss=MAE_test, test_mse= MSE_test)

        npzFiles = glob.glob(os.path.join(PATHNAME,"history_200Esubset"+str(fov_num)+"123c_Test.npz"))
        for f in npzFiles:
            fm = os.path.splitext(f)[0] + '.mat'
            d = np.load(f)
            savemat(fm, d)
            print('generated ', fm, 'from', f)
        del history, Yleft123, model123, Ytrain_subset123, Yvalid123

        ####################################################################################
        #### Channels 4
        ####################################################################################
        # y_samples4 = y_samples[:, :, :, [3]]  # 1
        Ytrain_subset4 = Ytrain_subset[:, :, :, [3]]
        Yvalid4 = Yvalid[:, :, :, [3]]
        model4 = create_model_with_skip(img_rows=x_samples.shape[1], img_cols=x_samples.shape[2],
                                        channel_out=Ytrain_subset4.shape[3])
        weight_file_path4 = os.path.join(folder_path, MODELPATH+"subset"+str(fov_num)+"_4channel_Test.h5")  # 1
        if LOAD_WEIGHTS == 1:
            model4.load_weights(weight_file_path4)



        train_gen = generate_batch(Xtrain_subset, Ytrain_subset4)
        valid_gen = generate_batch(Xvalid, Yvalid4)

        train_num_batches = len(Xtrain_subset) // BATCH_SIZE
        valid_num_batches = len(Xvalid) // BATCH_SIZE
        history = train_model(model4, weight_file_path=weight_file_path4, train_gen=train_gen, test_gen=valid_gen,
                              train_num_batches=train_num_batches, test_num_batches=valid_num_batches)

        num_samples = Xleft.shape[0]
        Yleft4=Yleft[:,:,:,[3]]
        loss_test, MAE_test, MSE_test=model4.evaluate(Xleft[0: num_samples - (num_samples % BATCH_SIZE)],Yleft4[0: num_samples - (num_samples % BATCH_SIZE)],batch_size=BATCH_SIZE)
        print("X_samples shape is: " + str(Xleft.shape))
        print("Y_samples shape is: " + str(Yleft4.shape))

        print(f'MAE_test: {MAE_test}')
        print(f'MSE_test: {MSE_test}')

        np.savez(os.path.join(PATHNAME, 'history_200Esubset'+str(fov_num)+'4c_Test.npz'), train_loss=history.history['loss'], train_mae=history.history['mae'],
                 train_mse=history.history['mse'], val_loss=history.history['val_loss'], val_mae=history.history['val_mae'],
                 val_mse=history.history['val_mse'], test_loss=MAE_test, test_mse= MSE_test)

        npzFiles = glob.glob(os.path.join(PATHNAME,"history_200Esubset"+str(fov_num)+"4c_Test.npz"))
        for f in npzFiles:
            fm = os.path.splitext(f)[0] + '.mat'
            d = np.load(f)
            savemat(fm, d)
            print('generated ', fm, 'from', f)



        del history, Yleft4, model4, Ytrain_subset4, Yvalid4

    ####################################################################################


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()

