from util_channel import Channel
from util_module import Attention_Encoder, Attention_Decoder
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np
import argparse
from dataset import dataset_imagenet
import os
import json
AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_kodak():
    images = np.empty(shape=[0,512,768,3])
    for i in range(1, 25):
        if i<10:
            image_path = 'dataset/kodak/kodim0' + str(i) + '.png'
        else:
            image_path = 'dataset/kodak/kodim' + str(i) + '.png'
        img_file = tf.io.read_file(image_path)
        image = tf.image.decode_png(img_file, channels=3)
        if image.shape[0] == 768:
            image = tf.transpose(image, [1, 0, 2])
        image = image[np.newaxis,:]
        images = np.append(images, image, axis=0)
    return images


def train(args, model):
    epoch_list = []
    loss_list = []
    val_loss_list = []
    min_loss = 10 ** 8
    if args.load_model_path is not None:
        model.load_weights(args.load_model_path)
    filename = os.path.basename(__file__).split('.')[0] + '_' + str(args.channel_type) + '_tcn' + str(
        args.transmit_channel_num) + '_snrdb' + str(args.snr_low_train) + 'to' + str(
        args.snr_up_train) + '_bs' + str(args.batch_size) + '_lr' + str(args.learning_rate)
    model_path = args.model_dir + filename + '.h5'
    cbk = ModelCheckpoint(model_path, monitor='loss', save_best_only=True, save_weights_only=True, save_freq=100)
    for epoch in range(0, args.epochs):
        train_ds, train_nums = dataset_imagenet.get_dataset_snr_range(args.snr_low_train, args.snr_up_train)
        train_ds = train_ds.batch(args.batch_size)
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        train_step = (
            train_nums // args.batch_size if train_nums % args.batch_size == 0 else train_nums // args.batch_size + 1)
        h = model.fit(train_ds, epochs=1, steps_per_epoch=train_step, callbacks=[cbk])
        his = h.history
        loss = his.get('loss')[0]
        if loss < min_loss:
            min_loss = loss
            model.save_weights(model_path)
            print('Epoch:', epoch + 1, ',loss=', loss, 'save')
        else:
            print('Epoch:', epoch + 1, ',loss=', loss)
        epoch_list.append(epoch)
        loss_list.append(loss)
        with open(args.loss_dir + filename + '.json', mode='w') as f:
            json.dump({'epoch': epoch_list, 'loss': loss_list, 'val_loss': val_loss_list}, f)


def eval(args, model):
    filename = os.path.basename(__file__).split('.')[0] + '_' + str(args.channel_type) + '_tcn' + str(
        args.transmit_channel_num) + '_snrdb' + str(args.snr_low_eval) + 'to' + str(
        args.snr_up_eval) + '_bs' + str(args.batch_size) + '_lr' + str(args.learning_rate)
    model_path = args.model_dir + filename + '.h5'
    model.load_weights(model_path)
    snr_list = []
    mse_list = []
    psnr_list = []
    kodak = get_kodak()
    for snrdb in range(args.snr_low_eval, args.snr_up_eval + 1):
        imse = []
        # test 10 times each snr
        for i in range(0, 100):
            mse = model.evaluate(x=[kodak, snrdb * np.ones((24,))], y=kodak)
            imse.append(mse)
        mse = np.mean(imse)
        psnr = 10 * np.log10(255 ** 2 / mse)
        snr_list.append(snrdb)
        mse_list.append(mse)
        psnr_list.append(psnr)
        with open(args.eval_dir + filename + '.json', mode='w') as f:
            json.dump({'snr': snr_list, 'mse': mse_list, 'psnr': psnr_list}, f)


def predict(args, model):
    filename = os.path.basename(__file__).split('.')[0] + '_' + str(args.channel_type) + '_tcn' + str(
        args.transmit_channel_num) + '_snrdb' + str(args.snr_low_train) + 'to' + str(
        args.snr_up_train) + '_bs' + str(args.batch_size) + '_lr' + str(args.learning_rate)
    model_path = args.model_dir + filename + '.h5'
    model.load_weights(model_path)
    img_name = 'kodim23'
    image_path = 'dataset/kodak/'+img_name+'.png'
    img_file = tf.io.read_file(image_path)
    image = tf.image.decode_png(img_file, channels=3)
    image = image[np.newaxis, :]
    r_image = model.predict(x=[image,args.snr_predict*np.ones([1,])])
    r_image = r_image[0]
    img = tf.cast(r_image, tf.uint8)
    cont = tf.image.encode_png(img)
    tf.io.write_file('predict_pic/' + img_name + '_adjscc_imagenet_snrdb' + str(args.snr_predict) + '.png', cont)



def main(args):
    # construct encoder-decoder model
    if args.command == 'train':
        input_imgs = Input(shape=(128, 128, 3))
    elif args.command == 'eval':
        input_imgs = Input(shape=(512, 768, 3))
    elif args.command == 'predict':
        input_imgs = Input(shape=(512, 768, 3))
    input_snrdb = Input(shape=(1,))
    normal_imgs = Lambda(lambda x: x / 255, name='normal')(input_imgs)
    encoder = Attention_Encoder(normal_imgs, input_snrdb, args.transmit_channel_num)
    rv = Channel(channel_type='awgn')(encoder, input_snrdb)
    decoder = Attention_Decoder(rv, input_snrdb)
    rv_imgs = Lambda(lambda x: x * 255, name='denormal')(decoder)
    model = Model(inputs=[input_imgs, input_snrdb], outputs=rv_imgs)
    model.compile(Adam(args.learning_rate), 'mse')
    model.summary()
    if args.command == 'train':
        train(args, model)
    elif args.command == 'eval':
        eval(args, model)
    elif args.command == 'predict':
        predict(args, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help='train/eval/eval_pic/')
    parser.add_argument("-ct", '--channel_type', help="awgn", default='awgn')
    parser.add_argument("-md", '--model_dir', help="dir for model", default='model/')
    parser.add_argument("-lmp", '--load_model_path', help="model path for loading")
    parser.add_argument("-bs", "--batch_size", help="Batch size for training", default=16, type=int)
    parser.add_argument("-e", "--epochs", help="epochs for training", default=2, type=int)
    parser.add_argument("-lr", "--learning_rate", help="learning_rate for training", default=0.0001, type=float)
    parser.add_argument("-tcn", "--transmit_channel_num", help="transmit_channel_num for djscc model", default=16,
                        type=int)
    parser.add_argument("-snr_low_train", "--snr_low_train", help="snr_low for training", default=0, type=int)
    parser.add_argument("-snr_up_train", "--snr_up_train", help="snr_up for training", default=20, type=int)
    parser.add_argument("-snr_low_eval", "--snr_low_eval", help="snr_low for evaluation", default=0, type=int)
    parser.add_argument("-snr_up_eval", "--snr_up_eval", help="snr_up for evaluation", default=20, type=int)
    parser.add_argument("-snr_eval", "--snr_eval", help="snr for evaluation", default=10, type=int)
    parser.add_argument("-snr_predict", "--snr_predict", help="snr for predict", default=10, type=int)
    parser.add_argument("-ldd", "--loss_dir", help="loss_dir for training", default='loss/')
    parser.add_argument("-ed", "--eval_dir", help="eval_dir", default='eval/')
    global args
    args = parser.parse_args()
    print("#######################################")
    print("Current execution paramenters:")
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value))
    print("#######################################")
    main(args)
