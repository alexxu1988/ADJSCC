import tensorflow_compression as tfc
from tensorflow.keras.layers import PReLU, Activation, GlobalAveragePooling2D, Dense, Concatenate, Conv2D, Multiply


def GFR_Encoder_Module(inputs, name_prefix, num_filter, kernel_size, stride, activation=None):
    conv = tfc.SignalConv2D(num_filter, kernel_size, corr=True, strides_down=stride, padding="same_zeros",
                            use_bias=True, activation=tfc.GDN(), name=name_prefix + '_conv')(inputs)
    if activation == 'prelu':
        conv = PReLU(shared_axes=[1,2], name=name_prefix + '_prelu')(conv)
    return conv


def Basic_Encoder(inputs, tcn):
    en1 = GFR_Encoder_Module(inputs, 'en1', 256, (9, 9), 2, 'prelu')
    en2 = GFR_Encoder_Module(en1, 'en2', 256, (5, 5), 2, 'prelu')
    en3 = GFR_Encoder_Module(en2, 'en3', 256, (5, 5), 1, 'prelu')
    en4 = GFR_Encoder_Module(en3, 'en4', 256, (5, 5), 1, 'prelu')
    en5 = GFR_Encoder_Module(en4, 'en5', tcn, (5, 5), 1)
    return en5


def GFR_Decoder_Module(inputs, name_prefix, num_filter, kernel_size, stride, activation=None):
    conv = tfc.SignalConv2D(num_filter, kernel_size, corr=False, strides_up=stride, padding="same_zeros", use_bias=True,
                            activation=tfc.GDN(inverse=True), name=name_prefix + '_conv')(inputs)
    if activation == 'prelu':
        conv = PReLU(shared_axes=[1,2], name=name_prefix + '_prelu')(conv)
    elif activation == 'sigmoid':
        conv = Activation('sigmoid', name=name_prefix + '_sigmoid')(conv)
    return conv


def Basic_Decoder(inputs):
    de1 = GFR_Decoder_Module(inputs, 'de1', 256, (5, 5), 1, 'prelu')
    de2 = GFR_Decoder_Module(de1, 'de2', 256, (5, 5), 1, 'prelu')
    de3 = GFR_Decoder_Module(de2, 'de3', 256, (5, 5), 1, 'prelu')
    de4 = GFR_Decoder_Module(de3, 'de4', 256, (5, 5), 2, 'prelu')
    de5 = GFR_Decoder_Module(de4, 'de5', 3, (9, 9), 2, 'sigmoid')
    return de5


def AF_Module(inputs, snr, name_prefix):
    (_, width, height, ch_num) = inputs.shape
    m = GlobalAveragePooling2D(name=name_prefix + '_globalpooling')(inputs)
    m = Concatenate(name=name_prefix + 'concat')([m, snr])
    m = Dense(ch_num//16, activation='relu', name=name_prefix + '_dense1')(m)
    m = Dense(ch_num, activation='sigmoid', name=name_prefix + '_dense2')(m)
    out = Multiply(name=name_prefix + 'mul')([inputs, m])
    return out


def Attention_Encoder(inputs, snr, tcn):
    en1 = GFR_Encoder_Module(inputs, 'en1', 256, (9, 9), 2, 'prelu')
    en1 = AF_Module(en1, snr, 'en1')
    en2 = GFR_Encoder_Module(en1, 'en2', 256, (5, 5), 2, 'prelu')
    en2 = AF_Module(en2, snr, 'en2')
    en3 = GFR_Encoder_Module(en2, 'en3', 256, (5, 5), 1, 'prelu')
    en3 = AF_Module(en3, snr, 'en3')
    en4 = GFR_Encoder_Module(en3, 'en4', 256, (5, 5), 1, 'prelu')
    en4 = AF_Module(en4, snr, 'en4')
    en5 = GFR_Encoder_Module(en4, 'en5', tcn, (5, 5), 1)
    return en5


def Attention_Decoder(inputs, snr):
    de1 = GFR_Decoder_Module(inputs, 'de1', 256, (5, 5), 1, 'prelu')
    de1 = AF_Module(de1, snr, 'de1')
    de2 = GFR_Decoder_Module(de1, 'de2', 256, (5, 5), 1, 'prelu')
    de2 = AF_Module(de2, snr, 'de2')
    de3 = GFR_Decoder_Module(de2, 'de3', 256, (5, 5), 1, 'prelu')
    de3 = AF_Module(de3, snr, 'de3')
    de4 = GFR_Decoder_Module(de3, 'de4', 256, (5, 5), 2, 'prelu')
    de4 = AF_Module(de4, snr, 'de4')
    de5 = GFR_Decoder_Module(de4, 'de5', 3, (9, 9), 2, 'sigmoid')
    return de5


def AF_Module_H(inputs, snr, h_real, h_imag, name_prefix):
    (_, width, height, ch_num) = inputs.shape
    m = GlobalAveragePooling2D(name=name_prefix + '_globalpooling')(inputs)
    m = Concatenate(name=name_prefix + 'concat')([m, snr, h_real, h_imag])
    m = Dense(ch_num//16, activation='relu', name=name_prefix + '_dense1')(m)
    m = Dense(ch_num, activation='sigmoid', name=name_prefix + '_dense2')(m)
    out = Multiply(name=name_prefix + 'mul')([inputs, m])
    return out


def Attention_Encoder_H(inputs, snr, h_real, h_imag, tcn):
    en1 = GFR_Encoder_Module(inputs, 'en1', 256, (9, 9), 2, 'prelu')
    en1 = AF_Module_H(en1, snr, h_real, h_imag, 'en1')
    en2 = GFR_Encoder_Module(en1, 'en2', 256, (5, 5), 2, 'prelu')
    en2 = AF_Module_H(en2, snr, h_real, h_imag, 'en2')
    en3 = GFR_Encoder_Module(en2, 'en3', 256, (5, 5), 1, 'prelu')
    en3 = AF_Module_H(en3, snr, h_real, h_imag, 'en3')
    en4 = GFR_Encoder_Module(en3, 'en4', 256, (5, 5), 1, 'prelu')
    en4 = AF_Module_H(en4, snr, h_real, h_imag, 'en4')
    en5 = GFR_Encoder_Module(en4, 'en5', tcn, (5, 5), 1)
    return en5


def Attention_Decoder_H(inputs, h_real, h_imag, snr):
    de1 = GFR_Decoder_Module(inputs, 'de1', 256, (5, 5), 1, 'prelu')
    de1 = AF_Module_H(de1, snr, h_real, h_imag, 'de1')
    de2 = GFR_Decoder_Module(de1, 'de2', 256, (5, 5), 1, 'prelu')
    de2 = AF_Module_H(de2, snr, h_real, h_imag, 'de2')
    de3 = GFR_Decoder_Module(de2, 'de3', 256, (5, 5), 1, 'prelu')
    de3 = AF_Module_H(de3, snr, h_real, h_imag, 'de3')
    de4 = GFR_Decoder_Module(de3, 'de4', 256, (5, 5), 2, 'prelu')
    de4 = AF_Module_H(de4, snr, h_real, h_imag, 'de4')
    de5 = GFR_Decoder_Module(de4, 'de5', 3, (9, 9), 2, 'sigmoid')
    return de5