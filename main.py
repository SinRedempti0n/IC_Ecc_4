import av
import numpy as np
import sys
import matplotlib.pyplot as plt

import ldpc.ldpc as ldpc
import ldpc.codec as codec

from time import time

def ldcpTransmition(frame):
    n = 200
    d_v = 3
    d_c = 4
    seed = 42
    H, G = ldpc.make_ldpc(n, d_v, d_c)

    snr = 8

    framesBin = codec.rgb2bin(frame)
    print("Frame shape: (%s, %s, %s)" % framesBin.shape)
    print("Frame Binary shape: (%s, %s, %s)" % framesBin.shape)

    framesBin_coded, frame_noisy = codec.encode_img(G, framesBin, snr)

    print("Coded Frame shape", framesBin_coded.shape)

    t = time()
    frame_decoded = codec.decode_img(G, H, framesBin_coded, snr, framesBin.shape)
    t = time() - t
    print("Tiger | Decoding time: ", t)

    error_decoded_tiger = abs(frame - frame_decoded).mean()
    error_noisy_tiger = abs(frame_noisy - frame).mean()

    plt.figure()
    plt.imshow(frame)
    plt.title('Original')
    plt.figure()
    plt.imshow(frame_noisy)
    plt.title('Noizy = %f' % error_noisy_tiger)
    plt.figure()
    plt.imshow(frame_decoded)
    plt.title('Decoded = %f' % error_decoded_tiger)
    plt.show()


def readFrames(videoName):
    container = av.open("%s" % videoName)
    framesIn = []
    for frame in container.decode(video=0):
        framesIn.append(np.array(frame.to_image()))
    return framesIn


def getYCbCr(image):
    RGB = np.asarray(image)
    YCbCr = np.zeros(np.shape(RGB))
    YCbCr[:, :, 0] = (0.299*RGB[:, :, 0])+(0.587*RGB[:, :, 1])+(0.144*RGB[:, :, 2])
    YCbCr[:, :, 1] = (0.587*RGB[:, :, 0])+(-0.331*RGB[:, :, 1])+(0.5*RGB[:, :, 2])
    YCbCr[:, :, 2] = (0.5*RGB[:, :, 0])+(-0.419*RGB[:, :, 1])+(-0.081*RGB[:, :, 2])
    return YCbCr


if __name__ == "__main__":
    frames = readFrames(sys.argv[1])
    YCbRc = getYCbCr(frames[0])
    Quant = (YCbRc // 2) + 128
    ldcpTransmition(frames[0])
    #plt.imshow(Quant[:, :, 0], cmap='Greys')
    #print('TODO')
    plt.show()
