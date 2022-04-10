import av
import numpy as np
import sys
import matplotlib.pyplot as plt

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
    plt.imshow(Quant[:, :, 0], cmap='Greys')
    plt.show()
    print('TODO')
