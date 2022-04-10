import av
import numpy as np

def readFrames(videoName):
    container = av.open("%s" % videoName)
    framesIn = []
    for frame in container.decode(video=0):
        framesIn.append(np.array(frame.to_image()))
    return framesIn


def getYCbCr(image):
    pix = np.asarray(image)
    Y = (0.299*pix[:, :, 0])+(0.587*pix[:, :, 1])+(0.144*pix[:, :, 2])
    Cb = 
    Cr = 


if __name__ == "__main__":
    readFrames("lr1_3.avi")
