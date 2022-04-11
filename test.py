# This is a Python script.

import math
import os
import av
from PIL import Image
import scipy.fftpack
import numpy as np
import sys
import matplotlib.pyplot as plt
import encode
import decode
import defs
cellSize = defs.cellSize;

#функция читает файл и формирует массив кадров в формате массива пикселей RGB
def readFrames(videoName):
    container = av.open("%s" % videoName)
    framesIn = []
    for frame in container.decode(video=0):
        framesIn.append(np.array(frame.to_image()))
    return framesIn

if __name__ == '__main__':
    filedir = "d:/Users/new/Documents/workspace/1programming/Artem_Kursovik2022/"
    fileInputName = "lr1_3.avi"
    fileOutputName = "lr1_3_encoded_decoded.avi"
    frames = readFrames(filedir+fileInputName)
    frameNum = frames.__len__()
    frameNum2 = len(frames)

    dct = []
    #тестирование функции
    #os.mkdir("decoded/")
    #os.mkdir("original/")
    for i in range(frameNum):
        #pieceOfFrame = frames[i]
        for col in range(defs.colors,3):
        #    pieceOfFrame[:, :, col] = 0
            frames[i][:, :, col] = 0
        #encodedData = encode.dctEncoder(pieceOfFrame)
        #pieceOfFrame = pieceOfFrame[cellSize:(2*cellSize),cellSize:(2*cellSize), :]
        encodedData = encode.dctEncoder(frames[i])

        decodedData = decode.dctDecoder(encodedData)
      #  error1 = pieceOfFrame - decode
       # error2 = abs(error1).max()
        im = Image.fromarray(decodedData)
        im.save("decoded/img_%04d.png" % i)
        im2 = Image.fromarray(pieceOfFrame)
        im2.save("original/img_%04d.png" % i)
      #  print(f'frame {i}: error {error2}')

   # writeFrames(filedir+fileInputName, restoredFrames)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
