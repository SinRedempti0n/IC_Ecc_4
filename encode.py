
import math
import av
from PIL import Image
import scipy.fftpack
import numpy as np
import sys
import matplotlib.pyplot as plt
import defs
cellSize = defs.cellSize;

#функция разбивает кадр на блоки 8х8, кодирует их,
#на входе массив
#на выходе структура с изначальными размерами и огромной байтовой простыней.
def dctEncoder(frame):
    height = math.ceil(frame.shape[0]/cellSize);
    width = math.ceil(frame.shape[1]/cellSize);
    encodeSize = height*width*cellSize*cellSize*defs.colors
    encode = {'width':width, 'height':height, 'data':np.zeros((encodeSize),np.float32)}
    iterator = 0;
    for col in range(defs.colors):
        for i in range(width):
            for j in range(height):
                #
                dataPiece = frame[j*cellSize:(j+1)*cellSize,i*cellSize:(i+1)*cellSize,col]
                encodePiece = scipy.fftpack.dct(scipy.fftpack.dct(dataPiece.T, norm="ortho").T, norm="ortho")
                for i1 in range(cellSize):
                    for j1 in range(cellSize):
                        tmp = (encodePiece[j1,i1])
                        encode['data'][iterator] = tmp
                        iterator +=1
    return encode
