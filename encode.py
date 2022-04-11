
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
    encode = np.zeros((encodeSize+2), np.float32)
    encode[0] = height
    encode[1] = width
    iterator = 2
    for col in range(defs.colors):
        for i in range(width):
            for j in range(height):
                #
                dataPiece = frame[j*cellSize:(j+1)*cellSize,
                                  i*cellSize:(i+1)*cellSize, col]
                encodePiece = scipy.fftpack.dct(scipy.fftpack.dct(dataPiece.T,
                                                                norm="ortho").T,
                                                norm="ortho")
                encode[iterator:iterator + cellSize*cellSize] = encodePiece.flatten()
                iterator+= cellSize*cellSize
    return encode

