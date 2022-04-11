
import math
import av
from PIL import Image
import scipy.fftpack
import numpy as np
import sys
import matplotlib.pyplot as plt
import defs
cellSize = defs.cellSize;

#функция восстанавливает кадр по каналам RGB
#на входе структура с изначальными размерами и огромной байтовой простыней.
#на выходе массив
def dctDecoder(encode):
    frame =np.zeros((encode['height']*cellSize, encode['width']*cellSize, 3), np.uint8)
    iterator = 0
    for col in range(defs.colors):
        for i in range(encode['width']):
            for j in range(encode['height']):
                #
                encodePiece = np.zeros((cellSize,cellSize), np.int8)
                for i1 in range(cellSize):
                    for j1 in range(cellSize):
                        encodePiece[i1,j1] = encode['data'][iterator]
                        iterator +=1
                dataPiece = scipy.fftpack.idct(scipy.fftpack.idct(encodePiece.T, norm='ortho').T, norm='ortho').T
                dataPiece = dataPiece.clip(0, 255)
                dataPiece = dataPiece.astype("uint8")
                for i1 in range(cellSize):
                    for j1 in range(cellSize):
                        frame[j*cellSize+j1, i*cellSize+i1, col] = dataPiece[j1, i1]
    #frame = frame.clip(0, 255)
    #frame = frame.astype("uint8")
    return frame
