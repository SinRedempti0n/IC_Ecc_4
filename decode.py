
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
    height = encode[0].astype("uint8")
    width = encode[1].astype("uint8")
    frame =np.zeros((height*cellSize, width*cellSize, 3), np.uint8)
    iterator = 2
    for col in range(defs.colors):
        for i in range(width):
            for j in range(height):
                #
                encodePiece = encode[iterator:iterator + cellSize*cellSize].reshape(cellSize,cellSize)
                iterator += cellSize*cellSize
                dataPiece = scipy.fftpack.idct(scipy.fftpack.idct(encodePiece.T,
                                                                norm='ortho').T,
                                                norm='ortho')
                dataPiece = dataPiece.clip(0, 255)
                dataPiece = dataPiece.astype("uint8")
                frame[j*cellSize:(j+1)*cellSize, i*cellSize:(i+1)*cellSize, col] = dataPiece
    #frame = frame.clip(0, 255)
    #frame = frame.astype("uint8")
    return frame
