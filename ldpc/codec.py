import numpy as np
import warnings
from ldpc.ldpc import binaryproduct
from numba import njit, int64, types, float64
import scipy


def encode_img(tG, img_bin, snr):
    n, k = tG.shape

    height, width, depth = img_bin.shape
    if depth not in [8, 24]:
        raise ValueError("The expected dimension of a binary image is "
                         "(width, height, 8) for grayscale images or "
                         "(width, height, 24) for RGB images; got %s"
                         % list(img_bin.shape))
    img_bin = img_bin.flatten()
    n_bits_total = img_bin.size
    n_blocks = n_bits_total // k
    residual = n_bits_total % k
    if residual:
        n_blocks += 1
    resized_img = np.zeros(k * n_blocks)
    resized_img[:n_bits_total] = img_bin

    codeword = encode(tG, resized_img.reshape(k, n_blocks), snr)
    noisy_img = (codeword.flatten()[:n_bits_total] < 0).astype(int)
    noisy_img = noisy_img.reshape(height, width, depth)
    noisy_img = bin2rgb(noisy_img)

    return codeword, noisy_img

def encode(tG, v, snr, seed=None):
    n, k = tG.shape

    d = binaryproduct(tG, v)
    x = (-1) ** d

    sigma = 10 ** (- snr / 20)
    e = np.random.randn(*x.shape) * sigma

    y = x + e

    return y


def rgb2bin(img):
    height, width, depth = img.shape

    if not depth == 3:
        raise ValueError("""{}\'s 3rd dimension must be equal to 3 (RGB).
                             Make sure it\'s an RGB image.""")

    img_bin = np.zeros(shape=(height, width, 24), dtype=int)

    for i in range(height):
        for j in range(width):
            r = int2bitarray(img[i, j, 0], 8)
            g = int2bitarray(img[i, j, 1], 8)
            b = int2bitarray(img[i, j, 2], 8)

            img_bin[i, j, :] = np.concatenate((r, g, b))

    return img_bin


def int2bitarray(n, k):
    binary_string = bin(n)
    length = len(binary_string)
    bitarray = np.zeros(k, 'int')
    for i in range(length - 2):
        bitarray[k - i - 1] = int(binary_string[length - i - 1])

    return bitarray



def bin2rgb(img_bin):
    """Convert a binary image to RGB."""
    height, width, depth = img_bin.shape
    img_rgb = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            r = bitarray2int(img_bin[i, j, :8])
            g = bitarray2int(img_bin[i, j, 8:16])
            b = bitarray2int(img_bin[i, j, 16:])

            img_rgb[i, j] = np.array([r, g, b], dtype=np.uint8)

    return img_rgb


def bitarray2int(bitarray):
    bitstring = "".join([str(i) for i in bitarray])
    return int(bitstring, 2)


def decode_img(tG, H, codeword, snr, img_shape, maxiter=100):
    n, k = tG.shape
    _, n_blocks = codeword.shape

    depth = img_shape[-1]
    if depth not in [8, 24]:
        raise ValueError("The expected dimension of a binary image is "
                         "(width, height, 8) for grayscale images or "
                         "(width, height, 24) for RGB images; got %s"
                         % list(img_shape))
    if len(codeword) != n:
        raise ValueError("The left dimension of `codeword` must be equal to "
                         "n, the number of columns of H.")

    systematic = True

    if not (tG[:k, :] == np.identity(k)).all():
        warnings.warn("""In LDPC applications, using systematic coding matrix
                         G is highly recommanded to speed up decoding.""")
        systematic = False

    codeword_solution = decode(H, codeword, snr, maxiter)
    if systematic:
        decoded = codeword_solution[:k, :]
    else:
        decoded = np.array([get_message(tG, codeword_solution[:, i])
                           for i in range(n_blocks)]).T
    decoded = decoded.flatten()[:np.prod(img_shape)]
    decoded = decoded.reshape(*img_shape)

    decoded_img = bin2rgb(decoded)

    return decoded_img


def decode(H, y, snr, maxiter=1000):

    m, n = H.shape

    bits_hist, bits_values, nodes_hist, nodes_values = _bitsandnodes(H)

    _n_bits = np.unique(H.sum(0))
    _n_nodes = np.unique(H.sum(1))

    if _n_bits * _n_nodes == 1:
        solver = _logbp_numba_regular
        bits_values = bits_values.reshape(n, -1)
        nodes_values = nodes_values.reshape(m, -1)

    else:
        solver = _logbp_numba

    var = 10 ** (-snr / 10)

    if y.ndim == 1:
        y = y[:, None]
    # step 0: initialization

    Lc = 2 * y / var
    _, n_messages = y.shape

    Lq = np.zeros(shape=(m, n, n_messages))

    Lr = np.zeros(shape=(m, n, n_messages))
    for n_iter in range(maxiter):
        Lq, Lr, L_posteriori = solver(bits_hist, bits_values, nodes_hist,
                                      nodes_values, Lc, Lq, Lr, n_iter)
        x = np.array(L_posteriori <= 0).astype(int)
        product = incode(H, x)
        if product:
            break
    if n_iter == maxiter - 1:
        warnings.warn("""Decoding stopped before convergence. You may want
                       to increase maxiter""")
    return x.squeeze()


def _bitsandnodes(H):
    if type(H) != scipy.sparse.csr_matrix:
        bits_indices, bits = np.where(H)
        nodes_indices, nodes = np.where(H.T)
    else:
        bits_indices, bits = scipy.sparse.find(H)[:2]
        nodes_indices, nodes = scipy.sparse.find(H.T)[:2]
    bits_histogram = np.bincount(bits_indices)
    nodes_histogram = np.bincount(nodes_indices)

    return bits_histogram, bits, nodes_histogram, nodes


def incode(H, x):
    return (binaryproduct(H, x) == 0).all()


output_type_log2 = types.Tuple((float64[:, :, :], float64[:, :, :],
                               float64[:, :]))


@njit(output_type_log2(int64[:], int64[:], int64[:], int64[:], float64[:, :],
                       float64[:, :, :],  float64[:, :, :], int64), cache=True)
def _logbp_numba(bits_hist, bits_values, nodes_hist, nodes_values, Lc, Lq, Lr,
                 n_iter):
    m, n, n_messages = Lr.shape
    # step 1 : Horizontal

    bits_counter = 0
    nodes_counter = 0
    for i in range(m):
        # ni = bits[i]
        ff = bits_hist[i]
        ni = bits_values[bits_counter: bits_counter + ff]
        bits_counter += ff
        for j in ni:
            nij = ni[:]

            X = np.ones(n_messages)
            if n_iter == 0:
                for kk in range(len(nij)):
                    if nij[kk] != j:
                        X *= np.tanh(0.5 * Lc[nij[kk]])
            else:
                for kk in range(len(nij)):
                    if nij[kk] != j:
                        X *= np.tanh(0.5 * Lq[i, nij[kk]])
            num = 1 + X
            denom = 1 - X
            for ll in range(n_messages):
                if num[ll] == 0:
                    Lr[i, j, ll] = -1
                elif denom[ll] == 0:
                    Lr[i, j, ll] = 1
                else:
                    Lr[i, j, ll] = np.log(num[ll] / denom[ll])

    # step 2 : Vertical
    for j in range(n):
        # mj = nodes[j]
        ff = nodes_hist[j]
        mj = nodes_values[nodes_counter: nodes_counter + ff]
        nodes_counter += ff
        for i in mj:
            mji = mj[:]
            Lq[i, j] = Lc[j]

            for kk in range(len(mji)):
                if mji[kk] != i:
                    Lq[i, j] += Lr[mji[kk], j]

    # LLR a posteriori:
    L_posteriori = np.zeros((n, n_messages))
    nodes_counter = 0
    for j in range(n):
        ff = nodes_hist[j]
        mj = nodes_values[nodes_counter: nodes_counter + ff]
        nodes_counter += ff
        L_posteriori[j] = Lc[j] + Lr[mj, j].sum(axis=0)

    return Lq, Lr, L_posteriori


@njit(output_type_log2(int64[:], int64[:, :], int64[:], int64[:, :],
                       float64[:, :], float64[:, :, :],  float64[:, :, :],
                       int64), cache=True)
def _logbp_numba_regular(bits_hist, bits_values, nodes_hist, nodes_values, Lc,
                         Lq, Lr, n_iter):
    m, n, n_messages = Lr.shape
    # step 1 : Horizontal
    for i in range(m):
        ni = bits_values[i]
        for j in ni:
            nij = ni[:]

            X = np.ones(n_messages)
            if n_iter == 0:
                for kk in range(len(nij)):
                    if nij[kk] != j:
                        X *= np.tanh(0.5 * Lc[nij[kk]])
            else:
                for kk in range(len(nij)):
                    if nij[kk] != j:
                        X *= np.tanh(0.5 * Lq[i, nij[kk]])
            num = 1 + X
            denom = 1 - X
            for ll in range(n_messages):
                if num[ll] == 0:
                    Lr[i, j, ll] = -1
                elif denom[ll] == 0:
                    Lr[i, j, ll] = 1
                else:
                    Lr[i, j, ll] = np.log(num[ll] / denom[ll])

    # step 2 : Vertical
    for j in range(n):
        mj = nodes_values[j]
        for i in mj:
            mji = mj[:]
            Lq[i, j] = Lc[j]

            for kk in range(len(mji)):
                if mji[kk] != i:
                    Lq[i, j] += Lr[mji[kk], j]

    # LLR a posteriori:
    L_posteriori = np.zeros((n, n_messages))
    for j in range(n):
        mj = nodes_values[j]
        L_posteriori[j] = Lc[j] + Lr[mj, j].sum(axis=0)

    return Lq, Lr, L_posteriori


def get_message(tG, x):
    n, k = tG.shape

    rtG, rx = utils.gausselimination(tG, x)

    message = np.zeros(k).astype(int)

    message[k - 1] = rx[k - 1]
    for i in reversed(range(k - 1)):
        message[i] = rx[i]
        message[i] -= utils.binaryproduct(rtG[i, list(range(i+1, k))],
                                          message[list(range(i+1, k))])

    return abs(message)
