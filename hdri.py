import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
images = [cv2.imread(x) for x in ['./hdr/img_15.jpg', './hdr/img_2.5.jpg', './hdr/img_0.25.jpg', './hdr/img_0.033.jpg']]
times = np.array([15.0, 2.5, 0.25, 0.033]).astype(np.float32)

alignMTB = cv2.createAlignMTB()
alignMTB.process(images, images)

calibrateDebevec = cv2.createCalibrateDebevec()
responseDebevec = calibrateDebevec.process(images, times)

def split(images):
    i_b, i_g, i_r = [], [], []
    for i in images:
        b, g, r = cv2.split(i)
        i_b.append(b)
        i_g.append(g)
        i_r.append(r)
    return i_b, i_g, i_r

def wei(z, zmin=0, zmax=255):
    zmid = (zmin + zmax) / 2
    return z - zmin if z <= zmid else zmax - z

def gsolve(Z, B, l, w):
    n = 256
    A = np.zeros(shape=(np.size(Z, 0) * np.size(Z, 1) + n + 1, n + np.size(Z, 0)), dtype=np.float32)
    b = np.zeros(shape=(np.size(A, 0), 1), dtype=np.float32)

    k = 0
    for i in range(np.size(Z, 0)):
        for j in range(np.size(Z, 1)):
            z = int(Z[i][j])
            wij = w[z]
            A[k][z] = wij
            A[k][n + i] = -wij
            b[k] = wij * B[j]
            k += 1
    A[k][128] = 1
    k += 1
    for i in range(n - 1):
        A[k][i] = l * w[i + 1]
        A[k][i + 1] = -2 * l * w[i + 1]
        A[k][i + 2] = l * w[i + 1]
        k += 1
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    g = x[:256]
    lE = x[256:]

    return g, lE

import random
def hdr_debvec(images, times, l, sample_num):
    global w
    B = np.log2(times)
    w = [wei(z) for z in range(256)]
    samples = [(random.randint(0, images[0].shape[0] - 1), random.randint(0, images[0].shape[1] - 1)) for i in
               range(sample_num)]
    Z = []
    for img in images:
        Z += [[img[r[0]][r[1]] for r in samples]]
    Z = np.array(Z).T
    return gsolve(Z, B, l, w)


def getCRF(images, times, l=10, sam_num=70):
    _b, _g, _r = split(images)
    image_gb, lEb = hdr_debvec(_b, times, l, sam_num)
    image_gg, lEg = hdr_debvec(_g, times, l, sam_num)
    image_gr, lEr = hdr_debvec(_r, times, l, sam_num)
    return [image_gb, image_gg, image_gr]

crf = getCRF(images, times)
plt.figure(figsize=(8, 6))
plt.plot(crf[0], range(256), 'b')
plt.plot(crf[1], range(256), 'g')
plt.plot(crf[2], range(256), 'r')
plt.show()

mergeDebevec = cv2.createMergeDebevec()
hdrDebevec = mergeDebevec.process(images, times, responseDebevec)

cv2.imwrite("hdrDebevec.hdr", hdrDebevec)
plt.imshow(hdrDebevec)

def OneMap(images, times, g):
    tizu = np.zeros((images[0].shape[0], images[0].shape[1]))
    sumDown = np.zeros((images[0].shape[0], images[0].shape[1]))
    w = np.array([wei(z) for z in range(256)])
    for k in range(len(times)):
        Zij = images[k]
        Wij = w[Zij]
        tizu += Wij * (g[Zij][:, :, 0] - np.log(times[k]))
        sumDown += Wij
    tizu = tizu / sumDown
    return tizu


def RadianceMap(images, times, crf):
    images_b, images_g, images_r = split(images)
    radiancemap = np.zeros((images[0].shape[0], images[0].shape[1], 3), dtype=np.float32)
    radiancemap[:, :, 0] = OneMap(images_b, times, crf[0])
    radiancemap[:, :, 1] = OneMap(images_g, times, crf[1])
    radiancemap[:, :, 2] = OneMap(images_r, times, crf[2])
    return radiancemap


radiancemap = RadianceMap(images, times, getCRF(images, times, l=50))

plt.figure(figsize=(8, 6))
plt.imshow(radiancemap)

plt.imshow(radiancemap)
plt.title('Radiance Map')
plt.show()

tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
ldrReinhard = tonemapReinhard.process(hdrDebevec)
plt.imshow(ldrReinhard)

ldrReinhard_recovered = tonemapReinhard.process(radiancemap)
plt.imshow(ldrReinhard_recovered)

cv2.imwrite("ldr-Reinhard.jpg", ldrReinhard_recovered * 255)
