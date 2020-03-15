
import matplotlib.image as mpimg
import PIL.Image
import glob
import matplotlib.pyplot as plt
import numpy as np
def normalize_staining(img):
    """
    Adopted from "Classification of breast cancer histology images using Convolutional Neural Networks",
    Teresa Araújo , Guilherme Aresta, Eduardo Castro, José Rouco, Paulo Aguiar, Catarina Eloy, António Polónia,
    Aurélio Campilho. https://doi.org/10.1371/journal.pone.0177544

    Performs staining normalization.

    # Arguments
        img: Numpy image array.
    # Returns
        Normalized Numpy image array.
    """
    Io = 240
    beta = 0.15
    alpha = 1
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])

    h, w, c = img.shape
    img = img.reshape(h * w, c)
    OD = -np.log((img.astype("uint16") + 1) / Io)
    ODhat = OD[(OD >= beta).all(axis=1)]
    W, V = np.linalg.eig(np.cov(ODhat, rowvar=False))

    Vec = -V.T[:2][::-1].T  # desnecessario o sinal negativo
    That = np.dot(ODhat, Vec)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    vMin = np.dot(Vec, np.array([np.cos(minPhi), np.sin(minPhi)]))
    vMax = np.dot(Vec, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    if vMin[0] > vMax[0]:
        HE = np.array([vMin, vMax])
    else:
        HE = np.array([vMax, vMin])

    HE = HE.T
    Y = OD.reshape(h * w, c).T

    C = np.linalg.lstsq(HE, Y)
    maxC = np.percentile(C[0], 99, axis=1)

    C = C[0] / maxC[:, None]
    C = C * maxCRef[:, None]
    Inorm = Io * np.exp(-np.dot(HERef, C))
    Inorm = Inorm.T.reshape(h, w, c).clip(0, 255).astype("uint8")

    return Inorm

if __name__ == "__main__":
   
    
    path='/home/fayyaz/Desktop/AsfandYaar/FYP_AsfandYaar/Image_data/TS2_data/train/Resistant/tiles_png/**'
    dpath='path to save the normalized tiles'
    folders=glob.glob(path)
    import os
    for i in range(len(folders)):
        dirName=folders[i].split('/')[-1]
        if not os.path.exists(dpath+dirName):
            os.makedirs(dpath+dirName)
        files=glob.glob(folders[i]+'/**')
        print(folders[i])
        for j in range(len(files)):
            print(files[j])
            rgb_image = PIL.Image.open(files[j])
            rgb_image = np.array(rgb_image)
            norm=normalize_staining(rgb_image)
            plt.imsave(dpath+dirName+'/'+files[j].split('/')[-1][:-3]+'png',norm)
            