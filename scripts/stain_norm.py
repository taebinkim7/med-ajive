import os
import numpy as np

from glob import glob
from PIL import Image

from cbcs_joint.Paths import Paths

def get_stain(img, Io, alpha, beta):  
    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log10((img.astype(np.float) + 1) / Io)
    
    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]
        
    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    # project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:, 1:3])
    
    phi = np.arctan2(That[:, 1],That[:, 0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    
    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    # a heuristic to make one vector corresponding to the first color and the 
    # other corresponding to the second
    if vMin[0] > vMax[0]:
        stain = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        stain = np.array((vMax[:, 0], vMin[:, 0])).T
    
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(stain,Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    
    return stain, C, maxC
    
def stain_norm(img, stainType, stainRef, maxCRef, Io, alpha, beta, saveFile=None):
    # define height and width of image
    h, w, _ = img.shape
    
    # get stain vectors
    stain, C, maxC = get_stain(img, Io, alpha, beta)
    
    # normalize stain concentrations
    if stainType == 'er':
        stainRef[:, 1] = stain[:, 1] # only normalize the eosin (blue) stain of 'er' images

    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])
    
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, 10**(-stainRef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    
    if saveFile is not None:
        Image.fromarray(Inorm).save(saveFile + '.png')

    return

    
def batch_stain_norm(stainType, inputPath, outputPath, Io, alpha, beta):
    # get images and references
    if stainType == 'he':
        imageList = glob('{}/*_he*'.format(inputPath))
        fileRef = glob('{}/he_ref*'.format(inputPath))[0]
    elif stainType == 'er':
        imageList = glob('{}/*_er*'.format(inputPath))
        fileRef = glob('{}/er_ref*'.format(inputPath))[0]
        
    imgRef = np.array(Image.open(fileRef))
    stainRef, _, maxCRef = get_stain(imgRef, Io, alpha, beta)
    
    # apply stain normalization to each image
    for file in imageList:
        fileName = os.path.basename(file)
        saveFile = outputPath + '/' + fileName[:-4] + '_restained'
        img = np.array(Image.open(file))
        stain_norm(img = img,
                   stainType = stainType,
                   stainRef = stainRef,
                   maxCRef = maxCRef,
                   Io = Io,
                   alpha = alpha,
                   beta = beta,
                   saveFile = saveFile)    
    return

                      
# inputPath = '/datastore/nextgenout5/share/labs/smarronlab/tkim/data/med-ajive_9344/raw_images'
# outputPath = '/datastore/nextgenout5/share/labs/smarronlab/tkim/data/med-ajive_9344/processed_images'

inputPath = Paths().raw_image_dir
outputPath = Paths().pro_image_dir
    
# batch_stain_norm('he', inputPath, outputPath, 240, 0.5, 0.15)
batch_stain_norm('er', inputPath, outputPath, 250, 0.5, 0.15)
