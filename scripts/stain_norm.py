import os
import numpy as np

from glob import glob
from PIL import Image

from cbcs_joint.Paths import Paths

def get_stain(img, Io, alpha=1, beta=0.15):
    # define height and width of image
    h, w, c = img.shape
    
    # reshape image
    img = img.reshape((-1,3))

    # calculate optical density
    OD = -np.log10((img.astype(np.float)+1)/Io)
    
    # remove transparent pixels
    ODhat = OD[~np.any(OD<beta, axis=1)]
        
    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3])
    
    phi = np.arctan2(That[:,1],That[:,0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    # a heuristic to make one vector corresponding to the first color and the 
    # other corresponding to the second
    if vMin[0] > vMax[0]:
        stain = np.array((vMin[:,0], vMax[:,0])).T
    else:
        stain = np.array((vMax[:,0], vMin[:,0])).T
    
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(stain,Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    
    return stain, C, maxC
    
def stain_norm(img, saveFile=None, stainRef, maxCRef, Io, alpha=1, beta=0.15):
#     if stainType == 'he':
# #         stainRef = np.array([[0.7787, 0.3303],    #he_r01c04
# #                              [0.5895, 0.9376],
# #                              [0.2149, 0.1091]])
# #         maxCRef = np.array([0.4361, 0.2255])  #0.4361, 0.2255
        
# #         stainRef = np.array([[0.7543, 0.2710],   #ref1
# #                              [0.6160, 0.9500],
# #                              [0.2272, 0.1551]])
# #         maxCRef = np.array([0.5062, 0.3787])  
#         Io = 240
#     elif stainType == 'er':
# #         stainRef = np.array([[0.7317, 0.3660],
# #                              [0.6296, 0.6324],
# #                              [0.2611, 0.6828]])
# #         maxCRef = np.array([0.4013, 0.2593])        
#         Io = 250
   
#     # define height and width of image
#     h, w, c = img.shape
    
#     # reshape image
#     img = img.reshape((-1,3))

#     # calculate optical density
#     OD = -np.log10((img.astype(np.float)+1)/Io)
    
#     # remove transparent pixels
#     ODhat = OD[~np.any(OD<beta, axis=1)]
        
#     # compute eigenvectors
#     eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
#     #project on the plane spanned by the eigenvectors corresponding to the two 
#     # largest eigenvalues    
#     That = ODhat.dot(eigvecs[:,1:3])
    
#     phi = np.arctan2(That[:,1],That[:,0])
    
#     minPhi = np.percentile(phi, alpha)
#     maxPhi = np.percentile(phi, 100-alpha)
    
#     vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
#     vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
#     # a heuristic to make one vector corresponding to the first color and the 
#     # other corresponding to the second
#     if vMin[0] > vMax[0]:
#         stain = np.array((vMin[:,0], vMax[:,0])).T
#     else:
#         stain = np.array((vMax[:,0], vMin[:,0])).T
    
#     # rows correspond to channels (RGB), columns to OD values
#     Y = np.reshape(OD, (-1, 3)).T
    
#     # determine concentrations of the individual stains
#     C = np.linalg.lstsq(stain,Y, rcond=None)[0]
    
    stain, C, maxC = get_stain(img, Io, alpha, beta)
    
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, 10**(-stainRef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    
    if saveFile is not None:
        Image.fromarray(Inorm).save(saveFile+'.png')

    return

    
def batch_stain_norm(stainType, inputPath, outputPath, alpha, beta):
    
    if stainType == 'he':
        imageList = glob('{}/*_he_*'.format(inputPath))
        Io = 240
        stainRef, _, maxCRef = 
    elif stainType == 'er':
        imageList = glob('{}/*_er_*'.format(inputPath))
        Io = 250
        stainRef, _, maxCRef = 
    
    for file in imageList:
        imageFile = file
        fileName = os.path.basename(file)
        saveFile = outputPath + '/' + fileName[:-4] + '_restained'
        img = np.array(Image.open(imageFile))
        stain_norm(stainType = stainType,
                   img = img,
                   saveFile = saveFile,
                   alpha = alpha,
                   beta = beta)    
    return

                      
# inputPath = '/datastore/nextgenout5/share/labs/smarronlab/tkim/data/med-ajive_9344/raw_images'
# outputPath = '/datastore/nextgenout5/share/labs/smarronlab/tkim/data/med-ajive_9344/processed_images'

inputPath = Paths().raw_image_dir
outputPath = Paths().pro_image_dir
    
batch_stain_norm('he', inputPath, outputPath, 1, 0.15)
# batch_stain_norm('er', inputPath, outputPath, 1, 0.15)
