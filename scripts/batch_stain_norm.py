import os
import numpy as np

from glob import glob
from PIL import Image

from cbcs_joint.Paths import Paths

def stain_norm(stainType, img, saveFile=None, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images
        
    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity
        
    Output:
        Inorm: normalized image
    
    Reference: 
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''     
    
    if stainType == 'he':
        stainRef = np.array([[0.7787, 0.3303],    #he_r01c04
                             [0.5895, 0.9376],
                             [0.2149, 0.1091]])
        maxCRef = np.array([0.4361, 0.2255])  
        
#         stainRef = np.array([[0.7360, 0.2765],   #he_r08c02
#                              [0.6201, 0.9555],
#                              [0.2717, 0.1023]])
#         maxCRef = np.array([0.5326, 0.4144])  
        Io = 240

    elif stainType == 'er':
        stainRef = np.array([[0.7317, 0.3660],
                             [0.6296, 0.6324],
                             [0.2611, 0.6828]])
        maxCRef = np.array([0.4013, 0.2593])        
        Io = 250
      
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
    
    # a heuristic to make the vector corresponding to the first color and the 
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
    elif stainType == 'er':
        imageList = glob('{}/*_er_*'.format(inputPath))
    
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
