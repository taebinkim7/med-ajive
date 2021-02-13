import os
import argparse
import numpy as np

from PIL import Image



# from cbcs_joint.Paths import Paths

def stain_norm(img, saveFile=None, Io=245, alpha=1, beta=0.15):
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

    HERef = np.array([[0.7415, 0.4114],
                      [0.5908, 0.9024],
                      [0.3180, 0.1283]])
        
    maxCRef = np.array([1.2007, 0.9412])
      
    # define height and width of image
    h, w, c = img.shape
    
    # reshape image
    img = img.reshape((-1,3))

    # calculate optical density
    OD = -np.log((img.astype(np.float)+1)/Io)
    
    # remove transparent pixels
    ODhat = OD[~np.any(OD<beta, axis=1)]
        
    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    #eigvecs *= -1
    
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3])
    
    phi = np.arctan2(That[:,1],That[:,0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:,0], vMax[:,0])).T
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T
    
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    
    # unmix hematoxylin and eosin
#    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
#    H[H>255] = 254
#    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
#    
#    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
#    E[E>255] = 254
#    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
    
    if saveFile is not None:
        Image.fromarray(Inorm).save(saveFile+'.png')
#        Image.fromarray(H).save(saveFile+'_H.png')
#        Image.fromarray(E).save(saveFile+'_E.png')

    return
    
# def get_ref():
    
    
def batch_stain_norm(input_path, output_path, Io, alpha, beta):
    for filename in os.listdir(input_path):
        imageFile = input_path + '/' + filename
        saveFile = output_path + '/' + filename[:-4] + '_restained'
        img = np.array(Image.open(imageFile))
        stain_norm(img = img,
                  saveFile = saveFile,
                  Io = Io,
                  alpha = alpha,
                  beta = beta)    
    return

                      
if __name__=='__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/datastore/nextgenout5/share/labs/smarronlab/tkim/data/med-ajive_9344/raw_images', help='RGB image file')
    parser.add_argument('--output_path', type=str, default='/datastore/nextgenout5/share/labs/smarronlab/tkim/data/med-ajive_9344/processed_images', help='save file')
    parser.add_argument('--Io', type=int, default=240)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=0.15)
    args = parser.parse_args()
    
    batch_stain_norm(args.input_path, args.output_path, args.Io, args.alpha, args.beta)
