  
# @author: msharrock
# version: 0.0.1

"""
Image Format Conversion Methods

"""

import os
import ants
import numpy as np
import nibabel as nib
import tempfile

def ants2nii(image):
    array_data = image.numpy()
    affine = np.hstack([image.direction*np.diag(image.spacing),np.array(image.origin).reshape(3,1)])
    affine = np.vstack([affine, np.array([0,0,0,1.])])
    nii_image = nib.Nifti1Image(array_data, affine)
    return nii_image

def nii2ants(image):
    ndim = image.ndim
    q_form = image.get_qform()
    spacing = image.header["pixdim"][1 : ndim + 1]

    origin = np.zeros((ndim))
    origin[:3] = q_form[:3, 3]

    direction = np.diag(np.ones(ndim))
    direction[:3, :3] = q_form[:3, :3] / spacing[:3]

    image_data = image.get_fdata().astype(np.float32)
    
    image = ants.from_numpy(
        data = image_data,
        origin = origin.tolist(),
        spacing = spacing.tolist(),
        direction = direction )
    return image

def ants2np(image):
    ants_params = [image.origin, image.spacing, image.direction]
    image = image.numpy().copy()
    image = np.nan_to_num(image)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    return image, ants_params

def nii2np(image):
    image = image.get_fdata().copy()
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    return image, image.affine

def np2ants(image, ants_params):
    n_class = image.shape[-1]
    image = np.squeeze(image, axis=0)
    images = np.split(image,indices_or_sections=n_class, axis=-1)
    #inverse = np.ones_like(np.squeeze(images[0]))
    #new_image = np.zeros_like(np.squeeze(images[0]))
    inverse_image = []
    new_image = []

    for i in range(n_class):
        class_i = images[i]
        class_i = np.squeeze(class_i)
        class_i = class_i > 0.5
        class_i = class_i * 2**i
        
        #print('pre inverse multiply:', np.unique(class_i))
        for j in range(len(inverse_image)):
            class_i = class_i * inverse_image[j]

        #print('post inverse multiply:', np.unique(class_i))
        inverse_image.append(class_i == 0)
        new_image.append(class_i)

    image = sum(new_image).astype(np.float32)
    #print('after sum:', np.unique(image))
    image = ants.from_numpy(image, origin = ants_params[0], spacing = ants_params[1], direction = ants_params[2])
    return image #image.astype('uint8')
