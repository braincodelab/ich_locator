# @author: msharrock
# version: 0.0.1

'''
Brain Extraction Tool per Muschelli et al. 
       
'''

import os
import tempfile
import nibabel as nib
from scipy import ndimage
from fsl.wrappers import fslmaths, bet

def brain(image, keep_mask=False):

        '''
        Brain Extraction with FSL 
        Params:
        - image: nifti object, scan to brain extract
        Output: 
        - brain_image: nifti object, extracted brain
        '''

        smooth = image.get_fdata().copy()
        smooth = ndimage.gaussian_filter(smooth, sigma=(2.0, 2.0, 1.0))
        smooth = nib.Nifti1Image(smooth, image.affine, image.header)


        #affine = image.affine
        #header = image.header
        tmpfile = tempfile.mkstemp(suffix='.nii.gz')[1]
        smooth.to_filename(tmpfile)

        # FSL calls
        mask = fslmaths(smooth).thr('-100.000000').uthr('100.000000').bin().fillh().run()
        fslmaths(smooth).mas(mask).run(tmpfile)
        bet(tmpfile, tmpfile, fracintensity = 0.01)
        mask = fslmaths(tmpfile).bin().fillh().run()
        image_data = mask.get_fdata() * image.get_fdata()
        #image = fslmaths(image).mas(mask).run()
        image = nib.Nifti1Image(image_data, image.affine, image.header)

        if keep_mask:
                mask = nib.Nifti1Image(mask.get_fdata(), image.affine, image.header)
                return image, mask
        else:
                return image

