#! /usr/bin/env python2.7


import argparse
import cv2
import skimage.transform
import os
import numpy as np


DEFAULT_DATA_PATH = '../../data/images/'
DEFAULT_SET_PATH = '../../data/sets/'
DEFAULT_SET = 'train'
DEFAULT_IMAGE_SIZE = 100


def main():
    # Get arguments
    args = parse_arguments()

    # Get paths
    data_path = args.data_path if args.data_path is not None else DEFAULT_DATA_PATH
    set_path = args.set_path if args.set_path is not None else DEFAULT_SET_PATH
    output_path = args.set_path if args.set_path is not None else DEFAULT_SET_PATH
    set = args.set if args.set is not None else DEFAULT_SET
    image_size = args.image_size if args.image_size is not None else DEFAULT_IMAGE_SIZE
    perturb = args.perturb if args.perturb is not None else False

    # Augment the data
    augment(data_path, set_path, output_path, set, (image_size, image_size), perturb)


def parse_arguments():
    """ 
        This function retrieves the command line parameters passed into this script.

        Parameters
        ----------
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data-path', type=str, help='Specify the path to the dataset')
    parser.add_argument('-sp', '--set-path', type=str, help='Specify the path to the set files')
    parser.add_argument('-op', '--output-path', type=str, help='Specify the path to save the augmented images')
    parser.add_argument('--set', type=str, help='The set to augment (train|dev|test)')
    parser.add_argument('-is', '--image-size', type=int, help='Specify the downsampling image size')
    parser.add_argument('--perturb', action='store_true', help='Specify whether or not to perturb the images')
    args = parser.parse_args()

    return args


def augment(data_path, set_path, output_path, set, target_size, perturb=False):
    """ 
        This function performs the image augmentation based on the parameters passed in

        Parameters
        ----------
        data_path : str
            the path to the images in the dataset
        set_path : str
            the path to the set text files 
        output_path : str
            the path to the store the augmented images
        set: (train|dev|test)
            the set to augment
        target_size: (int, int)
            the target size of the augmented images
        perturb: (true|false)
            whether or not to perform a random perturbation on the augmented images
    """

    # Specify the augmentation transforms to perform
    aug_transforms = {
        'zoom_range': (1.0, 1.1),
        'rotation_range': (0, 360),
        'shear_range': (0, 0),
        'translation_range': (-4, 4),
    }

    # Read in images for the the specified set
    for subdir, dirs, files in os.walk(set_path):
        for file in files:
            # Check if the file belongs to the desired set
            if set == file.split('_')[-1][:-4] and file.split('_')[-2] != 'aug':
                label = file.split('_')[0]

                # Read images from the set
                with open(os.path.join(set_path, file), 'r') as set_file:
                    for image_name in set_file:
                        # Read in the image as color
                        image_name = image_name.rstrip('\n')
                        image = cv2.imread(os.path.join(data_path, image_name), 1)

                        # If everything was successful, augment the image
                        if image is not None:
                            image = image.astype('float32')
                            tform_ds_cc = build_ds_transform((image.shape[0], image.shape[1]), target_size, 1.0)
                            tform_ds_cropped33 = build_ds_transform((image.shape[0], image.shape[1]), target_size, 3.0)
                            ds_transforms = [tform_ds_cc]

                            # Calculate the shift for, the transformations
                            center_shift = np.array((image.shape[0], image.shape[1])) / 2. - 0.5
                            tform_center = skimage.transform.SimilarityTransform(translation=-center_shift)
                            tform_uncenter = skimage.transform.SimilarityTransform(translation=center_shift)

                            # Perturb and downsample the image
                            if perturb:
                                aug_images = aug_and_dscrop(image, ds_transforms, aug_transforms, target_size, tform_center, tform_uncenter, perturb=True)

                                # Write file to aug set
                                output_file = file.replace('_', '_aug_')
                                perturb_image_name = image_name.replace('.', '_aug.')
                                for i, aug_img in enumerate(aug_images):
                                    perturb_image_name = perturb_image_name.replace('.', '_'+str(i)+'.')
                                    with open(os.path.join(set_path, output_file), 'a') as perturb_file:
                                        perturb_file.write(perturb_image_name)
                                    cv2.imwrite(data_path+perturb_image_name, aug_img)
                            # Only downsample the image
                            else:
                                # TODO fix to adjust for multiple transforms
                                aug_images = aug_and_dscrop(image, ds_transforms, target_size=target_size, perturb=False)
                                for i, aug_img in enumerate(aug_images):
                                    cv2.imwrite(data_path+image_name, aug_img)
                            

def aug_and_dscrop(img, ds_transforms, aug_params=None, target_size=None, tform_center=None, tform_uncenter=None, perturb=False):
    """ 
        This function augments an image with the appropriate d

        Parameters
        ----------
        img:
            the image to augment
        ds_transforms:
            the transforms to perform on the image
        aug_params:
        augmentation_transforms:
        target_size: (int, int)
        tfor_center:
        tform_uncenter:

        Copyright (c) 2014, Sander Dieleman
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright notice, this
          list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above copyright notice,
          this list of conditions and the following disclaimer in the documentation
          and/or other materials provided with the distribution.

        * Neither the name of the {organization} nor the names of its
          contributors may be used to endorse or promote products derived from
          this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    if target_size is None: # default
        target_sizes = [(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE) for _ in xrange(len(ds_transforms))]
    else:
        target_sizes = [target_size for _ in xrange(len(ds_transforms))]


    if perturb:
        tform_augment = random_perturbation_transform(tform_center, tform_uncenter, **aug_params)
    else:
        tform_augment = skimage.transform.AffineTransform() # this is an identity transform by default

    augmentations = []
    for tform_ds, target_size in zip(ds_transforms, target_sizes):
        augmentations.append(fast_warp(img, tform_ds+tform_augment, output_shape=target_size, mode='reflect').astype('float32'))

    return augmentations


def random_perturbation_transform(tform_center, tform_uncenter, zoom_range, rotation_range, shear_range, translation_range, do_flip=False):
    """ 
        This function reads the images and correspondings labels of the dataset.

        Parameters
        ----------
        tform_center:
        tform_uncenter:
        zoom_range:
            
        rotation_range:
        shear_range:
        translation_range:
        do_flip:

        Copyright (c) 2014, Sander Dieleman
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright notice, this
          list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above copyright notice,
          this list of conditions and the following disclaimer in the documentation
          and/or other materials provided with the distribution.

        * Neither the name of the {organization} nor the names of its
          contributors may be used to endorse or promote products derived from
          this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    # random shift [-4, 4] - shift no longer needs to be integer!
    shift_x = np.random.uniform(*translation_range)
    shift_y = np.random.uniform(*translation_range)
    translation = (shift_x, shift_y)

    # random rotation [0, 360]
    rotation = np.random.uniform(*rotation_range) # there is no post-augmentation, so full rotations here!

    # random shear [0, 5]
    shear = np.random.uniform(*shear_range)

    # # flip
    if do_flip and (np.random.randint(2) > 0): # flip half of the time
        shear += 180
        rotation += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    # random zoom [0.9, 1.1]
    # zoom = np.random.uniform(*zoom_range)
    log_zoom_range = [np.log(z) for z in zoom_range]
    zoom = np.exp(np.random.uniform(*log_zoom_range)) # for a zoom factor this sampling approach makes more sense.
    # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.

    return build_aug_transform(tform_center, tform_uncenter, zoom, rotation, shear, translation)


def build_aug_transform(tform_center, tform_uncenter, zoom=1.0, rotation=0, shear=0, translation=(0, 0)):
    """ 
        This function reads the images and correspondings labels of the dataset.

        Parameters
        ----------
        zoom: (float, float)
        rotation: (int, int)
        shear: (int, int
        translation: (int, int)
        tform_center:
        tform_uncenter:


        Copyright (c) 2014, Sander Dieleman
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright notice, this
          list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above copyright notice,
          this list of conditions and the following disclaimer in the documentation
          and/or other materials provided with the distribution.

        * Neither the name of the {organization} nor the names of its
          contributors may be used to endorse or promote products derived from
          this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    tform_augment = skimage.transform.AffineTransform(scale=(1/zoom, 1/zoom), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
    tform = tform_center + tform_augment + tform_uncenter # shift to center, augment, shift back (for the rotation/shearing)
    return tform


def build_ds_transform(orig_size, target_size, ds_factor=1.0, do_shift=True, subpixel_shift=False):
    """ 
        This function 

        Parameters
        ----------
        orig_size
        ds_factor:
        do_shift:
        subpixel_shift:
        orig_size:
        target_size:

        Copyright (c) 2014, Sander Dieleman
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright notice, this
          list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above copyright notice,
          this list of conditions and the following disclaimer in the documentation
          and/or other materials provided with the distribution.

        * Neither the name of the {organization} nor the names of its
          contributors may be used to endorse or promote products derived from
          this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    rows, cols = orig_size
    trows, tcols = target_size
    col_scale = row_scale = ds_factor
    src_corners = np.array([[1, 1], [1, rows], [cols, rows]]) - 1
    dst_corners = np.zeros(src_corners.shape, dtype=np.double)
    # take into account that 0th pixel is at position (0.5, 0.5)
    dst_corners[:, 0] = col_scale * (src_corners[:, 0] + 0.5) - 0.5
    dst_corners[:, 1] = row_scale * (src_corners[:, 1] + 0.5) - 0.5

    tform_ds = skimage.transform.AffineTransform()
    tform_ds.estimate(src_corners, dst_corners)

    if do_shift:
        if subpixel_shift: 
            # if this is true, we add an additional 'arbitrary' subpixel shift, which 'aligns'
            # the grid of the target image with the original image in such a way that the interpolation
            # is 'cleaner', i.e. groups of <ds_factor> pixels in the original image will map to
            # individual pixels in the resulting image.
            #
            # without this additional shift, and when the downsampling factor does not divide the image
            # size (like in the case of 424 and 3.0 for example), the grids will not be aligned, resulting
            # in 'smoother' looking images that lose more high frequency information.
            #
            # technically this additional shift is not 'correct' (we're not looking at the very center
            # of the image anymore), but it's always less than a pixel so it's not a big deal.
            #
            # in practice, we implement the subpixel shift by rounding down the orig_size to the
            # nearest multiple of the ds_factor. Of course, this only makes sense if the ds_factor
            # is an integer.

            cols = (cols // int(ds_factor)) * int(ds_factor)
            rows = (rows // int(ds_factor)) * int(ds_factor)
            # print "NEW ROWS, COLS: (%d,%d)" % (rows, cols)

        shift_x = cols / (2 * ds_factor) - tcols / 2.0
        shift_y = rows / (2 * ds_factor) - trows / 2.0
        tform_shift_ds = skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))
        return tform_shift_ds + tform_ds
    else:
        return tform_ds

def fast_warp(img, tf, output_shape=(53,53), mode='reflect'):
    """ 
    This wrapper function is about five times faster than skimage.transform.warp, for our use case.
    """
    m = tf._matrix
    img_wf = np.empty((output_shape[0], output_shape[1], 3), dtype='float32')
    for k in xrange(3):
        img_wf[..., k] = skimage.transform._warps_cy._warp_fast(img[..., k], m, output_shape=output_shape, mode=mode)
    return img_wf

if __name__ == '__main__':
    main()
