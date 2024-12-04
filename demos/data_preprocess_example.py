from SAMReg.tools.utils.med import save_image, load_sitk_vol, adjust_direction
from SAMReg.tools.utils.med import read_image, seg_bg_mask
import SimpleITK as sitk
import numpy as np
import argparse

def resample_nii(input_path, output_path, target_spacing=2.0, mode='image'):
    image = load_sitk_vol(input_path)
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    direction = image.GetDirection()
    new_size = [
        int(original_size[0] * (original_spacing[0] / target_spacing)),
        int(original_size[1] * (original_spacing[1] / target_spacing)),
        int(original_size[2] * (original_spacing[2] / target_spacing)),
    ]
    print(input_path, 'original_spacing', original_spacing,'original_size', original_size, 'new_size', new_size)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing((target_spacing, target_spacing, target_spacing))
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    if mode == 'image':
        resampler.SetInterpolator(sitk.sitkLinear) 
    elif mode == 'label':
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    resampled_image = resampler.Execute(image)

    vol = sitk.GetArrayFromImage(resampled_image)
    if not np.all(np.array(direction) == np.eye(3).ravel()):
        vol = adjust_direction(input_path, vol, direction).copy()

    info = dict(spacing = (2.0, 2.0, 2.0), origin = image.GetOrigin(), direction=np.eye(3).ravel())
    print(vol.shape)
    save_image(vol, output_path, info)
    return 


if __name__ == "__main__":
    """
    Data processing for image including 1) resampling to a spacing of 2x2x2 mm;
                                        2) orienting to the RAI direction;
                                        3) generating mask file.
    Arguments:
        --input_path/ -i: the path to the input CT image
        --output_path/ -o: the path of output CT image
        --output_mask_path/: the path of output mask data
    Note that change the mode to 'label' when resampling.
    """
    parser = argparse.ArgumentParser(
        description="An easy interface for data processing"
    )
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        type=str,
        default=None
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        type=str,
        default=None
    )
    parser.add_argument(
        "--mask_path",
        required=True,
        type=str,
        default=None
    )
    args = parser.parse_args()
    

    input_path = args.input_path
    output_path = args.output_path
    mask_path = args.mask_path

    resample_nii(input_path, output_path, target_spacing=2.0, mode='image')

    new_input = output_path
    img, img_info = read_image(new_input)
    mask = seg_bg_mask(np.clip(img, -900, 1000))[0].astype(np.int32)
    save_image(mask, mask_path, img_info)
    
