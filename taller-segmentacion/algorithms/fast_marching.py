from pathlib import Path
from typing import Union, List, Tuple
import itk
import os
import time

def create_sphere_mask(image, seed_indices, radius_voxels):
    import numpy as np
    size = image.GetBufferedRegion().GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = np.array(image.GetDirection()).reshape((3, 3))
    
    arr = np.zeros(tuple(reversed(size)), dtype=np.uint8)  # z, y, x
    for center in seed_indices:
        cz, cy, cx = center
        for z in range(size[2]):
            for y in range(size[1]):
                for x in range(size[0]):
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
                    if dist <= radius_voxels:
                        arr[z, y, x] = 255

    return itk.image_from_array(arr, is_vector=False)


def fast_marching_segmentation(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    seed_indices: List[Tuple[int, ...]],
    time_threshold: float,
    sigma: float = 1.0,
    alpha: float = -1.0,
    beta: float = 0.0,
    time_step: float = 0.125,
    conductance: float = 9.0,
    iterations: int = 5,
    dimensions: int = 3
) -> None:
    pixel_type_input = itk.F
    image_type = itk.Image[pixel_type_input, dimensions]
    output_pixel_type = itk.UC
    output_image_type = itk.Image[output_pixel_type, dimensions]
    
    # Reader
    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(str(input_file))
    reader.Update()
    
    image = reader.GetOutput()
    
    # Smoothing
    smoothing = itk.CurvatureAnisotropicDiffusionImageFilter[image_type, image_type].New()
    smoothing.SetInput(image)
    smoothing.SetTimeStep(time_step)
    smoothing.SetConductanceParameter(conductance)
    smoothing.SetNumberOfIterations(iterations)
    
    # Gradient magnitude
    gradient = itk.GradientMagnitudeRecursiveGaussianImageFilter[image_type, image_type].New()
    gradient.SetInput(smoothing.GetOutput())
    gradient.SetSigma(sigma)
    
    # Sigmoid
    sigmoid = itk.SigmoidImageFilter[image_type, image_type].New()
    sigmoid.SetInput(gradient.GetOutput())
    sigmoid.SetOutputMinimum(0.0)
    sigmoid.SetOutputMaximum(1.0)
    sigmoid.SetAlpha(alpha)
    sigmoid.SetBeta(beta)
    
    # Fast marching
    fast_marching = itk.FastMarchingImageFilter[image_type, image_type].New()
    fast_marching.SetInput(sigmoid.GetOutput())
    
    # Set seed points - Fixed approach for ITK Python wrapping
    trials = itk.VectorContainer[itk.UI, itk.LevelSetNode[pixel_type_input, dimensions]].New()
    trials.Initialize()
    
    for i, index_tuple in enumerate(seed_indices):
        node = itk.LevelSetNode[pixel_type_input, dimensions]()
        node.SetValue(0.0)
        idx = itk.Index[dimensions]()
        for j, val in enumerate(index_tuple):
            idx[j] = val
        node.SetIndex(idx)
        trials.InsertElement(i, node)
    
    fast_marching.SetTrialPoints(trials)
    fast_marching.SetStoppingValue(time_threshold)
    fast_marching.SetOutputSize(image.GetBufferedRegion().GetSize())
    

    # Threshold the output
    threshold_filter = itk.BinaryThresholdImageFilter[image_type, output_image_type].New()
    threshold_filter.SetInput(fast_marching.GetOutput())
    threshold_filter.SetLowerThreshold(0.0)
    threshold_filter.SetUpperThreshold(time_threshold)
    threshold_filter.SetInsideValue(255)
    threshold_filter.SetOutsideValue(0)
    
   
    
    # Save intermediate outputs for debugging
    base_output = Path(output_file).with_suffix('').as_posix()
    
    sphere_mask = create_sphere_mask(image, seed_indices, radius_voxels=20)
    itk.imwrite(sphere_mask, f"{base_output}_sphere_mask.nii.gz")
    itk.imwrite(smoothing.GetOutput(), f"{base_output}_smoothing.nii.gz")
    itk.imwrite(gradient.GetOutput(), f"{base_output}_gradient.nii.gz")
    itk.imwrite(sigmoid.GetOutput(), f"{base_output}_sigmoid.nii.gz")
    itk.imwrite(fast_marching.GetOutput(), f"{base_output}_fastmarching.nii.gz")
    itk.imwrite(threshold_filter.GetOutput(), f"{base_output}_thresholded.nii.gz")

if __name__ == "__main__":
    BASE_PATH = Path(os.getcwd()).parent
    names = ["MRBrainTumor", "MRBreastCancer", "MRLiverTumor"]
    short_names = ['BT', 'BstC', 'LT']
    time_steps = [0.050, 0.025, 0.0625]
    index = 0
    
    image_name = names[index]
    image_short_name = short_names[index]
    time_step = time_steps[index]
    extension = "nii.gz"
    input_image = BASE_PATH / f"inputs/{image_name}.{extension}"
    output_image = BASE_PATH / f"outputs/{image_short_name}_fastmarching.{extension}"
    
    start_time = time.time()

    seed_points = [
        (143, 100, 83),   # Another slight variation
        (145, 110, 84)
    ]
    
    fast_marching_segmentation(
        input_file=input_image,
        output_file=output_image,
        seed_indices=seed_points,
        time_step=0.008,
        alpha = -10.0,
        beta = 20.0,
        time_threshold = 20.0
    )
    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds")