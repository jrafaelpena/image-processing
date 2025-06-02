from pathlib import Path
from typing import Union
import itk
import os
import time


def watershed_segmentation(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    time_step: float = 0.0625,
    conductance: float = 1.0,
    iterations: int = 5,
    threshold: float = 0.01,
    level: float = 0.25,
    dimensions: int = 3
) -> None:
    pixel_type_input = itk.F
    image_type_input = itk.Image[pixel_type_input, dimensions]
    
    # Change this to a supported output type
    pixel_type_output = itk.UI  # Unsigned int (was UL which is not supported)
    image_type_output = itk.Image[pixel_type_output, dimensions]
    
    reader = itk.ImageFileReader[image_type_input].New()
    reader.SetFileName(str(input_file))
    image = reader.GetOutput()
    size = itk.size(image)
    
    # Denoising with GradientAnisotropicDiffusionImageFilter
    smoothing = itk.GradientAnisotropicDiffusionImageFilter[image_type_input, image_type_input].New()
    smoothing.SetInput(image)
    smoothing.SetTimeStep(time_step)
    smoothing.SetConductanceParameter(conductance)
    smoothing.SetNumberOfIterations(iterations)
    
    # Compute gradient magnitude
    gradient_filter = itk.GradientMagnitudeImageFilter[image_type_input, image_type_input].New()
    gradient_filter.SetInput(smoothing.GetOutput())
    
    # Apply Watershed filter
    watershed_filter = itk.WatershedImageFilter[image_type_input].New()
    watershed_filter.SetInput(gradient_filter.GetOutput())
    watershed_filter.SetThreshold(threshold)
    watershed_filter.SetLevel(level)
    watershed_filter.Update()
    
    # Cast the output to a supported type before writing
    cast_filter = itk.CastImageFilter[itk.Image[itk.UL, dimensions], image_type_output].New()
    cast_filter.SetInput(watershed_filter.GetOutput())
    cast_filter.Update()
    
    itk.imwrite(cast_filter.GetOutput(), str(output_file))
    return tuple(size)

if __name__ == "__main__":
    BASE_PATH = Path(os.getcwd()).parent
    names = ["MRBrainTumor", "MRBreastCancer", "MRLiverTumor"]
    short_names = ['BT', 'BstC', 'LT']
    time_steps = [0.050, 0.025, 0.0625]

    index = 2
    threshold=0.05
    level=0.065
    
    image_name = names[index]
    image_short_name = short_names[index]
    time_step = time_steps[index]
    extension = "nii.gz"

    input_image = BASE_PATH / f"inputs/{image_name}.{extension}"
    output_image = BASE_PATH / f"outputs/{image_short_name}_watershed_{threshold}_{level}.{extension}"
    
    start_time = time.time()
    
    watershed_segmentation(
        input_file=input_image,
        output_file=output_image,
        time_step=time_step,
        threshold=threshold,
        level=level
    )

    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds")