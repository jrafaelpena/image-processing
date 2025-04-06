from pathlib import Path
from typing import Union
import itk
import os
import time

def binary_threshold_filter(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    lower_threshold: int,
    upper_threshold: int,
    outside_value: int,
    inside_value: int,
    dimensions: int = 3
) -> None:
    
    pixel_type = itk.US
    image_type = itk.Image[pixel_type, dimensions]

    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(str(input_file))

    image = reader.GetOutput()
    size = itk.size(image)
    
    # Apply Binary Threshold filter
    threshold_filter = itk.BinaryThresholdImageFilter[image_type, image_type].New()
    threshold_filter.SetInput(image)
    threshold_filter.SetLowerThreshold(lower_threshold)
    threshold_filter.SetUpperThreshold(upper_threshold)
    threshold_filter.SetOutsideValue(outside_value)
    threshold_filter.SetInsideValue(inside_value)
    threshold_filter.Update()
    
    itk.imwrite(threshold_filter.GetOutput(), str(output_file))
    
    return tuple(size)


if __name__ == "__main__":
    BASE_PATH = Path(os.getcwd()).parent
    names = ["MRBrainTumor", "MRBreastCancer", "MRLiverTumor"]
    image_name = names[2]
    extension = "nii.gz"
    lower_threshold = 55
    upper_threshold = 89
    input_image = BASE_PATH / f"inputs/{image_name}.{extension}"
    output_image = BASE_PATH / f"outputs/{image_name}_binarizacion_{lower_threshold}_{upper_threshold}.{extension}"
    
    start_time = time.time()
    
    outside_value = 0
    inside_value = 255
    
    size = binary_threshold_filter(input_image, output_file=output_image, 
                                  lower_threshold=lower_threshold, 
                                  upper_threshold=upper_threshold, 
                                  outside_value=outside_value, 
                                  inside_value=inside_value)

    # Calculate total time
    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds")