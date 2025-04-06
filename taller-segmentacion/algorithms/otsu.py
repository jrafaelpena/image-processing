from pathlib import Path
from typing import Union
import itk
import os
import time

def otsu_threshold_filter(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    number_of_bins: int,
    dimensions: int = 3
) -> None:
    if number_of_bins <= 1:
        raise ValueError("Number of bins must be greater than 1")
    
    pixel_type = itk.US
    image_type = itk.Image[pixel_type, dimensions]

    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(str(input_file))

    image = reader.GetOutput()
    size = itk.size(image)
    
    # Apply Otsu threshold filter
    otsu_filter = itk.OtsuThresholdImageFilter[image_type, image_type].New()
    otsu_filter.SetInput(image)
    otsu_filter.SetNumberOfHistogramBins(number_of_bins)
    otsu_filter.SetInsideValue(1)
    otsu_filter.SetOutsideValue(0) 
    
    # Rescale the output to [0, 255]
    rescale_filter = itk.RescaleIntensityImageFilter[image_type, image_type].New()
    rescale_filter.SetInput(otsu_filter.GetOutput())
    rescale_filter.SetOutputMinimum(0)
    rescale_filter.SetOutputMaximum(255)
    rescale_filter.Update()
    
    itk.imwrite(rescale_filter.GetOutput(), str(output_file))
    
    return tuple(size)


if __name__ == "__main__":
    BASE_PATH = Path(os.getcwd()).parent
    names = ["MRBrainTumor", "MRBreastCancer", "MRLiverTumor"]
    image_name = names[2]
    extension = "nii.gz"

    bins_list = [7]

    for bins in bins_list:
        #bins = 8
        
        input_image = BASE_PATH / f"inputs/{image_name}.{extension}"
        output_image = BASE_PATH / f"outputs/{image_name}_otsu_{bins}.{extension}"
        
        # Start timer
        start_time = time.time()
        
        size = otsu_threshold_filter(input_image, output_image, bins)
    
        # Calculate total time
        total_time = time.time() - start_time
        print(f"Total time taken: {total_time:.2f} seconds")