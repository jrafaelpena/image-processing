from pathlib import Path
from typing import Union
import itk
import os
import sys
import time


def connected_threshold_segmentation(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    lower_threshold: int,
    upper_threshold: int,
    seed: tuple[int, int, int],
    dimensions: int = 3,
) -> None:
    if len(seed) != 3:
        raise ValueError("Seed must be a tuple of (x, y, z) coordinates.")

    pixel_type = itk.US
    image_type = itk.Image[pixel_type, dimensions]

    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(str(input_file))
    reader.Update()

    image = reader.GetOutput()
    region = image.GetLargestPossibleRegion()
    size = region.GetSize()
    print("Image size:", size)

    connected_threshold = itk.ConnectedThresholdImageFilter[image_type, image_type].New()
    connected_threshold.SetLower(lower_threshold)
    connected_threshold.SetUpper(upper_threshold)
    connected_threshold.SetReplaceValue(255)
    connected_threshold.SetSeed(seed)
    connected_threshold.SetInput(image)

    rescaler = itk.RescaleIntensityImageFilter[image_type, image_type].New()
    rescaler.SetInput(connected_threshold.GetOutput())
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(255)

    writer = itk.ImageFileWriter[image_type].New()
    writer.SetFileName(str(output_file))
    writer.SetInput(rescaler.GetOutput())
    writer.Update()


if __name__ == "__main__":
    BASE_PATH = Path(os.getcwd()).parent
    
    input_image = BASE_PATH / "inputs/A1_grayT1.nii.gz"
    output_image = BASE_PATH / "outputs/A1_grayT1_segmented.nii.gz"
    lower_threshold = 70
    upper_threshold = 180
    seed_coords = (145, 142, 98) 
    

    start_time = time.time()
    connected_threshold_segmentation(
        input_file=input_image,
        output_file=output_image,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        seed=seed_coords,
    )
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")