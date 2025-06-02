from pathlib import Path
from typing import Union
import itk
import os
import sys
import time


def confidence_connected_segmentation(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    number_of_iterations: int,
    multiplier: float,
    neighborhood_radius: int,
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
    size = itk.size(image)
    print("Image size:", size)

    confidence_filter = itk.ConfidenceConnectedImageFilter[image_type, image_type].New()
    confidence_filter.SetInitialNeighborhoodRadius(neighborhood_radius)
    confidence_filter.SetMultiplier(multiplier)
    confidence_filter.SetNumberOfIterations(number_of_iterations)
    confidence_filter.SetReplaceValue(255)
    confidence_filter.SetSeed(seed)
    confidence_filter.SetInput(image)

    rescaler = itk.RescaleIntensityImageFilter[image_type, image_type].New()
    rescaler.SetInput(confidence_filter.GetOutput())
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(255)

    itk.imwrite(rescaler.GetOutput(), str(output_file))


if __name__ == "__main__":
    BASE_PATH = Path(os.getcwd()).parent

    
    input_image = BASE_PATH / "inputs/A1_grayT1.nii.gz"
    output_image = BASE_PATH / "outputs/A1_grayT1.nii.gz"
    number_of_iterations = 2
    multiplier = 1.5
    neighborhood_radius = 2
    seed_coords = (145, 142, 98) 
    #seed_coords = (98, 142, 145) 

    start_time = time.time()
    confidence_connected_segmentation(
        input_file=input_image,
        output_file=output_image,
        number_of_iterations=number_of_iterations,
        multiplier=multiplier,
        neighborhood_radius=neighborhood_radius,
        seed=seed_coords,
    )
    total_time = time.time() - start_time
    print(f"Tiempo total tomado: {total_time:.2f} segundos")
