from pathlib import Path
from typing import Union
import itk
import os

def median_image_filter(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    radius_value: int,
    dimensions: int = 3
) -> None:
    if radius_value < 1:
        raise ValueError("El radio debe ser mayor que 1")
    
    pixel_type = itk.ctype("float")
    image_type = itk.Image[pixel_type, dimensions]

    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(str(input_file))

    image = reader.GetOutput()
    size = itk.size(image)
    
    median_filter = itk.MedianImageFilter[image_type, image_type].New()
    median_filter.SetInput(image)
    
    # Set radius
    radius = itk.Size[dimensions]()
    for i in range(dimensions):
        radius[i] = radius_value
    
    median_filter.SetRadius(radius)
    median_filter.Update()
    
    itk.imwrite(median_filter.GetOutput(), str(output_file))
    
    return tuple(size)


if __name__ == "__main__":
    BASE_PATH = Path(os.getcwd()).parent
    names = ["3_percent_noise", "9_percent_noise", "50_percent_impulsive_noise", "gaussian_noise"]
    image_name = names[3]
    extension = "nii.gz"
    input_image = BASE_PATH / f"inputs/{image_name}.{extension}"
    output_image = BASE_PATH / f"outputs/{image_name}_MF.{extension}"

    size = median_image_filter(input_image, output_image, 1)