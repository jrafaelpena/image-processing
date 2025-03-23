from pathlib import Path
from typing import Union
import itk

def median_image_filter(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    radius_value: int,
    dimensions: int = 3
) -> None:
    if radius_value < 1:
        raise ValueError("El radio debe ser mayor que 1")
    
    pixel_type = itk.ctype("unsigned char")
    image_type = itk.Image[pixel_type, dimensions]

    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(str(input_file))

    image = reader.GetOutput()
    size = itk.size(image)
    
    mean_filter = itk.MedianImageFilter[image_type, image_type].New()
    mean_filter.SetInput(image)
    
    # Set radius
    radius = itk.Size[dimensions]()
    for i in range(dimensions):
        radius[i] = radius_value
    
    mean_filter.SetRadius(radius)
    mean_filter.Update()
    
    itk.imwrite(mean_filter.GetOutput(), str(output_file))
    
    return tuple(size)
