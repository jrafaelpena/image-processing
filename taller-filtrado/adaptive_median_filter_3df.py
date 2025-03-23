import itk
import numpy as np
from pathlib import Path
from typing import Union
import os

def adaptive_median_filter_2d_slices(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    max_radius: int
) -> None:
    # leer la imagen 3D
    pixel_type = itk.ctype("float")
    image_type = itk.Image[pixel_type, 3]
    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(str(input_file))
    reader.Update()
    itk_image = reader.GetOutput()

    np_image = itk.GetArrayFromImage(itk_image)