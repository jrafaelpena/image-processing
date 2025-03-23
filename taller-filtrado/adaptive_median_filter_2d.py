import itk
import numpy as np
from pathlib import Path
from typing import Union
import os

def adaptive_median_filter(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    max_radius: int
) -> None:
    # Leer la imagen
    pixel_type = itk.ctype("unsigned char")
    image_type = itk.Image[pixel_type, 2]

    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(str(input_file))
    reader.Update()
    
    itk_image = reader.GetOutput()
    np_image = itk.GetArrayFromImage(itk_image)

    height, width = np_image.shape
    input_image = np.copy(np_image)
    output_image = np.zeros(np_image.shape)

    for y in range(height):
        for x in range(width):
            output_image[y, x] = process_pixel(input_image, y, x, max_radius)

    itk_output = itk.GetImageFromArray(output_image.astype(np.uint8))
    itk_output.SetSpacing(itk_image.GetSpacing())
    itk_output.SetOrigin(itk_image.GetOrigin())

    itk.imwrite(itk_output, str(output_file))
            
    return tuple(itk.size(itk_image)), input_image, output_image

def process_pixel(image: np.ndarray, y: int, x: int, max_radius: int) -> int:
    radius = 1
    while radius <= max_radius:
        window = get_window(image, y, x, radius)
        z_med = get_median(window)
        z_max = int(np.max(window))
        z_min = int(np.min(window))
        z_xy = int(image[y, x])

        # Nivel 1: Evaluar la mediana
        if z_min < z_med < z_max:
            # Nivel 2: Evaluar el valor del píxel original
            if z_min < z_xy < z_max:
                return z_xy
            else:
                return z_med
        else:
            radius += 1  # Aumentar el radio

    return z_xy  # Si se supera max_radius, devolver el píxel original

def get_window(image: np.ndarray, y: int, x: int, radius: int) -> np.ndarray:
    y_min, y_max = max(0, y - radius), min(image.shape[0], y + radius + 1)
    x_min, x_max = max(0, x - radius), min(image.shape[1], x + radius + 1)
    
    return image[y_min:y_max, x_min:x_max]

def get_median(image: np.ndarray):
    target_array = image.flatten()
    sorted_array = np.sort(target_array)
    return int(sorted_array[len(sorted_array) // 2])