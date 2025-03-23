import itk
import numpy as np
from pathlib import Path
from typing import Union
import os

BASE_PATH = Path(os.getcwd())

def adaptive_median_filter(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    max_radius: int
) -> None:
    itk_image = itk.imread(str(input_file))

    np_image = itk.GetArrayFromImage(itk_image)
    
    depth, height, width = np_image.shape
    output_image = np.copy(np_image)

    for z in range(depth):
        for y in range(height):
            for x in range(width):
                output_image[z, y, x] = process_pixel(np_image, x, y, z, max_radius)

    itk_output = itk.GetImageFromArray(output_image)
    itk_output.SetSpacing(itk_image.GetSpacing())  # Mantener la misma información espacial
    itk_output.SetOrigin(itk_image.GetOrigin())

    itk.imwrite(itk_output, str(output_file))

    return np_image, output_image


def process_pixel(image: np.ndarray, x: int, y: int, z: int, max_radius: int) -> int:
    """ Aplica el filtro de mediana adaptativo a un solo píxel """
    radius = 1
    while radius <= max_radius:
        window = get_window(image, x, y, z, radius)
        min_val, med_val, max_val = np.min(window), np.median(window), np.max(window)

        # Nivel 1: Evaluar la mediana
        if min_val < med_val < max_val:
            # Nivel 2: Evaluar el valor del píxel original
            pixel_value = image[z, y, x]
            if min_val < pixel_value < max_val:
                return pixel_value
            else:
                return med_val
        else:
            radius += 1  # Aumentar el radio

    return image[z, y, x]  # Si se supera max_radius, devolver el píxel original


def get_window(image: np.ndarray, x: int, y: int, z: int, radius: int) -> np.ndarray:
    """ Extrae la ventana 3d, se le suma uno al máximo dado que el slicer de numpy no incluye el límite superior"""
    z_min, z_max = max(0, z - radius), min(image.shape[0], z + radius + 1)
    y_min, y_max = max(0, y - radius), min(image.shape[1], y + radius + 1)
    x_min, x_max = max(0, x - radius), min(image.shape[2], x + radius + 1)
    
    return image[z_min:z_max, y_min:y_max, x_min:x_max]


# Ejemplo de uso
input_array, output_array = adaptive_median_filter(BASE_PATH / "inputs/9_percent_noise.nii.gz", "output.nii.gz", max_radius=3)