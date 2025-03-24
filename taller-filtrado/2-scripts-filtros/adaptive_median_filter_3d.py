import itk
import numpy as np
from pathlib import Path
from typing import Union
import os

def adaptive_median_filter_3d(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    max_radius: int
) -> None:
    # Leer imagen 3D con ITK
    pixel_type = itk.ctype("float")
    image_type = itk.Image[pixel_type, 3]
    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(str(input_file))
    reader.Update()

    # Convertir imagen a numpy array
    itk_image = reader.GetOutput()
    np_image = itk.GetArrayFromImage(itk_image)
    depth, height, width = np_image.shape
    input_image = np.copy(np_image)
    output_image = np.zeros(np_image.shape)
    
    # Se procesa cada voxel
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                output_image[z, y, x] = process_voxel(input_image, z, y, x, max_radius)
    
    # Convertir numpy array a imagen ITK y descargar
    itk_output = itk.GetImageFromArray(output_image)
    itk_output.SetSpacing(itk_image.GetSpacing())
    itk_output.SetOrigin(itk_image.GetOrigin())
    itk_output.SetDirection(itk_image.GetDirection())   # Configurar la dirección (IMPORTANTE PARA QUE NO QUEDE VOLTEADA)
    itk.imwrite(itk_output, str(output_file))
            
    return tuple(itk.size(itk_image)), input_image, output_image

def process_voxel(image: np.ndarray, z: int, y: int, x: int, max_radius: int) -> int:
    radius = 1
    while radius <= max_radius:
        window = get_window_3d(image, z, y, x, radius)
        v_med = get_median(window)
        v_max = np.max(window)
        v_min = np.min(window)
        v_xyz = image[z, y, x]
        
        # Nivel 1: Evalúa si la mediana está en el rango de valores de la ventana
        # Si la mediana está entre el mínimo y máximo, no se considera ruido impulsivo
        if v_min < v_med < v_max:
            # Nivel 2: Evalúa si el voxel central (original) está en el rango válido
            # Si el voxel no es un impulso, se conserva su valor original
            if v_min < v_xyz < v_max:
                return v_xyz
            # Si el voxel es un impulso pero la mediana no lo es, se reemplaza por la mediana
            else:
                return v_med
        # Si la mediana también es un impulso, se aumenta el radio de la ventana
        # para buscar una mediana más representativa del entorno
        else:
            radius += 1 
    # Si se agota el radio máximo sin encontrar una ventana adecuada,
    # se mantiene el valor original del voxel
    return v_xyz

def get_window_3d(image: np.ndarray, z: int, y: int, x: int, radius: int) -> np.ndarray:

    z_min, z_max = max(0, z - radius), min(image.shape[0], z + radius + 1)
    y_min, y_max = max(0, y - radius), min(image.shape[1], y + radius + 1)
    x_min, x_max = max(0, x - radius), min(image.shape[2], x + radius + 1)
    
    return image[z_min:z_max, y_min:y_max, x_min:x_max]

def get_median(image: np.ndarray):
    target_array = image.flatten()
    sorted_array = np.sort(target_array)
    return sorted_array[len(sorted_array) // 2]


if __name__ == "__main__":
    BASE_PATH = Path(os.getcwd()).parent
    names = ["3_percent_noise", "9_percent_noise", "50_percent_impulsive_noise", "gaussian_noise"]
    image_name = names[2]
    extension = "nii.gz"
    input_image = BASE_PATH / f"inputs/{image_name}.{extension}"
    output_image = BASE_PATH / f"outputs/{image_name}_AMF.{extension}"

    size = adaptive_median_filter_3d(input_image, output_image, 4)