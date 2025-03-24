import itk
import numpy as np
from pathlib import Path
from typing import Union, Tuple
import os


def add_salt_and_pepper_noise(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    noise_density: float,
) -> Tuple[int, int, int]:
    """
    Agrega ruido impulsivo tipo "salt and pepper" a una imagen 3D y la guarda.

    Parámetros:
        input_file (Union[str, Path]): Ruta de la imagen 3D de entrada.
        output_file (Union[str, Path]): Ruta del archivo donde se guardará la imagen ruidosa.
        noise_density (float): Densidad total del ruido (ruido blanco y negro combinados).
    """
    # Configuración de los tipos de datos de ITK
    pixel_type = itk.ctype("float")  # Tipo de píxel como flotante
    image_type = itk.Image[pixel_type, 3]  # Definición de imagen 3D en ITK

    # Leer la imagen 3D desde el archivo de entrada
    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(str(input_file))
    reader.Update()
    itk_image = reader.GetOutput()  # Imagen leída en formato ITK

    # Convertir la imagen de ITK a un array de NumPy para su manipulación
    np_image = itk.GetArrayFromImage(itk_image)  # Convertir ITK a NumPy
    noisy_image = np.copy(np_image)  # Crear una copia de la imagen para añadir ruido

    # Calcular el número total de píxeles a modificar con ruido
    total_voxels = np_image.size
    num_salt = int(np.ceil(noise_density * total_voxels / 2))  # Píxeles de "sal" (blanco)
    num_pepper = num_salt  # Píxeles de "pimienta" (negro)

    # Generar coordenadas aleatorias para los píxeles de "sal"
    salt_indices = np.random.choice(total_voxels, num_salt, replace=False)
    salt_coords = np.unravel_index(salt_indices, np_image.shape)
    noisy_image[salt_coords] = np.max(np_image)  # Asignar valor máximo (blanco)
    
    # Generar coordenadas aleatorias para los píxeles de "pimienta"
    pepper_indices = np.random.choice(total_voxels, num_pepper, replace=False)
    pepper_coords = np.unravel_index(pepper_indices, np_image.shape)
    noisy_image[pepper_coords] = np.min(np_image)  # Asignar valor mínimo (negro)

    # Convertir el array de numoy modificado de vuelta a una imagen ITK
    itk_noisy_image = itk.GetImageFromArray(noisy_image)
    itk_noisy_image.SetSpacing(itk_image.GetSpacing())
    itk_noisy_image.SetOrigin(itk_image.GetOrigin())
    itk_noisy_image.SetDirection(itk_image.GetDirection())  # Configurar la dirección (IMPORTANTE PARA QUE NO QUEDE VOLTEADA)

    # Guardar la imagen con ruido en el archivo de salida especificado
    itk.imwrite(itk_noisy_image, str(output_file))

if __name__ == "__main__":
    BASE_PATH = Path(os.getcwd()).parent
    image_name = "0_percent_noise"
    extension = "nii.gz"
    input_image = BASE_PATH / f"inputs/{image_name}.{extension}"
    output_image = BASE_PATH / f"outputs/50_percent_impulsive_noise.{extension}"

    size = add_salt_and_pepper_noise(input_image, output_image, 0.5)