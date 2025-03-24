import itk
import numpy as np
from pathlib import Path
from typing import Union
import os

def add_gaussian_noise(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    mean: float = 0.0,
    std_dev: float = 0.1,
) -> None:
    """
    Agrega ruido gaussiano a una imagen 3D médica y la guarda.
    Parámetros:
        input_file (Union[str, Path]): Ruta de la imagen 3D de entrada.
        output_file (Union[str, Path]): Ruta del archivo donde se guardará la imagen ruidosa.
        mean (float): Media del ruido gaussiano. Por defecto 0.0.
        std_dev (float): Desviación estándar del ruido gaussiano, controla la intensidad. Por defecto 0.1.
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
    
    # Calcular el rango de la imagen para normalizar la desviación estándar del ruido
    image_range = np.max(np_image) - np.min(np_image)
    noise_std_dev = std_dev * image_range
    
    # Generar el ruido gaussiano del mismo tamaño que la imagen
    noise = np.random.normal(mean, noise_std_dev, np_image.shape)
    
    # Añadir el ruido a la imagen
    noisy_image = np_image + noise
    
    # Opcional: Limitar los valores resultantes al rango original de la imagen
    # Esto evita que el ruido produzca valores fuera del rango esperado
    min_value = np.min(np_image)
    max_value = np.max(np_image)
    noisy_image = np.clip(noisy_image, min_value, max_value)
    
    # Convertir el array de numpy modificado de vuelta a una imagen ITK
    itk_noisy_image = itk.GetImageFromArray(noisy_image)
    
    # Preservar los metadatos de la imagen original
    itk_noisy_image.SetSpacing(itk_image.GetSpacing())
    itk_noisy_image.SetOrigin(itk_image.GetOrigin())
    itk_noisy_image.SetDirection(itk_image.GetDirection())  # Importante para mantener la orientación
    
    # Guardar la imagen con ruido en el archivo de salida especificado
    itk.imwrite(itk_noisy_image, str(output_file))


if __name__ == "__main__":
    BASE_PATH = Path(os.getcwd()).parent
    image_name = "0_percent_noise"
    extension = "nii.gz"
    input_image = BASE_PATH / f"inputs/{image_name}.{extension}"
    output_image = BASE_PATH / f"outputs/gaussian_noise.{extension}"
    
    # Añadir ruido gaussiano con desviación estándar de 0.1 (10% del rango)
    add_gaussian_noise(input_image, output_image, mean=0.0, std_dev=0.25)