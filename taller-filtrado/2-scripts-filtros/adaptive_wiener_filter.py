import itk
import numpy as np
from pathlib import Path
from typing import Union
import os
from scipy import signal

def adaptive_wiener_filter_3d(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    radius: int
) -> None:
    # Leer imagen 3D con ITK
    pixel_type = itk.ctype("float")
    image_type = itk.Image[pixel_type, 3]  # Changed dimension to 3
    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(str(input_file))
    reader.Update()

    # Convertir imagen a numpy array
    itk_image = reader.GetOutput()
    np_image = itk.GetArrayFromImage(itk_image)
    depth, height, width = np_image.shape
    input_image = np.copy(np_image)
    
    # Aplicar filtro de wiener adaptativo
    filtered_image = wiener3d(input_image, radius)
    
    # Convertir numpy array a imagen ITK y descargar
    output_itk_image = itk.GetImageFromArray(filtered_image)
    output_itk_image.SetOrigin(itk_image.GetOrigin())
    output_itk_image.SetSpacing(itk_image.GetSpacing())
    output_itk_image.SetDirection(itk_image.GetDirection()) # Configurar la dirección (IMPORTANTE PARA QUE NO QUEDE VOLTEADA)
    
    itk.imwrite(output_itk_image, str(output_file))

    return tuple(itk.size(itk_image))

def wiener3d(img, radius):
    
    # Se convierte la imagen a tipo float64 para asegurar precisión en los cálculos
    img = img.astype(np.float64)
    
    # Se calcula el tamaño de la ventana y se crea el kernel normalizado para la convolución
    window_size = 2 * radius + 1
    kernel = np.ones((window_size, window_size, window_size)) / (window_size**3)
    
    # Cálculo de la media local usando convolución 3D
    # fftconvolve aplica padding con ceros por defecto en modo 'same' para mantener las dimensiones
    local_mean = signal.fftconvolve(img, kernel, mode='same')
    
    # Cálculo de la varianza local:
    # Primero se obtiene el promedio de los valores al cuadrado
    local_squared_mean = signal.fftconvolve(img**2, kernel, mode='same')
    # Luego se aplica la fórmula: Var(X) = E[X²] - (E[X])²
    local_var = local_squared_mean - local_mean**2
    
    # Se asegura que la varianza no sea negativa (por errores de precisión numérica)
    local_var = np.maximum(local_var, 0)
    
    # Estimación de la varianza del ruido como el promedio de todas las varianzas locales
    noise_var = np.mean(local_var)
    
    # Cálculo del factor adaptativo de Wiener: max(0, (σ²ₗ - σ²ₙ)/σ²ₗ)
    # Donde σ²ₗ es la varianza local y σ²ₙ es la varianza del ruido
    factor = np.divide(
        (local_var - noise_var),
        np.maximum(local_var, 1e-10),  # Evita división por cero con un valor mínimo
        out=np.zeros_like(local_var),
        where=local_var > 0
    )
    
    # Limita el factor entre 0 y 1 para asegurar un comportamiento estable
    factor = np.clip(factor, 0, 1)
    
    # Aplicación del filtro mediante la fórmula:
    # resultado = media_local + factor * (valor_original - media_local)
    # Cuando factor=0 (zona homogénea), el resultado es la media local
    # Cuando factor=1 (zona con detalles), se preserva la diferencia con la media
    result = local_mean + factor * (img - local_mean)
    
    return result

if __name__ == "__main__":
    BASE_PATH = Path(os.getcwd()).parent
    names = ["3_percent_noise", "9_percent_noise", "50_percent_impulsive_noise", "gaussian_noise"]
    image_name = names[3]
    extension = "nii.gz"
    input_image = BASE_PATH / f"inputs/{image_name}.{extension}"
    output_image = BASE_PATH / f"outputs/{image_name}_AWF.{extension}"

    size = adaptive_wiener_filter_3d(input_image, output_image, 1)