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
    # Read the 3D image
    pixel_type = itk.ctype("float")
    image_type = itk.Image[pixel_type, 3]  # Changed dimension to 3
    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(str(input_file))
    reader.Update()
    
    itk_image = reader.GetOutput()
    np_image = itk.GetArrayFromImage(itk_image)
    depth, height, width = np_image.shape  # Now we have 3 dimensions
    input_image = np.copy(np_image)
    output_image = np.zeros(np_image.shape)
    
    # Process each voxel in the 3D volume
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                output_image[z, y, x] = process_voxel(input_image, z, y, x, max_radius)
    
    # Convert back to ITK image and save
    itk_output = itk.GetImageFromArray(output_image)
    itk_output.SetSpacing(itk_image.GetSpacing())
    itk_output.SetOrigin(itk_image.GetOrigin())
    itk_output.SetDirection(itk_image.GetDirection())  # Preserve direction matrix for 3D
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
        
        # Level 1: Evaluate the median
        if v_min < v_med < v_max:
            # Level 2: Evaluate the original voxel value
            if v_min < v_xyz < v_max:
                return v_xyz
            else:
                return v_med
        else:
            radius += 1  # Increase the radius
    
    return v_xyz  # If max_radius is exceeded, return the original voxel

def get_window_3d(image: np.ndarray, z: int, y: int, x: int, radius: int) -> np.ndarray:
    # Calculate bounds for the 3D window, accounting for volume boundaries
    z_min, z_max = max(0, z - radius), min(image.shape[0], z + radius + 1)
    y_min, y_max = max(0, y - radius), min(image.shape[1], y + radius + 1)
    x_min, x_max = max(0, x - radius), min(image.shape[2], x + radius + 1)
    
    # Return the 3D window
    return image[z_min:z_max, y_min:y_max, x_min:x_max]

def get_median(image: np.ndarray):
    target_array = image.flatten()
    sorted_array = np.sort(target_array)
    return sorted_array[len(sorted_array) // 2]