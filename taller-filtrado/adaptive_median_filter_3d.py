import itk
import numpy as np
from pathlib import Path
from typing import Union

def adaptive_median_filter_2d_slices(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    max_radius: int
) -> None:
    # Read the 3D image
    pixel_type = itk.ctype("float")
    image_type = itk.Image[pixel_type, 3]
    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(str(input_file))
    reader.Update()
    
    itk_image = reader.GetOutput()
    np_image = itk.GetArrayViewFromImage(itk_image)
    
    # Create output array
    output_image = np.zeros_like(np_image)
    
    depth, height, width = np_image.shape
    
    # Process each slice independently
    for z in range(depth):
        # Get the current 2D slice
        current_slice = np_image[z, :, :]
        
        # Process each pixel in the 2D slice
        for y in range(height):
            for x in range(width):
                # Apply adaptive median filter to this pixel
                output_image[z, y, x] = process_pixel_2d(current_slice, y, x, max_radius)
    
    # Convert back to ITK image
    output_itk = itk.GetImageFromArray(output_image)
    output_itk.SetSpacing(itk_image.GetSpacing())
    output_itk.SetOrigin(itk_image.GetOrigin())
    output_itk.SetDirection(itk_image.GetDirection())
    
    # # Convert to 8-bit for output
    # rescaler = itk.RescaleIntensityImageFilter[image_type, itk.Image[itk.UC, 3]].New()
    # rescaler.SetInput(output_itk)
    # rescaler.SetOutputMinimum(0)
    # rescaler.SetOutputMaximum(255)
    # rescaler.Update()
    
    # itk.imwrite(rescaler.GetOutput(), str(output_file))

    itk.imwrite(output_itk, str(output_file))
    
    return itk.size(itk_image)

def process_pixel_2d(image_slice: np.ndarray, y: int, x: int, max_radius: int) -> float:
    height, width = image_slice.shape
    pixel_value = image_slice[y, x]
    
    # Try windows of increasing size
    for radius in range(1, max_radius + 1):
        # Calculate window boundaries with edge handling
        y_min, y_max = max(0, y - radius), min(height, y + radius + 1)
        x_min, x_max = max(0, x - radius), min(width, x + radius + 1)
        
        # Extract window
        window = image_slice[y_min:y_max, x_min:x_max]
        window_flat = window.flatten()
        
        # Calculate statistics
        v_min = np.min(window_flat)
        v_med = np.median(window_flat)
        v_max = np.max(window_flat)
        
        # Level A: Test if median is between min and max (not an impulse)
        if v_min < v_med < v_max:
            # Level B: Test if center pixel is between min and max (not noise)
            if v_min < pixel_value < v_max:
                return pixel_value  # Not noise, keep original
            else:
                return v_med  # Noise detected, replace with median
        
        # If test fails and we've reached max radius
        if radius == max_radius:
            return v_med  # Use median as fallback
    
    # Shouldn't reach here, but just in case
    return pixel_value