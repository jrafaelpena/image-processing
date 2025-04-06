from pathlib import Path
from typing import Union
import itk
import os
import time

def kmeans_clustering_filter(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    initial_means: list,
    dimensions: int = 3
) -> dict:
    """
    Apply K-means clustering to an image
    
    Parameters:
    input_file: Input image file
    output_file: Output image file path
    initial_means: List of initial class means for K-means
    dimensions: Image dimensions (default: 3)
    
    Returns:
    Dictionary with clustering results
    """
    
    pixel_type = itk.US
    image_type = itk.Image[pixel_type, dimensions]
    
    # Load the input image
    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(str(input_file))
    reader.Update()
    image = reader.GetOutput()
    
    # Get original image size
    size = itk.size(image)
    
    # Apply K-means clustering
    kmeans_filter = itk.ScalarImageKmeansImageFilter[image_type, image_type].New()
    kmeans_filter.SetInput(image)
    kmeans_filter.SetUseNonContiguousLabels(True)
    
    # Add classes with initial means
    for mean in initial_means:
        kmeans_filter.AddClassWithInitialMean(mean)
    
    kmeans_filter.Update()
    
    # Get final estimated means
    estimated_means = kmeans_filter.GetFinalMeans()
    
    # Relabel components to ensure they are consecutive
    output_image_type = type(kmeans_filter.GetOutput())
    relabel_filter = itk.RelabelComponentImageFilter[output_image_type, output_image_type].New()
    relabel_filter.SetInput(kmeans_filter.GetOutput())
    
    # Rescale intensity to [0, 255]
    rescale_filter = itk.RescaleIntensityImageFilter[image_type, image_type].New()
    rescale_filter.SetInput(relabel_filter.GetOutput())
    rescale_filter.SetOutputMinimum(0)
    rescale_filter.SetOutputMaximum(255)
    rescale_filter.Update()
    
    # Get sizes of each class
    sizes = relabel_filter.GetSizeOfObjectsInPixels()
    
    # Save the output image
    itk.imwrite(rescale_filter.GetOutput(), str(output_file))
    
    # Prepare results to return
    results = {
        'estimated_means': [float(estimated_means[i]) for i in range(len(initial_means))],
        'class_sizes': [int(sizes[i]) for i in range(len(sizes))],
        'image_size': tuple(size)
    }
    
    return results


if __name__ == "__main__":
    BASE_PATH = Path(os.getcwd()).parent
    names = ["MRBrainTumor", "MRBreastCancer", "MRLiverTumor"]
    image_name = names[2]
    extension = "nii.gz"
    input_image = BASE_PATH / f"inputs/{image_name}.{extension}"
    output_image = BASE_PATH / f"outputs/{image_name}_KMeans.{extension}"
    
    # Start timer
    start_time = time.time()
    
    initial_means = [20, 40, 78, 255]
    
    # Run K-means clustering
    results = kmeans_clustering_filter(
        input_file=input_image,
        output_file=output_image,
        initial_means=initial_means
    )
    
    # Calculate total time
    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
    
    # Print results
    print("\nClustering Results:")
    print("-----------------")
    print("Initial means:")
    for i, mean in enumerate(initial_means):
        print(f"  Cluster[{i}] initial mean: {mean:.2f}")
    print("Estimated means:")
    for i, mean in enumerate(results['estimated_means']):
        print(f"  Cluster[{i}] estimated mean: {mean:.2f}")
    
    print("\nNumber of pixels per class:")
    for i, size in enumerate(results['class_sizes']):
        print(f"  Class {i} = {size} pixels")