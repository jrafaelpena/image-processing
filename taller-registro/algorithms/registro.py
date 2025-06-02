from pathlib import Path
from typing import Union, Optional, Tuple
import itk
import time
import os
import numpy as np

def rigid_registration(
    fixed_image,
    moving_image,
    number_of_samples: int,
    output_file: str = None):
    
    """Rigid registration using ITK v4 framework"""
    dimension = 3
    pixel_type = itk.F
    image_type = itk.Image[pixel_type, dimension]
    
    # Registration components - keeping your original choices but in v4
    transform = itk.VersorRigid3DTransform[itk.D].New()
    optimizer = itk.RegularStepGradientDescentOptimizerv4.New()  # v4 version
    metric = itk.MattesMutualInformationImageToImageMetricv4[image_type, image_type].New()  # v4 version
    registration = itk.ImageRegistrationMethodv4[image_type, image_type].New()  # v4 version
    
    # Set up registration - v4 style
    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    
    # Metric parameters - keeping your original values
    metric.SetNumberOfHistogramBins(50)
    
    # v4 uses sampling strategy - use integer value directly
    registration.SetMetricSamplingStrategy(0)  # 0 = RANDOM sampling
    registration.SetMetricSamplingPercentage(number_of_samples / (fixed_image.GetLargestPossibleRegion().GetSize()[0] * 
                                                                 fixed_image.GetLargestPossibleRegion().GetSize()[1] * 
                                                                 fixed_image.GetLargestPossibleRegion().GetSize()[2]))
    
    # Set random seed for reproducibility
    registration.MetricSamplingReinitializeSeed(76926294)
    
    # Optimizer parameters - keeping your original values
    optimizer.SetLearningRate(0.2)  # v4 uses LearningRate instead of MaximumStepLength
    optimizer.SetMinimumStepLength(0.001)
    optimizer.SetNumberOfIterations(200)
    optimizer.SetRelaxationFactor(0.9)
    
    # Initialize transform - v4 approach
    initializer = itk.CenteredTransformInitializer[
        type(transform), image_type, image_type
    ].New()
    initializer.SetTransform(transform)
    initializer.SetFixedImage(fixed_image)
    initializer.SetMovingImage(moving_image)
    initializer.MomentsOn()
    initializer.InitializeTransform()
    
    # Set the initialized transform and disable multi-resolution
    registration.SetInitialTransform(transform)
    registration.SetNumberOfLevels(1)  # Single resolution level only
    registration.SetShrinkFactorsPerLevel([1])  # No downsampling
    registration.SetSmoothingSigmasPerLevel([0])  # No smoothing
    
    # Add observer - keeping your original monitoring
    def iteration_update():
        iteration = optimizer.GetCurrentIteration()
        metric_value = optimizer.GetValue()
        print(f"  Rigid iteration {iteration}: {metric_value}")
    
    command = itk.PyCommand.New()
    command.SetCommandCallable(iteration_update)
    optimizer.AddObserver(itk.IterationEvent(), command)
    
    # Execute registration
    registration.Update()
    
    # Get final transform - v4 approach
    final_transform = registration.GetTransform()
    
    print(f"  Rigid final metric: {optimizer.GetValue()}")

    # If output file is specified, create and save the registered image
    if output_file is not None:
        print(f"  Creating registered image...")
        
        # Apply the transform to create the registered image
        resampler = itk.ResampleImageFilter.New(Input=moving_image)
        resampler.SetTransform(final_transform)
        resampler.SetReferenceImage(fixed_image)
        resampler.SetUseReferenceImage(True)
        resampler.SetDefaultPixelValue(0)
        resampler.Update()
        registered_image = resampler.GetOutput()
        itk.imwrite(registered_image, output_file)
        
    return final_transform

def similarity_registration(fixed_image, moving_image, initial_transform, number_of_samples: int):
    """Phase 2: Similarity (Rigid + Scale) registration using ITK v4 framework"""
    dimension = 3  # Fixed to 3D as requested
    pixel_type = itk.F
    image_type = itk.Image[pixel_type, dimension]
    
    # Use Similarity transform (rigid + uniform scaling) - fixed to 3D
    transform = itk.Similarity3DTransform[itk.D].New()
    optimizer = itk.RegularStepGradientDescentOptimizerv4.New()  # v4 version
    metric = itk.MattesMutualInformationImageToImageMetricv4[image_type, image_type].New()  # v4 version
    registration = itk.ImageRegistrationMethodv4[image_type, image_type].New()  # v4 version
    
    # Set up registration - v4 style
    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    
    # Initialize with rigid transform result
    transform.SetCenter(initial_transform.GetCenter())
    transform.SetTranslation(initial_transform.GetTranslation())
    transform.SetRotation(initial_transform.GetVersor())  # 3D only
    transform.SetScale(1.0)  # Start with no scaling
    
    # Metric parameters - v4 style
    metric.SetNumberOfHistogramBins(50)
    
    # v4 sampling approach
    registration.SetMetricSamplingStrategy(0)  # 0 = RANDOM sampling
    registration.SetMetricSamplingPercentage(number_of_samples / (fixed_image.GetLargestPossibleRegion().GetSize()[0] * 
                                                                 fixed_image.GetLargestPossibleRegion().GetSize()[1] * 
                                                                 fixed_image.GetLargestPossibleRegion().GetSize()[2]))
    
    # Set random seed for reproducibility
    registration.MetricSamplingReinitializeSeed(76926294)
    
    # Optimizer parameters - v4 style
    optimizer.SetLearningRate(0.1)  # v4 uses LearningRate instead of MaximumStepLength
    optimizer.SetMinimumStepLength(0.001)
    optimizer.SetNumberOfIterations(200)
    optimizer.SetRelaxationFactor(0.9)
    
    # Set the initialized transform and configure single-level registration
    registration.SetInitialTransform(transform)
    registration.SetNumberOfLevels(1)  # Single resolution level only
    registration.SetShrinkFactorsPerLevel([1])  # No downsampling
    registration.SetSmoothingSigmasPerLevel([0])  # No smoothing
    
    # Add observer
    def iteration_update():
        iteration = optimizer.GetCurrentIteration()
        metric_value = optimizer.GetValue()
        print(f"  Similarity iteration {iteration}: {metric_value}")
    
    command = itk.PyCommand.New()
    command.SetCommandCallable(iteration_update)
    optimizer.AddObserver(itk.IterationEvent(), command)
    
    # Execute registration
    registration.Update()
    
    # Get final transform - v4 approach
    final_transform = registration.GetTransform()
    
    print(f"  Similarity final metric: {optimizer.GetValue()}")
    return final_transform

def main():
    base_path = Path(os.getcwd()).parent
    inputs_path = base_path / "inputs"
    outputs_path = base_path / "outputs"
    
    # Input files
    ct_1_path = inputs_path / "CT_1.nrrd"
    ct_2_path = inputs_path / "CT_2.nrrd"
    pet_1_path = inputs_path / "PET_1.nrrd"
    pet_2_path = inputs_path / "PET_2.nrrd"

    # Output files
    step_1_rigid = outputs_path / "step_1_rigid.nrrd"

    # Readers parameters
    dimension = 3
    pixel_type = itk.F
    image_type = itk.Image[pixel_type, dimension]

    # Read input images
    print("Reading CT images...")
    fixed_reader = itk.ImageFileReader[image_type].New()
    moving_reader = itk.ImageFileReader[image_type].New()
    fixed_reader.SetFileName(str(ct_1_path))
    moving_reader.SetFileName(str(ct_2_path))

    # Update readers to get image information
    fixed_reader.Update()
    moving_reader.Update()
    
    fixed_image = fixed_reader.GetOutput()
    moving_image = moving_reader.GetOutput()

    number_samples = 100_000
    rigid_transform = rigid_registration(fixed_image, moving_image, number_samples, str(step_1_rigid))

if __name__ == "__main__":
    main()