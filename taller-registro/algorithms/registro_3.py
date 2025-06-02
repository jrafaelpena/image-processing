from pathlib import Path
from typing import Union, Optional, Tuple
import itk
import time
import os
import numpy as np
from utils import save_transform, load_transform, get_rigid_transform_as_versor

def rigid_registration(
    fixed_image,
    moving_image,
    number_of_samples: int,
    output_file: str = None):
    
    """Rigid registration using ITK v4 framework"""
    dimension = 3
    pixel_type = itk.F
    image_type = itk.Image[pixel_type, dimension]
    
    # Registration components
    transform = itk.VersorRigid3DTransform[itk.D].New()
    optimizer = itk.RegularStepGradientDescentOptimizerv4.New()  # v4 version
    metric = itk.MattesMutualInformationImageToImageMetricv4[image_type, image_type].New()  # v4 version
    registration = itk.ImageRegistrationMethodv4[image_type, image_type].New()  # v4 version
    
    # Set up registration
    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    
    # Metric parameters
    metric.SetNumberOfHistogramBins(50)
    
    # v4 uses sampling strategy - use integer value directly
    registration.SetMetricSamplingStrategy(0)  # 0 = RANDOM sampling
    registration.SetMetricSamplingPercentage(number_of_samples / (fixed_image.GetLargestPossibleRegion().GetSize()[0] * 
                                                                 fixed_image.GetLargestPossibleRegion().GetSize()[1] * 
                                                                 fixed_image.GetLargestPossibleRegion().GetSize()[2]))
    
    # Set random seed for reproducibility
    registration.MetricSamplingReinitializeSeed(76926294)
    
    # Optimizer parameters
    optimizer.SetLearningRate(0.2)
    optimizer.SetMinimumStepLength(0.001)
    optimizer.SetNumberOfIterations(1500)
    optimizer.SetRelaxationFactor(0.5)
    
    # Initialize transform
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
    
    # Add observer
    def iteration_update():
        iteration = optimizer.GetCurrentIteration()
        metric_value = optimizer.GetValue()
        print(f"  Rigid iteration {iteration}: {metric_value}")
    
    command = itk.PyCommand.New()
    command.SetCommandCallable(iteration_update)
    optimizer.AddObserver(itk.IterationEvent(), command)
    
    # Execute registration
    registration.Update()
    
    # Get final transform
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

def similarity_registration(fixed_image, moving_image, initial_transform, number_of_samples: int, output_file: str = None):
    """Phase 2: Similarity (Rigid + Scale) registration using ITK v4 framework"""
    dimension = 3  # Fixed to 3D as requested
    pixel_type = itk.F
    image_type = itk.Image[pixel_type, dimension]
    
    # Use Similarity transform (rigid + uniform scaling) - fixed to 3D
    transform = itk.Similarity3DTransform[itk.D].New()
    optimizer = itk.RegularStepGradientDescentOptimizerv4.New()  # v4 version
    metric = itk.MattesMutualInformationImageToImageMetricv4[image_type, image_type].New()  # v4 version
    registration = itk.ImageRegistrationMethodv4[image_type, image_type].New()  # v4 version
    
    # Set up registration
    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    
    # Initialize with rigid transform result - ensure we have a proper versor transform
    versor_transform = get_rigid_transform_as_versor(initial_transform)
    
    transform.SetCenter(versor_transform.GetCenter())
    transform.SetTranslation(versor_transform.GetTranslation())
    transform.SetRotation(versor_transform.GetVersor())  # 3D only
    transform.SetScale(1.0)  # Start with no scaling
    
    # Metric parameters
    metric.SetNumberOfHistogramBins(50)
    
    # v4 sampling approach
    registration.SetMetricSamplingStrategy(0)  # 0 = RANDOM sampling
    registration.SetMetricSamplingPercentage(number_of_samples / (fixed_image.GetLargestPossibleRegion().GetSize()[0] * 
                                                                 fixed_image.GetLargestPossibleRegion().GetSize()[1] * 
                                                                 fixed_image.GetLargestPossibleRegion().GetSize()[2]))
    
    # Set random seed for reproducibility
    registration.MetricSamplingReinitializeSeed(76926294)
    
    # Optimizer parameters - v4 style
    optimizer.SetLearningRate(0.1)
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
    
    # If output file is specified, create and save the registered image
    if output_file is not None:
        print(f"  Creating similarity registered image...")
        
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

def get_similarity_transform_as_similarity(transform):
    """Convert any similarity transform to Similarity3DTransform for compatibility"""
    similarity_transform = itk.Similarity3DTransform[itk.D].New()
    similarity_transform.SetParameters(transform.GetParameters())
    similarity_transform.SetFixedParameters(transform.GetFixedParameters())
    return similarity_transform

def affine_registration(fixed_image, moving_image, initial_transform, number_of_samples: int, output_file: str = None):
    """Phase 3: Affine registration using ITK v4 framework"""
    dimension = 3  # Fixed to 3D
    pixel_type = itk.F
    image_type = itk.Image[pixel_type, dimension]
    
    # Use Affine transform - fixed to 3D
    transform = itk.AffineTransform[itk.D, dimension].New()
    optimizer = itk.RegularStepGradientDescentOptimizerv4.New()  # v4 version
    metric = itk.MattesMutualInformationImageToImageMetricv4[image_type, image_type].New()  # v4 version
    registration = itk.ImageRegistrationMethodv4[image_type, image_type].New()  # v4 version
    
    # Set up registration
    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    
    # Initialize with similarity transform result - ensure we have a proper similarity transform
    similarity_transform = get_similarity_transform_as_similarity(initial_transform)
    
    # Initialize affine transform with similarity transform parameters
    transform.SetCenter(similarity_transform.GetCenter())
    transform.SetTranslation(similarity_transform.GetTranslation())
    
    # Convert similarity transform matrix to affine (3D only)
    matrix = transform.GetMatrix()
    rotation_matrix = similarity_transform.GetMatrix()
    scale = similarity_transform.GetScale()
    
    # Apply rotation and scale to create initial affine matrix
    # Convert matrices to numpy arrays for easier manipulation
    rotation_array = itk.array_from_matrix(rotation_matrix)
    matrix_array = itk.array_from_matrix(matrix)
    
    # Apply rotation and scale
    for i in range(dimension):
        for j in range(dimension):
            matrix_array[i, j] = rotation_array[i, j] * scale
    
    # Convert back to ITK matrix
    matrix = itk.matrix_from_array(matrix_array)
    
    transform.SetMatrix(matrix)
    
    # Metric parameters
    metric.SetNumberOfHistogramBins(50)
    
    # v4 sampling approach
    registration.SetMetricSamplingStrategy(0)  # 0 = RANDOM sampling
    registration.SetMetricSamplingPercentage(number_of_samples / (fixed_image.GetLargestPossibleRegion().GetSize()[0] * 
                                                                 fixed_image.GetLargestPossibleRegion().GetSize()[1] * 
                                                                 fixed_image.GetLargestPossibleRegion().GetSize()[2]))
    
    # Set random seed for reproducibility
    registration.MetricSamplingReinitializeSeed(76926294)
    
    # Optimizer parameters - v4 style
    optimizer.SetLearningRate(0.05)
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
        print(f"  Affine iteration {iteration}: {metric_value}")
    
    command = itk.PyCommand.New()
    command.SetCommandCallable(iteration_update)
    optimizer.AddObserver(itk.IterationEvent(), command)
    
    # Execute registration
    registration.Update()
    
    # Get final transform - v4 approach
    final_transform = registration.GetTransform()
    
    print(f"  Affine final metric: {optimizer.GetValue()}")
    
    # If output file is specified, create and save the registered image
    if output_file is not None:
        print(f"  Creating affine registered image...")
        
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

def main():
    base_path = Path(os.getcwd()).parent
    inputs_path = base_path / "inputs"
    outputs_path = base_path / "outputs"
    
    # Create outputs directory if it doesn't exist
    outputs_path.mkdir(exist_ok=True)
    
    # Input files
    ct_1_path = inputs_path / "CT_1.nrrd"
    ct_2_path = inputs_path / "CT_2.nrrd"
    pet_1_path = inputs_path / "PET_1.nrrd"
    pet_2_path = inputs_path / "PET_2.nrrd"

    # Output files
    step_1_rigid = outputs_path / "step_1_rigid.nrrd"
    step_2_similarity = outputs_path / "step_2_similarity.nrrd"
    step_3_affine = outputs_path / "step_3_affine.nrrd"
    
    # Transform files for saving/loading
    rigid_transform_file = outputs_path / "rigid_transform.txt"
    similarity_transform_file = outputs_path / "similarity_transform.txt"
    affine_transform_file = outputs_path / "affine_transform.txt"

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
    
    # STEP 1: Rigid Registration (with loading/saving capability)
    print("\n=== STEP 1: RIGID REGISTRATION ===")
    
    # Try to load existing rigid transform
    rigid_transform = load_transform(rigid_transform_file, itk.VersorRigid3DTransform[itk.D])
    
    if rigid_transform is None:
        print("No existing rigid transform found. Running rigid registration...")
        start_time = time.time()
        rigid_transform = rigid_registration(fixed_image, moving_image, number_samples, str(step_1_rigid))
        end_time = time.time()
        print(f"Rigid registration completed in {end_time - start_time:.2f} seconds")
        
        # Save the rigid transform for future use
        save_transform(rigid_transform, rigid_transform_file)
    else:
        print("Loaded existing rigid transform. Skipping rigid registration.")
        
        # Still create the registered image if it doesn't exist
        if not step_1_rigid.exists():
            print("Creating registered image from loaded transform...")
            resampler = itk.ResampleImageFilter.New(Input=moving_image)
            resampler.SetTransform(rigid_transform)
            resampler.SetReferenceImage(fixed_image)
            resampler.SetUseReferenceImage(True)
            resampler.SetDefaultPixelValue(0)
            resampler.Update()
            registered_image = resampler.GetOutput()
            itk.imwrite(registered_image, str(step_1_rigid))
    
    # STEP 2: Similarity Registration (Rigid + Scale)
    print("\n=== STEP 2: SIMILARITY REGISTRATION ===")
    
    # Try to load existing similarity transform
    similarity_transform = load_transform(similarity_transform_file, itk.Similarity3DTransform[itk.D])
    
    if similarity_transform is None:
        print("No existing similarity transform found. Running similarity registration...")
        start_time = time.time()
        similarity_transform = similarity_registration(
            fixed_image, moving_image, rigid_transform, number_samples, str(step_2_similarity)
        )
        end_time = time.time()
        print(f"Similarity registration completed in {end_time - start_time:.2f} seconds")
        
        # Save the similarity transform for future use
        save_transform(similarity_transform, similarity_transform_file)
    else:
        print("Loaded existing similarity transform. Skipping similarity registration.")
        
        # Still create the registered image if it doesn't exist
        if not step_2_similarity.exists():
            print("Creating similarity registered image from loaded transform...")
            resampler = itk.ResampleImageFilter.New(Input=moving_image)
            resampler.SetTransform(similarity_transform)
            resampler.SetReferenceImage(fixed_image)
            resampler.SetUseReferenceImage(True)
            resampler.SetDefaultPixelValue(0)
            resampler.Update()
            registered_image = resampler.GetOutput()
            itk.imwrite(registered_image, str(step_2_similarity))
    
    # STEP 3: Affine Registration
    print("\n=== STEP 3: AFFINE REGISTRATION ===")
    
    # Try to load existing affine transform
    affine_transform = load_transform(affine_transform_file, itk.AffineTransform[itk.D, 3])
    
    if affine_transform is None:
        print("No existing affine transform found. Running affine registration...")
        start_time = time.time()
        affine_transform = affine_registration(
            fixed_image, moving_image, similarity_transform, number_samples, str(step_3_affine)
        )
        end_time = time.time()
        print(f"Affine registration completed in {end_time - start_time:.2f} seconds")
        
        # Save the affine transform for future use
        save_transform(affine_transform, affine_transform_file)
    else:
        print("Loaded existing affine transform. Skipping affine registration.")
        
        # Still create the registered image if it doesn't exist
        if not step_3_affine.exists():
            print("Creating affine registered image from loaded transform...")
            resampler = itk.ResampleImageFilter.New(Input=moving_image)
            resampler.SetTransform(affine_transform)
            resampler.SetReferenceImage(fixed_image)
            resampler.SetUseReferenceImage(True)
            resampler.SetDefaultPixelValue(0)
            resampler.Update()
            registered_image = resampler.GetOutput()
            itk.imwrite(registered_image, str(step_3_affine))
    
    print(f"\n=== REGISTRATION COMPLETE ===")
    print(f"Rigid registered image: {step_1_rigid}")
    print(f"Similarity registered image: {step_2_similarity}")
    print(f"Affine registered image: {step_3_affine}")
    print(f"Rigid transform saved: {rigid_transform_file}")
    print(f"Similarity transform saved: {similarity_transform_file}")
    print(f"Affine transform saved: {affine_transform_file}")
    
    # Print final transform parameters for reference
    print(f"\nFinal Rigid Transform Parameters:")
    print(f"  Translation: {rigid_transform.GetTranslation()}")
    print(f"  Center: {rigid_transform.GetCenter()}")
    
    print(f"\nFinal Similarity Transform Parameters:")
    print(f"  Translation: {similarity_transform.GetTranslation()}")
    print(f"  Center: {similarity_transform.GetCenter()}")
    print(f"  Scale: {similarity_transform.GetScale()}")
    
    print(f"\nFinal Affine Transform Parameters:")
    # For affine transform, we need to access parameters differently
    # since registration.GetTransform() returns a generic transform
    try:
        # Try to cast to specific affine transform type
        specific_affine = itk.AffineTransform[itk.D, 3].New()
        specific_affine.SetParameters(affine_transform.GetParameters())
        specific_affine.SetFixedParameters(affine_transform.GetFixedParameters())
        
        print(f"  Translation: {specific_affine.GetTranslation()}")
        print(f"  Center: {specific_affine.GetCenter()}")
        print(f"  Matrix: {specific_affine.GetMatrix()}")
    except:
        # Fallback to raw parameters if casting fails
        print(f"  Parameters: {affine_transform.GetParameters()}")
        print(f"  Fixed Parameters: {affine_transform.GetFixedParameters()}")
        
        # Manual extraction of translation (last 3 parameters for 3D affine)
        params = affine_transform.GetParameters()
        if len(params) >= 12:  # 3x3 matrix + 3 translation = 12 parameters
            translation = [params[9], params[10], params[11]]  # Last 3 parameters
            print(f"  Translation (extracted): {translation}")
        
        # Manual extraction of center (from fixed parameters)
        fixed_params = affine_transform.GetFixedParameters()
        if len(fixed_params) >= 3:
            center = [fixed_params[0], fixed_params[1], fixed_params[2]]
            print(f"  Center (extracted): {center}")

if __name__ == "__main__":
    main()