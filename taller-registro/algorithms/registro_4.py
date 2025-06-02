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
    
    # Optimizer parameters
    optimizer.SetLearningRate(0.1)
    optimizer.SetMinimumStepLength(0.001)
    optimizer.SetNumberOfIterations(1500)
    optimizer.SetRelaxationFactor(0.5)
    
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
    
    # Get final transform
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
    
    # Optimizer parameters
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
    
    # Get final transform
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

def bspline_registration(fixed_image, moving_image, initial_transform, number_of_samples: int, 
                         grid_size: tuple = (5, 5, 3), output_file: str = None):
    """B-spline registration using RegularStepGradientDescentOptimizerv4 with improved parameters"""
    dimension = 3
    pixel_type = itk.F
    image_type = itk.Image[pixel_type, dimension]
    
    # Create B-spline transform with coarser grid
    transform = itk.BSplineTransform[itk.D, dimension, 3].New()
    
    # Set up the B-spline grid
    fixed_region = fixed_image.GetLargestPossibleRegion()
    fixed_spacing = fixed_image.GetSpacing()
    fixed_origin = fixed_image.GetOrigin()
    fixed_size = fixed_region.GetSize()
    
    # Calculate B-spline grid parameters - FIXED CALCULATION
    # The transform domain should cover the entire image
    transform.SetTransformDomainOrigin(fixed_origin)
    transform.SetTransformDomainDirection(fixed_image.GetDirection())
    
    # Physical dimensions should be the actual image size
    physical_dimensions = [
        fixed_spacing[i] * (fixed_size[i] - 1) for i in range(dimension)
    ]
    transform.SetTransformDomainPhysicalDimensions(physical_dimensions)
    
    # Mesh size is grid_size - spline_order (3 for cubic B-splines)
    mesh_size = [max(1, grid_size[i] - 3) for i in range(dimension)]
    transform.SetTransformDomainMeshSize(mesh_size)
    
    # Initialize B-spline parameters to zero
    num_parameters = transform.GetNumberOfParameters()
    parameters = itk.OptimizerParameters[itk.D](num_parameters)
    parameters.Fill(0.0)
    transform.SetParameters(parameters)
    
    print(f"  B-spline grid size: {grid_size}")
    print(f"  B-spline mesh size: {mesh_size}")
    print(f"  Number of B-spline parameters: {num_parameters}")
    
    # Create composite transform - FIXED APPROACH
    composite_transform = itk.CompositeTransform[itk.D, dimension].New()
    
    # Add the initial affine transform (this should be fixed/non-optimizable)
    try:
        # Create a copy of the affine transform
        affine_transform = itk.AffineTransform[itk.D, dimension].New()
        affine_transform.SetParameters(initial_transform.GetParameters())
        affine_transform.SetFixedParameters(initial_transform.GetFixedParameters())
        composite_transform.AddTransform(affine_transform)
        print("  Added affine transform to composite")
    except Exception as e:
        print(f"  Warning: Could not cast initial transform to affine: {e}")
        # Use identity if casting fails
        identity_affine = itk.AffineTransform[itk.D, dimension].New()
        identity_affine.SetIdentity()
        composite_transform.AddTransform(identity_affine)
        print("  Added identity affine transform to composite")
    
    # Add the B-spline transform (this will be optimized)
    composite_transform.AddTransform(transform)
    
    # CRITICAL: Only optimize the B-spline component (most recent)
    composite_transform.SetOnlyMostRecentTransformToOptimizeOn()
    
    # Verify the setup
    print(f"  Composite transform has {composite_transform.GetNumberOfTransforms()} transforms")
    print(f"  Total parameters: {composite_transform.GetNumberOfParameters()}")
    print(f"  Parameters to optimize: {composite_transform.GetNumberOfParameters()}")
    
    # Use RegularStepGradientDescentOptimizerv4 with B-spline specific parameters
    optimizer = itk.RegularStepGradientDescentOptimizerv4.New()
    
    # B-spline specific parameters - much more aggressive than before
    optimizer.SetLearningRate(0.5)  # Much larger learning rate for B-spline
    optimizer.SetMinimumStepLength(0.001)  # Smaller minimum step
    optimizer.SetNumberOfIterations(500)  # More iterations for B-spline
    optimizer.SetRelaxationFactor(0.8)  # Slightly more conservative
    optimizer.SetGradientMagnitudeTolerance(1e-8)  # Tighter convergence
    
    print(f"  Optimizer: RegularStepGradientDescent (B-spline tuned)")
    
    # Set up metric with better parameters for B-spline
    metric = itk.MattesMutualInformationImageToImageMetricv4[image_type, image_type].New()
    metric.SetNumberOfHistogramBins(50)
    
    # IMPORTANT: Use virtual domain from fixed image for B-spline
    metric.SetVirtualDomainFromImage(fixed_image)
    
    # Set up registration
    registration = itk.ImageRegistrationMethodv4[image_type, image_type].New()
    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetInitialTransform(composite_transform)
    
    # Sampling strategy - use more samples for B-spline
    registration.SetMetricSamplingStrategy(0)  # RANDOM sampling
    total_voxels = fixed_size[0] * fixed_size[1] * fixed_size[2]
    sampling_percentage = min(0.05, max(0.01, number_of_samples / total_voxels))  # 1-5%
    registration.SetMetricSamplingPercentage(sampling_percentage)
    registration.MetricSamplingReinitializeSeed(76926294)
    
    print(f"  Using sampling percentage: {sampling_percentage:.4f}")
    
    # Use multi-resolution for better convergence
    registration.SetNumberOfLevels(3)  # More levels for B-spline
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    
    # Progress tracking
    iteration_count = [0]
    last_metric_value = [float('inf')]
    
    def iteration_update():
        try:
            iteration = optimizer.GetCurrentIteration()
            metric_value = optimizer.GetValue()
            iteration_count[0] = iteration
            last_metric_value[0] = metric_value
            
            # Print progress every 10 iterations
            if iteration % 10 == 0 or iteration < 10:
                print(f"  B-spline iteration {iteration}: {metric_value:.6f}")
                    
        except Exception as e:
            print(f"  Iteration {iteration_count[0]}: Error - {e}")
    
    command = itk.PyCommand.New()
    command.SetCommandCallable(iteration_update)
    optimizer.AddObserver(itk.IterationEvent(), command)
    
    # Execute registration
    print(f"  Starting B-spline registration...")
    
    try:
        registration.Update()
        print(f"  B-spline registration completed successfully!")
    except Exception as e:
        print(f"  Registration failed: {e}")
        print(f"  Using current state of transform")
    
    # Get final transform
    final_composite_transform = registration.GetTransform()
    
    try:
        final_metric = optimizer.GetValue()
        print(f"  B-spline final metric: {final_metric:.6f}")
        print(f"  Total iterations completed: {iteration_count[0]}")
    except:
        print(f"  Final metric unavailable. Last known: {last_metric_value[0]:.6f}")
        print(f"  Last known iteration: {iteration_count[0]}")
    
    # DIAGNOSTIC: Check if B-spline parameters actually changed
    final_bspline = final_composite_transform.GetNthTransform(1)  # B-spline is second transform
    final_params = final_bspline.GetParameters()
    param_array = itk.array_from_vnl_vector(final_params.GetVnlVector())
    
    print(f"  B-spline parameter statistics:")
    print(f"    Min: {param_array.min():.6f}")
    print(f"    Max: {param_array.max():.6f}")
    print(f"    Mean: {param_array.mean():.6f}")
    print(f"    Std: {param_array.std():.6f}")
    print(f"    Non-zero parameters: {np.count_nonzero(param_array)}/{len(param_array)}")
    
    if param_array.std() < 1e-10:
        print(f"  WARNING: B-spline parameters barely changed - registration may not have worked!")
    
    # Create registered image if requested
    if output_file is not None:
        print(f"  Creating B-spline registered image...")
        resampler = itk.ResampleImageFilter.New(Input=moving_image)
        resampler.SetTransform(final_composite_transform)
        resampler.SetReferenceImage(fixed_image)
        resampler.SetUseReferenceImage(True)
        resampler.SetDefaultPixelValue(0)
        resampler.Update()
        registered_image = resampler.GetOutput()
        itk.imwrite(registered_image, output_file)
    
    return final_composite_transform

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
    step_4_bspline = outputs_path / "step_4_bspline.nrrd"
    
    # Transform files for saving/loading
    rigid_transform_file = outputs_path / "rigid_transform.txt"
    similarity_transform_file = outputs_path / "similarity_transform.txt"
    affine_transform_file = outputs_path / "affine_transform.txt"
    bspline_transform_file = outputs_path / "bspline_transform.txt"

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
    
    # STEP 4: B-spline Registration
    print("\n=== STEP 4: B-SPLINE REGISTRATION ===")
    
    # B-spline grid size as specified - reduced for faster computation
    bspline_grid_size = (7, 7, 5)  # Reduced from (11, 11, 7)
    
    # Try to load existing B-spline transform (composite transform)
    bspline_transform = None
    try:
        bspline_transform = load_transform(bspline_transform_file, itk.CompositeTransform[itk.D, 3])
    except:
        print("Could not load composite transform, will run B-spline registration...")
    
    if bspline_transform is None:
        print(f"No existing B-spline transform found. Running B-spline registration with grid size {bspline_grid_size}...")
        start_time = time.time()
        bspline_transform = bspline_registration(
            fixed_image, moving_image, affine_transform, number_samples, 
            bspline_grid_size, str(step_4_bspline)
        )
        end_time = time.time()
        print(f"B-spline registration completed in {end_time - start_time:.2f} seconds")
        
        # Save the B-spline transform for future use
        save_transform(bspline_transform, bspline_transform_file)
    else:
        print("Loaded existing B-spline transform. Skipping B-spline registration.")
        
        # Still create the registered image if it doesn't exist
        if not step_4_bspline.exists():
            print("Creating B-spline registered image from loaded transform...")
            resampler = itk.ResampleImageFilter.New(Input=moving_image)
            resampler.SetTransform(bspline_transform)
            resampler.SetReferenceImage(fixed_image)
            resampler.SetUseReferenceImage(True)
            resampler.SetDefaultPixelValue(0)
            resampler.Update()
            registered_image = resampler.GetOutput()
            itk.imwrite(registered_image, str(step_4_bspline))
    
    print(f"\n=== REGISTRATION COMPLETE ===")
    print(f"Rigid registered image: {step_1_rigid}")
    print(f"Similarity registered image: {step_2_similarity}")
    print(f"Affine registered image: {step_3_affine}")
    print(f"B-spline registered image: {step_4_bspline}")
    print(f"Rigid transform saved: {rigid_transform_file}")
    print(f"Similarity transform saved: {similarity_transform_file}")
    print(f"Affine transform saved: {affine_transform_file}")
    print(f"B-spline transform saved: {bspline_transform_file}")
    
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
    
    print(f"\nFinal B-spline Transform Parameters:")
    try:
        # B-spline transform is part of a composite transform
        print(f"  B-spline grid size used: {bspline_grid_size}")
        print(f"  Number of transforms in composite: {bspline_transform.GetNumberOfTransforms()}")
        print(f"  Total parameters: {bspline_transform.GetNumberOfParameters()}")
        
        # Try to get the B-spline component (should be the last transform in composite)
        if bspline_transform.GetNumberOfTransforms() >= 2:
            bspline_component = bspline_transform.GetNthTransform(1)  # Second transform (index 1)
            print(f"  B-spline component parameters: {bspline_component.GetNumberOfParameters()}")
            
            # Calculate deformation statistics
            bspline_params = bspline_component.GetParameters()
            if len(bspline_params) > 0:
                params_array = itk.array_from_vnl_vector(bspline_params.GetVnlVector())
                print(f"  B-spline deformation range: [{params_array.min():.4f}, {params_array.max():.4f}]")
                print(f"  B-spline deformation std: {params_array.std():.4f}")
        
    except Exception as e:
        print(f"  Could not extract B-spline parameters: {e}")
        print(f"  B-spline grid size used: {bspline_grid_size}")
        try:
            print(f"  Total composite parameters: {bspline_transform.GetNumberOfParameters()}")
        except:
            print("  Could not get parameter count")

if __name__ == "__main__":
    main()