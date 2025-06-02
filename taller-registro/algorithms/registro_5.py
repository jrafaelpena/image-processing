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
    
    dimension = 3
    pixel_type = itk.F
    image_type = itk.Image[pixel_type, dimension]
    
    # Set up registration components
    transform = itk.VersorRigid3DTransform[itk.D].New()
    optimizer = itk.RegularStepGradientDescentOptimizerv4.New()
    metric = itk.MattesMutualInformationImageToImageMetricv4[image_type, image_type].New()
    registration = itk.ImageRegistrationMethodv4[image_type, image_type].New()
    
    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    
    metric.SetNumberOfHistogramBins(50)
    
    registration.SetMetricSamplingStrategy(0)
    registration.SetMetricSamplingPercentage(number_of_samples / (fixed_image.GetLargestPossibleRegion().GetSize()[0] * 
                                                                 fixed_image.GetLargestPossibleRegion().GetSize()[1] * 
                                                                 fixed_image.GetLargestPossibleRegion().GetSize()[2]))
    
    registration.MetricSamplingReinitializeSeed(76926294)
    
    optimizer.SetLearningRate(0.2)
    optimizer.SetMinimumStepLength(0.01)
    optimizer.SetNumberOfIterations(1500)
    optimizer.SetRelaxationFactor(0.5)
    
    # Initialize transform center based on image moments
    initializer = itk.CenteredTransformInitializer[
        type(transform), image_type, image_type
    ].New()
    initializer.SetTransform(transform)
    initializer.SetFixedImage(fixed_image)
    initializer.SetMovingImage(moving_image)
    initializer.MomentsOn()
    initializer.InitializeTransform()
    
    registration.SetInitialTransform(transform)
    registration.SetNumberOfLevels(1)
    registration.SetShrinkFactorsPerLevel([1])
    registration.SetSmoothingSigmasPerLevel([0])
    
    def iteration_update():
        iteration = optimizer.GetCurrentIteration()
        metric_value = optimizer.GetValue()
        print(f"  Rigid iteration {iteration}: {metric_value}")
    
    command = itk.PyCommand.New()
    command.SetCommandCallable(iteration_update)
    optimizer.AddObserver(itk.IterationEvent(), command)
    
    registration.Update()
    
    final_transform = registration.GetTransform()
    
    print(f"  Rigid final metric: {optimizer.GetValue()}")

    if output_file is not None:
        print(f"  Creating registered image...")
        
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
    dimension = 3
    pixel_type = itk.F
    image_type = itk.Image[pixel_type, dimension]
    
    transform = itk.Similarity3DTransform[itk.D].New()
    optimizer = itk.RegularStepGradientDescentOptimizerv4.New()
    metric = itk.MattesMutualInformationImageToImageMetricv4[image_type, image_type].New()
    registration = itk.ImageRegistrationMethodv4[image_type, image_type].New()
    
    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    
    versor_transform = get_rigid_transform_as_versor(initial_transform)
    
    transform.SetCenter(versor_transform.GetCenter())
    transform.SetTranslation(versor_transform.GetTranslation())
    transform.SetRotation(versor_transform.GetVersor())
    transform.SetScale(1.0)
    
    metric.SetNumberOfHistogramBins(50)
    
    registration.SetMetricSamplingStrategy(0)
    registration.SetMetricSamplingPercentage(number_of_samples / (fixed_image.GetLargestPossibleRegion().GetSize()[0] * 
                                                                 fixed_image.GetLargestPossibleRegion().GetSize()[1] * 
                                                                 fixed_image.GetLargestPossibleRegion().GetSize()[2]))
    
    registration.MetricSamplingReinitializeSeed(76926294)
    
    optimizer.SetLearningRate(0.2)
    optimizer.SetMinimumStepLength(0.01)
    optimizer.SetNumberOfIterations(60)
    optimizer.SetRelaxationFactor(0.5)
    
    registration.SetInitialTransform(transform)
    registration.SetNumberOfLevels(1)
    registration.SetShrinkFactorsPerLevel([1])
    registration.SetSmoothingSigmasPerLevel([0])
    
    def iteration_update():
        iteration = optimizer.GetCurrentIteration()
        metric_value = optimizer.GetValue()
        print(f"  Similarity iteration {iteration}: {metric_value}")
    
    command = itk.PyCommand.New()
    command.SetCommandCallable(iteration_update)
    optimizer.AddObserver(itk.IterationEvent(), command)
    
    registration.Update()
    
    final_transform = registration.GetTransform()
    
    print(f"  Similarity final metric: {optimizer.GetValue()}")
    
    if output_file is not None:
        print(f"  Creating similarity registered image...")
        
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
    similarity_transform = itk.Similarity3DTransform[itk.D].New()
    similarity_transform.SetParameters(transform.GetParameters())
    similarity_transform.SetFixedParameters(transform.GetFixedParameters())
    return similarity_transform

def affine_registration(fixed_image, moving_image, initial_transform, number_of_samples: int, output_file: str = None):
    dimension = 3
    pixel_type = itk.F
    image_type = itk.Image[pixel_type, dimension]
    
    transform = itk.AffineTransform[itk.D, dimension].New()
    optimizer = itk.RegularStepGradientDescentOptimizerv4.New()
    metric = itk.MattesMutualInformationImageToImageMetricv4[image_type, image_type].New()
    registration = itk.ImageRegistrationMethodv4[image_type, image_type].New()
    
    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    
    similarity_transform = get_similarity_transform_as_similarity(initial_transform)
    
    transform.SetCenter(similarity_transform.GetCenter())
    transform.SetTranslation(similarity_transform.GetTranslation())
    
    # Convert similarity matrix to affine
    matrix = transform.GetMatrix()
    rotation_matrix = similarity_transform.GetMatrix()
    scale = similarity_transform.GetScale()
    
    rotation_array = itk.array_from_matrix(rotation_matrix)
    matrix_array = itk.array_from_matrix(matrix)
    
    for i in range(dimension):
        for j in range(dimension):
            matrix_array[i, j] = rotation_array[i, j] * scale
    
    matrix = itk.matrix_from_array(matrix_array)
    transform.SetMatrix(matrix)
    
    metric.SetNumberOfHistogramBins(50)
    
    registration.SetMetricSamplingStrategy(0)
    registration.SetMetricSamplingPercentage(number_of_samples / (fixed_image.GetLargestPossibleRegion().GetSize()[0] * 
                                                                 fixed_image.GetLargestPossibleRegion().GetSize()[1] * 
                                                                 fixed_image.GetLargestPossibleRegion().GetSize()[2]))
    
    registration.MetricSamplingReinitializeSeed(76926294)
    
    optimizer.SetLearningRate(0.1)
    optimizer.SetMinimumStepLength(0.01)
    optimizer.SetNumberOfIterations(60)
    optimizer.SetRelaxationFactor(0.5)
    
    registration.SetInitialTransform(transform)
    registration.SetNumberOfLevels(1)
    registration.SetShrinkFactorsPerLevel([1])
    registration.SetSmoothingSigmasPerLevel([0])
    
    def iteration_update():
        iteration = optimizer.GetCurrentIteration()
        metric_value = optimizer.GetValue()
        print(f"  Affine iteration {iteration}: {metric_value}")
    
    command = itk.PyCommand.New()
    command.SetCommandCallable(iteration_update)
    optimizer.AddObserver(itk.IterationEvent(), command)
    
    registration.Update()
    
    final_transform = registration.GetTransform()
    
    print(f"  Affine final metric: {optimizer.GetValue()}")
    
    if output_file is not None:
        print(f"  Creating affine registered image...")
        
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
    dimension = 3
    pixel_type = itk.F
    image_type = itk.Image[pixel_type, dimension]
    
    transform = itk.BSplineTransform[itk.D, dimension, 3].New()
    
    # Configure B-spline grid
    fixed_region = fixed_image.GetLargestPossibleRegion()
    fixed_spacing = fixed_image.GetSpacing()
    fixed_origin = fixed_image.GetOrigin()
    fixed_size = fixed_region.GetSize()
    
    transform.SetTransformDomainOrigin(fixed_origin)
    transform.SetTransformDomainDirection(fixed_image.GetDirection())
    
    physical_dimensions = [
        fixed_spacing[i] * (fixed_size[i] - 1) for i in range(dimension)
    ]
    transform.SetTransformDomainPhysicalDimensions(physical_dimensions)
    
    mesh_size = [max(1, grid_size[i] - 3) for i in range(dimension)]
    transform.SetTransformDomainMeshSize(mesh_size)
    
    num_parameters = transform.GetNumberOfParameters()
    parameters = itk.OptimizerParameters[itk.D](num_parameters)
    parameters.Fill(0.0)
    transform.SetParameters(parameters)
    
    print(f"  B-spline grid size: {grid_size}")
    print(f"  B-spline mesh size: {mesh_size}")
    print(f"  Number of B-spline parameters: {num_parameters}")
    
    # Create composite transform
    composite_transform = itk.CompositeTransform[itk.D, dimension].New()
    
    try:
        affine_transform = itk.AffineTransform[itk.D, dimension].New()
        affine_transform.SetParameters(initial_transform.GetParameters())
        affine_transform.SetFixedParameters(initial_transform.GetFixedParameters())
        composite_transform.AddTransform(affine_transform)
        print("  Added affine transform to composite")
    except Exception as e:
        print(f"  Warning: Could not cast initial transform to affine: {e}")
        identity_affine = itk.AffineTransform[itk.D, dimension].New()
        identity_affine.SetIdentity()
        composite_transform.AddTransform(identity_affine)
        print("  Added identity affine transform to composite")
    
    composite_transform.AddTransform(transform)
    
    composite_transform.SetOnlyMostRecentTransformToOptimizeOn()
    
    print(f"  Composite transform has {composite_transform.GetNumberOfTransforms()} transforms")
    print(f"  Total parameters: {composite_transform.GetNumberOfParameters()}")
    print(f"  Parameters to optimize: {composite_transform.GetNumberOfParameters()}")
    
    optimizer = itk.RegularStepGradientDescentOptimizerv4.New()
    
    optimizer.SetLearningRate(0.5)
    optimizer.SetMinimumStepLength(0.05)
    optimizer.SetNumberOfIterations(15)
    optimizer.SetRelaxationFactor(0.5)
    optimizer.SetGradientMagnitudeTolerance(1e-6)
    
    print(f"  Optimizer: RegularStepGradientDescent (B-spline tuned)")
    
    metric = itk.MattesMutualInformationImageToImageMetricv4[image_type, image_type].New()
    metric.SetNumberOfHistogramBins(50)
    
    metric.SetVirtualDomainFromImage(fixed_image)
    
    registration = itk.ImageRegistrationMethodv4[image_type, image_type].New()
    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetInitialTransform(composite_transform)
    
    registration.SetMetricSamplingStrategy(0)
    total_voxels = fixed_size[0] * fixed_size[1] * fixed_size[2]
    sampling_percentage = 0.1
    registration.SetMetricSamplingPercentage(sampling_percentage)
    registration.MetricSamplingReinitializeSeed(76926294)
    
    print(f"  Using sampling percentage: {sampling_percentage:.4f}")
    
    registration.SetNumberOfLevels(2)
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([1, 0])
    
    iteration_count = [0]
    last_metric_value = [float('inf')]
    
    def iteration_update():
        try:
            iteration = optimizer.GetCurrentIteration()
            metric_value = optimizer.GetValue()
            iteration_count[0] = iteration
            last_metric_value[0] = metric_value
            
            if iteration % 3 == 0 or iteration < 10:
                print(f"  B-spline iteration {iteration}: {metric_value:.6f}")
                    
        except Exception as e:
            print(f"  Iteration {iteration_count[0]}: Error - {e}")
    
    command = itk.PyCommand.New()
    command.SetCommandCallable(iteration_update)
    optimizer.AddObserver(itk.IterationEvent(), command)
    
    print(f"  Starting B-spline registration...")
    
    try:
        registration.Update()
        print(f"  B-spline registration completed successfully!")
    except Exception as e:
        print(f"  Registration failed: {e}")
        print(f"  Using current state of transform")
    
    # Get the transform from registration (this might be the optimized composite or just the B-spline)
    final_transform = registration.GetTransform()
    
    try:
        final_metric = optimizer.GetValue()
        print(f"  B-spline final metric: {final_metric:.6f}")
        print(f"  Total iterations completed: {iteration_count[0]}")
    except:
        print(f"  Final metric unavailable. Last known: {last_metric_value[0]:.6f}")
        print(f"  Last known iteration: {iteration_count[0]}")
    
    # Check parameter statistics - handle both composite and non-composite cases
    try:
        # Try to access as composite transform first
        if hasattr(final_transform, 'GetNthTransform'):
            print("  Retrieved composite transform")
            final_bspline = final_transform.GetNthTransform(1)  # B-spline is the second transform
            final_composite_transform = final_transform
        else:
            # If it's not composite, it might be just the B-spline transform
            print("  Retrieved single transform (likely B-spline)")
            final_bspline = final_transform
            # Reconstruct the composite transform
            final_composite_transform = itk.CompositeTransform[itk.D, dimension].New()
            try:
                affine_transform = itk.AffineTransform[itk.D, dimension].New()
                affine_transform.SetParameters(initial_transform.GetParameters())
                affine_transform.SetFixedParameters(initial_transform.GetFixedParameters())
                final_composite_transform.AddTransform(affine_transform)
            except:
                identity_affine = itk.AffineTransform[itk.D, dimension].New()
                identity_affine.SetIdentity()
                final_composite_transform.AddTransform(identity_affine)
            final_composite_transform.AddTransform(final_bspline)
        
        # Get B-spline parameters for statistics
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
            
    except Exception as e:
        print(f"  Could not analyze B-spline parameters: {e}")
        # If all else fails, use the original composite transform
        final_composite_transform = composite_transform
    
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

def resample_pet(pet_moving, pet_reference, ct_transform, output_file: str = None):
    """Resample PET using CT registration transform"""
    dimension = 3
    pixel_type = itk.F
    image_type = itk.Image[pixel_type, dimension]
    
    print(f"  Resampling PET using CT transform...")
    print(f"  PET moving image size: {pet_moving.GetLargestPossibleRegion().GetSize()}")
    print(f"  PET reference image size: {pet_reference.GetLargestPossibleRegion().GetSize()}")
    
    resampler = itk.ResampleImageFilter.New(Input=pet_moving)
    resampler.SetTransform(ct_transform)
    resampler.SetReferenceImage(pet_reference)
    resampler.SetUseReferenceImage(True)
    resampler.SetDefaultPixelValue(0)
    
    # Create the interpolator and set the input image correctly
    interpolator = itk.LinearInterpolateImageFunction[image_type, itk.D].New()
    interpolator.SetInputImage(pet_moving)
    resampler.SetInterpolator(interpolator)
    
    resampler.Update()
    registered_pet = resampler.GetOutput()
    
    print(f"  PET resampling completed")
    print(f"  Output PET image size: {registered_pet.GetLargestPossibleRegion().GetSize()}")
    
    if output_file is not None:
        print(f"  Saving resampled PET image to: {output_file}")
        itk.imwrite(registered_pet, output_file)
    
    return registered_pet

def main():
    base_path = Path(os.getcwd()).parent
    inputs_path = base_path / "inputs"
    outputs_path = base_path / "outputs"
    
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
    step_5_pet_resampled = outputs_path / "step_5_pet_resampled.nrrd"
    
    # Transform files
    rigid_transform_file = outputs_path / "rigid_transform.txt"
    similarity_transform_file = outputs_path / "similarity_transform.txt"
    affine_transform_file = outputs_path / "affine_transform.txt"
    bspline_transform_file = outputs_path / "bspline_transform.txt"

    dimension = 3
    pixel_type = itk.F
    image_type = itk.Image[pixel_type, dimension]

    print("Reading CT images...")
    fixed_reader = itk.ImageFileReader[image_type].New()
    moving_reader = itk.ImageFileReader[image_type].New()
    fixed_reader.SetFileName(str(ct_1_path))
    moving_reader.SetFileName(str(ct_2_path))

    fixed_reader.Update()
    moving_reader.Update()
    
    fixed_image = fixed_reader.GetOutput()
    moving_image = moving_reader.GetOutput()

    print("Reading PET images...")
    pet_1_reader = itk.ImageFileReader[image_type].New()
    pet_2_reader = itk.ImageFileReader[image_type].New()
    pet_1_reader.SetFileName(str(pet_1_path))
    pet_2_reader.SetFileName(str(pet_2_path))
    
    pet_1_reader.Update()
    pet_2_reader.Update()
    
    pet_1_image = pet_1_reader.GetOutput()
    pet_2_image = pet_2_reader.GetOutput()

    number_samples = 100_000
    
    # STEP 1: Rigid Registration
    print("\n=== STEP 1: RIGID REGISTRATION ===")
    
    rigid_transform = load_transform(rigid_transform_file, itk.VersorRigid3DTransform[itk.D])
    
    if rigid_transform is None:
        print("No existing rigid transform found. Running rigid registration...")
        start_time = time.time()
        rigid_transform = rigid_registration(fixed_image, moving_image, number_samples, str(step_1_rigid))
        end_time = time.time()
        print(f"Rigid registration completed in {end_time - start_time:.2f} seconds")
        
        save_transform(rigid_transform, rigid_transform_file)
    else:
        print("Loaded existing rigid transform. Skipping rigid registration.")
        
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
    
    # STEP 2: Similarity Registration
    print("\n=== STEP 2: SIMILARITY REGISTRATION ===")
    
    similarity_transform = load_transform(similarity_transform_file, itk.Similarity3DTransform[itk.D])
    
    if similarity_transform is None:
        print("No existing similarity transform found. Running similarity registration...")
        start_time = time.time()
        similarity_transform = similarity_registration(
            fixed_image, moving_image, rigid_transform, number_samples, str(step_2_similarity)
        )
        end_time = time.time()
        print(f"Similarity registration completed in {end_time - start_time:.2f} seconds")
        
        save_transform(similarity_transform, similarity_transform_file)
    else:
        print("Loaded existing similarity transform. Skipping similarity registration.")
        
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
    
    affine_transform = load_transform(affine_transform_file, itk.AffineTransform[itk.D, 3])
    
    if affine_transform is None:
        print("No existing affine transform found. Running affine registration...")
        start_time = time.time()
        affine_transform = affine_registration(
            fixed_image, moving_image, similarity_transform, number_samples, str(step_3_affine)
        )
        end_time = time.time()
        print(f"Affine registration completed in {end_time - start_time:.2f} seconds")
        
        save_transform(affine_transform, affine_transform_file)
    else:
        print("Loaded existing affine transform. Skipping affine registration.")
        
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
    
    print("\n=== STEP 4: B-SPLINE REGISTRATION ===")
    
    bspline_grid_size = (5, 5, 3)
    
    bspline_transform = None
    
    # Try multiple transform types for loading
    if bspline_transform_file.exists():
        print(f"B-spline transform file exists: {bspline_transform_file}")
        
        # Try loading as CompositeTransform first
        try:
            bspline_transform = load_transform(bspline_transform_file, itk.CompositeTransform[itk.D, 3])
            print("Successfully loaded as CompositeTransform")
        except Exception as e:
            print(f"Could not load as CompositeTransform: {e}")
            
            # Try loading as BSplineTransform
            try:
                bspline_transform = load_transform(bspline_transform_file, itk.BSplineTransform[itk.D, 3, 3])
                print("Successfully loaded as BSplineTransform")
            except Exception as e:
                print(f"Could not load as BSplineTransform: {e}")
                
                # Try loading as generic Transform
                try:
                    bspline_transform = itk.transformread(str(bspline_transform_file))[0]
                    print("Successfully loaded using itk.transformread")
                except Exception as e:
                    print(f"Could not load with itk.transformread: {e}")
    else:
        print("B-spline transform file does not exist")
    
    if bspline_transform is None:
        print(f"No existing B-spline transform found. Running B-spline registration with grid size {bspline_grid_size}...")
        start_time = time.time()
        bspline_transform = bspline_registration(
            fixed_image, moving_image, affine_transform, number_samples, 
            bspline_grid_size, str(step_4_bspline)
        )
        end_time = time.time()
        print(f"B-spline registration completed in {end_time - start_time:.2f} seconds")
        
        save_transform(bspline_transform, bspline_transform_file)
    else:
        print("Loaded existing B-spline transform. Skipping B-spline registration.")
        
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
    
    # STEP 5: Resample PET
    print("\n=== STEP 5: PET RESAMPLING ===")
    
    if not step_5_pet_resampled.exists():
        print("Resampling PET_2 using CT registration transform...")
        start_time = time.time()
        resampled_pet = resample_pet(pet_2_image, pet_1_image, bspline_transform, str(step_5_pet_resampled))
        end_time = time.time()
        print(f"PET resampling completed in {end_time - start_time:.2f} seconds")
    else:
        print("PET resampled image already exists. Skipping PET resampling.")
    
    print(f"\n=== REGISTRATION COMPLETE ===")
    print(f"Rigid registered image: {step_1_rigid}")
    print(f"Similarity registered image: {step_2_similarity}")
    print(f"Affine registered image: {step_3_affine}")
    print(f"B-spline registered image: {step_4_bspline}")
    print(f"PET resampled image: {step_5_pet_resampled}")
    print(f"Rigid transform saved: {rigid_transform_file}")
    print(f"Similarity transform saved: {similarity_transform_file}")
    print(f"Affine transform saved: {affine_transform_file}")
    print(f"B-spline transform saved: {bspline_transform_file}")
    
    print(f"\nFinal Rigid Transform Parameters:")
    print(f"  Translation: {rigid_transform.GetTranslation()}")
    print(f"  Center: {rigid_transform.GetCenter()}")
    
    print(f"\nFinal Similarity Transform Parameters:")
    print(f"  Translation: {similarity_transform.GetTranslation()}")
    print(f"  Center: {similarity_transform.GetCenter()}")
    print(f"  Scale: {similarity_transform.GetScale()}")
    
    print(f"\nFinal Affine Transform Parameters:")
    print(f"  Translation: {affine_transform.GetTranslation()}")
    print(f"  Center: {affine_transform.GetCenter()}")
    print(f"  Matrix:\n{affine_transform.GetMatrix()}")


if __name__ == "__main__":
    main()