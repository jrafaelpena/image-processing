from pathlib import Path
from typing import Union, Optional, Tuple
import itk
import time
import os
import numpy as np


def multi_phase_ct_registration(
    fixed_ct_path: Union[str, Path],
    moving_ct_path: Union[str, Path],
    output_ct_path: Union[str, Path],
    transform_output_path: Union[str, Path],
    dimension: int = 3,
    number_of_samples: int = 400000,
    bspline_grid_size: Tuple[int, int, int] = (11, 11, 7)
) -> itk.Transform:
    """
    Multi-phase CT registration: Rigid -> Rigid+Scale -> Affine -> BSpline
    Following Slicer's General Registration (BRAINS) workflow
    """
    print("Starting multi-phase CT registration...")
    
    # Define pixel and image types
    pixel_type = itk.F
    image_type = itk.Image[pixel_type, dimension]
    
    # Read input images
    print("Reading CT images...")
    fixed_reader = itk.ImageFileReader[image_type].New()
    moving_reader = itk.ImageFileReader[image_type].New()
    fixed_reader.SetFileName(str(fixed_ct_path))
    moving_reader.SetFileName(str(moving_ct_path))
    
    # Update readers to get image information
    fixed_reader.Update()
    moving_reader.Update()
    
    fixed_image = fixed_reader.GetOutput()
    moving_image = moving_reader.GetOutput()
    
    print(f"Fixed image size: {fixed_image.GetLargestPossibleRegion().GetSize()}")
    print(f"Moving image size: {moving_image.GetLargestPossibleRegion().GetSize()}")
    
    # Phase 1: Rigid Registration
    print("\nPhase 1: Rigid Registration")
    rigid_transform = rigid_registration(fixed_image, moving_image, number_of_samples)
    
    # Phase 2: Rigid + Scale Registration (Similarity Transform)
    print("\nPhase 2: Rigid + Scale Registration")
    similarity_transform = similarity_registration(fixed_image, moving_image, rigid_transform, number_of_samples)
    
    # Phase 3: Affine Registration
    print("\nPhase 3: Affine Registration")
    affine_transform = affine_registration(fixed_image, moving_image, similarity_transform, number_of_samples)
    
    # Phase 4: BSpline Registration
    print("\nPhase 4: BSpline Registration")
    bspline_transform = bspline_registration(
        fixed_image, moving_image, affine_transform, 
        number_of_samples, bspline_grid_size
    )
    
    # Apply final transformation and save registered CT
    print("\nApplying final transformation...")
    resample_and_save_ct(moving_image, fixed_image, bspline_transform, output_ct_path)
    
    # Save transform
    print(f"Saving transform to: {transform_output_path}")
    writer = itk.TransformFileWriter.New()
    writer.SetInput(bspline_transform)
    writer.SetFileName(str(transform_output_path))
    writer.Update()
    
    print("CT registration completed successfully!")
    return bspline_transform


def rigid_registration(fixed_image, moving_image, number_of_samples: int):
    """Phase 1: Rigid registration"""
    dimension = fixed_image.GetImageDimension()
    pixel_type = itk.F
    image_type = itk.Image[pixel_type, dimension]
    
    # Registration components
    transform = itk.Euler3DTransform[itk.D].New() if dimension == 3 else itk.Euler2DTransform[itk.D].New()
    optimizer = itk.RegularStepGradientDescentOptimizer.New()
    metric = itk.MattesMutualInformationImageToImageMetric[image_type, image_type].New()
    interpolator = itk.LinearInterpolateImageFunction[image_type, itk.D].New()
    registration = itk.ImageRegistrationMethod[image_type, image_type].New()
    
    # Set up registration
    registration.SetTransform(transform)
    registration.SetOptimizer(optimizer)
    registration.SetMetric(metric)
    registration.SetInterpolator(interpolator)
    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    registration.SetFixedImageRegion(fixed_image.GetBufferedRegion())
    
    # Metric parameters
    metric.SetNumberOfHistogramBins(50)
    metric.SetNumberOfSpatialSamples(number_of_samples)
    metric.ReinitializeSeed(76926294)
    
    # Optimizer parameters
    optimizer.SetMaximumStepLength(0.2)
    optimizer.SetMinimumStepLength(0.001)
    optimizer.SetNumberOfIterations(200)
    optimizer.SetRelaxationFactor(0.9)
    
    # Initialize transform
    initializer = itk.CenteredTransformInitializer[
        type(transform), image_type, image_type
    ].New()
    initializer.SetTransform(transform)
    initializer.SetFixedImage(fixed_image)
    initializer.SetMovingImage(moving_image)
    initializer.MomentsOn()
    initializer.InitializeTransform()
    
    registration.SetInitialTransformParameters(transform.GetParameters())
    
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
    final_transform = type(transform).New()
    final_transform.SetParameters(registration.GetLastTransformParameters())
    final_transform.SetFixedParameters(transform.GetFixedParameters())
    
    print(f"  Rigid final metric: {optimizer.GetValue()}")
    return final_transform


    """Phase 2: Similarity (Rigid + Scale) registration"""
    dimension = fixed_image.GetImageDimension()
    pixel_type = itk.F
    image_type = itk.Image[pixel_type, dimension]
    
    # Use Similarity transform (rigid + uniform scaling)
    transform = itk.Similarity3DTransform[itk.D].New() if dimension == 3 else itk.Similarity2DTransform[itk.D].New()
    optimizer = itk.RegularStepGradientDescentOptimizer.New()
    metric = itk.MattesMutualInformationImageToImageMetric[image_type, image_type].New()
    interpolator = itk.LinearInterpolateImageFunction[image_type, itk.D].New()
    registration = itk.ImageRegistrationMethod[image_type, image_type].New()
    
    # Set up registration
    registration.SetTransform(transform)
    registration.SetOptimizer(optimizer)
    registration.SetMetric(metric)
    registration.SetInterpolator(interpolator)
    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    registration.SetFixedImageRegion(fixed_image.GetBufferedRegion())
    
    # Initialize with rigid transform result
    transform.SetCenter(initial_transform.GetCenter())
    transform.SetTranslation(initial_transform.GetTranslation())
    if dimension == 3:
        transform.SetRotation(initial_transform.GetVersor())
    else:
        transform.SetAngle(initial_transform.GetAngle())
    transform.SetScale(1.0)  # Start with no scaling
    
    # Metric parameters
    metric.SetNumberOfHistogramBins(50)
    metric.SetNumberOfSpatialSamples(number_of_samples)
    metric.ReinitializeSeed(76926294)
    
    # Optimizer parameters
    optimizer.SetMaximumStepLength(0.1)
    optimizer.SetMinimumStepLength(0.001)
    optimizer.SetNumberOfIterations(200)
    optimizer.SetRelaxationFactor(0.9)
    
    registration.SetInitialTransformParameters(transform.GetParameters())
    
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
    final_transform = type(transform).New()
    final_transform.SetParameters(registration.GetLastTransformParameters())
    final_transform.SetFixedParameters(transform.GetFixedParameters())
    
    print(f"  Similarity final metric: {optimizer.GetValue()}")
    return final_transform


    def affine_registration(fixed_image, moving_image, initial_transform, number_of_samples: int):
        """Phase 3: Affine registration"""
        dimension = fixed_image.GetImageDimension()
        pixel_type = itk.F
        image_type = itk.Image[pixel_type, dimension]
        
        transform = itk.AffineTransform[itk.D, dimension].New()
        optimizer = itk.RegularStepGradientDescentOptimizer.New()
        metric = itk.MattesMutualInformationImageToImageMetric[image_type, image_type].New()
        interpolator = itk.LinearInterpolateImageFunction[image_type, itk.D].New()
        registration = itk.ImageRegistrationMethod[image_type, image_type].New()
        
        # Set up registration
        registration.SetTransform(transform)
        registration.SetOptimizer(optimizer)
        registration.SetMetric(metric)
        registration.SetInterpolator(interpolator)
        registration.SetFixedImage(fixed_image)
        registration.SetMovingImage(moving_image)
        registration.SetFixedImageRegion(fixed_image.GetBufferedRegion())
        
        # Initialize with similarity transform result
        transform.SetCenter(initial_transform.GetCenter())
        transform.SetTranslation(initial_transform.GetTranslation())
        
        # Convert similarity transform matrix to affine
        matrix = transform.GetMatrix()
        if dimension == 3:
            # Get rotation matrix from similarity transform
            rotation_matrix = initial_transform.GetMatrix()
            scale = initial_transform.GetScale()
            for i in range(dimension):
                for j in range(dimension):
                    matrix[i, j] = rotation_matrix[i, j] * scale
        else:
            # 2D case
            angle = initial_transform.GetAngle()
            scale = initial_transform.GetScale()
            matrix[0, 0] = scale * np.cos(angle)
            matrix[0, 1] = -scale * np.sin(angle)
            matrix[1, 0] = scale * np.sin(angle)
            matrix[1, 1] = scale * np.cos(angle)
        
        transform.SetMatrix(matrix)
        
        # Metric parameters
        metric.SetNumberOfHistogramBins(50)
        metric.SetNumberOfSpatialSamples(number_of_samples)
        metric.ReinitializeSeed(76926294)
        
        # Optimizer parameters
        optimizer.SetMaximumStepLength(0.05)
        optimizer.SetMinimumStepLength(0.001)
        optimizer.SetNumberOfIterations(200)
        optimizer.SetRelaxationFactor(0.9)
        
        registration.SetInitialTransformParameters(transform.GetParameters())
        
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
        final_transform = itk.AffineTransform[itk.D, dimension].New()
        final_transform.SetParameters(registration.GetLastTransformParameters())
        final_transform.SetFixedParameters(transform.GetFixedParameters())
        
        print(f"  Affine final metric: {optimizer.GetValue()}")
        return final_transform


def bspline_registration(fixed_image, moving_image, initial_transform, 
                        number_of_samples: int, grid_size: Tuple[int, int, int]):
    """Phase 4: BSpline deformable registration"""
    dimension = fixed_image.GetImageDimension()
    pixel_type = itk.F
    image_type = itk.Image[pixel_type, dimension]
    
    # BSpline transform
    transform = itk.BSplineTransform[itk.D, dimension, 3].New()
    optimizer = itk.LBFGSBOptimizer.New()
    metric = itk.MattesMutualInformationImageToImageMetric[image_type, image_type].New()
    interpolator = itk.LinearInterpolateImageFunction[image_type, itk.D].New()
    registration = itk.ImageRegistrationMethod[image_type, image_type].New()
    
    # Set up BSpline grid
    physical_dimensions = []
    mesh_size = []
    for i in range(dimension):
        physical_dimensions.append(
            fixed_image.GetSpacing()[i] * 
            (fixed_image.GetLargestPossibleRegion().GetSize()[i] - 1)
        )
        mesh_size.append(grid_size[i] if i < len(grid_size) else grid_size[-1])
    
    transform.SetTransformDomainOrigin(fixed_image.GetOrigin())
    transform.SetTransformDomainPhysicalDimensions(physical_dimensions)
    transform.SetTransformDomainMeshSize(mesh_size)
    transform.SetTransformDomainDirection(fixed_image.GetDirection())
    
    # Initialize BSpline parameters
    number_of_parameters = transform.GetNumberOfParameters()
    parameters = itk.OptimizerParameters[itk.D](number_of_parameters)
    parameters.Fill(0.0)
    transform.SetParameters(parameters)
    
    # Set up registration
    registration.SetTransform(transform)
    registration.SetOptimizer(optimizer)
    registration.SetMetric(metric)
    registration.SetInterpolator(interpolator)
    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    registration.SetFixedImageRegion(fixed_image.GetBufferedRegion())
    
    # Metric parameters
    metric.SetNumberOfHistogramBins(50)
    metric.SetNumberOfSpatialSamples(number_of_samples)
    metric.ReinitializeSeed(76926294)
    
    # Optimizer parameters
    optimizer.SetGradientConvergenceTolerance(1e-6)
    optimizer.SetLineSearchAccuracy(0.9)
    optimizer.SetDefaultStepLength(1.5)
    optimizer.TraceOn()
    optimizer.SetMaximumNumberOfIterations(200)
    
    # Set bounds for BSpline parameters (optional regularization)
    bounds_selection = itk.OptimizerParameters[itk.UC](number_of_parameters)
    bounds_selection.Fill(0)  # 0 = unbounded, 1 = lower bound, 2 = both bounds
    
    lower_bounds = itk.OptimizerParameters[itk.D](number_of_parameters)
    upper_bounds = itk.OptimizerParameters[itk.D](number_of_parameters)
    lower_bounds.Fill(-10.0)
    upper_bounds.Fill(10.0)
    
    optimizer.SetBoundSelection(bounds_selection)
    optimizer.SetLowerBound(lower_bounds)
    optimizer.SetUpperBound(upper_bounds)
    
    registration.SetInitialTransformParameters(transform.GetParameters())
    
    # Add observer
    def iteration_update():
        iteration = optimizer.GetCurrentIteration()
        metric_value = optimizer.GetValue()
        print(f"  BSpline iteration {iteration}: {metric_value}")
    
    command = itk.PyCommand.New()
    command.SetCommandCallable(iteration_update)
    optimizer.AddObserver(itk.IterationEvent(), command)
    
    # Execute registration
    registration.Update()
    
    # Get final transform - create composite transform
    composite_transform = itk.CompositeTransform[itk.D, dimension].New()
    composite_transform.AddTransform(initial_transform)
    
    # Create final BSpline transform with optimized parameters
    final_bspline = type(transform).New()
    final_bspline.SetTransformDomainOrigin(transform.GetTransformDomainOrigin())
    final_bspline.SetTransformDomainPhysicalDimensions(transform.GetTransformDomainPhysicalDimensions())
    final_bspline.SetTransformDomainMeshSize(transform.GetTransformDomainMeshSize())
    final_bspline.SetTransformDomainDirection(transform.GetTransformDomainDirection())
    final_bspline.SetParameters(registration.GetLastTransformParameters())
    
    composite_transform.AddTransform(final_bspline)
    
    print(f"  BSpline final metric: {optimizer.GetValue()}")
    return composite_transform


def resample_and_save_ct(moving_image, fixed_image, transform, output_path: Union[str, Path]):
    """Resample moving CT using final transform and save"""
    dimension = moving_image.GetImageDimension()
    pixel_type = itk.F
    input_image_type = itk.Image[pixel_type, dimension]
    output_pixel_type = itk.SS  # Signed short for CT
    output_image_type = itk.Image[output_pixel_type, dimension]
    
    # Resample filter
    resampler = itk.ResampleImageFilter[input_image_type, input_image_type].New()
    resampler.SetInput(moving_image)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(itk.LinearInterpolateImageFunction[input_image_type, itk.D].New())
    
    # Use fixed image properties
    resampler.SetSize(fixed_image.GetLargestPossibleRegion().GetSize())
    resampler.SetOutputOrigin(fixed_image.GetOrigin())
    resampler.SetOutputSpacing(fixed_image.GetSpacing())
    resampler.SetOutputDirection(fixed_image.GetDirection())
    resampler.SetDefaultPixelValue(-1000)  # Air value for CT
    
    # Cast to appropriate output type
    caster = itk.CastImageFilter[input_image_type, output_image_type].New()
    caster.SetInput(resampler.GetOutput())
    
    # Write output
    writer = itk.ImageFileWriter[output_image_type].New()
    writer.SetInput(caster.GetOutput())
    writer.SetFileName(str(output_path))
    writer.Update()


def resample_pet_with_transform(
    pet_image_path: Union[str, Path],
    reference_pet_path: Union[str, Path],
    transform_path: Union[str, Path],
    output_pet_path: Union[str, Path],
    dimension: int = 3
) -> None:
    """
    Phase 3: Resample PET using the transform from CT registration
    """
    print(f"Resampling PET image using transform: {transform_path}")
    
    pixel_type = itk.F
    image_type = itk.Image[pixel_type, dimension]
    
    # Read PET images
    moving_reader = itk.ImageFileReader[image_type].New()
    reference_reader = itk.ImageFileReader[image_type].New()
    moving_reader.SetFileName(str(pet_image_path))
    reference_reader.SetFileName(str(reference_pet_path))
    
    moving_reader.Update()
    reference_reader.Update()
    
    moving_pet = moving_reader.GetOutput()
    reference_pet = reference_reader.GetOutput()
    
    # Read transform
    transform_reader = itk.TransformFileReader.New()
    transform_reader.SetFileName(str(transform_path))
    transform_reader.Update()
    
    transform_list = transform_reader.GetTransformList()
    transform = transform_list[0]
    
    # Resample moving PET
    resampler = itk.ResampleImageFilter[image_type, image_type].New()
    resampler.SetInput(moving_pet)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(itk.LinearInterpolateImageFunction[image_type, itk.D].New())
    
    # Use reference PET properties
    resampler.SetSize(reference_pet.GetLargestPossibleRegion().GetSize())
    resampler.SetOutputOrigin(reference_pet.GetOrigin())
    resampler.SetOutputSpacing(reference_pet.GetSpacing())
    resampler.SetOutputDirection(reference_pet.GetDirection())
    resampler.SetDefaultPixelValue(0)  # Background value for PET
    
    # Write output
    output_pixel_type = itk.SS
    output_image_type = itk.Image[output_pixel_type, dimension]
    caster = itk.CastImageFilter[image_type, output_image_type].New()
    caster.SetInput(resampler.GetOutput())
    
    writer = itk.ImageFileWriter[output_image_type].New()
    writer.SetInput(caster.GetOutput())
    writer.SetFileName(str(output_pet_path))
    writer.Update()
    
    print(f"Resampled PET saved to: {output_pet_path}")


def main():
    """
    Main function implementing the complete Slicer workflow:
    Phase 1: Load & Display (handled by file I/O)
    Phase 2: Co-register CT (multi-phase registration)
    Phase 3: Resample PET (using CT registration transform)
    """
    base_path = Path(os.getcwd())
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
    ct_2_registered_path = outputs_path / "CT_2_Xf2.nrrd"
    pet_2_registered_path = outputs_path / "PET_2_Xf2.nrrd"
    transform_path = outputs_path / "Xf2_CT21.h5"  # HDF5 format for composite transforms
    
    start_time = time.time()
    
    print("="*60)
    print("SLICER CASE #20: Intra-subject whole-body PET-CT Registration")
    print("="*60)
    
    # Phase 2: Co-register CT (multi-phase: Rigid -> Rigid+Scale -> Affine -> BSpline)
    print("\nPHASE 2: CT Co-registration")
    print("-" * 30)
    
    try:
        ct_transform = multi_phase_ct_registration(
            fixed_ct_path=ct_1_path,
            moving_ct_path=ct_2_path,
            output_ct_path=ct_2_registered_path,
            transform_output_path=transform_path,
            dimension=3,
            number_of_samples=400000,  # Reduce to 200000 for faster performance
            bspline_grid_size=(11, 11, 7)
        )
        
        # Phase 3: Resample PET using CT registration transform
        print("\nPHASE 3: PET Resampling")
        print("-" * 30)
        
        resample_pet_with_transform(
            pet_image_path=pet_2_path,
            reference_pet_path=pet_1_path,
            transform_path=transform_path,
            output_pet_path=pet_2_registered_path,
            dimension=3
        )
        
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("REGISTRATION COMPLETED SUCCESSFULLY!")
        print(f"Total processing time: {elapsed_time:.2f} seconds")
        print(f"Registered CT saved to: {ct_2_registered_path}")
        print(f"Registered PET saved to: {pet_2_registered_path}")
        print(f"Transform saved to: {transform_path}")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR during registration: {str(e)}")
        print("Please check that input files exist and are in the correct format.")
        raise


if __name__ == "__main__":
    main()