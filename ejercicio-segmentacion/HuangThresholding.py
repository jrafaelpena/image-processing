import itk
import os
from pathlib import Path

def apply_huang_threshold(input_image, output_image, num_bins):
    
    Dimension = 3
    PixelType = itk.US
    ImageType = itk.Image[PixelType, Dimension]

    # Read the input image
    ReaderType = itk.ImageFileReader[ImageType]
    reader = ReaderType.New()
    reader.SetFileName(input_image)

    # Apply Huang Threshold Filter
    FilterType = itk.HuangThresholdImageFilter[ImageType, ImageType]
    imfilter = FilterType.New()
    imfilter.SetInput(reader.GetOutput())
    imfilter.SetNumberOfHistogramBins(num_bins)

    # Rescale the intensity of the image
    RescaleType = itk.RescaleIntensityImageFilter[ImageType, ImageType]
    rescaler = RescaleType.New()
    rescaler.SetInput(imfilter.GetOutput())
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(255)

    # Write the output image
    WriterType = itk.ImageFileWriter[ImageType]
    writer = WriterType.New()
    writer.SetFileName(output_image)
    writer.SetInput(rescaler.GetOutput())
    writer.Update()


if __name__ == "__main__":
    
    input_image = "inputs/USProstate.nii.gz"
    num_bins = 128
    output_image = f"outputs/USProstate_proccessed_{num_bins}.nii.gz"
    

    apply_huang_threshold(input_image, output_image, num_bins)
