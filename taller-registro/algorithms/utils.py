# utils.py
from typing import Union, Optional, Tuple
import itk
import os

def save_transform(transform, filename):
    """Save ITK transform to file"""
    writer = itk.TransformFileWriterTemplate[itk.D].New()
    writer.SetInput(transform)
    writer.SetFileName(str(filename))
    writer.Update()
    print(f"  Transform saved to: {filename}")

def load_transform(filename, transform_type):
    """Load ITK transform from file"""
    if not os.path.exists(filename):
        return None
    
    reader = itk.TransformFileReaderTemplate[itk.D].New()
    reader.SetFileName(str(filename))
    reader.Update()
    
    transform_list = reader.GetTransformList()
    if len(transform_list) > 0:
        # Create a new transform of the specified type and copy parameters
        loaded_transform = transform_type.New()
        # The loaded transform should be compatible
        source_transform = transform_list[0]
        loaded_transform.SetParameters(source_transform.GetParameters())
        loaded_transform.SetFixedParameters(source_transform.GetFixedParameters())
        print(f"  Transform loaded from: {filename}")
        return loaded_transform
    return None

def get_rigid_transform_as_versor(transform):
    """Convert any rigid transform to VersorRigid3DTransform for compatibility"""
    versor_transform = itk.VersorRigid3DTransform[itk.D].New()
    versor_transform.SetParameters(transform.GetParameters())
    versor_transform.SetFixedParameters(transform.GetFixedParameters())
    return versor_transform