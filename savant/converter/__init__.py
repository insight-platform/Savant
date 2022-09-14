"""Model output postprocessing."""
from .classifier import TensorToLabelConverter
from .scale import scale_rbbox
from .vector_attribute import TensorToVectorConverter, TensorToItemConverter
