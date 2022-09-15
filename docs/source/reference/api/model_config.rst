
Model configuration
===================

.. include:: ../../includes/dataclasses_note.rst

Base model configuration entities
---------------------------------

Savant base models
^^^^^^^^^^^^^^^^^^

.. automodule:: savant.base.model

.. _model_hierarchy_base:

.. inheritance-diagram:: Model ObjectModel AttributeModel ComplexModel
    :parts: 1
    :caption: Model child classes

.. inheritance-diagram:: ModelOutput ObjectModelOutput AttributeModelOutput ComplexModelOutput
    :parts: 1
    :caption: ModelOutput child classes

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/class.rst

    Model
    ModelInput
    ModelOutput
    PreprocessObjectTensor

    ObjectModel
    ObjectModelOutput
    ObjectModelOutputObject

    AttributeModel
    AttributeModelOutput
    AttributeModelOutputAttribute

    ComplexModel
    ComplexModelOutput

.. autoclass:: savant.base.model.ModelPrecision
    :members:

.. autoclass:: savant.base.model.ModelColorFormat
    :members:

Deepstream base models
^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: savant.deepstream.nvinfer.model

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/class.rst

    NvInferModel

Deepstream model configuration
------------------------------

.. automodule:: savant.deepstream.nvinfer.model

.. _model_hierarchy_nvinfer:

.. inheritance-diagram:: savant.base.model.Model NvInferModel NvInferDetector NvInferRotatedObjectDetector NvInferInstanceSegmentation NvInferAttributeModel NvInferComplexModel
    :parts: 1
    :caption: Model hierarchy with nvinfer entities

.. inheritance-diagram:: savant.base.model.ComplexModelOutput NvInferObjectModelOutput NvInferRotatedObjectModelOutput
    :parts: 1
    :caption: Model output hierarchy with nvinfer entities

.. note::

    The following classes specify configuration templates for fully realized
    concrete model types that can be included in the pipeline.

    For example, base model types :py:class:`~savant.base.model.Model` or :py:class:`NvInferModel`
    cannot be used in configuration, while their descendant :py:class:`NvInferDetector` can.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/class.rst

    NvInferDetector
    NvInferRotatedObjectDetector
    NvInferAttributeModel
    NvInferComplexModel

    NvInferModelInput

    NvInferObjectModelOutput
    NvInferRotatedObjectModelOutput
    NvInferObjectModelOutputObject

.. autoclass:: savant.deepstream.nvinfer.model.NvInferModelFormat
    :members:

.. autoclass:: savant.deepstream.nvinfer.model.NvInferModelType
    :members:
