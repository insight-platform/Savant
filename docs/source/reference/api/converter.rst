Converters
==========

Base converters
---------------

.. inheritance-diagram:: savant.base.converter.BaseObjectModelOutputConverter savant.base.converter.BaseAttributeModelOutputConverter savant.base.converter.BaseComplexModelOutputConverter
    :parts: 1
    :caption: Base converters hierarchy
    :top-classes: savant.base.pyfunc.BasePyFuncCallableImpl

.. automodule:: savant.base.converter

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/class.rst

    BaseObjectModelOutputConverter
    BaseAttributeModelOutputConverter
    BaseComplexModelOutputConverter

Savant model converters
-----------------------

.. automodule:: savant.converter

Attribute model converters
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: savant.converter.TensorToLabelConverter savant.converter.TensorToVectorConverter savant.converter.TensorToItemConverter
    :parts: 1
    :caption: Attribute converters hierarchy
    :top-classes: savant.base.converter.BaseAttributeModelOutputConverter

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/class.rst

    TensorToLabelConverter
    TensorToVectorConverter
    TensorToItemConverter

Object model converters
^^^^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: savant.converter.yolo.TensorToBBoxConverter savant.converter.yolo_v4.TensorToBBoxConverter savant.converter.yolo_x.TensorToBBoxConverter savant.converter.rapid.TensorToBBoxConverter
    :parts: 2
    :caption: Object converters hierarchy
    :top-classes: savant.base.converter.BaseObjectModelOutputConverter

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/module_class.rst

    yolo.TensorToBBoxConverter
    yolo_v4.TensorToBBoxConverter
    yolo_x.TensorToBBoxConverter
    rapid.TensorToBBoxConverter
