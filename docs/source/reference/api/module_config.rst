Module configuration
====================

.. include:: ../../includes/dataclasses_note.rst

Main module configuration entities
----------------------------------

.. automodule:: savant.config.schema

.. _pipeline_element_hierarchy:

.. inheritance-diagram:: PipelineElement ModelElement PyFuncElement DrawBin
    :parts: 1
    :caption: PipelineElement hierarchy

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/class.rst

    Module
    Pipeline
    PipelineElement
    PyFuncElement
    ModelElement
    DrawBin
    DynamicGstProperty

Module configuration manager
----------------------------

.. automodule:: savant.config

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/class.rst

    ModuleConfig

OmegaConf resolvers
-------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/function.rst

    initializer_resolver
    calc_resolver
    json_resolver
