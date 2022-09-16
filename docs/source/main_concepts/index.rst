General overview
================

Let's define some basic notions, terms, and concepts that will be used below
to understand how :repo-link:`Savant` works and how to correctly generate a configuration file.

Glossary
--------

.. glossary::

  **Video stream**
    A time sequence of frames of a specific format encoded into a bit stream.

  **Frame**
    A single image of a video stream.

  **Module**
    A processing unit that receives video streams and generates or enriches meta information
    about objects and events in those video streams.
    A module can be deployed on edge devices or in the cloud.

  **Adapter**
    An input or output unit. Provides data for a module or receives module output and transforms it to task-specific format:
    video files, window display, etc.

  **Pipeline**
    The main entity of the module that consists of a sequence of elements that implement required frame processing. This sequence has one input (src) and one output (sink).

  **Element**
    A unit of processing logic for a batch of frames that adds new or enriches existing meta information for each frame. This entity is a part of the Pipeline. An element can be an inference model, a deepstream plugin or a custom function.

  **Object**
    A problem-specific entity that is physically present on a frame, eg. a person or a car. An object is described using special structures in the frame meta data. An object's position is specified using bounding boxes of either of two types: axis-aligned bounding box or rotated bounding box.

  **Attribute**
    An arbitrary property of an object. For example: gender, age, color, object class, identifier, etc.

:ref:`Figure 1 <fig-1-general>` illustrates the general video analytics application scheme.
The yellow box denotes a module, which is the part of the processing that can be implemented using the Savant framework.
Green box denotes the entire pipeline and blue one highlights the user-cutomizable part of the pipeline.

.. _fig-1-general:

.. figure:: images/pic1.png
   :width: 600
   :align: center

   Figure 1. Video Analytics Application Scheme


Learn more about:

* Module and its configuration in :doc:`module`.
* Pipeline and its contents in :doc:`pipeline`.
* The most important type of pipeline element in :doc:`model_elements`.
* Adapter details in :doc:`adapters`
