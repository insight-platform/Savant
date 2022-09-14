Pipeline overview
=================

.. _fig-1-pipeline:

.. figure:: images/pic1.png
   :width: 600
   :align: center

   Figure 1. Video Analytics Application Scheme

Pipeline is a sequence of frame processing elements. This sequence has one input (src) and one output (sink).
:ref:`Figure 1 <fig-1-pipeline>` illustrates pipeline's place in a module, green box denotes the entire pipeline
and blue one highlights the user-customizable part of it that is generated
by the ``elements`` section of the module configuration file.

.. note::

   Note that ``ZeroMQ Source`` / ``Muxer`` / ``ZeroMQ Sink`` parts of the pipeline are fixed,
   and are not supposed to be customized by the user. This is a beneficial feature related to
   simplifying pipeline data input and output.

   By design, sources and sinks are decoupled from the pipeline main processing and are instead
   realized in separate units called adapters. This increases the stability of the pipeline
   and avoids unnecessary reloads in case of source failures.

   One of notable benefits of this approach is the ability to easily mix together data from various sources
   in a single pipeline.

Default Savant module config includes definition for the :py:attr:`~savant.config.schema.Pipeline.source`
and :py:attr:`~savant.config.schema.Pipeline.sink` nodes:

.. literalinclude:: ../../../savant/config/default.yml
  :language: YAML
  :lines: 50-

It is possible to redefine these values, but it is important to remember that the main
operation mode for a Savant module is with ZeroMQ source and sink.

Usually, when writing a module, user would only define pipeline :py:attr:`~savant.config.schema.Pipeline.elements`.
All elements in the pipeline are enumerated in a list. The following element types can be added to pipelines:

#. detector
#. rotated_object_detector
#. classifier
#. attribute_model
#. complex_model
#. pyfunc
#. other plugins (except for sinks or sources) from NVIDIA Deepstream.

These elements' configuration is discussed in more detail in :doc:`model_elements`.

.. _fig-2:

.. figure:: images/pic2.png
   :width: 600
   :align: center

   Figure 2. Example of pipeline execution and data flow. 'S2 – F2: Object 1' notation
   represents an object with unique id 1 from the frame 2 of the source 2.

Each element in the pipeline processes frame data or frame meta information and
adds new or enriches existing meta information. This process is illustrated on :ref:`Figure 2 <fig-2>`.

Pipeline input is a stream of frames from one or more sources possibly accompanied by metadata from other modules.

.. note::

   Module relies on source adapters for its input. Savant includes several useful source (and sink) adapters,
   and it's possible to implement custom ones should the user need them.

Frames from different sources are combined into batches (the first object model configuration determines the batch size)
and, together with metadata, pass sequentially through the elements of the pipeline.
Each element can change the metadata, add/delete/change objects and their attributes.

In :ref:`Figure 2 <fig-2>`, the first element specified by the user is a detection model.

This detector locates objects on the frame and adds information about them to the frame metadata.
Savant uses bounding boxes of 2 types to describe locations of detected objects:
axis-aligned boxes or rotated boxes.

.. note::

   Rotated boxes have to be supported by the used detection models.

In addition to location, detected objects receive labels based on a combination of the model name and object class label.

For example, the detection model in :ref:`Figure 2 <fig-2>` is named ``yolo``, and it can detect
3 classes of objects: ``car``, ``dog`` and ``cat``. Therefore, detected objects will be added
to the meta information with labels ``yolo.car``, ``yolo.dog``, ``yolo.cat``. This property allows the
user to specify the input :py:attr:`~savant.base.model.ModelInput.object` for any subsequent models.

Frame meta information always holds an object that corresponds to the whole frame.
This object's label name is set to ``frame``. If you do not specify which
objects should be used as input then the ``frame`` object will be used by default, i.e. the whole frame.

The next element of the pipeline in :ref:`Figure 2 <fig-2>` is a classification model.

Let's say, this model determines the color of a car and shouldn't process all the objects detected previously.
To configure the model to work only on objects that are labeled ``yolo.car`` this label must be set in the
:py:attr:`~savant.base.model.ModelInput.object` field in the configuration file. Savant will then automatically
filter all objects present in metadata and only cars will be used as input for the classification model.

The classification results will be added to the ``yolo.car`` objects as additional meta-information (attribute).

In this way, as frames go through the pipeline, new objects are detected and added as metadata,
which is then being extended with various attributes.

After the frame was processed by all the elements of the pipeline the meta-information for each object in the frame
is passed into ZeroMQ Sink element in the output format (:ref:`reference/avro:VideoFrameMetadata Schema`).
