Metadata Operations
===================

A Savant module's :doc:`pipeline <29_pipeline>` is comprised of units that process objects. Every object is represented by its metadata. Metadata defines an object label, location and size, identifier, and other properties. There are two kinds of metadata in Savant: primary (DeepStream-related) metadata and extended metadata.

Simply speaking, every unit can produce metadata or modify it. Let us begin with a bare image frame coming into a pipeline to show you how it happens. In the first step, Savant creates a unique "frame" object whose metadata object corresponds to the whole viewport (except paddings). This object is somehow "static"; however, you may create a ``pyfunc`` unit and modify its ROI or remove it.

Next, the metadata moves to the next unit, let it be a detector. The detector detects some objects and fills metadata for them, specifying labels, ROIs, etc.

Thus, the pipeline adds, removes, and changes metadata for a frame along the way. Finally, the metadata are sent to an external system.

In Savant, we support two types of metadata: the basic one provided by DeepStream (we call it DS meta or basic meta) and the extended one used only by Savant. DeepStream-based units cannot benefit from extended metadata because they are not aware of it; however, you may use ``pyfunc`` units to use it to change the workflow by analyzing extended attributes or modifying the basic metadata.

.. image:: ../_static/img/24_metadata.png

We will discuss operations on metadata in detail in :doc:`75_working_with_metadata`.

When we started this topic, we claimed that initially, the metadata was absent. It is true when vanilla DeepStream is used; with Savant, it is not always like that because Savant's protocol allows for injecting already-existing metadata from an external source.

E.g., we have an edge device where you run a Savant module that does people detection. This module is chained with a Savant module in the Datacenter which doesn't run people detectors but utilizes already-known bounding boxes to run classification without the detection step.

Detailed Metadata Workflow
--------------------------

.. _fig-2:

.. figure:: ../_static/img/24_metadata_workflow.png
   :width: 600
   :align: center

   Pic 2. Example of a pipeline execution and data flow. 'S2 – F2: Object 1' notation
   represents an object with unique id ``1`` from the frame ``2`` of the source ``2``.

Each element in the pipeline processes frame data or frame meta information, adds new or enriches existing meta information. This process is illustrated on :ref:`Figure 2 <fig-2>`.

Pipeline input is a stream of frames from one or more sources possibly accompanied by metadata from other modules.

.. note::

   Module relies on source adapters for its input. Savant includes several useful source (and sink) adapters, and it's possible to implement custom ones should the user need them.

Frames from different sources are combined into batches (the first object model configuration determines the batch size) and, together with metadata, pass sequentially through the elements of the pipeline. Each element can change the metadata, add/delete/change objects and their attributes.

In :ref:`Figure 2 <fig-2>`, the first element specified by the user is a detection model.

This detector locates objects on the frame and adds information about them to the frame metadata.
Savant uses bounding boxes of 2 types to describe locations of detected objects: axis-aligned boxes or rotated boxes.

.. note::

   Rotated boxes have to be supported by the used detection models.

In addition to location, detected objects receive labels based on a combination of the model name and object class label.

For example, the detection model in :ref:`Figure 2 <fig-2>` is named ``yolo``, and it can detect 3 classes of objects: ``car``, ``dog`` and ``cat``. Therefore, detected objects will be added to the meta information with labels ``yolo.car``, ``yolo.dog``, ``yolo.cat``. This property allows the user to specify the input :py:attr:`~savant.base.model.ModelInput.object` for any subsequent models.

Frame meta information always holds an object that corresponds to the whole frame. This object's label name is set to ``frame``. If you do not specify which objects should be used as input then the ``frame`` object will be used by default, i.e. the whole frame.

The next element of the pipeline in :ref:`Figure 2 <fig-2>` is a classification model.

Let's say, this model determines the color of a car and shouldn't process all the objects detected previously. To configure the model to work only on objects that are labeled ``yolo.car`` this label must be set in the :py:attr:`~savant.base.model.ModelInput.object` field in the configuration file. Savant will then automatically filter all objects present in metadata and only cars will be used as input for the classification model.

The classification results will be added to the ``yolo.car`` objects as additional meta-information (attribute).

In this way, as frames go through the pipeline, new objects are detected and added as metadata,
which is then being extended with various attributes.

After the frame was processed by all the elements of the pipeline the meta-information for each object in the frame is passed into ZeroMQ Sink element in the output format.

.. todo::

   Describe output format.
