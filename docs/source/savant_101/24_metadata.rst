Metadata
========

Units defined within a pipeline process objects. Every object is represented by its metadata. Metadata defines an object label, location and size, identifier, and other properties. There are two kinds of metadata in Savant: primary (DeepStream-related) metadata and extended metadata.

Simply speaking, every unit can produce metadata or modify it. Let us begin with a bare image frame coming into a pipeline to show you how it happens. In the first step, Savant creates a unique "frame" object whose metadata object corresponds to the whole viewport (except paddings). This object is somehow "static"; however, you may create a ``pyfunc`` unit and modify its ROI or remove it.

Next, the metadata move to the next unit, let it be a detector. The detector detects some objects and fills metadata for them, specifying labels, ROIs, etc.

Thus, the pipeline adds, removes, and changes metadata for a frame along the way. Finally, the metadata are sent to an external system.

In Savant, we support two types of metadata: the basic one provided by DeepStream (we call it DS meta or basic meta) and the extended one used only by Savant. DeepStream-based units cannot benefit from extended metadata because they are not aware of it; however, you may use ``pyfunc`` units to use it to change the workflow by analyzing extended attributes or modifying the basic metadata.

.. image:: ../_static/img/24_metadata.png

We will discuss operations on metadata in detail in :doc:`65_extended_metadata`.

When we started this topic, we claimed that initially, the metadata was absent. It is true when vanilla DeepStream is used; with Savant, it is not always like that because Savant's protocol allows for injecting already-existing metadata from outer space.

E.g., we have an edge device where you run a Savant module that does people detection. This module is chained with a Savant module in the Datacenter which doesn't run people detectors but utilizes already-known bounding boxes to run classification without the detection step.
