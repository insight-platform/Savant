Pipeline
========

.. _fig-1-pipeline:

.. figure:: ../_static/img/29_pipeline.png
   :width: 600
   :align: center

   Pic 1. Video Analytics Application Scheme

A pipeline is a sequence of processing elements (units). The sequence has one input (src) and one output (sink).

:ref:`Pic 1 <fig-1-pipeline>` illustrates pipeline's place within the module: the green box represents entire pipeline and the blue box highlights a user-defined parts that are generated based on the ``pipeline.elements`` section.

Default module configuration file already defines the :py:attr:`~savant.config.schema.Pipeline.source` and :py:attr:`~savant.config.schema.Pipeline.sink` sections, so user don't need to redefine them in a practical pipeline because they are derived:

.. literalinclude:: ../../../savant/config/default.yml
  :language: YAML
  :lines: 131-

It is possible to redefine them, but the encouraged operation mode assumes the use of ZeroMQ source and sink.

When writing a module, a user normally defines only pipeline :py:attr:`~savant.config.schema.Pipeline.elements`. All supported units are listed as follows:

#. detector model;
#. rotated detector model;
#. classifier model;
#. attribute model;
#. complex model;
#. pyfunc unit;
#. DeepStream tracker unit;
#. other DeepStream plugins (except for sinks or sources).

The units are discussed in detail in the following sections.

Along with the listed units, pipeline definition may include :py:class:`~savant.config.schema.ElementGroup` nodes, which are used to introduce a condition on including the elements into the pipeline. Read more about this in the :doc:`54_conditional_unit_init` section.

Frame Processing Workflow
-------------------------

Savant's pipeline is linear. It doesn't support tree-like conditional processing. Every frame goes from the beginning of the pipeline to the end of the pipeline. However, it doesn't mean that every pipeline unit handles every object.

To get an idea of how the frame is processed, let us take a look at the following **pseudocode**, which corresponds to the Savant's logic of operation:

.. code-block:: python

    # run SSD model on a whole frame, by searching its ROI
    objects = inference.meta_filter(meta, frame_object=True)
    if objects:
        inference.run(objects, PeopleNet_detector)
        inference.filter_results(configured_conditions)
        inference.update_meta()

    # find 'person' objects and pass them to classify the gender
    objects = inference.meta_filter(meta, class='person')
    if objects:
        inference.run(objects, Gender_classifier)
        inference.filter_results(configured_conditions)
        inference.update_meta()

    # again find 'person' objects and pass them to determine their age
    objects = inference.meta_filter(meta, class='person')
    if objects:
        inference.run(objects, Age_model)
        inference.filter_results(configured_conditions)
        inference.update_meta()

    # call pyfunc and do something with meta collected
    pyfunc.run('func.name.ClassName', meta)

    ...
    ...
    # draw objects selected to be drawn
    objects = meta.filter(need_draw=True)
    if objects:
        draw_objects(objects)

So, basically every unit in the pipeline runs a unit-specific selection on metadata available. If metadata match the configured unit requirements, the unit is executed on those matched objects. Certain units like ``pyfunc`` don't filter objects beforehand.

In other words, the pipeline doesn't support branching directly, but it enables conditional call for units based on selection. Also, a developer can implement sophisticated "shadowing" hiding objects from a unit with a specially designed pyfunc. Our experience shows that such a functionality is enough to make complex pipelines without significant limitations.

Finally, the metadata and resulting frames are encoded in Savant protocol message and sent to the sink socket. This is done by the framework.
