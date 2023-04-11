Data Processing
===============

Savant's pipeline is linear. It doesn't support tree-like conditional processing. Every frame goes from the beginning of the pipeline to the end of the pipeline. However, it doesn't mean that every pipeline block handles every object.

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


So, basically every unit in the pipeline runs a unit-specific selection on metadata available. If metadata match the configured unit requirements, the unit is executed on those matched objects. Certain units like ``pyfunc`` doesn't filter objects beforehand.

Other words, the pipeline doesn't support branching directly, but it enables conditional call for units based on selection. Also, a developer can implement sophisticated "shadowing" hiding objects from a unit with a specially designed pyfunc. Our experience shows that such a functionality is enough to craft complex pipelines without significant limitations.

Finally, the metadata and resulting frames are encoded in Savant protocol message and sent to the sink socket. This is done by the framework.
