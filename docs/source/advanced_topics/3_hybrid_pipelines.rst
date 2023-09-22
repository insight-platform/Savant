Building Hybrid Pipelines
=========================

The section discusses how to pack multiple pipelines in a single pipeline, separating their compute spaces with custom primary ROI. In Savant, a developer can pack multiple sub-pipelines into a single pipeline, working either on all frames or conditionally based on ROI.

Unconditional Processing
------------------------

By default, the primary model analyzes the whole frame; however, under the hood, Savant creates the :doc:`default <../savant_101/25_top_level_roi>` top-level object covering the whole frame; thus, the models without specified input constraints can analyze it. It allows placing multiple primary models one after another and then their secondary models. The only requirement is non-overlapping unit names to avoid object collisions.

Normally, there is no difference how to place units if there are no cross dependencies between units; thus, the ordering is important only between elements of sub-pipelines.

.. image:: ../_static/img/3_building_hybrid_pipelines_unconditional.png

.. tip:: consider placing sub-pipelines in :doc:`element groups <2_element_group>`: it helps to develop and debug them independently.

Conditional Processing
----------------------

When you need to process frames conditionally, based on per-stream information, e.g., handle `cam-1` with a car processing sub-pipeline and `cam-2` with a person processing sub-pipeline, a developer must place a special ROI-modifying custom pyfunc before other pipeline elements.

.. image:: ../_static/img/3_building_hybrid_pipelines_conditional.png

That pyfunc must modify ROI based on ``source-id`` or other knowledge like per-frame attributes:

- when ``source-id`` is unknown it can :ref:`be deleted <delete_default_roi>` to ensure the frame is not processed;
- when ``source-id`` is known and relates to the car processing sub-pipeline, :ref:`set <create_custom_roi>` it to ``car.roi``;
- when ``source-id`` is known and relates to the person processing sub-pipeline, :ref:`set <create_custom_roi>` it to ``person.roi``;

The primary models must accept corresponding ROIs rather than work on default ROI:

.. code-block:: yaml

    ...
    element: nvinfer@detector
      name: CarDetector
      model:
        input:
          object: car.roi
    ...
    element: nvinfer@detector
      name: PersonDetector
      model:
        input:
          object: person.roi

Pros & Cons Of Hybrid Pipelines
-------------------------------

Pros:

- easier to maintain deployments;
- efficient processing;
- easier to route video streams (no stream duplication is needed);

Cons:

- more difficult to develop and troubleshoot, consider :doc:`element groups <2_element_group>`;
- increases end-to-end delay;
- more difficult to plan compute resources when real-time processing is required.
