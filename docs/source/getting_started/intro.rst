Introduction
============

What is it used for
-------------------

:repo-link:`Savant` is built with the goal of simplifying the common tasks encountered when building a video analytics pipeline. With Gstreamer technicalities being hidden by Savant all that's left for the user to write is code and configs specific to their video analytics application.

How is it used
--------------

Here's an example how to configure pipeline for the basic case of applying one regular Deepstream detector model:

.. code-block:: yaml

  pipeline:
    elements:
      - element: nvinfer@detector
        name: primary_detector
        output:
          objects:
            - class_id: 0
              label: car

To add a model that classifies objects detected by the primary model it's enough to simply extend `elements` section of the config:

.. code-block:: yaml

  elements:
    - element: nvinfer@detector
      name: primary_detector
      output:
        objects:
          - class_id: 0
            label: car
    - element: nvinfer@attribute_model
      name: secondary_carcolor
      input:
        objects: primary_detector.car
      output:
        attributes:
          - name: car_color

Next, to add custom user processing for pipeline results, eg count the number of detected cars of each color, it's possible to extend the pipeline with PyFunc element:

.. code-block:: yaml

  elements:
    - element: nvinfer@detector
      name: primary_detector
      output:
        objects:
          - class_id: 0
            label: car
    - element: nvinfer@attribute_model
      name: secondary_carcolor
      input:
        objects: primary_detector.car
      output:
        attributes:
          - name: car_color
    - element: pyfunc
      name: car_counter
      module: module.car_counter
      class_name: CarCounter

And the contents of ``module/car_counter.py`` Python script in this example would be:

.. code-block:: python

    from collections import Counter
    from savant.deepstream.meta.frame import NvDsFrameMeta

    counter = Counter()

    CAR_COLOR_ELEMENT_NAME = 'secondary_carcolor'
    CAR_COLOR_ATTR_NAME = 'car_color'

    class CarCounter(NvDsPyFunc):
        def process_frame_meta(self, frame_meta: NvDsFrameMeta):
            for obj_meta in frame_meta.objects:
                car_color_attr = obj_meta.get_attr_meta(CAR_COLOR_ELEMENT_NAME, CAR_COLOR_ATTR_NAME)
                counter[car_color_attr.value] += 1

Next steps
----------

Check out :doc:`installation` and :doc:`running` technicalities or go straight to :doc:`examples`.
