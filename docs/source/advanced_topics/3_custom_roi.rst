ROI Customization
=================

By default, Savant creates the automatic default ROI, which covers all the frame space. Sometimes developers need to reconfigure the ROI globally or per frame. It can be done with custom ``pyfunc`` units.

Create Custom ROI
-----------------

:ref:`savant_101/25_top_level_roi:Top-Level ROI` is an object on a frame with a predefined label. If you want to create an additional custom ROI for models, you need to create an object with the custom `BBox <https://insight-platform.github.io/savant-rs/modules/savant_rs/primitives_geometry.html#savant_rs.primitives.geometry.BBox>`__.

In order to do this, you need to implement a custom :ref:`savant_101/70_python:Python Function Unit`. In pyfunc you need to create an :py:class:`savant.meta.object.ObjectMeta` with a `BBox <https://insight-platform.github.io/savant-rs/modules/savant_rs/primitives_geometry.html#savant_rs.primitives.geometry.BBox>`__. This BBox should have the coordinates and size of custom ROI. Also you have to set the element name and label. Then you need to add this object to the frame.

**Example of creating a custom ROI**

The custom ROI is created for the left half of the frame.

.. code-block:: python

    from savant_rs.primitives.geometry import BBox

    from savant.deepstream.meta.frame import NvDsFrameMeta
    from savant.deepstream.pyfunc import NvDsPyFuncPlugin
    from savant.gstreamer import Gst
    from savant.meta.object import ObjectMeta


    class CreateCustomROI(NvDsPyFuncPlugin):
        def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):

            custom_roi_bbox = BBox(
                frame_meta.video_frame.width // 4,
                frame_meta.video_frame.height // 2,
                frame_meta.video_frame.width // 2,
                frame_meta.video_frame.height,
            )

            object_meta = ObjectMeta(
                element_name="custom_roi",
                label="left_half",
                bbox=custom_roi_bbox
            )

            frame_meta.add_obj_meta(object_meta)

The configuration file (module.yml) from Car Detection and Classification `demo <https://github.com/insight-platform/Savant/tree/develop/samples/nvidia_car_classification>`__ is taken as base for example. The pyfunc element is added before the detection model in the configuration file to create the custom ROI with the element name ``custom_roi`` and the label ``left_half`` on each frame. Then the custom ROI is set as an input object for detection model element using element_name and label.

.. code-block:: YAML

    ...
    pipeline:
      elements:
        - element: pyfunc
          module: samples.nvidia_car_classification.custom_roi
          class_name: CreateCustomROI

        # detector
        - element: nvinfer@detector
          name: Primary_Detector
          model:
            format: caffe
            remote:
              url: s3://savant-data/models/Primary_Detector/Primary_Detector.zip
              checksum_url: s3://savant-data/models/Primary_Detector/Primary_Detector.md5
              parameters:
                endpoint: https://eu-central-1.linodeobjects.com
            model_file: resnet10.caffemodel
            batch_size: 1
            precision: int8
            int8_calib_file: cal_trt.bin
            label_file: labels.txt
            input:
              object: custom_roi.left_half
              scale_factor: 0.0039215697906911373
            output:
              num_detected_classes: 4
              layer_names: [conv2d_bbox, conv2d_cov/Sigmoid]
              objects:
                - class_id: 0
                  label: Car
                - class_id: 2
                  label: Person

    ...

Change Default ROI
------------------

You can also create a custom ROI by modifying the default ROI. In this case you do not create an additional object.

.. note::
    Be careful, because in this case all model elements for which no input objects are specified will make inference on the modified ROI.

**Example of changing the default ROI**

A new default ROI is created for the left half of the frame.

.. code-block:: python

    from savant_rs.primitives.geometry import BBox

    from savant.deepstream.meta.frame import NvDsFrameMeta
    from savant.deepstream.pyfunc import NvDsPyFuncPlugin
    from savant.gstreamer import Gst
    from savant.meta.object import ObjectMeta

    class ChangeROI(NvDsPyFuncPlugin):
        def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):

            frame_meta.roi = BBox(
                frame_meta.video_frame.width // 4,
                frame_meta.video_frame.height // 2,
                frame_meta.video_frame.width // 2,
                frame_meta.video_frame.height,
            )

The configuration file (module.yml) from Car Detection and Classification `demo <https://github.com/insight-platform/Savant/tree/develop/samples/nvidia_car_classification>`__ is taken as base for example. In the configuration file, a pyfunc element is added before the detection model to change the default ROI

.. code-block:: YAML

    ...
    pipeline:
      elements:
        - element: pyfunc
          module: samples.nvidia_car_classification.custom_roi
          class_name: ChangeROI

        # detector
        - element: nvinfer@detector
          name: Primary_Detector
          model:
            format: caffe
            remote:
              url: s3://savant-data/models/Primary_Detector/Primary_Detector.zip
              checksum_url: s3://savant-data/models/Primary_Detector/Primary_Detector.md5
              parameters:
                endpoint: https://eu-central-1.linodeobjects.com
            model_file: resnet10.caffemodel
            batch_size: 1
            precision: int8
            int8_calib_file: cal_trt.bin
            label_file: labels.txt
            input:
              scale_factor: 0.0039215697906911373
            output:
              num_detected_classes: 4
              layer_names: [conv2d_bbox, conv2d_cov/Sigmoid]
              objects:
                - class_id: 0
                  label: Car
                - class_id: 2
                  label: Person
    ...


Delete Default ROI
------------------

An example of how to delete the default ROI object from a frame is demonstrated in the Traffic Meter `demo <https://github.com/insight-platform/Savant/tree/develop/samples/traffic_meter>`__. In the demo, when the lines are not configured for a source, the default ROI is removed from a frame.

.. literalinclude:: ../../../samples/traffic_meter/line_crossing.py
  :language: YAML
  :lines: 12-35

