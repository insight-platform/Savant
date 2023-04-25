Complex Model Unit
==================

The complex model unit is used for inferring complex models that perform both detection and determine model attributes simultaneously, i.e.,

.. code-block:: text

    complex model = detection model + attribute model


This element combines both the detection and attribute model units. Below is an example of defining such a unit for inferring a model that detects faces and simultaneously finds facial keypoints.

.. code-block:: yaml

    - element: nvinfer@complex_model
      name: face_detector
      model:
        format: onnx
        onnx-file: retinaface_resnet50.onnx
        batch-size: 16
        precision: fp16
        input:
          object: person_detector.person
          shape: [3, 192, 192]
          offsets: [104.0, 117.0, 123.0]
        output:
          layer_names: ['bboxes', 'scores', 'landmarks']
          converter:
            module: customer_analysis.retinaface_converter
            class_name: RetinafaceConverter
          objects:
            - class_id: 0
              label: face
              selector:
                module: savant.selector
                class_name: BBoxSelector
                kwargs:
                  confidence_threshold: 0.991
                  nms_iou_threshold: 0.4
                  min_height: 70
                  min_width: 90
          attributes:
            - name: landmarks


We will not describe the parameters for the input section, as they are similar to those described in :doc:`30_dm`. The output section is of particular interest, we specify both the ``objects`` section (described in the :doc:`30_dm`) and the ``attributes`` section (described in the :doc:`43_am`).

The converter must be implemented by specifying :py:class:`~savant.deepstream.nvinfer.model.BaseComplexModelOutputConverter` as the parent class. The converter for this example is provided below.

.. code-block:: python

    class RetinafaceConverter(BaseComplexModelOutputConverter):
        def __call__(
            self,
            *output_layers: np.ndarray,
            model: ComplexModel,
            roi: Tuple[float, float, float, float]
        ) -> Tuple[np.ndarray, List[List[Tuple[Any, float]]]]:
            """Converts raw model output tensors to savant format.

            :param output_layers: Model output layer tensors
            :param model: Complex model, required parameters: input tensor shape, maintain_aspect_ratio flag
            :param roi_width: width of the rectangle on which the model infers
            :param roi_height: height of the rectangle on which the model infers
            :return: BBox tensor BBox tensor (class_id, confidence, xc, yc, width, height, [angle])
                offset by roi upper left and scaled by roi width and height,
                and list of attributes values with confidences
            """

            bboxes, scores, landmarks = detector_decoder(
                roi,
                *output_layers,  # bboxes  # scores # landmarks
            )

            bbox_tensor = np.concatenate(
                (
                    np.zeros((len(bboxes), 1)),
                    scores.reshape(-1, 1),
                    bboxes,
                ),
                axis=1,
            )

            attrs = [[(model.output.attributes[0].name, x.tolist(), None)] for x in landmarks]
            return bbox_tensor, attrs


The model used in the example has three outputs. Two are related to detections, and the third returns the coordinates of the facial keypoints for the detected face. The converter processes the first two outputs with the names ``bboxes`` and ``scores`` to obtain the boxes, while the third output with the name ``landmarks`` returns the keypoints, which are returned as attributes for each detected object. Note that the number of boxes and the length of the attribute list for each box must match.

The ``detector_decoder`` is a separate function specifically written to process the outputs of the RetinaNet model and is not provided here, as it does not affect the overall understanding of the principles of writing converters.


