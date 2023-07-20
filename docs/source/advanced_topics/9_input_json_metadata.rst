Using of object metadata files in Image and Video source adapters
=================================================================

:ref:`ifsa Image File Source Adapter` and :ref:`vfsa Video File Source Adapter`  support sending metadata
along with frames. The READ_METADATA option must be specified when starting the adapters.

The input data must have a certain structure and format, so that the adapter can
correctly read and match images or video files with metadata in json format.

Data structure:

* json file names, image or video names must be identical.
* The files must be in the same directory

Example:

.. code-block:: yaml

    /input_data/0000000000000785.jpg
    /input_data/0000000000000785.json

Or

.. code-block:: yaml

    /input_data/street_people.json
    /input_data/street_people.mp4

As input meta-information can be used:

* json files generated with sink adapters
* prepare json files with metainformation in accordance with the format requirements.

Requirements for json files:

General requirements:

* In the file, each individual line must be a json format record for one image or frame
* For images there should be only 1 line in the file. For video files, the number of lines should be equal to the number of frames in the video file.

The JSON format:

Each json record in the file must be of the following format.

.. code-block:: json

    {
        "metadata":
        {
            "objects": []
        }
    }

- ``metadata`` - mandatory key
- ``objects`` - list of objects on the frame. If there are no objects on the frame or image, the list should be empty.

Each item in the ``objects`` list must satisfy the format and data type of the AVRO specification
schema for the metadata of the object `link <https://docs.savant-ai.io/reference/avro.html#object-schema>`__.

Example of metadata for one object

.. code-block:: json

    {
        "model_name": "coco",
        "label": "person",
        "object_id": 1,
        "bbox":
        {
            "xc": 390.14,
            "yc": 218.07,
            "width": 218.7,
            "height": 346.68,
            "angle": 0.0
        },
        "confidence": 1,
        "attributes": [],
        "parent_model_name": null,
        "parent_label": null,
        "parent_object_id": null
    }

- ``model_name`` - name of the model that created this object. If you're converting some data you can specify any name you want;
- ``label`` - object label;
- ``object_id`` - unique object identifier within one frame or unique object track number;
- ``bbox`` - bbox coordinates and angle of the object.
- ``confidence`` - object confidence
- ``attributes`` - list of object attributes. The list of attributes can be empty. Each attribute must correspond to the AVRO schema (`link <https://docs.savant-ai.io/reference/avro.html#attribute-schema>`__)
- ``parent_model_name`` - name of the model that created the parent object. If you're converting some data you can specify any name you want;
- ``parent_label`` - parent object label;
- ``parent_object_id`` - unique object identifier within one frame or unique object track number.

If you specify a parent object, it must necessarily be in the list of objects.

Example of attribute setting:

.. code-block:: json

    {
        "element_name": "age_model",
        "name": "age",
        "value": 69,
        "confidence": 0.9
    }

- ``element_name`` - name of the element that created this attribute. If you are converting some data, you can set any name you want.
- ``name`` - attribute name
- ``value`` - attribute value
- ``confidence`` - attribute confidence

A complete example json file with metadata for an image file:

.. code-block:: json

    {
        "metadata":
        {
            "objects":
            [
                {
                    "model_name": "coco",
                    "label": "person",
                    "object_id": 1,
                    "bbox":
                    {
                        "xc": 390.14,
                        "yc": 218.07,
                        "width": 218.7,
                        "height": 346.68,
                        "angle": 0.0
                    },
                    "confidence": 1,
                    "attributes": [
                        {
                            "element_name": "age_model",
                            "name": "age",
                            "value": 69,
                            "confidence": 0.9
                        }
                    ],
                    "parent_model_name": null,
                    "parent_label": null,
                    "parent_object_id": null
                }
            ]
        }
    }


A complete example json file with metadata for an video file with two frames:

.. code-block:: json

    {
        "metadata":
        {
            "objects":
            [
                {
                    "model_name": "yolov8",
                    "label": "person",
                    "object_id": 1,
                    "bbox":
                    {
                        "xc": 390.14,
                        "yc": 218.07,
                        "width": 218.7,
                        "height": 346.68,
                        "angle": 0.0
                    },
                    "confidence": 0.99,
                    "attributes": [
                        {
                            "element_name": "age_model",
                            "name": "age",
                            "value": 69,
                            "confidence": 0.9
                        }
                    ],
                    "parent_model_name": null,
                    "parent_label": null,
                    "parent_object_id": null
                }
            ]
        }
    }
    {
            "metadata":
            {
                "objects":
                [
                    {
                        "model_name": "yolov8",
                        "label": "person",
                        "object_id": 1,
                        "bbox":
                        {
                            "xc": 393.14,
                            "yc": 219.07,
                            "width": 218.7,
                            "height": 346.68,
                            "angle": 0.0
                        },
                        "confidence": 0.99,
                        "attributes": [
                            {
                                "element_name": "age_model",
                                "name": "age",
                                "value": 68,
                                "confidence": 0.93
                            }
                        ],
                        "parent_model_name": null,
                        "parent_label": null,
                        "parent_object_id": null
                    }
                ]
            }
        }
