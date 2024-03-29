Source JSON Injection
=====================

.. warning::

    Subject to change. We recommend using :doc:`Client SDK </advanced_topics/10_client_sdk>` instead.

:ref:`savant_101/10_adapters:Image File Source Adapter` and :ref:`savant_101/10_adapters:Video File Source Adapter` support sending metadata along with frames. The ``READ_METADATA`` option must be specified when starting the adapters.

The input data must have a certain structure and format, so that the adapter can correctly read and match images or video files with metadata in json format.

File structure
^^^^^^^^^^^^^^

JSON metadata must follow the following requirements:

* file names must be identical to corresponding images/videos;
* the directory must be the same.

Example:

.. code-block::

    ./input_data/0000000000000785.json
    ./input_data/0000000000000785.jpg

Or

.. code-block::

    ./input_data/street_people.json
    ./input_data/street_people.mp4

Metadata files can be generated by the Savant sink adapters; these files must be used together with the generated video recorded by the sink adapter.

Manually by the user. JSON files must be prepared to follow the format requirements. The Video files must be without b-frames or you must sort rows in JSON file with correct frame ordering considering PTS/DTS.

JSON Files Requirements
^^^^^^^^^^^^^^^^^^^^^^^

General requirements:

The JSON file must be in `Newline-Delimited_JSON <https://en.wikipedia.org/wiki/JSON_streaming#Newline-Delimited_JSON>`_ format or Pretty-Print Newline-Delimited JSON format. The number of records must be equal to the number of frames in a file (for images - 1 record).

Each JSON record must be of the following format.

.. code-block:: json

    {
        "metadata":
        {
            "objects": []
        }
    }

- ``metadata``: a mandatory key, ``{}``;
- ``metadata.objects``: a list of objects in the frame; If there are no objects in the frame or image, the list should be empty.

Each item in the ``objects`` list must satisfy the defined format.

Example of metadata for one object:

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

- ``model_name``: a semantically meaningful name of the model created this object;
- ``label``: the object label;
- ``object_id``: the unique object identifier within a frame or unique object track number;
- ``bbox``: bbox coordinates and angle of the object;
- ``confidence``: the object confidence;
- ``attributes``: the list of the object attributes; the list can be empty, each attribute must be in the defined format;
- ``parent_model_name``: a semantically meaningful name of the model created the parent object;
- ``parent_label``: the parent object label;
- ``parent_object_id``: the unique object identifier within a frame or unique object track number.

If you specify a parent object, it must be in the list of objects.

Example of attribute specification:

.. code-block:: json

    {
        "element_name": "age_model",
        "name": "age",
        "value": 69,
        "confidence": 0.9
    }

- ``element_name``: a semantically meaningful name of the element created the attribute;
- ``name``: the attribute name;
- ``value``: the attribute value;
- ``confidence``: the attribute confidence.

A complete example JSON with metadata for a single frame:

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


A complete example JSON file with metadata for a video with two frames:

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
