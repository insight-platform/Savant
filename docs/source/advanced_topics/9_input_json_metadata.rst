Source adapters with metadata files
===================================

:ref:`savant_101/10_adapters:Image File Source Adapter` and :ref:`savant_101/10_adapters:Video File Source Adapter` support sending metadata
along with frames. The READ_METADATA option must be specified when starting the adapters.

The input data must have a certain structure and format, so that the adapter can
correctly read and match images or video files with metadata in json format.

File structure
^^^^^^^^^^^^^^

The location and naming of files must meet the following requirements:

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

Metadata files can be generated:

* By the Savant sink adapters. These files should be used together with the generated video recorded by the sink adapter.
* Manually by the user. Json files should be prepared accordance to the format requirements. Video files should be without b-frames or you should prepare a json file with correct frame order before decoding. Otherwise there will be incorrect mapping of metadata and frames.

Requirements for json files
^^^^^^^^^^^^^^^^^^^^^^^^^^^

General requirements:

* The JSON file must be in `Newline-Delimited_JSON <https://en.wikipedia.org/wiki/JSON_streaming#Newline-Delimited_JSON>`_ format or Pretty-Print Newline-Delimited JSON format
* For images there should be only with 1 records. For video files, the number of records should be equal to the number of frames in the video file.

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

Each item in the ``objects`` list must satisfy the specified format.

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
- ``attributes`` - list of object attributes. The list of attributes can be empty. Each attribute must correspond to the specified format;
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


A complete example json file with metadata for a video file with two frames:

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
