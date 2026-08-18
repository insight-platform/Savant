Working With Metadata
=====================

Units handle two types of data when the pipeline is running: images and their corresponding metadata. Some of the metadata is read-only, some is modifiable, and some metadata can be deleted or added.

Metadata interaction varies by unit type. In the ``input`` section of a model inference unit — where the model's input data is defined — you specify which objects to process by filtering on the ``element_name`` and ``label`` metadata fields:

.. code-block:: yaml

    input:
      object: person_detector.person

In this example, we indicated that between all the objects that exist on the frame, we want to select those objects that have ``element_name==person_detector`` and ``label==person`` in their metadata.

Similarly, the ``output`` section defines how metadata is written to the model results, including attribute names. It can also apply filters to exclude specific metadata values from the output.

The Python Function unit provides full access to all metadata. Within this unit, you can implement custom processing to read frame and object metadata, modify writable metadata, and delete or add metadata as needed.

For example, you may want to remove all objects that belong to red-colored cars without an identified car plate or make the areas with license plates blurred.

Before examining the API, it is useful to understand the available metadata types. Each type has its own access rules: some metadata are read-only, some allow writing new values, some can only be extended without modification or deletion, and others permit deletion.

There are three types of metadata:

* metadata for the entire frame
* metadata for objects on the frame
* metadata for object attributes

Entire Frame Metadata
---------------------

Let us first look at the frame metadata. Note than the access restrictions for metadata are shown in square brackets:

* The ``source_id [read]`` attribute is a unique identifier of the video stream source. With this identifier, you can understand which source, the frame metadata you received belongs to. Most often, this identifier is used to be able to store some state, which must be unique for each video stream. In the TrafficMeter example, this property is used to count people in different video streams (`link <https://github.com/insight-platform/Savant/blob/documentation-initial-update/samples/traffic_meter/line_crossing.py>`__).

* The ``frame_num [read]`` attribute is the frame number of a particular source.

* The ``roi [read, write]`` attribute stores meta-information about the region of the image that serves as the default input area for the detection units, if no object is specified for the area where detection will be done.

* The ``objects_number [read]`` represents the total number of objects on the frame.

* The ``tags [read, extend]`` attribute stores auxiliary information about the frame as an extensible dictionary. These tags can hold various details; for example, standard video file adapters expose the video's relative path under the ``location`` key. If you implement a method that determines frame brightness, you might classify it as ``light``, ``regular``, or ``dark`` and record this under the ``illumination`` key for later use in the pipeline.

* The ``pts [read]`` attribute stores the presentation timestamp. This is the information from the source video stream timestamp.

* The ``duration [read]`` attribute provides the frame duration. If unavailable, it returns ``None``.

* The ``framerate [read]`` attribute records the source stream's FPS value as a string, for example ``20/1``.

The second type of metadata is object data. All metadata of this type is a single list that you can iterate over.

Per-Object Metadata
-------------------

* The ``label [read, write]`` attribute stores the object's class (e.g., car, person, flower).

* The ``track_id [read, write]`` attribute is a unique object identifier used to track objects. If ``track_id`` is equal to the max ``uint64`` value, it means the object is not tracked. This corresponds to the DeepStream's `constant <https://docs.nvidia.com/metropolis/deepstream/dev-guide/sdk-api/group__metadata__structures.html#ga23a0088be46b70720415bc25e8c85c7f>`__.

* The ``element_name [read]`` attribute is the name of the unit that added this object. If the object is a result of the detection model, ``element_name`` is the name of the unit (the name field of the unit defined in the configuration file). For user-created objects, ``element_name`` is a mandatory constructor argument.

* The ``confidence [read]`` attribute is the numeric value denoting the probability that the object of the class is specified in the label field. Typically set by a detector, in cases described in `NvDsObjectMeta <https://docs.nvidia.com/metropolis/deepstream/dev-guide/python-api/PYTHON_API/NvDsMeta/NvDsObjectMeta.html>`__ a special value ``-0.1`` is possible.

* The ``bbox [read, write]`` attribute is meta-information about the object's position on the frame. This attribute can have two types: an aligned bounding box (sides of the box are parallel to the coordinate axes) and an oriented bounding box (the box can have an angle).  The position is set by the coordinates of the center of the box, the width and height of the box, and the rotation angle if it is an oriented bounding box.

* The ``uid [read]`` attribute is a unique identifier of the box. This identifier is assigned when adding an object to the list of frame objects and does not change throughout the existence of meta-information about the object, in contrast to the ``track_id`` which can change for the object.

* The ``parent [read, write]`` attribute stores a reference to the parent object. This value can be ``None``, if there is no parent object. The parent reference can be used to associate objects with each other. For example, the model can detect a face only within the area related to the human body, which forms the relationship between the "human" and "face" objects. If a model produces both face and person detections simultaneously, these objects exist at the same hierarchy level and require manual association.

* The ``is_primary [read]`` attribute shows whether this metadata structure describes the main frame object. You can read more about the main frame object later, in the context of associating metadata to each other.

Object Attributes
-----------------

And the third type of metadata represents object attributes:

* The ``element_name [read, write]`` is the name of the unit that added this attribute. For example, if the attribute is a result of an attribute model, then ``element_name`` is the name of the unit (the ``name`` field specified in the configuration file). For attributes created manually by the user, the name of the element is a mandatory constructor argument.

* The ``name [read, write]`` is the name of the attribute. It is necessary for future access to this attribute, given that one element can add more than one attribute to the object. For the attributes created manually by the user, the attribute name is a mandatory constructor argument.

* The ``value [read, write]`` is the value of the attribute. The value can be a string, a numeric value, or an array of numeric values. For the attributes created manually by the user, the attribute's value is a mandatory constructor argument.

* The ``confidence [read, write]`` is a numeric value with the probability that the attribute for the object is true. It is usually obtained as a result of the attribute model inference. For attributes created manually by user, it is an optional argument (by default ``1.0``).

The different types of metadata are related to each other. Frame metadata allows access to an iterator on objects on that frame, and object metadata allows a list of attributes of that object.

In addition to this hierarchy, there is also a relationship between the metadata of different objects: an object can have a reference to a parent object located on the frame (the ``parent`` property).

In Savant, unlike DeepStream, objects usually have a parent, even if they are objects obtained from the detector inference on the whole frame. For the purpose of flexible application of different models (for example, if you need to specify the region of interest or skip the inference by a user condition), Savant always creates one object on the frame equal to the whole frame; the default class label of such pseudo-object is ``frame``. See also :doc:`25_top_level_roi`.

All pipeline models configured without specifying an input object receive this pseudo-object, also called the primary object, as input. Then, in the case of detectors, the resulting objects will have the ``frame`` as a parent by default.

To work with metadata, it is necessary to get a frame metadata iterator in the batch from ``Gst.Buffer``. You can see details on how to do this in the code at the `link <https://github.com/insight-platform/Savant/blob/develop/savant/deepstream/pyfunc.py#L37-L49>`__, but Savant simplifies working with GStreamer/DeepStream structures, so the Python Function unit provides a simple API described below.

Frame metadata is of type :py:class:`~savant.deepstream.meta.frame.NvDsFrameMeta`. The ``objects`` property gives access to the iterator on the meta-information of objects on that frame. For example,

.. code-block:: python

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        for obj_meta in frame_meta.objects:
            # use ObjectMeta API to process object metadata
            pass

The ``add_obj_meta`` method of frame metadata allows you to add a new object to the frame. This object will be completely similar to the objects obtained as a result of inference of detection models, i.e., it can serve as an input for subsequent processing steps in the pipeline, including other detection models, attribute models, etc.

.. code-block:: python

    def add_obj_meta(self, object_meta: ObjectMeta)

The method ``remove_obj_meta`` of frame metadata allows removing the object's metadata from the metadata list.

.. code-block:: python

    def remove_obj_meta(self, object_meta: ObjectMeta)

For example, the ``remove_obj_meta`` method can be used to disable the detector inference by some condition by removing the main frame object:

.. code-block:: python

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        primary_meta_object = None
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                primary_meta_object = obj_meta
                break
        condition = True

        if condition and primary_meta_object:
            frame_meta.remove_obj_meta(primary_meta_object)

Object metadata is of :py:class:`~savant.deepstream.meta.object.ObjectMeta` type. Initialization of a new ObjectMeta structure to describe a user object is defined as follows:

.. code-block:: python

    def __init__(
    	self,
    	element_name: str,
    	label: str,
    	bbox: Union[BBox, RBBox],
    	confidence: Optional[float] = DEFAULT_CONFIDENCE,
    	track_id: int = UNTRACKED_OBJECT_ID,
    	parent: Optional['ObjectMeta'] = None,
    	attributes: Optional[List[AttributeMeta]] = None,
    )

For the new object, be sure to specify the ``element_name`` and ``label`` attributes described above, and the ``bbox`` structure, defining the object's position on the frame.

The ``bbox`` parameter can be one of the two types described above in ``bbox``. To create an aligned ``bbox``, you must specify the coordinates of the center and the size of the bounding box, for example:

.. code-block:: python

    from savant_rs.primitives.geometry import BBox
    primary_bbox = BBox(
        xc=400,
        yc=300,
        width=200,
        height=100,
    )

To create an oriented ``bbox``, in addition to the coordinates of the center and dimensions, you also need to specify the angle of rotation, given in degrees, for example:

.. code-block:: python

    from savant_rs.primitives.geometry import RBBox
    primary_bbox = RBBox(
        xc=400,
        yc=300,
        width=200,
        height=100,
        angle=45
    )

Thus, an example of adding metadata about a new object to the frame is as follows:

.. code-block:: python

    from savant_rs.primitives.geometry import BBox
    from savant.deepstream.meta.frame import NvDsFrameMeta
    from savant.meta.object import ObjectMeta
    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        new_obj_meta = ObjectMeta(
            element_name='my_element_name',
            label='my_obj_class_label',
            bbox=BBox(
                xc=400,
                yc=300,
                width=200,
                height=100,
            ),
        )
    frame_meta.add_obj_meta(new_obj_meta)

It is not necessary to specify any parent, including the primary object, for objects added to the frame manually.

Next, let's look at the methods of working with object attributes. The methods ``get_attr_meta`` and ``get_attr_meta_list`` are defined as follows:

.. code-block:: python

    def get_attr_meta(self, element_name: str, attr_name: str) -> Optional[AttributeMeta]

    def get_attr_meta_list(self, element_name: str, attr_name: str) -> Optional[List[AttributeMeta]]

These methods return an attribute (or list of attributes in case of multi-label classification) with the specified name, created by the specified element, or ``None`` in case there is no such attribute.

The result is intended for reading only: neither the returned list nor the ``AttributeMeta`` objects in it should be modified in place. Use ``add_attr_meta``, ``replace_attr_meta_list`` and ``remove_attr_meta_list`` to change the attributes of an object.

For example, in the `nvidia_car_classification <https://github.com/insight-platform/Savant/tree/develop/samples/nvidia_car_classification>`__ sample, the attributes created by the classifiers are read in the user rendering procedure:

.. code-block:: python

    for obj_meta in frame_meta.objects:
        attr_meta = obj_meta.get_attr_meta('Secondary_CarColor', 'car_color')
        if attr_meta is not None:
            # use attr_meta.value to get attribute value

The ``add_attr_meta`` method allows adding a new attribute to an object. There is no need for a separate initialization for the metadata structure for the new attribute; all the properties described above are passed as arguments to ``add_attr_meta``.

.. code-block:: python

    def add_attr_meta(
        self,
        element_name: str,
        name: str,
        value: Any,
        confidence: float = 1.0,
    )

For example, in the `traffic_meter <https://github.com/insight-platform/Savant/tree/develop/samples/traffic_meter>`__ sample, the counters resulting from the custom processing are added to the main frame object using arbitrary strings as ``element_name`` and ``name`` attributes:

.. code-block:: python

    primary_meta_object.add_attr_meta(
        'analytics', 'entries_n', self.entry_count[frame_meta.source_id]
    )
    primary_meta_object.add_attr_meta(
        'analytics', 'exits_n', self.exit_count[frame_meta.source_id]
    )
