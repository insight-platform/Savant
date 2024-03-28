Draw Function Usage & Customization
===================================

Often, the pipeline must display metadata on frames. The developer may utilize OpenCV CUDA functionality within a custom ``pyfunc`` to do that. However, Savant provides a special facility called ``draw_func`` that is designed for displaying overlay information on frames.

There are two ways to use ``draw_func``:

- predefined ``draw_func``, configurable in a declarative manner;
- custom ``draw_func`` implementation in Python.

Declarative Configuration
-------------------------

The predefined ``draw_func`` supports many of operations required. Usually, it is enough for pipelines when there is no need to display advanced graphics.

To set the parameters for the ``draw_func`` operation, the general parameters section is used, since the rendering step always occurs in a fixed place: after the completion of all elements of the pipeline, before passing frames and metadata to external consumers.

.. code-block:: yaml

    parameters:
      draw_func:
        module: draw_func.module
        class_name: CustomDrawFuncClass
        kwargs: {}

In the example above, a custom ``draw_func`` implementation is used. In simple cases you may use a predefined ``draw_func`` implementation.

.. note::

    To disable ``draw_func`` functionality, remove ``parameters.draw_func`` from the manifest completely.

It is important to note that rendering is not performed if no frame encoding scheme is configured for the pipeline. To set the frame coding scheme, you must specify the codec in the ``parameters.output_frame.codec`` parameter to one of the following values: ``jpeg``, ``h264``, ``hevc``, ``raw-rgba``, ``raw-rgb24``.

.. code-block:: yaml

    parameters:
      output_frame:
        codec: jpeg

Savant has a default implementation of ``draw_func`` that is used when ``draw_func`` parameter is set to an empty dictionary:

.. code-block:: yaml

    parameters:
      draw_func: {}

Another way to use the default implementation is to populate the ``rendered_objects`` section, which the default implementation uses to define the objects of which classes are going to be rendered and their render specifications. The ``rendered_objects`` section is a dictionary of the following structure:

.. code-block:: yaml

    parameters:
      draw_func:
        rendered_objects:
          <unit_name>:
            <class_label>:
              bbox:
                backgound_color: <color_str>
                border_color: <color_str>
                thickness: <int>
                padding:
                  left: <int>
                  top: <int>
                  right: <int>
                  bottom: <int>
              label:
                background_color: <color_str>
                border_color: <color_str>
                font_color: <color_str>
                font_scale: <float>
                thickness: <int>
                format:
                  - "Label: {label}"
                  - "Confidence: {confidence:.2f}"
                  - "Track ID: {track_id}"
                  - "Model: {model}"
                padding:
                  left: <int>
                  top: <int>
                  right: <int>
                  bottom: <int>
                position:
                  position: TopLeftInside / TopLeftOutside / Center
                  margin_x: <int>
                  margin_y: <int>
              central_dot:
                color: <color_str>
                radius: <int>
              blur: <true/false>

where:

* ``<unit_name>`` the name of the unit defining the objects;
* ``<class_label>`` the label of the object class set by a detector, draw label is used in place of the class label if it is set by the user;
* ``<color_str>`` color used to draw the specified element, color is defined as a RGBA hex string (without the '#' as it marks a comment in YAML), e.g. ``"00ff00ff"`` for green;

Any of the elements in the render specification (``bbox``, ``label``, ``central_dot``, ``blur``) can be omitted, if the corresponding element is not required to be rendered. Blur is ``false`` by default.

Label format is defined as a list of strings, where each string is a format string that can contain the following placeholders: ``{label}``, ``{confidence}``, ``{track_id}``, ``{model}``. Each string in the list is rendered on a separate line. The 4 line config above is provided as an example.

Customization
-------------

Besides the standard ``draw_func``, it is also possible to use a custom draw function. In this case, the function must inherit the :py:class:`~savant.deepstream.drawfunc.NvDsDrawFunc` class, overriding the ``draw_on_frame`` method or ``override_draw_spec`` method in it.

.. code-block:: python

    class CustomFunc(NvDsDrawFunc):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # todo

        def override_draw_spec(
            self, object_meta: ObjectMeta, specification: ObjectDraw
        ) -> ObjectDraw:
            # todo
            return specification

        def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
            # todo

In the ``draw_on_frame`` method, by processing meta-information, you can select the objects of interest to the user and, using the values of various object properties (class, coordinates, track id), add graphics to the frame through the methods of the :py:class:`~savant.utils.artist.artist_gpumat.Artist` object.

The ``override_draw_spec`` method is a simpler way to customize drawing of objects. It allows overriding the configured drawing specification for a given object. The method receives the object meta and the default drawing specification and returns the changed drawing specification. The returned drawing specification is then used to draw the object. There's no need to learn the :py:class:`~savant.utils.artist.artist_gpumat.Artist` object API to use this method.

The ``draw_func`` feature uses :py:class:`~savant.utils.artist.artist_gpumat.Artist` object which implements displaying a number of primitives like text labels, bounding boxes, etc. The :py:class:`~savant.utils.artist.artist_gpumat.Artist` object also can be used directly from any ``pyfunc``, but in current section we discuss the use of the ``draw_func`` function.

Artist Methods
--------------

Add_text Method
^^^^^^^^^^^^^^^

The ``add_text`` method allows you to add text to the frame, with a given value, position, text color and background color:

.. code-block:: python

    def add_text(
        self,
        text: str,
        anchor: Tuple[int, int],
        font_scale: float = 0.5,
        font_thickness: int = 1,
        font_color: Tuple[int, int, int, int] = (255, 255, 255, 255),  # white
        border_width: int = 0,
        border_color: Tuple[int, int, int, int] = (255, 0, 0, 255),  # red
        bg_color: Optional[Tuple[int, int, int, int]] = (0, 0, 0, 255),  # black
        padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
        anchor_point_type: Position = Position.CENTER,
    ) -> int:

For example, such a call will add white text on a black background to the upper left corner of detected objects with the name of the object class.

.. code-block:: python

    for obj_meta in frame_meta.objects:
        artist.add_text(
            text=obj_meta.label,
            anchor=(int(obj_meta.bbox.left), int(obj_meta.bbox.top)),
            anchor_point=Position.LEFT_TOP,
        )

Add_bbox Method
^^^^^^^^^^^^^^^

The ``add_bbox`` method allows you to add a frame to the frame with specified coordinates, thickness, frame color, and background color inside the frame.

.. code-block:: python

    def add_bbox(
        self,
        bbox: Union[BBox, RBBox],
        border_width: int = 3,
        border_color: Tuple[int, int, int, int] = (0, 255, 0, 255),  # RGBA, Green
        bg_color: Optional[Tuple[int, int, int, int]] = None,  # RGBA
        padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ):

For example, the following call will add a green border around each detected object.

.. code-block:: python

    for obj_meta in frame_meta.objects:
        artist.add_bbox(obj_meta.bbox)

Add_rounded_rect Method
^^^^^^^^^^^^^^^^^^^^^^^

The ``add_rounded_rect`` method allows you to add a rectangle with rounded corners of the specified color to the frame.

.. code-block:: python

    def add_rounded_rect(
        self,
        bbox: BBox,
        radius: int,
        bg_color: Tuple[int, int, int, int],  # RGBA
    ):

For example, the following call will add a blue rounded square with a width and height of ``100`` px in the top left corner of the frame.

.. code-block:: python

    from savant_rs.primitives.geometry import BBox


    artist.add_rounded_rect(
        bbox=BBox(50, 50, 100, 100),
        radius=4,
        bg_color=(0, 0, 255, 255),
    )

Add_circle Method
^^^^^^^^^^^^^^^^^

The ``add_circle`` method allows you to add a circle to the frame with the given coordinates, radius, and color.

.. code-block:: python

    def add_circle(
        self,
        center: Tuple[int, int],
        radius: int,
        color: Tuple[int, int, int, int],
        thickness: int,
        line_type: int = cv2.LINE_AA,
    ):

For example, the following call will add a red round bullet of radius 3 to the center of each object:

.. code-block:: python

    import cv2


    for obj_meta in frame_meta.objects:
        center = round(obj_meta.bbox.xc), round(obj_meta.bbox.yc)
        artist.add_circle(center, 3, (255, 0, 0, 255), cv2.FILLED)

Add_polygon Method
^^^^^^^^^^^^^^^^^^

The ``add_polygon`` method allows you to add an arbitrary polygon to the frame, defined by a sequence of points, with a specified outline thickness, outline color, and background color.

.. code-block:: python

    def add_polygon(
        self,
        vertices: List[Tuple[int, int]],
        line_width: int = 3,
        line_color: Tuple[int, int, int, int] = (255, 0, 0, 255),  # RGBA, Red
        bg_color: Optional[Tuple[int, int, int, int]] = None,  # RGBA
    ):

For example, the following call will add a red line segment to the frame between two points with given coordinates.

.. code-block:: python

    pt1 = (100, 100)
    pt2 = (200, 200)
    artist.add_polygon([pt1, pt2])

Add_graphic Method
^^^^^^^^^^^^^^^^^^

The ``add_graphic`` method allows you to add an arbitrary sprite to the frame, previously loaded in OpenCV CUDA GpuMat, at a given position defined by the coordinates of the upper left corner.

.. code-block:: python

    def add_graphic(self, img: cv2.cuda.GpuMat, origin: Tuple[int, int])

For example, the following call will add to the frame an image read from a file at the given path, with the upper left corner of the image placed in the upper left corner of the frame.

.. code-block:: python

    import cv2

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img , cv2.COLOR_BGRA2RGBA)
    img = cv2.cuda.GpuMat(img)
    artist.add_graphic(img , (0, 0))

Blur Method
^^^^^^^^^^^

The ``blur`` method allows you to apply Gaussian blur to a given area of the frame with the ability to set the standard deviation value.

.. code-block:: python

    def blur(
        self,
        bbox: BBox,
        padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
        sigma: Optional[float] = None,
    ):

For example, the following call will apply a blur to the objects detected on the frame, while the sigma for each object will be calculated automatically based on its size.

.. code-block:: python

    for obj_meta in frame_meta.objects:
        artist.blur(obj_meta.bbox)
