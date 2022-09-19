Model elements
==============

Model elements are the heart of any Savant pipeline, where the model inference happens.

Currently, the only supported inference backend is NVIDIA Deepstream.

Model configuration
-------------------

All the model-specific configuration parameters are defined in the model configuration section of each :py:class:`~savant.config.schema.ModelElement` in the module config file.

Deepstream models
^^^^^^^^^^^^^^^^^

Deepstream model configuration in :repo-link:`Savant` is based on
`nvinfer model configuration <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html#id2>`_
and most of model config parameters can still be provided by specifiying the nvinfer model
:py:attr:`~savant.deepstream.nvinfer.model.NvInferModel.config_file` in Savant model configuration. But it may be more convenient
to configure the model entirely through Savant config.

Look for detailed config specification for Savant-supported Deepstream model types in API refrence:

#. ``- element: nvinfer@detector``

   definition in :py:class:`~savant.deepstream.nvinfer.model.NvInferDetector`

#. ``- element: nvinfer@rotated_object_detector``

   definition in :py:class:`~savant.deepstream.nvinfer.model.NvInferRotatedObjectDetector`

#. ``- element: nvinfer@attribute_model`` or ``- element: nvinfer@classifier``

   definition in :py:class:`~savant.deepstream.nvinfer.model.NvInferAttributeModel`

#. ``- element: nvinfer@complex_model``

   definition in :py:class:`~savant.deepstream.nvinfer.model.NvInferComplexModel`

.. note::

   Deepstream inference backend requires model to be in one of the following formats:

   #. ONNX
   #. UFF
   #. Caffe
   #. Nvidia TAO toolkit
   #. Custom CUDA engine

Preprocessing
-------------

Preprocessing prepares the input data before calling the model. Preprocessing produces
images based on object metadata and these images are used as inputs for the model.
Any image preprocessing always includes:

#. Scaling the object image to the input size of the model
#. Normalizing the image using the mean and standard deviation calculated on training data.

These preprocessing steps are always performed automatically in :repo-link:`Savant`.
The parameters are set based on the model's configuration file.

The user can also implement his own preprocessing function which allows to prepare data
for the model according to the requirements of the selected model. You should use the configuration
section :py:class:`~savant.base.model.ModelInput` to enable user preprocessing. There are two types of preprocessing:

* Meta preprocessing of object information (specified in the :py:attr:`~savant.base.model.ModelInput.preprocess_object_meta` attribute).
* Image preprocessing (specified in the :py:attr:`~savant.base.model.ModelInput.preprocess_object_tensor` attribute).

Meta preprocessing
^^^^^^^^^^^^^^^^^^

Preprocessing object meta information allows to change the object's meta information
for current model only (subsequent models will still use meta unchanged by the current
model preprocessing). For example, the input for a classification model can only include
the top part of the object's bounding box. This method of preprocessing is preferable
because it gives higher performance.

.. note::

   There are some predefined preprocessing functions in Savant
   (:ref:`reference/api/input_preproc:Input preprocessing`)
   and additionaly user-defined functions are easy to implement.

A custom preprocessing should be implemented as a subclass of
:py:class:`~savant.base.input_preproc.BasePreprocessObjectMeta`.
The user must implement one required method ``__call__``

.. code-block:: python

   from savant.base.input_preproc import BasePreprocessObjectMeta

   class CustomPreprocessObjectMeta(BasePreprocessObjectMeta):
       def __call__(self, bbox: pyds.NvBbox_Coords, **kwargs) -> pyds.NvBbox_Coords:
           # custom preprocessing code
           return bbox

Image preprocessing
^^^^^^^^^^^^^^^^^^^

Object image preprocessing is a function that processes the object image. For example,
you can cut out ⅓ of the top of an object and ⅓ of the bottom of an object
and combine them into an image that will be used as a model's input.
The function can be implemented either in Python or C/C++.

This method provides higher flexibility for input preprocessing since it allows you
to work directly with the image itself.

For example, this preprocessing will be useful in the following case: after a person's face has been detected,
it may be required to align it to eliminate distortion and build a more stable identifier.

In order to implement custom preprocessing user needs to write a function that would implement
the following signature:

Python

.. code-block:: python

   def function_name(image: pysavantboost.Image) -> pysavantboost.Image:

or

C++

.. code-block:: c

   savantboost::Image* function_name(savantboost::Image* image);

Postprocessing
--------------

After the model has returned its results, it is required to post-process and convert them
into Savant framework format. Currently, :repo-link:`Savant` supports two types of models: detection models
and attribute models (e.g. classification). Models can further be divided into two types
according to the output format: regular models (Deepstream-native models) and custom models.

A custom model outputs are required to be converted into the Savant format
in order to use this model's results in subsequent pipeline elements. To do this, user needs to implement a **converter**.

.. note::

   Converters are called and executed immediately after a custom model returns its tensor result.

**Detection model**

A custom converter should be implemented as a subclass
of :py:class:`~savant.base.converter.BaseObjectModelOutputConverter`.
The ``__call__`` function of the converter class should return a two-dimensional array
with **class_id**, **confidence**, **xc**, **yc**, **width**, **height** and **angle** values in each row, where:

* **class_id** – numeric class ID
* **confidence** – object's confidence score
* **xc**, **yc** – absolute coordinates of the box center
* **width**, **height** – width and height of the box, respectively
* **angle** – box rotation angle. This value is optional, if the detection model does not
  support rotated boxes then it can be omitted.

Postprocessing for detection models also includes a filtering step.
For example, if a detection model was trained to detect 80 types of objects,
but to solve the current problem you only need objects of a select few types,
then there is no need to return all the objects found.
You can specify which object classes get added to metadata in
the :py:attr:`~savant.base.model.ObjectModelOutput.objects` section.
Limiting the amount of objects in metadata is useful because extra information can
reduce the speed of the entire pipeline.

Additionally, the user can implement the **selector** function, which will allow him to more
accurately filter the necessary objects, for example apply detection confidence filter or nms.
A custom selector function should implement as
a subclass of :py:class:`~savant.base.selector.BaseSelector` with one required method ``__call__``.

**Attribute model**

A converter for an attribute model should be implemented as a subclass
of :py:class:`~savant.base.converter.BaseAttributeModelOutputConverter`.
Check the class decscription for details.

**Complex model**

Complex model converter returns a combination of the object model converter output
and attribute model converter BaseObjectModelOutputConverter, implemenent a subclass of
:py:class:`~savant.base.converter.BaseComplexModelOutputConverter`, check the class description for details.

PyFunc elements
---------------

You can add a custom Python class as one of the elements in the pipeline.
Usually it is enough to overload a ``process_frame`` method of the :py:class:`~savant.deepstream.pyfunc.NvDsPyFuncPlugin`

.. code-block:: python

   import numpy as np
   from savant.deepstream.meta.frame import NvDsFrameMeta
   from savant.deepstream.pyfunc import NvDsPyFuncPlugin

   class PyClass(NvDsPyFuncPlugin):
      def process_frame(self, frame_meta: NvDsFrameMeta, frame: np.ndarray):
         ...

User can get access to all meta information at the lowest level and image data through the function parameters.

Also you can access the raw ``Gst.Buffer`` with ``process_buffer`` method.
See base class :py:class:`~savant.deepstream.pyfunc.NvDsPyFuncPlugin` for details.

Pyfunc can be used to:

* correct metadata (for subsequent elements or display)
* linking objects (for example, linking a found face to a person)
* deleting unnecessary metadata
* create new objects/attributes for subsequent elements (ROI for scene change)
* adding a non-neural model or a model that is not yet supported as a separate element (easyOCR)
* implement analytical tool with connection to a third-party service
