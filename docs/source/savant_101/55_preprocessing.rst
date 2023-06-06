Preprocessing for models
========================


Often, you may require the use of non-standard preprocessing methods.
For instance, when working with a detection model that can identify rotated objects,
you might come across an object that requires special preprocessing.
In such cases, for the classification model, it is necessary to extract the object
and rotate it to enhance the classification accuracy.

These scenarios call for the implementation of non-standard image processing
algorithms to prepare the image before inferring the model.

There are two ways of preprocessing in the Savant framework: meta information
preprocessing (bounding boxes) and image preprocessing.


Preprocessing object meta
-------------------------

If you don't need to make any changes to the image, or if you are working with
aligned bounding boxes, then often all you need to do is change or fix the bounding box.
For example, take its top half or add padding. In this case it is enough to write your
own class inherited from `BasePreprocessObjectMeta <https://insight-platform.github.io/Savant/reference/api/generated/savant.base.input_preproc.BasePreprocessObjectMeta.html#basepreprocessobjectmeta>`_ class and
implement __call__ magic method.

Before model inference, custom preprocessing will be applied to all objects on frames. A copy of all
object meta-information is passed to this method. Note that changes in object meta-information
in preprocessing are not passed and do not affect the meta-information that will be after
model inference. The method is called for each object from the list of objects selected
for inference, based on the information you specified in the ``object`` of ``input`` section.

You can read about the metadata methods in the `"Working With Metadata"
<https://insight-platform.github.io/Savant/savant_101/75_working_with_metadata.html>`_ section.
The method must return the instance of the class
`BBox <https://insight-platform.github.io/Savant/reference/api/generated/savant.meta.bbox.BBox.html#bbox>`_
(The class of aligned bounding box).

After you implemented your own preprocessing class, you just need to specify the module in the
input section in the preprocess_object_meta subsection (this is the name of your module and
the file with the class. In the example person_detector is the name of the module and
input_preproc is the file with the class) and the name of the class
(in the example the TopCrop class).

Example config:

.. code-block:: yaml

    input:
        object: person_detector.person
        preprocess_object_meta:
            module: person_detector.input_preproc
            class_name: TopCrop

Preprocessing object image
--------------------------

This kind of preprocessing opens up more possibilities, because it allows you to
interact directly with the image of an object. To implement your own image
preprocessing you need to create your own class inherited from
`BasePreprocessObjectImage
<https://insight-platform.github.io/Savant/reference/api/generated/savant.base.input_preproc.BasePreprocessObjectImage.html#basepreprocessobjectimage>`_
and implement the __call__ magic method.

.. code-block:: python

    class BasePreprocessObjectImage(BasePyFuncCallableImpl):
    """Object image preprocessing interface."""

        @abstractmethod
        def __call__(
            self,
            object_meta: ObjectMeta,
            frame_image: GpuImage,
            cuda_stream: cv2.cuda.Stream
        ) -> GpuImage:
        """Transforms object image.

        :param object_meta: object meta
        :param frame_image: original image
        :return: changed image
        """

The method passes meta information about the object, the whole frame (image),
as an instance of the GpuImage class and cuda stream. You can use cuda stream
to call asynchronous functions from OpenCV library. No additional synchronization
is required from you to complete all operations, it will be done automatically before
transferring images to the inference model. Each object uses its own thread for
processing. This allows the processing of objects on the same frame in parallel
with the most efficient use of GPU resources.

After you write your own preprocessing, you just need to specify the module in
the input section in the preprocess_object_image sub-section
(this is the name of your module and the file with the class.
In the example person_detector is the name of the module and input_preproc
is the file with the class) and the name of the class.


Example config:

.. code-block:: yaml

    input:
        object: person_detector.person
        preprocess_object_image:
            module: person_detector.input_preproc
            class_name: TopCrop
            output_image:
                width: 32
                height: 140
                method: scale # fit | scale
                interpolation: nearest # linear | cubic | area | lanczos4

You can also optionally specify the image dimensions, the resizing method and
the interpolation method for the final transformation, after which the image
will be transferred to the inference model.

GPUImage is a special wrapper class which allows you to simplify the work with
the image on the GPU. A detailed specification of the methods can be found in
the `documentation <https://insight-platform.github.io/Savant/reference/api/generated/savant.utils.image.GPUImage.html#gpuimage>`_.
Let's review the basic methods of this class, which will allow you
to perform basic operations on the GPU

`GPUImage <https://insight-platform.github.io/Savant/reference/reference/api/generated/savant.utils.image.GPUImage.html#gpuimage>`_ class properties:

- gpu_mat - returns an instance of the `GpuMat <https://docs.opencv.org/4.x/d0/d60/classcv_1_1cuda_1_1GpuMat.html>`_ class from OpenCV.
- width - image width in pixels.
- height - image height in pixels.

GPUImage class methods:

- to_cpu - copies image from GPU memory into RAM. The image is returned as instance of CPUImage class.
- сut - cuts out of the image part defined by normal or rotated box. If a rotated box is specified, it cuts out part of the image by the rectangle enclosing the rotated box. The method returns the cropped part of the image and the box with coordinates relative to the new image.
- concat - allows you to combine two images into one. The first image is the image from which this method is called, the second is the image that is passed to the method. You can specify whether images should be vertically or horizontally joined.
- paste - inserts the image into the current image. The insertion place is specified as a point on the upper left corner of the inserted image.
- rotate - rotates the image by a specified angle. You can also pass an object bounding box to the method, so that it is rotated together with the image. The method returns the rotated image and the box.
- resize - resizes the image and returns the result as a new image. You can specify the resize method. Fit - the image will be resized without aspect ratio preservation, scale - the image will be resized with aspect ratio preservation and indentation. You can also specify the interpolation method.


`CPUImage <https://insight-platform.github.io/Savant/reference/api/generated/savant.utils.image.CPUImage.html#cpuimage>`_ has the same methods as GPUImage, but they work with images in RAM,
instead gpu_nat property it has `np_array` property, which returns an instance of the numpy array
and instead to_cpu method it has to_gpu method, which copies image from RAM into GPU memory.

Using these basic methods you can do most of the necessary transformations.
