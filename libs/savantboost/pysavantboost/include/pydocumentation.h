#pragma once


namespace pydsdoc
{
    namespace usermeta{
        namespace RBboxCoords
        {
            constexpr const char* descr = R"pyds(
                Holds rotated bbox coordinates for an object.
                
                :ivar x_center: *float*, Holds the box's x coordinate of center
                :ivar y_center: *float*, Holds the box's y coordinate of center
                :ivar width: *float*, Holds the box's width 
                :ivar height: *float*, Holds the box's height
                :ivar angle: *float*, Holds the box's angle in degrees.)pyds";

            constexpr const char* cast=R"pyds(cast given object/data to :class:`RBboxCoords`, call pysavantboost.RBboxCoords.cast(data))pyds";
        }
    };

    namespace functionDoc
	{
        constexpr const char* acquire_rbbox =R"pyds(
                After acquiring and filling object metadata user must add it to the object metadata with API.)pyds"; 
    };

    namespace preprocessing
    {
        namespace ObjectsPreprocessing {
            constexpr const char* class_desc =R"pyds(The object is intended for preprocessing 
                the detectable objects for the classification models on the CPU and GPU, 
                as well as frame management (due to the peculiarities of preprocessing in deepstream.)pyds"; 
            constexpr const char* restore_frame =R"pyds(
                A method for restoring the original frame after model inference from the frame cache.
                :ivar gst_buffer: *GstBuffer**, pointer on gst buffer  ,  
                :ivar batchID: *int*, Batch id
                .)pyds"; 
            constexpr const char* preprocessing_rotated=R"pyds(
                A method for preprocessing objects and placing them on a frame for the classification model inference.)pyds"; 
        }
        namespace Image{
             constexpr const char* class_desc =R"pyds(Class for image representation.)pyds"; 
        }
    }
}