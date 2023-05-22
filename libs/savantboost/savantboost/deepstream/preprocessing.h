#include <unordered_map>
#include "glib.h"
#include <gst/base/gstbasetransform.h>
#include "nvsurfaceptr.h"
#include "gstnvdsmeta.h"
#include "dsrbbox_meta.h"
#include <functional>
#include "image.h"
#include <string>
#include <list>
#include "dsrbbox_meta.h"


typedef std::unordered_map<int, void *> tframe_map;


class Object{
    public:
        int width, height;
};

class CpuObject: public Object
{
    public:
        unsigned char* data = nullptr;
        CpuObject(int width, int height);
        ~CpuObject();
};


class GpuObject: public Object
{
    public:
        unsigned char* data = nullptr;
        GpuObject(int width, int height);
        ~GpuObject();
};

class ObjectsCutter{
    public:
        ObjectsCutter(size_t gst_buffer, int batchID);
};

class ObjectsPreprocessing{
    private:
        std::unordered_map<size_t, tframe_map> frames_map;
        std::unordered_map<std::string, std::function<savantboost::Image* (savantboost::Image *)>> preprocessing_functions;
        std::list<void *> custom_lib_list;
        bool hasEnding (std::string const &fullString, std::string const &ending);

    public:
        ObjectsPreprocessing();
        ~ObjectsPreprocessing();
        void add_preprocessing_function(
            std::string element_name,
            const std::function<savantboost::Image* (savantboost::Image *)> &preprocessing_func
        );
        void add_preprocessing_function(
            std::string element_name,
            std::string custom_lib,
            std::string custom_func           
        );
        GstFlowReturn restore_frame(GstBuffer* gst_buffer);
        GstFlowReturn preprocessing_rect(GstBuffer * inbuf, gint model_uid, gint class_id, gint padding_width, gint padding_height);
        GstFlowReturn preprocessing(
            std::string element_name,
            GstBuffer * inbuf, 
            gint model_uid, 
            gint class_id, 
            gint padding_width=0, 
            gint padding_height=0
        );
};
