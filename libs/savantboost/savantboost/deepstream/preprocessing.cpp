#include "preprocessing.h"
#include "user_meta_type.h"
#include "bbox/rotatebbox.h"
#include "cuda/cudaCrop.h"
#include <iostream>
#include <stdexcept>
#include <dlfcn.h>


#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if (e!=cudaSuccess) { \
        GST_ERROR("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(0); \
    } \
}


GST_DEBUG_CATEGORY_STATIC(gst_framemanager_debug);
#define GST_CAT_DEFAULT gst_framemanager_debug


ObjectsPreprocessing::ObjectsPreprocessing()
{
    GST_DEBUG_CATEGORY_INIT (
          gst_framemanager_debug,
          "framemanager",
          150,
          "framemanager");
}


ObjectsPreprocessing::~ObjectsPreprocessing()
{
    for (auto handle_custom_lib=this->custom_lib_list.begin(); handle_custom_lib!=this->custom_lib_list.end(); handle_custom_lib++)
    {
        dlclose(*handle_custom_lib);
    }
    this->custom_lib_list.clear();
}

bool ObjectsPreprocessing::hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

void ObjectsPreprocessing::add_preprocessing_function(
    std::string element_name,
    const std::function<savantboost::Image* (savantboost::Image *)> &preprocessing_func
){
    preprocessing_functions.insert({element_name, preprocessing_func});
}

void ObjectsPreprocessing::add_preprocessing_function(
    std::string element_name,
    std::string custom_lib,
    std::string custom_func           
)
{
    std::function<savantboost::Image *(savantboost::Image* )> custom_preprocess_func;
    /* open custom library */
    auto handle_custom_lib = dlopen(custom_lib.c_str(), RTLD_LAZY);
    if (handle_custom_lib == nullptr) throw std::runtime_error("Can't open custom library: " + custom_lib);

    /* find the address of function and data objects */
    custom_preprocess_func = (savantboost::Image*  (*)(savantboost::Image* )) dlsym(handle_custom_lib , custom_func.c_str());
    if (custom_preprocess_func == nullptr) throw std::runtime_error("Can't find function: " + custom_func);
    // dlclose(handle_custom_lib);

    this->add_preprocessing_function(element_name, custom_preprocess_func);
    this->custom_lib_list.push_back(handle_custom_lib);
}


GstFlowReturn ObjectsPreprocessing::preprocessing(
    std::string element_name,
    GstBuffer * inbuf, 
    gint model_uid, 
    gint class_id, 
    gint padding_width,
    gint padding_height
){
    
    GstMapInfo in_map_info;
    GstFlowReturn flow_ret = GST_FLOW_ERROR;
    NvBufSurface *surface;
    NvDsBatchMeta *batch_meta;
    NvDsFrameMeta *frame_meta;
    NvDsMetaList *frame_meta_list_item;
    NvDsObjectMetaList *object_meta_list;
    NvDsObjectMeta *object_meta;

    Npp8u *ref_frame, *copy_frame;
    NvRBboxCoords* rbbox;
    savantboost::Image *preproc_object;
    std::string file_name;

    int frame_height, frame_width;
    NppiSize ref_frame_size;
    gboolean status;
    status = gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ);
    const std::function<savantboost::Image* (savantboost::Image *)> &custom_preprocess_func = this->preprocessing_functions[element_name];
    if (status)
    {
        surface = (NvBufSurface *) in_map_info.data;
        batch_meta = gst_buffer_get_nvds_batch_meta(inbuf);

        if (batch_meta == nullptr) {
            return GST_FLOW_ERROR;
        }

        frames_map.insert({(size_t) inbuf, tframe_map()});

        for (frame_meta_list_item = batch_meta->frame_meta_list; frame_meta_list_item != nullptr; frame_meta_list_item = frame_meta_list_item->next)
        {
            frame_meta = (NvDsFrameMeta *) (frame_meta_list_item->data);
            DSCudaMemory ds_cuda_memory = DSCudaMemory(surface, frame_meta->batch_id);
            frame_height = (int) surface->surfaceList[frame_meta->batch_id].planeParams.height[0];
            frame_width = (int) surface->surfaceList[frame_meta->batch_id].planeParams.width[0];
            ref_frame_size = {frame_width, frame_height};
            ref_frame = ds_cuda_memory.GetMapCudaPtr();
            const size_t ref_image_bytes= ref_frame_size.width*ref_frame_size.height*sizeof(Npp8u)*4;

            cudaMalloc((void **)&copy_frame, ref_image_bytes);
            cudaCheckError()
            cudaMemcpy((void *) copy_frame, (void *) ref_frame, ref_image_bytes, cudaMemcpyDeviceToDevice);
            cudaCheckError()
            frames_map[(size_t) inbuf].insert({frame_meta->batch_id, (void *) copy_frame});
            
            int left=0, top=0, row_height=0;
            for (object_meta_list = frame_meta->obj_meta_list; object_meta_list != nullptr; object_meta_list = object_meta_list -> next)
            {
                object_meta = (NvDsObjectMeta *) (object_meta_list->data);
                if ((object_meta ->unique_component_id == model_uid) && (object_meta->class_id ==  class_id))
                {
                    auto rect_params = object_meta->rect_params;
                    if ((rect_params.left == 0) && (rect_params.top == 0) && (rect_params.width == 0) && (rect_params.height == 0))
                    {
                        rbbox = get_rbbox(object_meta);
                        if (rbbox)
                        {
                            RotateBBox rotated_bbox = RotateBBox(rbbox->x_center, rbbox->y_center, rbbox->width, rbbox->height, rbbox->angle);
                            preproc_object = rotated_bbox.CutFromFrame(copy_frame, ref_frame_size, padding_width, padding_height);
                        }
                        else {
                            GST_ERROR("Rbbox don't found rotated bbox for %ld", object_meta->object_id);
                        }
                    }
                    else
                    {
                        NppiRect crop_rect = {
                            .x =static_cast<int>(rect_params.left),
                            .y=static_cast<int>(rect_params.top),
                            .width = static_cast<int>(rect_params.width),
                            .height = static_cast<int>(rect_params.height)
                        };
                        NppiSize src_image_size = {.width = frame_width, .height = frame_height};

                        preproc_object = new savantboost::Image(rect_params.width, rect_params.height);
                        NppiSize dst_image_size = {
                            .width = static_cast<int>(rect_params.width),
                            .height = static_cast<int>(rect_params.height)
                            };

                        cudaCrop(
                            (Npp8u *) copy_frame,
                            src_image_size,
                            crop_rect,
                            (Npp8u*) preproc_object->getDataPtr(),
                            dst_image_size,
                            0,
                            0
                            );
                    }
                    if (custom_preprocess_func)
                    {
                        savantboost::Image *tmp_preproc_object;
                        tmp_preproc_object = custom_preprocess_func(preproc_object);
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
                        delete preproc_object;
                        preproc_object = tmp_preproc_object;
                    }

                    NppiRect crop_rect = {.x =0, .y=0, .width = preproc_object->getWidth(), .height = preproc_object->getHeight()};
                    NppiSize clc_image_size = {.width = preproc_object->getWidth(), .height = preproc_object->getHeight()};
                    
                    if (left + preproc_object->getWidth() > ref_frame_size.width) 
                    {
                        left = 0;
                        top += row_height;
                        row_height = 0;
                        if (top >= ref_frame_size.height) throw std::runtime_error("There is no place on frame to put object image.");
                    }
                    if (top + preproc_object->getHeight() > ref_frame_size.height)
                        throw std::runtime_error("There is no place on frame to put object image.");
                    if (preproc_object->getHeight() > row_height) row_height = preproc_object->getHeight();

                    cudaCrop(
                        (Npp8u *) preproc_object->getDataPtr(),
                        clc_image_size,
                        crop_rect,
                        ref_frame,
                        ref_frame_size,
                        left,
                        top
                    );

                    object_meta->rect_params.left = (float) left;
                    object_meta->rect_params.top = (float) top;
                    object_meta->rect_params.width = (float) preproc_object->getWidth();
                    object_meta->rect_params.height = (float) preproc_object->getHeight();

                    left += preproc_object->getWidth() ;
                    if (preproc_object!=nullptr) delete preproc_object;
                }
            }
            ds_cuda_memory.UnMapCudaPtr();
        }
        gst_buffer_unmap (inbuf, &in_map_info);
        return GST_FLOW_OK;
    }
    else
    {
        GST_ERROR("Failed to map gst buffer.");
        gst_buffer_unmap (inbuf, &in_map_info);
        return flow_ret;
    }
}

GstFlowReturn ObjectsPreprocessing::restore_frame(GstBuffer* gst_buffer){
    
    GstMapInfo in_map_info;
    gboolean status;
    NvDsBatchMeta *batch_meta;
    NvDsFrameMeta *frame_meta;
    NvDsFrameMetaList *frame_list_item;
    NvBufSurface *surface;
    int frame_height, frame_width;
    NppiSize frame_size;
    size_t frame_bytes;
    Npp8u *frame, *ref_frame;

    status = gst_buffer_map (gst_buffer, &in_map_info, GST_MAP_READ);
    if (status)
    {
        batch_meta = gst_buffer_get_nvds_batch_meta(gst_buffer);
        surface = (NvBufSurface *) in_map_info.data;
        for (frame_list_item = batch_meta->frame_meta_list; frame_list_item != nullptr; frame_list_item = frame_list_item->next)
        {
            frame_meta = (NvDsFrameMeta *) (frame_list_item->data);
            DSCudaMemory ds_cuda_memory = DSCudaMemory(surface, frame_meta->batch_id);
            frame_height = (int) surface->surfaceList[frame_meta->batch_id].planeParams.height[0];
            frame_width = (int) surface->surfaceList[frame_meta->batch_id].planeParams.width[0];
            frame_size = {frame_width, frame_height};
            frame = ds_cuda_memory.GetMapCudaPtr();
            frame_bytes = frame_width*frame_height*sizeof(Npp8u)*4;

            ref_frame = (Npp8u *) frames_map[(size_t) gst_buffer][frame_meta->batch_id];

            cudaMemcpy((void *) frame, (void *) ref_frame, frame_bytes, cudaMemcpyDeviceToDevice);
            cudaCheckError()
            cudaFree(ref_frame);
            cudaCheckError()
            frames_map[(size_t) gst_buffer].erase(frame_meta->batch_id);
            ds_cuda_memory.UnMapCudaPtr();
        }
        if (frames_map[(size_t) gst_buffer].empty())
        {
            frames_map.erase((size_t) gst_buffer);
        }
        gst_buffer_unmap (gst_buffer, &in_map_info);
        return GST_FLOW_OK;
    }
    else{
        GST_ERROR("Failed to map gst buffer.");
        gst_buffer_unmap (gst_buffer, &in_map_info);
        return GST_FLOW_ERROR;
    }
}
