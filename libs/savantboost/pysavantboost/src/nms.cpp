#include "nms.h"


#define BBOX_NUM_VALUES 6


int pencil_count=0;

namespace pysavantboost {
    py::array py_nms(py::array_t<float, py::array::c_style> raw_input_bboxes,
                    float nms_thresh,
                    float conf_thresh,
                    int topk)
    {
        int num_detections_end;
        // check input dimensions
        if ( raw_input_bboxes.ndim()     != 2 )
            throw std::runtime_error("Input should be 2-D NumPy array");
        if ( raw_input_bboxes.shape()[1] != BBOX_NUM_VALUES ) {
            std::string errorMessage =
                    std::string("Input should have size [N,6") + std::to_string(BBOX_NUM_VALUES) + std::string("]");
            throw std::runtime_error(errorMessage);
        }
        auto* output_bboxes = new float[raw_input_bboxes.size()];
        num_detections_end  = nms((void *) raw_input_bboxes.data(), output_bboxes,
                                (int) raw_input_bboxes.shape(0), nms_thresh, conf_thresh, topk);
        auto* result = new float[num_detections_end*BBOX_NUM_VALUES];
        std::memcpy(result,(float *) output_bboxes, num_detections_end * BBOX_NUM_VALUES *sizeof(float));

        ssize_t              ndim    = 2;
        std::vector<ssize_t> shape   = { num_detections_end , BBOX_NUM_VALUES };
        std::vector<ssize_t> strides = { sizeof(float)*BBOX_NUM_VALUES , sizeof(float) };

        delete [] output_bboxes;

        return py::array(py::buffer_info(
                (float*) result,                               /* data as contiguous array  */
                sizeof(float),                                 /* size of one scalar        */
                py::format_descriptor<float>::format(), /* data type                 */
                ndim,                                          /* number of dimensions      */
                shape,                                 /* shape of the matrix       */
                strides                                /* strides for each axis     */
        ));
    }


    py::array cut_rotated_bbox(float left, float top, float width, float height, float angle, 
    int padding_width, int padding_height, size_t gst_buffer, int batchID)
    {
        Npp8u* ref_frame;
        GstMapInfo inmap;
        savantboost::Image *object_image;
        auto *buffer = reinterpret_cast<GstBuffer *>(gst_buffer);
        
        gst_buffer_map(buffer, &inmap, GST_MAP_READ);
        auto *inputnvsurface = reinterpret_cast<NvBufSurface *>(inmap.data);
        DSCudaMemory ds_cuda_memory = DSCudaMemory(inputnvsurface, batchID);
        gst_buffer_unmap(buffer, &inmap);

        int frame_height = (int) inputnvsurface->surfaceList[batchID].planeParams.height[0];
        int frame_width = (int) inputnvsurface->surfaceList[batchID].planeParams.width[0];
        
        NppiSize ref_frame_size = {frame_width, frame_height};

        ref_frame = ds_cuda_memory.GetMapCudaPtr();
        RotateBBox rotated_bbox = RotateBBox(left + width/2, top+height/2, width, height, angle);
        
        object_image = rotated_bbox.CutFromFrame(ref_frame, ref_frame_size, padding_width, padding_height);

        const unsigned int pencil_image_elements = object_image->getWidth() * object_image->getHeight() * 4;

        Npp8u *pencil_image = new Npp8u[pencil_image_elements];
        cudaMemcpy(pencil_image, (void *) object_image->getDataPtr(), pencil_image_elements * sizeof(Npp8u), cudaMemcpyDeviceToHost);

        ds_cuda_memory.UnMapCudaPtr();
        ssize_t              ndim    = 3;
        std::vector<ssize_t> shape   = { object_image->getHeight(), object_image->getWidth(), 4};
        std::vector<ssize_t> strides = {
            (ssize_t) (sizeof(Npp8u) * object_image->getWidth() * 4),
            (ssize_t) (sizeof(Npp8u)*4),
            sizeof(Npp8u)
            };

        return py::array(py::buffer_info(
                (Npp8u*) pencil_image,                               /* data as contiguous array  */
                sizeof(Npp8u),                                 /* size of one scalar        */
                py::format_descriptor<uint8_t>::format(), /* data type                 */
                ndim,                                          /* number of dimensions      */
                shape,                                 /* shape of the matrix       */
                strides                                /* strides for each axis     */
        ));
    }



    void bindnms(py::module &m) {
        m.def("nms", &py_nms, "Returns bboxes after nms module");
        m.def("cut_rotated_bbox", &cut_rotated_bbox, "Returns numpy array with rotatted object");
    }

}
