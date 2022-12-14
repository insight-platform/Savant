#include "image.h"
#include <iostream>

#ifdef ENABLE_DEBUG
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#endif


savantboost::Image::Image(Npp32u *data, gint width, gint height)
{
    _data = data;
    _width = width;
    _height = height;
    _pitch = sizeof(Npp32u) * width;
    _size = sizeof(Npp32u) * width * height;
    free_memory = false;
}

savantboost::Image::Image(Npp32u *data, gint width, gint height, gboolean copy2gpu)
{
    if (copy2gpu)
    {
        cudaMalloc((void **) &_data, width * height * sizeof(Npp32u));
        cudaMemcpy(_data, data, width * height * sizeof(Npp32u), cudaMemcpyHostToDevice);
        free_memory = true;
    }
    else
    {
        _data = data;
        free_memory = false;
    }
    _width = width;
    _height = height;
    _pitch = sizeof(Npp32u) * width;
    _size = sizeof(Npp32u) * width * height;
    
}

savantboost::Image::Image(gint width, gint height){
    _width = width;
    _height = height;
    _pitch = sizeof(Npp32u) * width;
    _size = sizeof(Npp32u) * width * height;
    cudaMalloc(&_data, _size);
    free_memory = true;
}

savantboost::Image::~Image(){
    if ((free_memory) && (_data)) cudaFree(_data);
    if (_cpu_data) 
    {
        free(_cpu_data);
    }
}

#ifdef ENABLE_DEBUG
void savantboost::Image::save_image(std::string filepath){
    if (_data) 
    {
        cv::Mat cv_image = cv::Mat (_height, _width, CV_8UC4, this->getCPUDataPtr(), _pitch);
        cv::cvtColor(cv_image, cv_image, cv::COLOR_BGR2RGB);
        cv::imwrite(filepath, cv_image);
    }
}
#endif

Npp32u* savantboost::Image::getDataPtr(){
    return _data;
}

Npp32u* savantboost::Image::getCPUDataPtr(){
    if (_cpu_data)
    {
        return _cpu_data;
    }
    else
    {
        _cpu_data = (Npp32u*) calloc(this->getWidth()*this->getHeight(), sizeof(Npp32u));
        cudaMemcpy(_cpu_data, this->_data, this->getByteSize(), cudaMemcpyDeviceToHost);
        return _cpu_data;
    }
}


gint savantboost::Image::getWidth(){
    return _width;
}

gint savantboost::Image::getHeight(){
    return _height;
}

gint savantboost::Image::getPitch(){
    return _pitch;
}

gint savantboost::Image::getByteSize(){
    return _size;
}
