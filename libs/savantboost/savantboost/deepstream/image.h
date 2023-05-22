#pragma  once

#include <glib.h>
#include "npp.h"
#include <memory>
#include <string>

namespace savantboost
{

    class Image{
        private:
            Npp32u *_data = nullptr;;
            Npp32u *_cpu_data = nullptr;
            gint _width = 0, _height = 0, _pitch = 0, _size = 0;
            gboolean free_memory;

        public:
            Image(Npp32u *data, gint width, gint height,  gboolean cuda_copy);
            Image(Npp32u *data, gint width, gint height);
            Image(gint width, gint height);
            Npp32u* getDataPtr();
            Npp32u* getCPUDataPtr();
            
            #ifdef ENABLE_DEBUG
            void save_image(std::string filepath);
            #endif
            
            gint getWidth();
            gint getHeight();
            gint getPitch();    
            gint getByteSize();
            ~Image();
    };

} // namespace savantboost

