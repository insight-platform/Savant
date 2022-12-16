#pragma  once

#include "pysavantboost.h"


#include "nvdsparserapid.h"
#include "cuda/cudaCrop.h"
#include "bbox/rotatebbox.h"
#include "deepstream/nvsurfaceptr.h"

#include "pydocumentation.h"
#include "deepstream/dsrbbox_meta.h"

#ifdef ENABLE_DEBUG
    #include "opencv2/imgproc/imgproc.hpp"
    #include "opencv2/highgui/highgui.hpp"
#endif

namespace pysavantboost {
    void bindnms(py::module &m);
}