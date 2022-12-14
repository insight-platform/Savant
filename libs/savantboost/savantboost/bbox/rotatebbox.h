#ifndef DEEPSTREAM_PENCILS_ROTATEBBOX_H
#define DEEPSTREAM_PENCILS_ROTATEBBOX_H

#include <nppdefs.h>
#include "gstnvdsmeta.h"
#include "deepstream/image.h"

/**
 * A class for storing rotated boxes and cutting objects out of the frame by box.
 */
class RotateBBox
{
private:
    float _x_center, _y_center, _width, _height, _angle, _confidence;
public:
    /**
     * @brief Construct a new rotate bounding box (rbbox)
     * 
     * @param x_center the x coordinate of the rbbox central point
     * @param y_center the y coordinate of the rbbox central point
     * @param width the width of the rbbox 
     * @param height the heightof the rbbox
     * @param angle the angle of the rbbox in degree
     */
    RotateBBox(float x_center, float y_center, float width, float height, float angle);
    
    /**
     * @brief Construct a new predicted rotate bounding box (rbbox)
     * 
     * @param x_center the x coordinate of the predicted rbbox central point
     * @param y_center the y coordinate of the predicted rbbox central point
     * @param width the width of the predicted rbbox 
     * @param height the heightof the predicted rbbox
     * @param angle the angle of the rbbox in predicted degree
     * @param confidence the confidence of the predicted rbbox in degree
     */
    RotateBBox(float x_center, float y_center, float width, float height, float angle, float confidence);
    
    // RotateBBox(const NvDsObjectMeta *object_meta);

    /**
     * @brief Cut object from the frame 
     * 
     * @param frame pointer to the frame in GPu memory
     * @param frame_size frame size 
     * @param padding_width Padding size (pixels) by the width of the box with which the box will be cut from the frame
     * @param padding_height Padding size (pixels) by the height of the box with which the box will be cut from the frame
     * @return savantboost::Image* Pointer to Image object
     */
    savantboost::Image* CutFromFrame(Npp8u* frame, NppiSize frame_size, float padding_width, float padding_height);
};

#endif //DEEPSTREAM_PENCILS_ROTATEBBOX_H
