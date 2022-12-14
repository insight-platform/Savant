#pragma  once

#include "gstnvdsmeta.h"
#include <iterator>

/**
 * Holds unclipped bounding box coordinates of the object.
 */
typedef struct _NvRBboxCoords {
  float x_center;            /**< Holds the box's x coordinate of center in pixels. */
  float y_center;             /**< Holds the box's y coordinate of center in pixels. */
  float width;           /**< Holds the box's width in pixels. */
  float height;          /**< Holds the box's height in pixels. */
  float angle;           /**< Holds the box's angle in degrees. */
} NvRBboxCoords;

NvRBboxCoords* acquire_rbbox();
void add_rbbox_to_object_meta(NvDsBatchMeta *batch_meta, NvDsObjectMeta* obj_meta, NvRBboxCoords* rbbox);
NvRBboxCoords* get_rbbox(NvDsObjectMeta *object_meta);

class RBBoxIterator: public std::iterator<std::input_iterator_tag, NvRBboxCoords >
{  
  public:
    RBBoxIterator(NvDsUserMetaList* user_meta);
    RBBoxIterator(const RBBoxIterator& it);

    bool operator!=(RBBoxIterator const& other) const;
    bool operator==(RBBoxIterator const& other) const; //need for BOOST_FOREACH
    NvRBboxCoords* operator*() const;
    RBBoxIterator& operator++();
private:
    NvDsUserMetaList* user_meta_item;
};
