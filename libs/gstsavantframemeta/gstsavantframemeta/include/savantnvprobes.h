#ifndef _SAVANT_NV_PROBES_
#define _SAVANT_NV_PROBES_

#include <gst/gst.h>
#include "nvdsmeta.h"

/**
 * A probe for the nvtracker src pad to remove the objects created by the nvtracker element.
 *
 * @param pad GStreamer pad.
 * @param info Pad probe info.
 * @param user_data Pointer to user data.
 */
GstPadProbeReturn remove_tracker_objs_pad_probe(GstPad *pad,
                                                GstPadProbeInfo *info,
                                                gpointer user_data);

/**
 * A helper function to check if the object was created by the nvtracker element.
 *
 * @param obj_meta Deepstream object meta structure.
 */
bool obj_is_tracker_created(NvDsObjectMeta *obj_meta);

#endif //_SAVANT_NV_PROBES_
