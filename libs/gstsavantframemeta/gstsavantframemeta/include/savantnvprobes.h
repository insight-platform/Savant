#ifndef _SAVANT_NV_PROBES_
#define _SAVANT_NV_PROBES_

#include <gst/gst.h>

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

#endif //_SAVANT_NV_PROBES_
