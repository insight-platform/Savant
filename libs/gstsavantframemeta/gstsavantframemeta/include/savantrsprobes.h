#ifndef _SAVANT_RS_PROBES_
#define _SAVANT_RS_PROBES_

#include <cstdint>
#include <gst/gst.h>

/**
 * Wrapper for pad probe data.
 *
 * @param handle Handle for VideoPipeline object.
 * @param stage Target pipeline stage.
 */
typedef struct _SavantRsPadProbeData_ {
    uintptr_t handle;
    char *stage;
} SavantRsPadProbeData;

/**
 * Create SavantRsPadProbeData.
 *
 * @param handle Handle for VideoPipeline object.
 * @param stage Target pipeline stage.
 * @return Pointer to SavantRsPadProbeData.
 */
SavantRsPadProbeData *
create_savant_rs_pad_probe_data(uintptr_t handle, const char *stage);

/**
 * Release SavantRsPadProbeData.
 *
 * @param data Pointer to SavantRsPadProbeData.
 */
void release_savant_rs_pad_probe_data(SavantRsPadProbeData *data);

/**
 * Pad probe to move frame between pipeline stages.
 *
 * @param pad GStreamer pad.
 * @param info Pad probe info.
 * @param user_data Pointer to SavantRsPadProbeData.
 */
GstPadProbeReturn move_frame_as_is_pad_probe(GstPad *pad,
                                             GstPadProbeInfo *info,
                                             gpointer user_data);

/**
 * Pad probe to move frames between pipeline stages and pack them into batch.
 *
 * @param pad GStreamer pad.
 * @param info Pad probe info.
 * @param user_data Pointer to SavantRsPadProbeData.
 */
GstPadProbeReturn move_and_pack_frames_pad_probe(GstPad *pad,
                                                 GstPadProbeInfo *info,
                                                 gpointer user_data);

/**
 * Pad probe to move batch between pipeline stages.
 *
 * @param pad GStreamer pad.
 * @param info Pad probe info.
 * @param user_data Pointer to SavantRsPadProbeData.
 */
GstPadProbeReturn move_batch_as_is_pad_probe(GstPad *pad,
                                             GstPadProbeInfo *info,
                                             gpointer user_data);

/**
 * Pad probe to move batch between pipeline stages and unpack it into frames.
 *
 * @param pad GStreamer pad.
 * @param info Pad probe info.
 * @param user_data Pointer to SavantRsPadProbeData.
 */
GstPadProbeReturn move_and_unpack_batch_pad_probe(GstPad *pad,
                                                  GstPadProbeInfo *info,
                                                  gpointer user_data);

#endif //_SAVANT_RS_PROBES_
