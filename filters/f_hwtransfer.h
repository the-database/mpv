#pragma once

#include "filter.h"

// A filter which uploads sw frames to hw. Ignores hw frames.
struct mp_hwupload {
    // Indicates if the filter was successfully initialised, or not.
    // If not, the state of other members is undefined.
    bool successful_init;

    // The filter to use for uploads. NULL if none is required.
    struct mp_filter *f;

    // The underlying format of uploaded frames
    int selected_sw_imgfmt;
};

// allowed_sw_fmts (optional): if non-empty, the uploader will only offer these
// surface sub-formats as upload targets, so that when a conversion before upload
// is required it lands on a format the consumer can ingest, instead of
// best-matching to an arbitrary device-supported format. Pass NULL/0 for the
// default (unrestricted) behavior.
struct mp_hwupload mp_hwupload_create(struct mp_filter *parent, int hw_imgfmt,
                                       int sw_imgfmt, bool src_is_same_hw,
                                       const int *allowed_sw_fmts,
                                       int num_allowed_sw_fmts);

// A filter which downloads sw frames from hw. Ignores sw frames.
struct mp_hwdownload {
    struct mp_filter *f;

    struct mp_image_pool *pool;
};

struct mp_hwdownload *mp_hwdownload_create(struct mp_filter *parent);
