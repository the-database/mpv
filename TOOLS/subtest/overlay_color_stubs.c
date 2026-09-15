// CPU data substitutes for the production color helper / overlay emitter.
// No libplacebo conversion, GPU output, or absolute display luminance is tested.
#include <stdbool.h>
#include <stdio.h>

enum { TRC_SRGB, TRC_BT1886, TRC_PQ, TRC_HLG };
enum { PL_OVERLAY_MONOCHROME, PL_ALPHA_INDEPENDENT };
enum pl_overlay_coords { PL_OVERLAY_COORDS_DST_FRAME };
struct pl_hdr_metadata { float max_luma; };
struct pl_color_space {
    int transfer, primaries;
    struct pl_hdr_metadata hdr;
};
static const struct pl_color_space pl_color_space_srgb = {
    .transfer = TRC_SRGB, .primaries = 709,
};
static bool pl_color_transfer_is_hdr(int transfer)
{
    return transfer == TRC_PQ || transfer == TRC_HLG;
}
static bool pl_color_space_is_hdr(const struct pl_color_space *color)
{
    return pl_color_transfer_is_hdr(color->transfer);
}
struct pl_overlay_part { int unused; };
struct pl_overlay {
    int mode, tex, num_parts;
    enum pl_overlay_coords coords;
    struct pl_color_space color;
    struct { int alpha; } repr;
    struct pl_overlay_part *parts;
};
struct pl_frame { int num_overlays; };
struct osd_state { struct pl_overlay overlays[8]; };
struct osd_entry {
    int result_tex, num_run_parts, num_spill_parts;
    struct pl_overlay_part *run_parts, *spill_parts;
    int *spill_links;
};
struct sub_bitmaps { bool video_color_space; };
struct mp_image { struct { struct pl_color_space color; } params; };
struct next_opts { float sub_hdr_peak; };
struct priv { struct next_opts *next_opts; int trans_chain[2]; };

/* PRODUCTION_FUNCTIONS */

static int failures;
static void check(const char *name, struct priv *p, struct osd_entry *entry,
                  struct sub_bitmaps *item, struct mp_image *src,
                  float ref_luma, float expected_peak, int expected_transfer)
{
    struct pl_frame frame = {0};
    struct osd_state state = {0};
    emit_composed_overlays(p, item, entry, &frame, &state,
                          PL_OVERLAY_COORDS_DST_FRAME, src /* REF_LUMA_ARGUMENT */);
    bool ok = frame.num_overlays == (entry->num_run_parts ? 3 : 2);
    for (int i = 0; i < frame.num_overlays; i++) {
        const struct pl_color_space *color = &state.overlays[i].color;
        ok &= color->hdr.max_luma == expected_peak;
        ok &= color->transfer == expected_transfer;
    }
    printf("%s: %s (peak %.0f, expected %.0f; %d overlays)\n", name,
           ok ? "PASS" : "FAIL", state.overlays[0].color.hdr.max_luma,
           expected_peak, frame.num_overlays);
    failures += !ok;
}

int main(void)
{
    struct next_opts opts = {.sub_hdr_peak = 100};
    struct priv p = {.next_opts = &opts, .trans_chain = {2, 3}};
    struct pl_overlay_part parts[4] = {0};
    int links[] = {0, 0, 1};
    struct osd_entry entry = {
        .result_tex = 1, .num_run_parts = 1, .run_parts = parts,
        .num_spill_parts = 3, .spill_parts = parts + 1, .spill_links = links,
    };
    struct sub_bitmaps item = {0};
    struct mp_image src = {.params.color = {.transfer = TRC_PQ, .primaries = 2020}};
    check("PQ explicit 100", &p, &entry, &item, &src, 203, 100, TRC_SRGB);
    // Reuse the exact textures/parts, changing only the current display option.
    opts.sub_hdr_peak = 203;
    check("PQ cached content, live 203", &p, &entry, &item, &src, 203, 203, TRC_SRGB);
    src.params.color.transfer = TRC_HLG;
    opts.sub_hdr_peak = 100;
    check("HLG explicit 100", &p, &entry, &item, &src, 203, 100, TRC_SRGB);
    opts.sub_hdr_peak = 0;
    check("HDR automatic reference", &p, &entry, &item, &src, 150, 150, TRC_SRGB);
    entry.num_run_parts = 0;
    check("all-spill reference", &p, &entry, &item, &src, 150, 150, TRC_SRGB);
    entry.num_run_parts = 1;
    src.params.color.transfer = TRC_BT1886;
    item.video_color_space = true;
    opts.sub_hdr_peak = 100;
    check("SDR video colorspace", &p, &entry, &item, &src, 80, 80, TRC_BT1886);
    check("no source reference", &p, &entry, &item, NULL, 80, 80, TRC_SRGB);
    check("no source default", &p, &entry, &item, NULL, 0, 0, TRC_SRGB);
    return failures ? 1 : 0;
}
