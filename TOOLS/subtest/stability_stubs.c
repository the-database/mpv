// CPU resource substitutes for test_stability.py. No native driver calls.
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#ifdef _WIN32
#include <windows.h>
#endif
#define MPMAX(a,b) ((a)>(b)?(a):(b))
#define MPMIN(a,b) ((a)<(b)?(a):(b))
#define MP_TIME_S_TO_NS(s) ((s) * INT64_C(1000000000))
#define HAVE_ASS_OUTLINE_DEFERRED 1
#define PL_FMT_CAP_STORABLE 1
#define SUBBITMAP_LIBASS 0
#define RASTER_RUN_MARGIN 512
#define RASTER_RESULT_H_MULT 4
#define RASTER_COVER_K 4
#define RASTER_SEGS_PER_TILE 8
#define WORK_TEX_W 8192
#define EDGE_TEX_W 8192
#define NUM_OVERLAY_BUFS 3
struct fmt { int caps; }; typedef const struct fmt *pl_fmt;
struct gpu { struct { int max_tex_2d_dim; bool buf_transfer,callbacks; } limits; };
typedef const struct gpu *pl_gpu;
struct pl_tex_params { pl_fmt format; int w,h; bool storable,sampleable,blit_src,blit_dst,host_writable; };
struct texture { struct pl_tex_params params; }; typedef const struct texture *pl_tex;
struct bufpar { size_t size; bool host_writable; };
struct buffer { struct bufpar params; }; typedef const struct buffer *pl_buf;
#define pl_tex_params(...) (&(struct pl_tex_params){__VA_ARGS__})
#define pl_buf_params(...) (&(struct bufpar){__VA_ARGS__})
struct pl_tex_transfer_params {
    pl_tex tex; struct { int x0,y0,x1,y1; } rc; size_t row_pitch,buf_offset;
    pl_buf buf; const void *ptr; void (*callback)(void *); void *priv;
};
#define pl_tex_transfer_params(...) (&(struct pl_tex_transfer_params){__VA_ARGS__})
struct entry { pl_tex result_tex,tex; };
struct image { unsigned char *planes[1]; int stride[1]; };
struct item { struct image *packed; int packed_w,packed_h; };
struct gmiss { int w,h,ax,ay; unsigned char *src; };
struct options { int sub_present_guard_ms; };
struct priv {
    pl_gpu gpu; pl_fmt osd_fmt[1],osd_acc_fmt; struct { int w,h; } osd_res;
    pl_tex run_acc,run_tmp,run_cov_f,run_cov_b,work_tex,edge_tex;
    struct { struct { struct entry entries[2]; } states[2]; } osd_guard;
    pl_buf edge_stage,work_stage,overlay_bufs[3],glyph_stage[3];
    int overlay_buf_idx,glyph_stage_idx,stage_frame_base;
    int64_t cnt_staging_grow,cnt_overlay_buf_grow,cnt_staging_wrap;
    size_t gstage_cpu_sz; unsigned char *gstage_cpu; void *stats;
    struct options *next_opts;
};
struct vo_frame { double duration,approx_duration,ideal_frame_duration; bool display_synced; };
static struct texture textures[64]; static struct buffer buffers[64];
static int nt,nb,tex_calls,buf_calls,fail_tex,fail_buf,uploads,buffer_uploads,pointer_uploads;
static struct fmt format={1};
static bool pl_tex_recreate(pl_gpu gpu,pl_tex *t,const struct pl_tex_params *params) {
    if (++tex_calls==fail_tex) return false;
    textures[nt].params=*params; *t=&textures[nt++]; return true;
}
static pl_fmt pl_find_named_fmt(pl_gpu gpu,const char *name) { return &format; }
static bool pl_buf_recreate(pl_gpu gpu,pl_buf *b,const struct bufpar *params) {
    if (++buf_calls==fail_buf) return false;
    buffers[nb].params=*params; *b=&buffers[nb++]; return true;
}
static void pl_buf_write(pl_gpu gpu,pl_buf b,size_t offset,const void *src,size_t size) {
    assert(b && b->params.size>=offset+size && src);
}
static bool pl_tex_upload(pl_gpu gpu,const struct pl_tex_transfer_params *tp) {
    uploads++;
    assert((tp->buf!=NULL) != (tp->ptr!=NULL));
    if (tp->buf) { assert(gpu->limits.buf_transfer); buffer_uploads++; }
    else pointer_uploads++;
    if (tp->callback) tp->callback(tp->priv);
    return true;
}
static void vo_alloc_bump(struct priv *p,int64_t *count) { (*count)++; }
#define talloc_realloc(ctx,ptr,type,n) ((type*)realloc(ptr,sizeof(type)*(n)))
#define talloc_array(ctx,type,n) ((type*)malloc(sizeof(type)*(n)))
static void talloc_free(void *p) { free(p); }
static void *mp_image_new_ref(struct image *i) { return malloc(1); }
#define stats_time_start(...) ((void)0)
#define stats_time_end(...) ((void)0)

/* PRODUCTION_FUNCTIONS */

static bool upload_overlay(struct priv *p,struct item *item,struct entry *entry) {
    bool ok=false;
    /* PRODUCTION_OVERLAY_UPLOAD */
    return ok;
}
static int64_t deadline(struct priv *p,struct vo_frame *frame) {
    /* PRODUCTION_DEADLINE */
}

int main(int argc,char **argv) {
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);
#endif
    assert(argc==2);
    struct gpu gpu={.limits={16384,true,false}};
    struct priv p={.gpu=&gpu,.osd_fmt={&format},.osd_acc_fmt=&format,.osd_res={1920,1080}};
    if (!strncmp(argv[1],"prealloc",8)) {
        if (!strcmp(argv[1],"prealloc-work-fail")) fail_tex=9;
        if (!strcmp(argv[1],"prealloc-edge-fail")) fail_tex=10;
        gc_prealloc_pools(&p);
        if (fail_tex) assert(buf_calls==0);
        else assert(p.work_tex && p.edge_tex && buf_calls==2);
    } else if (!strcmp(argv[1],"timing")) {
        struct options opts={-1}; p.next_opts=&opts;
        struct vo_frame frames[]={
            {.duration=41666666},
            {.duration=-1,.approx_duration=1.0/24,.ideal_frame_duration=1.0/24,.display_synced=true},
            {.duration=-1,.approx_duration=1.0/24,.ideal_frame_duration=1.0/48,.display_synced=true},
            {.duration=-1,.approx_duration=1.0/120},
            {.duration=-1},
        };
        int64_t want[]={41666666,41666666,20833333,8333333,42000000};
        for (int i=0;i<5;i++) assert(deadline(&p,&frames[i])==want[i]);
        opts.sub_present_guard_ms=0; assert(deadline(&p,&frames[1])==0);
        opts.sub_present_guard_ms=8; assert(deadline(&p,&frames[1])==8000000);
        puts("7 timing cases passed");
    } else {
        int mode=atoi(strrchr(argv[1],'-')+1);
        gpu.limits.buf_transfer=mode>=2;
        if (mode==2) fail_buf=1;
        struct buffer existing={.params={128,true}};
        if (mode==1 || mode==4) p.overlay_bufs[0]=p.glyph_stage[0]=p.edge_stage=&existing;
        unsigned char pixels[8]={1,2,3,4,5,6,7,8};
        struct texture tex={.params={.w=4,.h=2}};
        if (!strncmp(argv[1],"overlay",7)) {
            struct image img={.planes={pixels},.stride={4}};
            struct item item={.packed=&img,.packed_w=4,.packed_h=2};
            struct entry entry={.tex=&tex};
            assert(upload_overlay(&p,&item,&entry));
        } else if (!strncmp(argv[1],"staged",6)) {
            assert(gc_staged_tex_upload(&p,&p.edge_stage,&tex,4,2,4,pixels));
        } else {
            struct gmiss miss={.w=4,.h=2,.src=pixels};
            gc_flush_misses(&p,&miss,1,8,4,&tex);
            assert(!memcmp(p.gstage_cpu,pixels,8)); free(p.gstage_cpu);
        }
        assert(uploads==1);
        assert(buffer_uploads==(mode>=3));
        assert(pointer_uploads==(mode<3));
        if (!gpu.limits.buf_transfer) assert(buf_calls==0);
    }
    printf("PASS %s\n",argv[1]); return 0;
}
