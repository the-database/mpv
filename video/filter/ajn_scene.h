/* SPDX-License-Identifier: LGPL-2.1-or-later */
#ifndef MPV_AJN_SCENE_H
#define MPV_AJN_SCENE_H
#include <stdbool.h>
struct mp_image;
struct mp_log;
struct ajn_scene;
#ifdef _WIN32
struct ajn_scene *ajn_scene_create(struct mp_log *log);
bool ajn_scene_connect(struct ajn_scene *s, const char *name);
bool ajn_scene_active(struct ajn_scene *s);
void ajn_scene_reset(struct ajn_scene *s);
void ajn_scene_destroy(struct ajn_scene *s);
int ajn_scene_decide(struct ajn_scene *s, struct mp_image *a, struct mp_image *b,
                     bool supported);
#else
static inline struct ajn_scene *ajn_scene_create(struct mp_log *log) { return 0; }
static inline bool ajn_scene_connect(struct ajn_scene *s, const char *n) { return false; }
static inline bool ajn_scene_active(struct ajn_scene *s) { return false; }
static inline void ajn_scene_reset(struct ajn_scene *s) { }
static inline void ajn_scene_destroy(struct ajn_scene *s) { }
static inline int ajn_scene_decide(struct ajn_scene *s, struct mp_image *a,
    struct mp_image *b, bool supported) { return -1; }
#endif
#endif
