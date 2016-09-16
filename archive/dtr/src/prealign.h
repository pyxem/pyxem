/*
 * prealign.h
 *
 * Initial alignment of images
 *
 * (c) 2007-2008 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */
 
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef PREALIGN_H
#define PREALIGN_H

#include "image.h"

extern void prealign_do_series(ControlContext *ctx);
extern void prealign_sum_stack(ImageList *list, int have_centres, int sum_stack);
extern void prealign_fine_centering(ImageList *list, int sum_stack);
extern void prealign_feature_centering(ImageList *list);

#endif	/* PREALIGN_H */

