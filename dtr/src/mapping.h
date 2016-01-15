/*
 * mapping.h
 *
 * 3D Mapping
 *
 * (c) 2007-2008 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */
 
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef MAPPING_H
#define MAPPING_H

#include "reflections.h"
#include "control.h"
#include "image.h"

extern void mapping_rotate(double x, double y, double z, double *ddx, double *ddy, double *ddz,
			   double omega, double tilt);
extern int mapping_map_to_space(ImageFeature *refl, double *ddx, double *ddy, double *ddz, double *twotheta);
extern int mapping_scale(ImageFeature *refl, double *ddx, double *ddy);
extern double mapping_scale_bar_length(ImageRecord *image);

extern void mapping_adjust_axis(ControlContext *ctx, double offset);
extern void mapping_map_features(ControlContext *ctx);

#endif	/* MAPPING_H */

