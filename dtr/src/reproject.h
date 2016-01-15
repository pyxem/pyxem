/*
 * reproject.h
 *
 * Synthesize diffraction patterns
 *
 * (c) 2007-2008 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */
 
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef REPROJECT_H
#define REPROJECT_H

#include "control.h"
#include "image.h"

extern ImageFeatureList *reproject_get_reflections(ImageRecord *image, ReflectionList *reflectionlist);
extern void reproject_partner_features(ImageFeatureList *rflist, ImageRecord *image);
extern void reproject_cell_to_lattice(ControlContext *ctx);
extern void reproject_lattice_changed(ControlContext *ctx);

#endif	/* REPROJECT_H */

