/*
 * itrans-threshold.h
 *
 * Threshold and adaptive threshold peak searches
 *
 * (c) 2007 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */
 
#ifndef ITRANS_THRESHOLD_H
#define ITRANS_THRESHOLD_H

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "image.h"

extern ImageFeatureList *itrans_peaksearch_threshold(ImageRecord *image);
extern ImageFeatureList *itrans_peaksearch_adaptive_threshold(ImageRecord *image);

#endif	/* ITRANS_THRESHOLD_H */

