/*
 * itrans.h
 *
 * Parameterise features in an image for reconstruction
 *
 * (c) 2007 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */
 
#ifndef ITRANS_H
#define ITRANS_H

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "control.h"
#include "image.h"

extern ImageFeatureList *itrans_process_image(ImageRecord *image, PeakSearchMode psmode);
extern void itrans_quantify_features(ImageRecord *image);

#endif	/* ITRANS_H */

