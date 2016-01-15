/*
 * itrans-zaefferer.h
 *
 * Zaefferer peak search
 *
 * (c) 2007 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */
 
#ifndef ITRANS_ZAEFFERER_H
#define ITRANS_ZAEFFERER_H

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "image.h"

extern ImageFeatureList *itrans_peaksearch_zaefferer(ImageRecord *image);

#endif	/* ITRANS_ZAEFFERER_H */

