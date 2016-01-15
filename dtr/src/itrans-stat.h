/*
 * itrans-stat.h
 *
 * Peak detection by iterative statistical analysis and processing
 *
 * (c) 2007 Gordon Ball <gfb21@cam.ac.uk>
 *	    Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */
 
#ifndef ITRANS_STAT_H
#define ITRANS_STAT_H

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "image.h"

extern ImageFeatureList *itrans_peaksearch_stat(ImageRecord *imagerecord);

#endif /* ITRANS_STAT_H */

