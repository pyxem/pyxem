/*
 * cache.c
 *
 * Save a list of features from images to save recalculation
 *
 * (c) 2007 Gordon Ball <gfb21@cam.ac.uk>
 *	    Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */
 
#ifndef CACHE_H
#define CACHE_H

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "image.h"

extern int cache_load(ImageList *images, const char *filename);
extern int cache_save(ImageList *images, const char *filename);

#endif /*CACHE_H */

