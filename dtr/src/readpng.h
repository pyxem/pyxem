/*
 * readpng.h
 *
 * Read images
 *
 * (c) 2007 Thomas White <taw27@cam.ac.uk>
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

 
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef READPNG_H
#define READPNG_H

#include "control.h"
extern int readpng_read(const char *filename, double tilt, ControlContext *ctx);

#endif	/* READPNG_H */
