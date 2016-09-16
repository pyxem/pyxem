/*
 * intensities.h
 *
 * Extract integrated intensities by relrod estimation
 *
 * (c) 2007 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */
 
#ifndef INTENSITIES_H
#define INTENSITIES_H

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "control.h"

extern void intensities_extract(ControlContext *ctx);
extern void intensities_save(ControlContext *ctx);

#endif	/* INTENSITIES_H */

