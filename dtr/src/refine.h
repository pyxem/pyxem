/*
 * refine.h
 *
 * Refine the reconstruction
 *
 * (c) 2007-2008 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */
 
#ifndef REFINE_H
#define REFINE_H

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "control.h"

extern void refine_do_sequence(ControlContext *ctx);
extern double refine_do_cell(ControlContext *ctx);

#endif	/* REFINE_H */

