/*
 * qdrp.h
 *
 * Handle QDRP-style control files
 *
 * (c) 2007 Thomas White <taw27@cam.ac.uk>
 *  dtr - Diffraction Tomography Reconstruction
 *
 */
 
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef QDRP_H
#define QDRP_H

#include "control.h"

extern int qdrp_read(ControlContext *ctx);
extern unsigned int qdrp_is_qdrprc(const char *filename);

#endif	/* QDRP_H */

