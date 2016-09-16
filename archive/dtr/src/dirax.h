/*
 * dirax.h
 *
 * Invoke the DirAx auto-indexing program
 * also: handle DirAx input files
 *
 * (c) 2007 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */
 
#ifndef DIRAX_H
#define DIRAX_H

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "control.h"
#include "reflections.h"

extern void dirax_invoke(ControlContext *ctx);
extern void dirax_rerun(ControlContext *ctx);
extern void dirax_stop(ControlContext *ctx);
extern ReflectionList *dirax_load(const char *filename);
extern int dirax_is_drxfile(const char *filename);

#endif	/* DIRAX_H */

