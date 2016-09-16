/*
 * main.h
 *
 * The Top Level Source File
 *
 * (c) 2007 Thomas White <taw27@cam.ac.uk>
 *  dtr - Diffraction Tomography Reconstruction
 *
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef MAIN_H
#define MAIN_H

#include "control.h"

extern void main_do_reconstruction(ControlContext *ctx);

#endif	/* MAIN_H */
