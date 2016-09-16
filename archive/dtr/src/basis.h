/*
 * basis.h
 *
 * Handle basis structures
 *
 * (c) 2007-2009 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#ifndef BASIS_H
#define BASIS_H

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "control.h"

typedef struct {

	double x;
	double y;
	double z;

} Vector;

typedef struct basis_struct {
	Vector a;
	Vector b;
	Vector c;
} Basis;

typedef struct cell_struct {
	double a;
	double b;
	double c;
	double alpha;
	double beta;
	double gamma;
} UnitCell;

extern UnitCell basis_get_cell(Basis *cell);
extern void basis_save(ControlContext *ctx);
extern void basis_load(ControlContext *ctx);

#endif	/* BASIS_H */
