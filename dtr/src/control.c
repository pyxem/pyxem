/*
 * control.c
 *
 * Common control structure
 *
 * (c) 2007-2008 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#include <inttypes.h>
#include <stdlib.h>
#include <math.h>

#include "control.h"
#include "image.h"

ControlContext *control_ctx_new() {

	ControlContext *ctx;
	
	ctx = malloc(sizeof(ControlContext));

	ctx->x_centre = 0;
	ctx->y_centre = 0;
	ctx->have_centres = 0;
	ctx->cell = NULL;
	ctx->dirax = NULL;
	ctx->images = image_list_new();
	ctx->reflectionlist = NULL;
	ctx->refine_window = NULL;
	ctx->cell_lattice = NULL;
	ctx->integrated = NULL;
	ctx->cache_filename = NULL;
	
	return ctx;

}

