/*
 * mapping.c
 *
 * 3D Mapping
 *
 * (c) 2007-2008 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "reflections.h"
#include "control.h"
#include "image.h"
#include "displaywindow.h"
#include "cache.h"
#include "utils.h"

void mapping_rotate(double x, double y, double z, double *ddx, double *ddy, double *ddz, double omega, double tilt) {

	double nx, ny, nz;
	double x_temp, y_temp, z_temp;

	/* First: rotate image clockwise until tilt axis is aligned horizontally. */
	nx = x*cos(omega) + y*sin(omega);
	ny = -x*sin(omega) + y*cos(omega);
	nz = z;

	/* Now, tilt about the x-axis ANTICLOCKWISE around +x, i.e. the "wrong" way.
		This is because the crystal is rotated in the experiment, not the Ewald sphere. */
	x_temp = nx; y_temp = ny; z_temp = nz;
	nx = x_temp;
	ny = cos(tilt)*y_temp + sin(tilt)*z_temp;
	nz = -sin(tilt)*y_temp + cos(tilt)*z_temp;

	/* Finally, reverse the omega rotation to restore the location of the image in 3D space */
	x_temp = nx; y_temp = ny; z_temp = nz;
	nx = x_temp*cos(-omega) + y_temp*sin(-omega);
	ny = -x_temp*sin(-omega) + y_temp*cos(-omega);
	nz = z_temp;

	*ddx = nx;
	*ddy = ny;
	*ddz = nz;

}

int mapping_map_to_space(ImageFeature *refl, double *ddx, double *ddy, double *ddz, double *twotheta) {

	/* "Input" space */
	double d, x, y;
	ImageRecord *imagerecord;

	/* Angular description of reflection */
	double theta, psi, k;

	/* Reciprocal space */
	double tilt;
	double omega;

	double x_temp, y_temp, z_temp;

	imagerecord = refl->parent;
	x = refl->x - imagerecord->x_centre;	y = refl->y - imagerecord->y_centre;
	tilt = imagerecord->tilt;
	omega = imagerecord->omega;
	k = 1/imagerecord->lambda;

	/* Calculate an angular description of the reflection */
	if ( imagerecord->fmode == FORMULATION_CLEN ) {
		x /= imagerecord->resolution;
		y /= imagerecord->resolution;	/* Convert pixels to metres */
		d = sqrt((x*x) + (y*y));
		theta = atan2(d, imagerecord->camera_len);
	} else if (imagerecord->fmode == FORMULATION_PIXELSIZE ) {
		x *= imagerecord->pixel_size;
		y *= imagerecord->pixel_size;	/* Convert pixels to metres^-1 */
		d = sqrt((x*x) + (y*y));
		theta = atan2(d, k);
	} else {
		fprintf(stderr, "Unrecognised formulation mode in mapping_map_to_space.\n");
		return -1;
	}
	psi = atan2(y, x);

	x_temp = k*sin(theta)*cos(psi);
	y_temp = k*sin(theta)*sin(psi);
	z_temp = k - k*cos(theta);

	mapping_rotate(x_temp, y_temp, z_temp, ddx, ddy, ddz, omega, tilt);

	*twotheta = theta;	/* Radians.  I've used the "wrong" nomenclature above */

	return 0;

}

int mapping_scale(ImageFeature *refl, double *ddx, double *ddy) {

	double x, y;
	ImageRecord *imagerecord;
	double k;

	imagerecord = refl->parent;
	x = refl->x - imagerecord->x_centre;
	y = refl->y - imagerecord->y_centre;
	k = 1/imagerecord->lambda;

	if ( imagerecord->fmode == FORMULATION_CLEN ) {
		x /= imagerecord->resolution;
		y /= imagerecord->resolution;	/* Convert pixels to metres */
		*ddx = x * k / imagerecord->camera_len;
		*ddy = y * k / imagerecord->camera_len;
	} else if (imagerecord->fmode == FORMULATION_PIXELSIZE ) {
		*ddx = x * imagerecord->pixel_size;
		*ddy = y * imagerecord->pixel_size;	/* Convert pixels to metres^-1 */
	} else {
		fprintf(stderr, "Unrecognised formulation mode in mapping_scale.\n");
		return -1;
	}

	return 0;

}

/* Return the length of a 1 nm^-1 scale bar in the given image (in pixels)
 *  Result only strictly valid at the centre of the image */
double mapping_scale_bar_length(ImageRecord *image) {

	switch ( image->fmode ) {
		case FORMULATION_PIXELSIZE : return 1.0e9/image->pixel_size;
		case FORMULATION_CLEN : return 1.0e9*image->resolution*image->camera_len*image->lambda;
		default : fprintf(stderr, "Unrecognised formulation mode in mapping_scale_bar_length.\n");
	}

	return 0.0;

}

void mapping_map_features(ControlContext *ctx) {

	int i;

	/* Create reflection list for measured reflections */
	if ( ctx->reflectionlist ) reflectionlist_free(ctx->reflectionlist);
	ctx->reflectionlist = reflectionlist_new();

	printf("MP: Mapping to 3D..."); fflush(stdout);
	for ( i=0; i<ctx->images->n_images; i++ ) {

		int j;

		/* Iterate over the features in this image */
		for ( j=0; j<ctx->images->images[i].features->n_features; j++ ) {

			double nx, ny, nz, twotheta;

			if ( !mapping_map_to_space(&ctx->images->images[i].features->features[j],
											&nx, &ny, &nz, &twotheta) ) {
				reflection_add(ctx->reflectionlist, nx, ny, nz,
					ctx->images->images[i].features->features[j].intensity, REFLECTION_NORMAL);
			} else {
				printf("Couldn't map\n");
			}

		}

	}
	printf("done.\n");

}

void mapping_adjust_axis(ControlContext *ctx, double offset) {

	int i;

	for ( i=0; i<ctx->images->n_images; i++ ) {
		printf("Image #%3i: old omega=%f deg, new omega=%f deg\n", i, rad2deg(ctx->images->images[i].omega),
							rad2deg(ctx->images->images[i].omega+offset));
		ctx->images->images[i].omega += offset;
	}

	mapping_map_features(ctx);
	if ( ctx->dw ) {
		displaywindow_update_imagestack(ctx->dw);
		displaywindow_update(ctx->dw);
	}

}
