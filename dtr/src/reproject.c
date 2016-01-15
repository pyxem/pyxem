/*
 * reproject.c
 *
 * Synthesize diffraction patterns
 *
 * (c) 2007-2008 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#include <stdlib.h>
#include <math.h>

#include "control.h"
#include "reflections.h"
#include "utils.h"
#include "imagedisplay.h"
#include "displaywindow.h"
#include "image.h"
#include "mapping.h"

/* Attempt to find partners from the feature list of "image" for each feature in "flist". */
void reproject_partner_features(ImageFeatureList *rflist, ImageRecord *image) {

	int i;

	for ( i=0; i<rflist->n_features; i++ ) {

		//if ( !reflection_is_easy(rflist->features[i].reflection) ) continue;

		double d = 0.0;
		ImageFeature *partner;
		int idx;

		partner = image_feature_closest(image->features, rflist->features[i].x, rflist->features[i].y, &d, &idx);

		if ( (d <= 20.0) && partner ) {
			rflist->features[i].partner = partner;
			rflist->features[i].partner_d = d;
			image->features->features[idx].partner =
				&rflist->features[i];
		} else {
			rflist->features[i].partner = NULL;
		}

	}

}

ImageFeatureList *reproject_get_reflections(ImageRecord *image,
						ReflectionList *reflectionlist)
{
	ImageFeatureList *flist;
	Reflection *reflection;
	double smax = 0.2e9;
	double tilt, omega, k;
	double nx, ny, nz; /* "normal" vector */
	double kx, ky, kz; /* Electron wavevector ("normal" times 1/lambda */
	double ux, uy, uz; /* "up" vector */
	double rx, ry, rz; /* "right" vector */

	flist = image_feature_list_new();

	tilt = image->tilt;
	omega = image->omega;
	k = 1.0/image->lambda;

	/* Calculate the (normalised) incident electron wavevector */
	mapping_rotate(0.0, 0.0, 1.0, &nx, &ny, &nz, omega, tilt);
	kx = nx / image->lambda;
	ky = ny / image->lambda;
	kz = nz / image->lambda;	/* This is the centre of the Ewald sphere */

	/* Determine where "up" is */
	mapping_rotate(0.0, 1.0, 0.0, &ux, &uy, &uz, omega, tilt);

	/* Determine where "right" is */
	mapping_rotate(1.0, 0.0, 0.0, &rx, &ry, &rz, omega, tilt);

	reflection = reflectionlist->reflections;
	do {

		double xl, yl, zl;
		double a, b, c;
		double s1, s2, s, t;
		double g_sq, gn;

		/* Get the coordinates of the reciprocal lattice point */
		xl = reflection->x;
		yl = reflection->y;
		zl = reflection->z;
		g_sq = modulus_squared(xl, yl, zl);
		gn = xl*nx + yl*ny + zl*nz;

		/* Next, solve the relrod equation to calculate
		 * the excitation error */
		a = 1.0;
		b = 2.0*(gn - k);
		c = -2.0*gn*k + g_sq;
		t = -0.5*(b + sign(b)*sqrt(b*b - 4.0*a*c));
		s1 = t/a;
		s2 = c/t;
		if ( fabs(s1) < fabs(s2) ) s = s1; else s = s2;

		/* Skip this reflection if s is large */
		if ( fabs(s) <= smax ) {

			double xi, yi, zi;
			double gx, gy, gz;
			double theta;
			double x, y;

			/* Determine the intersection point */
			xi = xl + s*nx;  yi = yl + s*ny;  zi = zl + s*nz;

			/* Calculate Bragg angle */
			gx = xi - kx;
			gy = yi - ky;
			gz = zi - kz;	/* This is the vector from the centre of
					*  the sphere to the intersection */
			theta = angle_between(-kx, -ky, -kz, gx, gy, gz);

			if ( theta > 0.0 ) {

				double dx, dy, psi;

				/* Calculate azimuth of point in image
				 * (anticlockwise from +x) */
				dx = xi*rx + yi*ry + zi*rz;
				dy = xi*ux + yi*uy + zi*uz;
				psi = atan2(dy, dx);

				/* Get image coordinates from polar
				 * representation */
				if ( image->fmode == FORMULATION_CLEN ) {
					x = image->camera_len*tan(theta)*cos(psi);
					y = image->camera_len*tan(theta)*sin(psi);
					x *= image->resolution;
					y *= image->resolution;
				} else if ( image->fmode==FORMULATION_PIXELSIZE ) {
					x = tan(theta)*cos(psi) / image->lambda;
					y = tan(theta)*sin(psi) / image->lambda;
					x /= image->pixel_size;
					y /= image->pixel_size;
				} else {
					fprintf(stderr,
						"Unrecognised formulation mode "
						"in reproject_get_reflections\n");
					return NULL;
				}

				x += image->x_centre;
				y += image->y_centre;

				/* Sanity check */
				if ( (x>=0) && (x<image->width)
				  && (y>=0) && (y<image->height) ) {

					/* Record the reflection
					 *  Intensity should be multiplied by
					 *  relrod spike function except
					 *  reprojected reflections aren't used
					 *  quantitatively for anything
					 */
					image_add_feature_reflection(flist, x,
						y, image, reflection->intensity,
						reflection);

				} /* else it's outside the picture somewhere */

			} /* else this is the central beam */

		}

		reflection = reflection->next;

	} while ( reflection );

	/* Partner features only if the image has a feature list.  This allows
	 *  the test programs to use this function to generate simulated data
	 */
	if ( image->features != NULL ) {
		reproject_partner_features(flist, image);
	}

	return flist;

}

/* Ensure ctx->cell_lattice matches ctx->cell */
void reproject_cell_to_lattice(ControlContext *ctx) {

	int i;

	if ( ctx->cell_lattice ) {
		reflection_list_from_new_cell(ctx->cell_lattice, ctx->cell);
	} else {
		ctx->cell_lattice = reflection_list_from_cell(ctx->cell);
	}

	displaywindow_enable_cell_functions(ctx->dw, TRUE);

	/* Invalidate all the reprojected feature lists */
	for ( i=0; i<ctx->images->n_images; i++ ) {

		ImageRecord *image;
		int j;

		image = &ctx->images->images[i];
		if ( image->rflist != NULL ) {
			image_feature_list_free(image->rflist);
			image->rflist = NULL;
		}

		for ( j=0; j<image->features->n_features; j++ ) {
			image->features->features[i].partner = NULL;
		}

	}

}

/* Notify that ctx->cell has changed (also need to call displaywindow_update) */
void reproject_lattice_changed(ControlContext *ctx) {

	int i;
	int total_reprojected = 0;
	int total_measured = 0;
	int partnered_reprojected = 0;
	int partnered_measured = 0;

	reproject_cell_to_lattice(ctx);

	for ( i=0; i<ctx->images->n_images; i++ ) {

		ImageRecord *image;
		int j;

		image = &ctx->images->images[i];

		/* Perform relrod projection */
		image->rflist = reproject_get_reflections(image,
							ctx->cell_lattice);

		/* Loop over reprojected reflections */
		for ( j=0; j<image->rflist->n_features; j++ ) {

			double d;
			ImageFeature f;
			ImageFeature *p;

			f = image->rflist->features[j];
			p = image->rflist->features[j].partner;

			if ( p != NULL ) {
				d = distance(f.x, f.y, p->x, p->y);
				partnered_reprojected++;
			}
		}
		total_reprojected += image->rflist->n_features;

		/* Loop over measured reflections */
		for ( j=0; j<image->features->n_features; j++ ) {

			double d;
			ImageFeature f;
			ImageFeature *p;

			f = image->features->features[j];
			p = image->features->features[j].partner;

			if ( p != NULL ) {
				d = distance(f.x, f.y, p->x, p->y);
				partnered_measured++;
			}
		}
		total_measured += image->features->n_features;
	}
	printf("%i images\n", ctx->images->n_images);
	printf("%i measured reflections\n", total_measured);
	printf("%i reprojected reflections\n", total_reprojected);
	printf("%i partnered measured reflections\n", partnered_measured);
	printf("%i partnered reprojected reflections\n", partnered_reprojected);

	displaywindow_update_imagestack(ctx->dw);
}
