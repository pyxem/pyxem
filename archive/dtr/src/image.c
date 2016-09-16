/*
 * image.c
 *
 * Handle images and image features
 *
 * (c) 2007-2008 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#define _GNU_SOURCE 1
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "control.h"
#include "image.h"
#include "utils.h"

int image_add(ImageList *list, uint16_t *image, int width, int height, double tilt, ControlContext *ctx) {

	if ( list->images ) {
		list->images = realloc(list->images, (list->n_images+1)*sizeof(ImageRecord));
	} else {
		assert(list->n_images == 0);
		list->images = malloc(sizeof(ImageRecord));
	}
	
	list->images[list->n_images].tilt = tilt;
	list->images[list->n_images].omega = ctx->omega;
	list->images[list->n_images].image = image;
	list->images[list->n_images].width = width;
	list->images[list->n_images].height = height;
	list->images[list->n_images].lambda = ctx->lambda;
	list->images[list->n_images].fmode = ctx->fmode;
	list->images[list->n_images].x_centre = ctx->x_centre;
	list->images[list->n_images].y_centre = ctx->y_centre;
	list->images[list->n_images].slop = 0.0;
	list->images[list->n_images].features = NULL;
	list->images[list->n_images].rflist = NULL;
	
	if ( ctx->fmode == FORMULATION_PIXELSIZE ) {
		list->images[list->n_images].pixel_size = ctx->pixel_size;
		list->images[list->n_images].camera_len = 0;
		list->images[list->n_images].resolution = 0;
	} else if ( ctx->fmode == FORMULATION_CLEN ) {
		list->images[list->n_images].pixel_size = 0;
		list->images[list->n_images].camera_len = ctx->camera_length;
		list->images[list->n_images].resolution = ctx->resolution;
	}
	
	list->n_images++;
	
	return list->n_images - 1;

}

ImageList *image_list_new() {

	ImageList *list;
	
	list = malloc(sizeof(ImageList));
	
	list->n_images = 0;
	list->images = NULL;
	
	return list;

}

void image_add_feature_reflection(ImageFeatureList *flist, double x, double y, ImageRecord *parent, double intensity,
					Reflection *reflection) {

	if ( flist->features ) {
		flist->features = realloc(flist->features, (flist->n_features+1)*sizeof(ImageFeature));
	} else {
		assert(flist->n_features == 0);
		flist->features = malloc(sizeof(ImageFeature));
	}
	
	flist->features[flist->n_features].x = x;
	flist->features[flist->n_features].y = y;
	flist->features[flist->n_features].intensity = intensity;
	flist->features[flist->n_features].parent = parent;
	flist->features[flist->n_features].partner = NULL;
	flist->features[flist->n_features].partner_d = 0.0;
	flist->features[flist->n_features].reflection = reflection;
	
	flist->n_features++;

}

void image_add_feature(ImageFeatureList *flist, double x, double y, ImageRecord *parent, double intensity) {
	image_add_feature_reflection(flist, x, y, parent, intensity, NULL);
}

ImageFeatureList *image_feature_list_new() {

	ImageFeatureList *flist;
	
	flist = malloc(sizeof(ImageFeatureList));
	
	flist->n_features = 0;
	flist->features = NULL;
	
	return flist;
}

void image_feature_list_free(ImageFeatureList *flist) {

	if ( !flist ) return;
	
	if ( flist->features ) free(flist->features);
	free(flist);

}

ImageFeature *image_feature_closest(ImageFeatureList *flist, double x, double y, double *d, int *idx) {

	int i;
	double dmin = +HUGE_VAL;
	int closest = 0;
	
	for ( i=0; i<flist->n_features; i++ ) {
	
		double d;
		
		d = distance(flist->features[i].x, flist->features[i].y, x, y);
	
		if ( d < dmin ) {
			dmin = d;
			closest = i;
		}
	
	}
	
	if ( dmin < +HUGE_VAL ) {
		*d = dmin;
		*idx = closest;
		return &flist->features[closest];
	}
	
	*d = +INFINITY;
	return NULL;

}

ImageFeature *image_feature_second_closest(ImageFeatureList *flist, double x, double y, double *d, int *idx) {

	int i;
	double dmin = +HUGE_VAL;
	int closest = 0;
	double dfirst;
	int idxfirst;
	
	image_feature_closest(flist, x, y, &dfirst, &idxfirst);
	
	for ( i=0; i<flist->n_features; i++ ) {
	
		double d;
		
		d = distance(flist->features[i].x, flist->features[i].y, x, y);
	
		if ( (d < dmin) && (i != idxfirst) ) {
			dmin = d;
			closest = i;
		}
	
	}
	
	if ( dmin < +HUGE_VAL ) {
		*d = dmin;
		*idx = closest;
		return &flist->features[closest];
	}
	
	return NULL;

}

