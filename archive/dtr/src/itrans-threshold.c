/*
 * itrans-threshold.c
 *
 * Threshold and adaptive threshold peak searches
 *
 * (c) 2007 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdint.h>
#include <assert.h>

#include "image.h"

ImageFeatureList *itrans_peaksearch_threshold(ImageRecord *imagerecord, ControlContext *ctx) {

	int width, height;
	int x, y;
	uint16_t *image = imagerecord->image;
	uint16_t max = 0;
	ImageFeatureList *flist;
	
	flist = image_feature_list_new();
	
	width = imagerecord->width;
	height = imagerecord->height;
	
	/* Simple Thresholding */
	for ( y=0; y<height; y++ ) {
		for ( x=0; x<width; x++ ) {
			if ( image[x + width*y] > max ) max = image[x + width*y];
		}
	}
	
	for ( y=0; y<height; y++ ) {
		for ( x=0; x<width; x++ ) {
			if ( image[x + width*y] > max/3 ) {
				assert(x<width);
				assert(y<height);
				assert(x>=0);
				assert(y>=0);
				image_add_feature(flist, x, y, imagerecord, image[x + width*y]);
			}
		}
	}
	
	return flist;
	
}

ImageFeatureList *itrans_peaksearch_adaptive_threshold(ImageRecord *imagerecord, ControlContext *ctx) {

	uint16_t max_val = 0;
	int width, height;
	uint16_t *image;
	uint16_t max;
	int x, y;
	ImageFeatureList *flist;
	
	flist = image_feature_list_new();
	
	image = imagerecord->image;
	width = imagerecord->width;
	height = imagerecord->height;
	
	max = 0;
	for ( y=0; y<height; y++ ) {
		for ( x=0; x<width; x++ ) {
			if ( image[x + width*y] > max ) max = image[x + width*y];
		}
	}
	
	/* Adaptive Thresholding */
	do {
	
		int max_x = 0;
		int max_y = 0;;
		
		/* Locate the highest point */
		max_val = 0;
		for ( y=0; y<height; y++ ) {
			for ( x=0; x<width; x++ ) {

				if ( image[x + width*y] > max_val ) {
					max_val = image[x + width*y];
					max_x = x;  max_y = y;
				}
			
			}
		}
		
		if ( max_val > max/10 ) {
			assert(max_x<width);
			assert(max_y<height);
			assert(max_x>=0);
			assert(max_y>=0);
			image_add_feature(flist, x, y, imagerecord, image[x + width*y]);
			
			/* Remove it and its surroundings */
			for ( y=((max_y-10>0)?max_y-10:0); y<((max_y+10)<height?max_y+10:height); y++ ) {
				for ( x=((max_x-10>0)?max_x-10:0); x<((max_x+10)<width?max_x+10:width); x++ ) {
					image[x + width*y] = 0;
				}
			}
		}
		
	} while ( max_val > 50 );
	
	return flist;

}

