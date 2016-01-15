/*
 * itrans-zaefferer.c
 *
 * Zaefferer peak search
 *
 * (c) 2007-2008 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdint.h>
#include <assert.h>

#include "utils.h"
#include "image.h"

#define PEAK_WINDOW_SIZE 6

ImageFeatureList *itrans_peaksearch_zaefferer(ImageRecord *imagerecord) {

	int x, y;
	int width, height;
	uint16_t *image;
	ImageFeatureList *flist;

	flist = image_feature_list_new();

	image = imagerecord->image;
	width = imagerecord->width;
	height = imagerecord->height;

	for ( x=1; x<width-1; x++ ) {
		for ( y=1; y<height-1; y++ ) {

			double dx1, dx2, dy1, dy2;
			double dxs, dys;
			double grad;

			/* Get gradients */
			dx1 = image[x+width*y] - image[(x+1)+width*y];
			dx2 = image[(x-1)+width*y] - image[x+width*y];
			dy1 = image[x+width*y] - image[(x+1)+width*(y+1)];
			dy2 = image[x+width*(y-1)] - image[x+width*y];

			/* Average gradient measurements from both sides */
			dxs = ((dx1*dx1) + (dx2*dx2)) / 2;
			dys = ((dy1*dy1) + (dy2*dy2)) / 2;

			/* Calculate overall gradient */
			grad = dxs + dys;
			/*Change this value to improve sampling of weak peaks*/
			if ( grad >10 ) {
				//printf("%i, %i,  %f, %f, %f\n", x, y, dxs, dys, grad );
				int mask_x, mask_y;
				int sx, sy;
				double max;
				unsigned int did_something = 1;

				mask_x = x;
				mask_y = y;

				while ( (did_something) && (distance(mask_x, mask_y, x, y)<50) ) {

					max = image[mask_x+width*mask_y];
					did_something = 0;

					for ( sy=biggest(mask_y-PEAK_WINDOW_SIZE/2, 0);
					      sy<smallest(mask_y+PEAK_WINDOW_SIZE/2, height);
					      sy++ ) {

						for ( sx=biggest(mask_x-PEAK_WINDOW_SIZE/2, 0);
						      sx<smallest(mask_x+PEAK_WINDOW_SIZE/2, width);
						      sx++ ) {

							if ( image[sx+width*sy] > max ) {
								max = image[sx+width*sy];
								mask_x = sx;
								mask_y = sy;
								did_something = 1;
							}

						}

					}

				}

				if ( !did_something ) {

					double d;
					int idx;

					assert(mask_x<width);
					assert(mask_y<height);
					assert(mask_x>=0);
					assert(mask_y>=0);

					if ( distance(mask_x, mask_y, x, y)
					      > 50.0 ) {
						printf("Too far\n");
						continue;
					}

					/* Check for a feature at exactly the
					 * same coordinates */
					image_feature_closest(flist, mask_x,
								mask_y, &d,
								&idx);
					/*Change this value to prevent oversampling the central spot*/
					if ( d > 1.0 ) {
						image_add_feature(flist, mask_x,
						 mask_y, imagerecord,
						 image[mask_x + width*mask_y]);
					}

				}

			}
		}
	}

	return flist;

}
