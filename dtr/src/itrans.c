/*
 * itrans.c
 *
 * Parameterise features in an image for reconstruction
 *
 * (c) 2007-2009 Thomas White <taw27@cam.ac.uk>
 * (c) 2007      Gordon Ball <gfb21@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <math.h>

#include "image.h"
#include "itrans-threshold.h"
#include "itrans-zaefferer.h"
#include "itrans-stat.h"

/* Return an empty feature list */
ImageFeatureList *itrans_peaksearch_none(ImageRecord *imagerecord)
{
	return image_feature_list_new();
}

/* Return a feature list for the given image */
ImageFeatureList *itrans_process_image(ImageRecord *image,
						PeakSearchMode psmode)
{
	ImageFeatureList *list;

	switch ( psmode ) {
		case PEAKSEARCH_NONE : {
			list = itrans_peaksearch_none(image);
			break;
		}
		case PEAKSEARCH_THRESHOLD : {
			list = itrans_peaksearch_threshold(image);
			break;
		}
		case PEAKSEARCH_ADAPTIVE_THRESHOLD : {
			list = itrans_peaksearch_adaptive_threshold(image);
			break;
		}
		case PEAKSEARCH_ZAEFFERER : {
			list = itrans_peaksearch_zaefferer(image);
			break;
		}
		case PEAKSEARCH_STAT : {
			list = itrans_peaksearch_stat(image);
			break;
		}
		default: list = NULL;
	}

	return list;
}

/* Quantification radius */
#define R 20

/* Quantify each feature in the image's feature list */
void itrans_quantify_features(ImageRecord *image)
{
	int i;

	for ( i=0; i<image->features->n_features; i++ ) {

		double theta;
		double background, total;
		int n;
		int xi, yi;

		background = 0.0;
		n = 0;
		for ( theta=0; theta<2*M_PI; theta+=2*M_PI/10 ) {

			double x, y;
			long int xd, yd;

			x = image->features->features[i].x + R*sin(theta);
			y = image->features->features[i].y + R*cos(theta);
			xd = rint(x);
			yd = rint(y);

			if ( (xd>=0) && (yd>=0)
			  && (xd<image->width) && (yd<image->height) ) {
				background += image->image[xd+yd*image->width];
				n++;
			}

		}
		background /= n;

		total = 0.0;
		for ( xi=-20; xi<=20; xi++ ) {
			for ( yi=-20; yi<=20; yi++ ) {

				double x, y;
				long int xd, yd;

				if ( xi*xi + yi*yi > 20 ) continue;

				x = image->features->features[i].x + xi;
				y = image->features->features[i].y + yi;

				xd = rint(x);
				yd = rint(y);

				if ( (xd>=0) && (yd>=0)
				  && (xd<image->width) && (yd<image->height) ) {
					total +=
						image->image[xd+yd*image->width]
								- background;
				}

			}
		}

		image->features->features[i].intensity = total;

	}
}
