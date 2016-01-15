/*
 * prealign.c
 *
 * Initial alignment of images
 *
 * (c) 2007-2008 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "control.h"
#include "imagedisplay.h"
#include "main.h"
#include "image.h"
#include "utils.h"

typedef struct {
	int		n;
	ControlContext	*ctx;
	ImageDisplay	*id;
} PreAlignBlock;

static gint prealign_clicked(GtkWidget *widget, GdkEventButton *event, PreAlignBlock *pb) {

	double xoffs, yoffs, scale;
	double x, y;
	
	x = event->x;  y = event->y;
	xoffs = (pb->id->drawingarea_width - pb->id->view_width) / 2;
	yoffs = (pb->id->drawingarea_height - pb->id->view_height) / 2;
	scale = (double)pb->id->view_width/pb->id->imagerecord.width;
	x -= xoffs;	y -= yoffs;
	x /= scale;	y /= scale;
	y = pb->id->imagerecord.height - y;
	pb->ctx->images->images[pb->n].x_centre = x;
	pb->ctx->images->images[pb->n].y_centre = y;
	
	pb->n++;
	if ( pb->n >= pb->ctx->images->n_images ) {
		/* Finished */
		imagedisplay_close(pb->id);
		main_do_reconstruction(pb->ctx);
		free(pb);
	} else {
		/* Display the next pattern */
		imagedisplay_put_data(pb->id, pb->ctx->images->images[pb->n]);
	}
	
	return 0;

}

/* No peak-detection nor 3D mapping has been done yet.
	Ask the user to give a rough idea (i.e. as accurately as possible...)
	of the centre of each image. */
void prealign_do_series(ControlContext *ctx) {

	PreAlignBlock *pb;

	ctx->have_centres = 1;	/* Inhibit "centre-finding by stacking" */
	pb = malloc(sizeof(PreAlignBlock));
	pb->n = 0;
	pb->ctx = ctx;
	pb->id = imagedisplay_open_with_message(ctx->images->images[pb->n], "Image Pre-alignment",
				"Click the centre of the zero-order beam as accurately as you can.",
				IMAGEDISPLAY_QUIT_IF_CLOSED, G_CALLBACK(prealign_clicked), pb);

}

/* Sum the image stack, taking pre-existing centres into account if available.
 * If no centres available, select the brightest pixel from the sum and assign
 *  that as the centre to all the images. */
void prealign_sum_stack(ImageList *list, int have_centres, int sum_stack) {

	int twidth, theight;
	int mnorth, msouth, mwest, meast;
	int x, y, i;
	uint16_t *image_total;
	ImageDisplay *sum_id;
	ImageRecord total_record;
	
	/* Determine maximum size of image to accommodate, and allocate memory */
	mnorth = 0;  msouth = 0;  mwest = 0;  meast = 0;
	for ( i=0; i<list->n_images; i++ ) {
		if ( list->images[i].width-list->images[i].x_centre > meast ) {
			meast = list->images[i].width-list->images[i].x_centre;
		}
		if ( list->images[i].x_centre > mwest ) {
			mwest = list->images[i].x_centre;
		}
		if ( list->images[i].height-list->images[i].y_centre > mnorth ) {
			mnorth = list->images[i].height-list->images[i].y_centre;
		}
		if ( list->images[i].y_centre > msouth ) {
			msouth = list->images[i].y_centre;
		}
	}
	twidth = mwest + meast;
	theight = mnorth + msouth;
	
	image_total = malloc(twidth * theight * sizeof(uint16_t));
	memset(image_total, 0, twidth * theight * sizeof(uint16_t));
	
	/* Add the image stack together */
	if ( !have_centres ) {
		
		int max_x, max_y;
		uint16_t max_val;
		
		for ( i=0; i<list->n_images; i++ ) {
			int xoffs, yoffs;
			xoffs = (twidth - list->images[i].width)/2;
			yoffs = (theight - list->images[i].height)/2;
			for ( y=0; y<list->images[i].height; y++ ) {
				for ( x=0; x<list->images[i].width; x++ ) {
					assert(x+xoffs < twidth);
					assert(y+yoffs < theight);
					assert(x+xoffs >= 0);
					assert(y+yoffs >= 0);
					image_total[(x+xoffs) + twidth*(y+yoffs)] +=
						list->images[i].image[x + list->images[i].width*y]/list->n_images;
				}
			}
		}
		
		/* Locate the highest point */
		max_val = 0;	max_x = 0;	max_y = 0;
		for ( y=0; y<theight; y++ ) {
			for ( x=0; x<twidth; x++ ) {
				if ( image_total[x + twidth*y] > max_val ) {
					max_val = image_total[x + twidth*y];
					max_x = x;  max_y = y;
				}
			}
		}
		
		/* Record this measurement on all images */
		for ( i=0; i<list->n_images; i++ ) {
			list->images[i].x_centre = max_x;
			list->images[i].y_centre = max_y;
		}
		total_record.x_centre = max_x;
		total_record.y_centre = max_y;
		total_record.omega = list->images[0].omega;
	
	} else {
		
		/* Just sum the stack */
		for ( i=0; i<list->n_images; i++ ) {
		
			int xoffs, yoffs;
			
			xoffs = mwest - list->images[i].x_centre;
			yoffs = msouth - list->images[i].y_centre;
			
			for ( y=0; y<list->images[i].height; y++ ) {
				for ( x=0; x<list->images[i].width; x++ ) {
					assert(x+xoffs < twidth);
					assert(y+yoffs < theight);
					assert(x+xoffs >= 0);
					assert(y+yoffs >= 0);
					image_total[(x+xoffs) + twidth*(y+yoffs)] +=
						list->images[i].image[x + list->images[i].width*y]/list->n_images;
				}
			}
			
		}
		
		total_record.omega = list->images[0].omega;
		total_record.x_centre = mwest;
		total_record.y_centre = msouth;
	}
	
	/* Display */
	if ( sum_stack ) {
		total_record.image = image_total;
		total_record.width = twidth;
		total_record.height = theight;
		sum_id = imagedisplay_open(total_record, "Sum of All Images",
					IMAGEDISPLAY_SHOW_CENTRE | IMAGEDISPLAY_SHOW_TILT_AXIS | IMAGEDISPLAY_FREE);
	}

}

#define CENTERING_WINDOW_SIZE 50

void prealign_fine_centering(ImageList *list, int sum_stack) {

	int i;
	
	for ( i=0; i<list->n_images; i++ ) {
	
		int sx, sy;
		double max;
		unsigned int did_something = 1;
		int mask_x, mask_y;
		int width, height;
		
		width = list->images[i].width;
		height = list->images[i].height;
		mask_x = list->images[i].x_centre;
		mask_y = list->images[i].y_centre;
		
		while ( (did_something) &&
			(distance(mask_x, mask_y, list->images[i].x_centre, list->images[i].y_centre)<100) ) {
		
			double nmax, nmask_x, nmask_y;
		
			nmax = 0.0;
			nmask_x = 0;
			nmask_y = 0;
			max = list->images[i].image[mask_x+width*mask_y];
			did_something = 0;
			
			for ( sy=biggest(mask_y-CENTERING_WINDOW_SIZE/2, 0);
			      sy<smallest(mask_y+CENTERING_WINDOW_SIZE/2, height);
			      sy++ ) {
				for ( sx=biggest(mask_x-CENTERING_WINDOW_SIZE/2, 0);
				      sx<smallest(mask_x+CENTERING_WINDOW_SIZE/2, width);
				      sx++ ) {
					
					if ( list->images[i].image[sx+width*sy] > nmax ) {
						nmax = list->images[i].image[sx+width*sy];
						nmask_x = sx;
						nmask_y = sy;
					}
					
				}
			}
			
			if ( nmax > max ) {
				max = nmax;
				mask_x = nmask_x;
				mask_y = nmask_y;
				did_something = 1;
			}
			
		}
		
		if ( !did_something ) {
		
			assert(mask_x<width);
			assert(mask_y<height);
			assert(mask_x>=0);
			assert(mask_y>=0);
			
			printf("AL: Image %3i: centre offset by %f,%f\n", i,
				mask_x-list->images[i].x_centre, mask_y-list->images[i].y_centre);
			
			list->images[i].x_centre = mask_x;
			list->images[i].y_centre = mask_y;
			
		}
	
	}
	
	prealign_sum_stack(list, TRUE, sum_stack);

}

void prealign_feature_centering(ImageList *list) {

	int i;
	
	for ( i=0; i<list->n_images; i++ ) {
	
		double d1, d2;
		ImageFeature *feature1;
		ImageFeature *feature2;
		int idx;
		
		feature1 = image_feature_closest(list->images[i].features, list->images[i].x_centre,
							list->images[i].y_centre, &d1, &idx);
		feature2 = image_feature_second_closest(list->images[i].features, list->images[i].x_centre,
							list->images[i].y_centre, &d2, &idx);
		
		printf("AL: Image %i, d1=%f, d2=%f\n", i, d1, d2);
		
		if ( (fabs(d2-d1) <= 19.0) && feature1 && feature2 ) {
			list->images[i].x_centre = (feature1->x + feature2->x)/2;
			list->images[i].y_centre = (feature1->y + feature2->y)/2;
		} else {
			if ( feature1 ) {
				list->images[i].x_centre = feature1->x;
				list->images[i].y_centre = feature1->y;
			} else {
				printf("AL: Couldn't find centre feature for image %i\n", i);
			}
		}
		
	}

}

