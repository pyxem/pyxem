/*
 * itrans-stat.c
 *
 * Peak detection by iterative statistical analysis and processing
 *
 * (c) 2007 Gordon Ball <gfb21@cam.ac.uk>
 *	    Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <math.h>
#include <gsl/gsl_matrix.h>

#include "reflections.h"
#include "control.h"
#include "imagedisplay.h"
#include "utils.h"

/* Renormalise a gsl_matrix to 0->1
 * Re-written to use gsl_matrix native functions */
static void matrix_renormalise(gsl_matrix *m) {

	double max,min;

	gsl_matrix_minmax(m,&min,&max);
	gsl_matrix_add_constant(m,0.-min);
	gsl_matrix_scale(m,1./(max-min));

}

/*
 * Create a gsl_matrix for performing image operations on
 * from a raw image and control context
 * Converting to gsl_matrix because many of the required operations
 * can be done as matrices to save effort
 * Renormalises matrix to 0->1
 */
static gsl_matrix *itrans_peaksearch_stat_createImageMatrix(uint16_t *image,
							int width, int height)
{
	gsl_matrix *raw;
	int i, j;

	raw = gsl_matrix_alloc(width, height);
	for ( i=0; i<raw->size1; i++ ) {
		for ( j=0; j<raw->size2; j++ ) {
			gsl_matrix_set(raw, i, j, (double)image[i+j*width]);
		}
	}
	matrix_renormalise(raw);

	return raw;
}

/* Find and return the mean value of the matrix */
static double itrans_peaksearch_stat_image_mean(gsl_matrix *m)
{
	int i, j;
	double mean = 0.;

	for ( i=0; i<m->size1; i++ ) {
		for ( j=0; j<m->size2; j++ ) {
			mean += gsl_matrix_get(m,i,j);
		}
	}

	return mean / (m->size1 * m->size2);
}

/*
 * Return the standard deviation, sigma, of the matrix values
 * \sqrt(\sum((x-mean)^2)/n)
 */
static double itrans_peaksearch_stat_image_sigma(gsl_matrix *m, double mean)
{
	int i,j;
	double diff2 = 0;

	for ( i=0; i<m->size1; i++ ) {
		for ( j=0; j<m->size2; j++ ) {
			diff2 += (gsl_matrix_get(m,i,j)-mean)
				 * (gsl_matrix_get(m,i,j)-mean);
		}
	}

	return sqrt(diff2/(m->size1 * m->size2));
}

/*
 * Filter all pixels with value < mean + k*sigma to 0
 * Set all matching pixels to 1 d
 */
static void itrans_peaksearch_stat_sigma_filter(gsl_matrix *m, double k)
{
	double mean,sigma;
	int i,j;

	mean = itrans_peaksearch_stat_image_mean(m);
	sigma = itrans_peaksearch_stat_image_sigma(m,mean);

	for ( i=0; i<m->size1; i++ ) {
		for ( j=0; j<m->size2; j++ ) {
			if (gsl_matrix_get(m,i,j) >= mean+k*sigma) {
				gsl_matrix_set(m, i, j, 1.);
			} else {
				gsl_matrix_set(m, i, j, 0.);
			}
		}
	}
}

/*
 * Calculate the mean within a circle centred (x,y) of radius r
 *
 * TODO: Use a mask instead of calculating valid points
 */
static double itrans_peaksearch_stat_circle_mean(gsl_matrix *m, int x, int y,
						 int r, gsl_matrix *mask)
{
	double mean = 0.;
	int i, j;
	int n = 0;

	for ( i=x-r; i<=x+r; i++ ) {
		for ( j=y-r; j<=y+r; j++ ) {
			if ( gsl_matrix_get(mask,i-x+r,j-y+r)>0. ) {
				mean += gsl_matrix_get(m, i, j);
				n++;
			}
		}
	}

	//printf("cm: (%d,%d) summean=%lf n=%d\n",x,y,mean,n);
	return mean/n;

}

/*
 * Calculate sigma within a circle centred (x,y) of radius r
 *
 * TODO: Use a mask instead of calculating valid points
 */
static double itrans_peaksearch_stat_circle_sigma(gsl_matrix *m, int x, int y,
					int r, gsl_matrix *mask, double mean)
{
	double diff2 = 0.;
	int i, j;
	int n = 0;

	for ( i=x-r; i<=x+r; i++ ) {
		for ( j=y-r; j<=y+r; j++ ) {
			if ( gsl_matrix_get(mask, i-x+r, j-y+r) > 0 ) {
				diff2 += (gsl_matrix_get(m,i,j)-mean)
					 * (gsl_matrix_get(m,i,j)-mean);
				n++;
			}
		}
	}

	return sqrt(diff2/n);
}

/*
 * Calculate a circular mask to save recalculating it every time
 */
static gsl_matrix *itrans_peaksearch_stat_circle_mask(int r)
{
	gsl_matrix *m;
	int i,j;

	m = gsl_matrix_calloc(2*r+1, 2*r+1);
	for ( i=0; i<2*r+1; i++ ) {
		for ( j=0; j<2*r+1; j++ ) {
			if ( sqrt((r-i)*(r-i)+(r-j)*(r-j)) <= r ) {
				 gsl_matrix_set(m, i, j, 1.);
			}

		}
	}

	return m;
}

/*
 * Variation on the above filter where instead of using
 * sigma and mean for the whole image, it is found for a local
 * circle of radius r pixels.
 * The central point is also calculated as an approximately gaussian
 * average over local pixels rather than just a single pixel.
 * This takes a long time - 20-30 seconds for a 1024^2 image and r=10,
 * obviously large values of r will make it even slower.
 * The appropriate r value needs to be considered carefully - it needs to
 * be somewhat smaller than the average inter-reflection seperation.
 * 10 pixels worked well for the sapphire.mrc images, but this might
 * not be the case for other images. Images for which this would be very
 * large probably need to be resampled to a lower resolution.
 *
 * TODO: Pass a mask to the ancilliary functions instead of having
 * them calculate several hundred million sqrts
 *
 * self-referencing problem being dealt with - output being written onto the
 * array before the next point it computed problem carried over from the OO
 * version where a new object was created by each operation
 */
static void itrans_peaksearch_stat_local_sigma_filter(gsl_matrix *m, int r,
							double k)
{
	double mean,sigma;
	double local;
	gsl_matrix *mask;
	gsl_matrix *new;
	int i,j;

	mask = itrans_peaksearch_stat_circle_mask(r);
	new = gsl_matrix_alloc(m->size1, m->size2);

	for ( i=0; i<m->size1; i++ ) {
		for ( j=0; j<m->size2; j++ ) {

			if ( ((i >= r) && (i < m->size1-r))
			      && ((j >= r) && (j < m->size2-r)) ) {

				mean = itrans_peaksearch_stat_circle_mean(m,
							i, j, r, mask);
				sigma = itrans_peaksearch_stat_circle_sigma(m,
							i, j, r, mask, mean);

				local = gsl_matrix_get(m, i, j);
				local += gsl_matrix_get(m, i+1, j)
					+ gsl_matrix_get(m, i-1, j)
					+ gsl_matrix_get(m, i, j+1)
					+ gsl_matrix_get(m, i, j-1);
				local += .5*(gsl_matrix_get(m, i+1, j+1)
					+ gsl_matrix_get(m, i-1, j+1)
					+ gsl_matrix_get(m, i+1, j-1)
					+ gsl_matrix_get(m, i-1, j-1));
				local /= 7.;

				if ( local > mean+k*sigma ) {
					gsl_matrix_set(new, i, j, 1.);
				} else {
					gsl_matrix_set(new, i, j, 0.);
				}

			} else {
				gsl_matrix_set(new, i, j, 0.);
			}

		}
	}

	gsl_matrix_memcpy(m, new);
	gsl_matrix_free(new);
}

/*
 * Apply an arbitary kernel to the image - each point takes the value
 * of the sum of the kernel convolved with the surrounding pixels.
 * The kernel needs to be a (2n+1)^2 array (centred on (n,n)). It needs
 * to be square and should probably be normalised.
 * Don't do daft things like rectangular kernels or kernels larger
 * than the image - this doesn't check such things and will cry.
 *
 * Also suffers from self-reference problem
 */
static void itrans_peaksearch_stat_apply_kernel(gsl_matrix *m,
						gsl_matrix *kernel)
{
	int size;
	int half;
	gsl_matrix *l;
	gsl_matrix_view lv;
	gsl_matrix *new;
	double val;
	int i, j, x, y;

	size = kernel->size1;
	half = (size-1)/2;
	new = gsl_matrix_calloc(m->size1, m->size2);

	for ( i=0; i<m->size1; i++ ) {
		for ( j=0; j<m->size2; j++ ) {
			if ( ((i >= half) && (i < m->size1-half))
				&& ((j >= half) && (j < m->size2-half)) ) {

				lv = gsl_matrix_submatrix(m, i-half, j-half,
								size, size);
				l = &lv.matrix;
				val = 0.;
				for ( x=0; x<size; x++ ) {
					for ( y=0; y<size; y++ ) {
						val += gsl_matrix_get(l, x, y)
						 * gsl_matrix_get(kernel, x, y);
					}
				}
				gsl_matrix_set(new,i,j,val);

			}
		}
	}

	gsl_matrix_memcpy(m,new);
	gsl_matrix_free(new);
}

/* Generate the simplist possible kernel - a flat one */
static gsl_matrix *itrans_peaksearch_stat_generate_flat_kernel(int half)
{
	gsl_matrix *k;

	k = gsl_matrix_alloc(2*half+1,2*half+1);
	gsl_matrix_set_all(k,1./((2*half+1)*(2*half+1)));

	return k;
}

/* Expands or contracts a gsl_matrix by copying the columns to a new one */
static gsl_matrix *itrans_peaksearch_stat_matrix_expand(gsl_matrix *m,
				unsigned int oldsize, unsigned int newsize)
{
	gsl_matrix *new;
	int j;

	new = gsl_matrix_calloc(3, newsize);

	for ( j=0; j<oldsize; j++) {
		if ( j < newsize ) {
			gsl_matrix_set(new, 0, j, gsl_matrix_get(m, 0, j));
			gsl_matrix_set(new, 1, j, gsl_matrix_get(m, 1, j));
			gsl_matrix_set(new, 2, j, gsl_matrix_get(m, 2, j));
		}
	}

	gsl_matrix_free(m);

	return new;
}

/*
 * Stack-based flood-fill iteration routine
 * have to return a pointer to com each time because if the matrix size has to
 * be changed then we need to keep track of the location of the resized instance
 */
static gsl_matrix *itrans_peaksearch_stat_do_ff(int i, int j, int* mask,
			double threshold, gsl_matrix *m, gsl_matrix *com,
			int *com_n, int *com_size, double *val)
{
	if ( (i >= 0) && (i < m->size1) ) {
		if ( (j >= 0) && (j < m->size2) ) {
			if ( mask[i+j*m->size1] == 0 ) {
				if ( gsl_matrix_get(m, i, j) > threshold ) {

					*val += gsl_matrix_get(m, i, j);
					gsl_matrix_set(com, 0, *com_n, i);
					gsl_matrix_set(com, 1, *com_n, j);
					*com_n = *com_n + 1;
					if ( *com_n == *com_size ) {

	com = itrans_peaksearch_stat_matrix_expand(com, *com_size, *com_size*2);

						*com_size *= 2;
					}
					mask[i+j*m->size1] = 1;

	com = itrans_peaksearch_stat_do_ff(i+1, j, mask,threshold, m, com,
						com_n, com_size, val);
	com = itrans_peaksearch_stat_do_ff(i-1, j, mask,threshold, m, com,
						com_n, com_size, val);
	com = itrans_peaksearch_stat_do_ff(i, j+1, mask,threshold, m, com,
						com_n, com_size, val);
	com = itrans_peaksearch_stat_do_ff(i, j-1, mask,threshold, m, com,
						com_n, com_size, val);

				}
			}
		}
	}

	return com;

}

/*
 * Find points by floodfilling all pixels above threshold
 * Implements a stack-based flood-filling method which may
 * cause stack overflows if the blocks are extremely large -
 * dependent on implementation (default 4MiB stack for unix?)
 * Implements a crude variable-sized array method which hopefully works
 * Returns a gsl_matrix with x coordinates in row 0 and y
 * coordinates in row 1, which should be of the right length.
 * Intensities (sum of included pixel values) in row 2.
 * Variable count is set to the number of points found
 */
static gsl_matrix *itrans_peaksearch_stat_floodfill(gsl_matrix *m,
					double threshold, int *count)
{
	int *mask;
	int size, com_size, i, j, k, n;
	int com_n;
	gsl_matrix *p;
	gsl_matrix *com;
	double com_x, com_y;

	mask = calloc(m->size1*m->size2, sizeof(int));
	size = 32;
	n = 0;
	p = gsl_matrix_calloc(3, size);

	for ( i=0; i<m->size1; i++ ) {
		for ( j=0; j<m->size2; j++ ) {
			if ( gsl_matrix_get(m, i, j) > threshold ) {
				if ( mask[i+j*m->size1] == 0 ) {

					double val;

					com_size = 32;
					com_n = 0;
					com_x = com_y = 0.;
					com = gsl_matrix_calloc(3, com_size);
					val = 0;
		com = itrans_peaksearch_stat_do_ff(i, j, mask, threshold, m,
						com, &com_n, &com_size, &val);
					for ( k=0; k<com_n; k++ ) {
						com_x += gsl_matrix_get(com,
									0, k);
						com_y += gsl_matrix_get(com,
									1, k);
					}
					com_x /= com_n;
					com_y /= com_n;

					/* Now add it to the list */
					gsl_matrix_set(p, 0, n, com_x);
					gsl_matrix_set(p, 1, n, com_y);
					gsl_matrix_set(p, 2, n, val);
					n++;
					if ( n == size ) {

		p = itrans_peaksearch_stat_matrix_expand(p, size, size*2);
						size *= 2;
					}

				}
			}
		}
	}

	*count = n;
	if ( n > 0 ) {
		p = itrans_peaksearch_stat_matrix_expand(p, size, n);
	}

	return p;

}

/* Implements the iteration based automatic method
 * returns a gsl_matrix formatted as described in flood-fill */
static gsl_matrix *itrans_peaksearch_stat_iterate(gsl_matrix *m,
							unsigned int *count)
{
	int old;
	int cur;
	double k;
	double mean,sigma;
	gsl_matrix *p;
	gsl_matrix *kernel;

	old = m->size1*m->size2;

	kernel = itrans_peaksearch_stat_generate_flat_kernel(1);
	itrans_peaksearch_stat_local_sigma_filter(m, 10, 1.);

	while ( 1 ) {

		itrans_peaksearch_stat_apply_kernel(m, kernel);
		itrans_peaksearch_stat_apply_kernel(m, kernel);
		mean = itrans_peaksearch_stat_image_mean(m);
		sigma = itrans_peaksearch_stat_image_sigma(m,mean);
		k = (0.5-mean)/sigma;
		itrans_peaksearch_stat_sigma_filter(m,k);
		p = itrans_peaksearch_stat_floodfill(m, 0, &cur);

		/* Check for convergence of the number of peaks found */
		if ( old < 1.05*cur ) break;
		old = cur;

	}

	gsl_matrix_free(kernel);
	*count = cur;

	return p;
}

ImageFeatureList *itrans_peaksearch_stat(ImageRecord *imagerecord)
{
	unsigned int n_reflections;
	gsl_matrix *m;
	gsl_matrix *p;
	int i;
	uint16_t *image = imagerecord->image;
	ImageFeatureList *flist;

	flist = image_feature_list_new();

	m = itrans_peaksearch_stat_createImageMatrix(image, imagerecord->width,
							imagerecord->height);
	p = itrans_peaksearch_stat_iterate(m, &n_reflections);

	for ( i=0; i<n_reflections; i++ ) {

		double px, py, pi;

		px = gsl_matrix_get(p, 0, i);
		py = gsl_matrix_get(p, 1, i);
		pi = gsl_matrix_get(p, 2, i);
		image_add_feature(flist, px, py, imagerecord, pi);

	}

	gsl_matrix_free(m);
	gsl_matrix_free(p);

	return flist;
}
