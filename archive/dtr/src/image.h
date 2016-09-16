/*
 * image.h
 *
 * Handle images and image features
 *
 * (c) 2007-2008 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */
 
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef IMAGE_H
#define IMAGE_H

#include "control.h"

typedef struct imagefeature_struct {

	struct imagerecord_struct	*parent;
	double				x;
	double				y;
	double				intensity;
	
	struct imagefeature_struct	*partner;	/* Partner for this feature (in another feature list) or NULL */
	double				partner_d;	/* Distance between this feature and its partner, if any. */
	
	struct reflection_struct	*reflection;	/* The reflection this was projected from, if any */
	
} ImageFeature;

typedef struct {

	ImageFeature		*features;
	int			n_features;

} ImageFeatureList;

typedef struct imagerecord_struct {

	uint16_t		*image;
	double			tilt;		/* Radians.  Defines where the pattern lies in reciprocal space */
	double			omega;		/* Radians.  Defines where the pattern lies in reciprocal space */
	double			slop;		/* Radians.  Defines where the pattern lies "on the negative" */
	
	FormulationMode		fmode;
	double			pixel_size;
	double			camera_len;
	double			lambda;
	double			resolution;
	
	int			width;
	int			height;
	double			x_centre;
	double			y_centre;
	
	ImageFeatureList	*features;	/* "Experimental" features */
	ImageFeatureList	*rflist;	/* "Predicted" features */

} ImageRecord;

typedef struct imagelist_struct {

	int		n_images;
	ImageRecord	*images;

} ImageList;

#include "reflections.h"

extern ImageList *image_list_new(void);
extern int image_add(ImageList *list, uint16_t *image, int width, int height, double tilt, ControlContext *ctx);
extern ImageFeatureList *image_feature_list_new(void);
extern void image_feature_list_free(ImageFeatureList *flist);
extern void image_add_feature(ImageFeatureList *flist, double x, double y, ImageRecord *parent, double intensity);
extern void image_add_feature_reflection(ImageFeatureList *flist, double x, double y, ImageRecord *parent, double intensity, Reflection *reflection);
extern ImageFeature *image_feature_closest(ImageFeatureList *flist, double x, double y, double *d, int *idx);
extern ImageFeature *image_feature_second_closest(ImageFeatureList *flist, double x, double y, double *d, int *idx);

#endif	/* IMAGE_H */

