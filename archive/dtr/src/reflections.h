/*
 * reflections.h
 *
 * Data structures in 3D space
 *
 * (c) 2007-2008 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */
 
#ifndef REFLECTION_H
#define REFLECTION_H

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

typedef enum {
	REFLECTION_NORMAL,	/* Measured - expressed as x, y, z position */
	REFLECTION_GENERATED,	/* Generated and intensity-measured - expressed as h, k, l index */
	REFLECTION_MARKER,
	REFLECTION_VECTOR_MARKER_1,
	REFLECTION_VECTOR_MARKER_2,
	REFLECTION_VECTOR_MARKER_3
} ReflectionType;

typedef struct reflection_struct {

	double		x;
	double		y;
	double		z;
	double 		intensity;

	signed int	h;
	signed int	k;
	signed int	l;

	ReflectionType	type;
	
	/* Stuff used when finding bases */
	int		found;		/* This reflection has been used in the seed-finding process */
	int		lfom;		/* Line FoM for this seed */

	struct reflection_struct *next;	/* MUST BE LAST in order for caching to work */

} Reflection;

typedef struct reflectionlist_struct {

	Reflection *reflections;
	Reflection *last_reflection;
	
	unsigned int n_reflections;
	unsigned int list_capped;
	
} ReflectionList;

extern ReflectionList *reflectionlist_new(void);
extern void reflectionlist_clear(ReflectionList *reflectionlist);
extern void reflectionlist_clear_markers(ReflectionList *reflectionlist);
extern void reflectionlist_free(ReflectionList *reflectionlist);

extern Reflection *reflection_add(ReflectionList *reflectionlist, double x, double y, double z, double intensity, ReflectionType type);
extern int reflection_is_easy(Reflection *reflection);

extern double reflectionlist_largest_g(ReflectionList *reflectionlist);
extern Reflection *reflectionlist_find_nearest(ReflectionList *reflectionlist, double x, double y, double z);
extern Reflection *reflectionlist_find_nearest_longer_unknown(ReflectionList *reflectionlist, double x, double y, double z, double min_distance);
extern Reflection *reflectionlist_find_nearest_type(ReflectionList *reflectionlist, double x, double y, double z, ReflectionType type);
extern Reflection *reflectionlist_find(ReflectionList *reflectionlist, signed int h, signed int k, signed int l);

#include "basis.h"
extern ReflectionList *reflection_list_from_cell(Basis *basis);
extern void reflection_list_from_new_cell(ReflectionList *ordered, Basis *basis);

#endif	/* REFLECTION_H */

