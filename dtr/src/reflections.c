/*
 * reflections.c
 *
 * Data structures in 3D space
 *
 * (c) 2007-2008 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#define _GNU_SOURCE
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "reflections.h"
#include "utils.h"

static void reflectionlist_init(ReflectionList *reflectionlist) {
	reflectionlist->n_reflections = 0;
	reflectionlist->list_capped = 0;
	reflectionlist->reflections = NULL;
	reflectionlist->last_reflection = NULL;
}

ReflectionList *reflectionlist_new() {

	ReflectionList *reflectionlist = malloc(sizeof(ReflectionList));

	reflectionlist_init(reflectionlist);

	return reflectionlist;

}

void reflectionlist_clear_markers(ReflectionList *reflectionlist) {

	Reflection *reflection = reflectionlist->reflections;
	Reflection *prev = NULL;
	int del = 0;

	 while ( reflection ) {
		Reflection *next = reflection->next;

		if ( (reflection->type == REFLECTION_MARKER) || (reflection->type == REFLECTION_GENERATED)
		 || (reflection->type == REFLECTION_VECTOR_MARKER_1) || (reflection->type == REFLECTION_VECTOR_MARKER_2)
		 || (reflection->type == REFLECTION_VECTOR_MARKER_3) ) {
			free(reflection);
			del++;
			if ( prev ) {
				prev->next = next;
			} else {
				reflectionlist->reflections = next;
			}
		} else {
			prev = reflection;
		}

		reflection = next;

	};

	reflectionlist->n_reflections -= del;
	reflectionlist->last_reflection = prev;

}
void reflectionlist_clear(ReflectionList *reflectionlist) {

	Reflection *reflection = reflectionlist->reflections;
	while ( reflection ) {
		Reflection *next = reflection->next;
		free(reflection);
		reflection = next;
	};

	reflectionlist_init(reflectionlist);

}

void reflectionlist_free(ReflectionList *reflectionlist) {
	reflectionlist_clear(reflectionlist);
	free(reflectionlist);
}

Reflection *reflection_add(ReflectionList *reflectionlist, double x, double y, double z, double intensity, ReflectionType type) {

	Reflection *new_reflection;

	if ( reflectionlist->list_capped ) return NULL;

	if ( reflectionlist->n_reflections > 1e7 ) {
		fprintf(stderr, "More than 10 million reflections on list.  I think this is silly.\n");
		fprintf(stderr, "No further reflections will be stored.  Go and fix the peak detection.\n");
		reflectionlist->list_capped = 1;
	}

	new_reflection = malloc(sizeof(Reflection));
	new_reflection->next = NULL;
	new_reflection->x = x;
	new_reflection->y = y;
	new_reflection->z = z;
	new_reflection->h = 999;
	new_reflection->k = 333;
	new_reflection->l = 111;
	new_reflection->intensity = intensity;
	new_reflection->type = type;
	new_reflection->found = 0;

	if ( reflectionlist->last_reflection ) {
		reflectionlist->last_reflection->next = new_reflection;
		reflectionlist->last_reflection = new_reflection;
	} else {
		reflectionlist->reflections = new_reflection;
		reflectionlist->last_reflection = new_reflection;
	}
	reflectionlist->n_reflections++;

	return new_reflection;

}

double reflectionlist_largest_g(ReflectionList *reflectionlist) {

	double max = 0.0;
	Reflection *reflection;

	reflection = reflectionlist->reflections;
	while ( reflection ) {
		if ( reflection->type == REFLECTION_NORMAL ) {
			double mod;
			mod = modulus(reflection->x, reflection->y, reflection->z);
			if ( mod > max ) max = mod;
		}
		reflection = reflection->next;
	};

	return max;

}

Reflection *reflectionlist_find_nearest(ReflectionList *reflectionlist, double x, double y, double z) {

	double max = +INFINITY;
	Reflection *reflection;
	Reflection *best = NULL;

	reflection = reflectionlist->reflections;
	while ( reflection ) {
		if ( reflection->type == REFLECTION_NORMAL ) {
			double mod;
			mod = modulus(x - reflection->x, y - reflection->y, z - reflection->z);
			if ( mod < max ) {
				max = mod;
				best = reflection;
			}
		}
		reflection = reflection->next;
	};

	return best;

}

Reflection *reflectionlist_find_nearest_longer_unknown(ReflectionList *reflectionlist,
								double x, double y, double z, double min_distance) {

	double max = +INFINITY;
	Reflection *reflection;
	Reflection *best = NULL;

	reflection = reflectionlist->reflections;
	while ( reflection ) {
		if ( (reflection->type == REFLECTION_NORMAL) && (!reflection->found) ) {
			double mod;
			mod = modulus(x - reflection->x, y - reflection->y, z - reflection->z);
			if ( (mod < max) && (mod >= min_distance) ) {
				max = mod;
				best = reflection;
			}
		}
		reflection = reflection->next;
	};

	return best;

}

Reflection *reflectionlist_find_nearest_type(ReflectionList *reflectionlist, double x, double y, double z,
						ReflectionType type) {

	double max = +INFINITY;
	Reflection *reflection;
	Reflection *best = NULL;

	reflection = reflectionlist->reflections;
	while ( reflection ) {
		if ( reflection->type == type ) {
			double mod;
			mod = modulus(x - reflection->x, y - reflection->y, z - reflection->z);
			if ( mod < max ) {
				max = mod;
				best = reflection;
			}
		}
		reflection = reflection->next;
	};

	return best;

}

/* Generate a list of reflections from a unit cell */
ReflectionList *reflection_list_from_cell(Basis *basis) {

	ReflectionList *ordered;
	double max_res;
	signed int h, k, l;
	int max_order_a, max_order_b, max_order_c;

	ordered = reflectionlist_new();

	max_res = 21e9;
	do {
		max_order_a = max_res/modulus(basis->a.x, basis->a.y, basis->a.z);
		max_order_b = max_res/modulus(basis->b.x, basis->b.y, basis->b.z);
		max_order_c = max_res/modulus(basis->c.x, basis->c.y, basis->c.z);
		max_res -= 1e9;
	} while ( (max_order_a * max_order_b * max_order_c * 8) > 1e4 );
	printf("Selected maximum resolution %8.5f nm^-1\n", max_res/1e9);

	for ( h=-max_order_a; h<=max_order_a; h++ ) {
		for ( k=-max_order_b; k<=max_order_b; k++ ) {
			for ( l=-max_order_c; l<=max_order_c; l++ ) {

				double x, y, z;

				x = h*basis->a.x + k*basis->b.x + l*basis->c.x;
				y = h*basis->a.y + k*basis->b.y + l*basis->c.y;
				z = h*basis->a.z + k*basis->b.z + l*basis->c.z;

				if ( ( x*x + y*y + z*z ) <= max_res*max_res ) {
					if ( (h!=0) || (k!=0) || (l!=0) ) {
						Reflection *ref;
						ref = reflection_add(ordered,
							x, y, z, 1.0,
							REFLECTION_GENERATED);
						if ( ref ) {
							ref->h = h;
							ref->k = k;
							ref->l = l;
						}
					}
				}

			}
		}
	}

	return ordered;

}

void reflection_list_from_new_cell(ReflectionList *ordered, Basis *basis) {

	Reflection *ref;

	ref = ordered->reflections;

	while ( ref ) {

		signed int h, k, l;

		h = ref->h;	k = ref->k;	l = ref->l;

		ref->x = h*basis->a.x + k*basis->b.x + l*basis->c.x;
		ref->y = h*basis->a.y + k*basis->b.y + l*basis->c.y;
		ref->z = h*basis->a.z + k*basis->b.z + l*basis->c.z;

		ref = ref->next;

	}

}

/* Return true if the reflection is of type h00, 0k0 or 0l0 */
int reflection_is_easy(Reflection *reflection) {

	if ( reflection->h ) return !(reflection->k || reflection->l);
	if ( reflection->k ) return !(reflection->h || reflection->l);
	if ( reflection->l ) return !(reflection->h || reflection->k);

	return 0;	/* 000 */

}

Reflection *reflectionlist_find(ReflectionList *reflectionlist, signed int h, signed int k, signed int l) {

	Reflection *reflection;

	reflection = reflectionlist->reflections;
	while ( reflection ) {
		if ( (reflection->h==h) && (reflection->k==k) && (reflection->l==l) ) {
			return reflection;
		}
		reflection = reflection->next;
	};

	return NULL;

}
