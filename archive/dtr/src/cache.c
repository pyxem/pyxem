/*
 * cache.c
 *
 * Save a list of features from images to save recalculation
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

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "image.h"

typedef struct {
	char		top[16];
	uint32_t	n_images;
} CacheHeader;


int cache_save(ImageList *images, char *cache_filename) {

	FILE		*fh;
	CacheHeader	ch;
	uint32_t	*index;
	size_t		index_size;
	int		i;
	
	fh = fopen(cache_filename, "wb");
	if ( !fh ) {
		fprintf(stderr, "Couldn't open feature cache file.\n");
		return -1;
	}
	
	memcpy(&ch.top, "DTR-1.0.6\0\0\0\0\0\0\0", 16);
	ch.n_images = images->n_images;
	fwrite(&ch, sizeof(CacheHeader), 1, fh);
	
	index_size = images->n_images * 2 * sizeof(uint32_t);
	index = malloc(index_size);
	/* Dummy write to reserve space in the file */
	memset(index, 0, index_size);
	if ( fwrite(index, index_size, 1, fh) != 1 ) {
		fprintf(stderr, "Couldn't perform dummy index write.\n");
		fclose(fh);
		return -1;
	}
	
	for ( i=0; i<images->n_images; i++ ) {
		
		uint32_t size;
		
		size = images->images[i].features->n_features * sizeof(ImageFeature);
		index[2*i] = ftell(fh);
		index[2*i + 1] = images->images[i].features->n_features;
		
		if ( fwrite(images->images[i].features->features, size, 1, fh) != 1 ) {
			fprintf(stderr, "Couldn't write feature list for image %i.\n", i);
			fclose(fh);
			return -1;
		}
		
	}
	
	/* Actual write */
	if ( fseek(fh, sizeof(CacheHeader), SEEK_SET) ) {
		fprintf(stderr, "Couldn't seek for index write.\n");
		fclose(fh);
		return -1;
	}
	if ( fwrite(index, index_size, 1, fh) != 1 ) {
		fprintf(stderr, "Couldn't write index write.\n");
		fclose(fh);
		return -1;
	}
	
	fclose(fh);
	
	return 0;
	
}

int cache_load(ImageList *images, const char *filename) {

	FILE		*fh;
	CacheHeader	ch;
	uint32_t	*index;
	size_t		index_size;
	int		i;
	
	fh = fopen(filename, "rb");
	if ( !fh ) {
		fprintf(stderr, "Couldn't open cache file.\n");
		return -1;
	}
	
	if ( fread(&ch, sizeof(CacheHeader), 1, fh) != 1 ) {
		fprintf(stderr, "Couldn't read cache header.\n");
		fclose(fh);
		return -1;
	}
	
	/* Format check */
	if ( strncmp(ch.top, "DTR-1.0.6\0", 8) != 0 ) {
		fprintf(stderr, "Can't read this cache format.\n");
		fclose(fh);
		return -1;
	}
	
	/* Muppet check */
	if ( ch.n_images != images->n_images ) {
		fprintf(stderr, "Number of images in cache doesn't match.\n");
		fclose(fh);
		return -1;
	}
	
	index_size = images->n_images * 2 * sizeof(uint32_t);
	index = malloc(index_size);
	if ( fread(index, index_size, 1, fh) != 1 ) {
		fprintf(stderr, "Couldn't read index.\n");
		fclose(fh);
		free(index);
		return -1;
	}
	
	for ( i=0; i<images->n_images; i++ ) {
	
		uint32_t size;
		uint32_t offs;
		int j;
		
		offs = index[2*i];
		size = index[2*i + 1];
		
		if ( fseek(fh, offs, SEEK_SET) ) {
			fprintf(stderr, "Couldn't seek to feature list for image %i.\n", i);
			fclose(fh);
			return -1;
		}
		
		if ( images->images[i].features ) image_feature_list_free(images->images[i].features);
		images->images[i].features = image_feature_list_new();
		if ( !images->images[i].features ) {
			fprintf(stderr, "Couldn't allocate feature list for image %i.\n", i);
			fclose(fh);
			return -1;
		}
		
		assert(images->images[i].features->features == NULL);
		images->images[i].features->features = malloc(size*sizeof(ImageFeature));
		if ( !images->images[i].features->features ) {
			fprintf(stderr, "Couldn't allocate feature list block for image %i.\n", i);
			fclose(fh);
			return -1;
		}
		
		if ( fread(images->images[i].features->features, size*sizeof(ImageFeature), 1, fh) != 1 ) {
			fprintf(stderr, "Couldn't read feature list for image %i.\n", i);
			fclose(fh);
			return -1;
		}
		
		images->images[i].features->n_features = size;
		
		/* Set "parent" fields for all the features */
		for ( j=0; j<images->images[i].features->n_features; j++ ) {
			images->images[i].features->features[j].parent = &images->images[i];
		}
		
	}
	
	fclose(fh);
	free(index);
	
	return 0;
}

