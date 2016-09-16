/*
 * readpng.c
 *
 * Read PNG images
 *
 * (c) 2007-2009 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <png.h>
#include <math.h>

#include "control.h"
#include "reflections.h"
#include "itrans.h"

int readpng_read(const char *filename, double tilt, ControlContext *ctx)
{
	FILE *fh;
	png_bytep header;
	png_structp png_ptr;
	png_infop info_ptr;
	png_infop end_info;
	unsigned int width;
	unsigned int height;
	unsigned int bit_depth;
	unsigned int channels;
	png_bytep *row_pointers;
	unsigned int x;
	unsigned int y;
	uint16_t *image;

	/* Open file */
	fh = fopen(filename, "rb");
	if ( !fh ) {
		printf("RI: Couldn't open file '%s'\n", filename);
		return -1;
	}

	/* Check it's actually a PNG file */
	header = malloc(8);
	fread(header, 1, 8, fh);
	if ( png_sig_cmp(header, 0, 8)) {
		printf("RI: Not a PNG file.\n");
		free(header);
		fclose(fh);
		return -1;
	}
	free(header);

	png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
					 NULL, NULL, NULL);
	if ( !png_ptr ) {
		printf("RI: Couldn't create PNG read structure.\n");
		fclose(fh);
		return -1;
	}

	info_ptr = png_create_info_struct(png_ptr);
	if ( !info_ptr ) {
		png_destroy_read_struct(&png_ptr, (png_infopp)NULL,
					(png_infopp)NULL);
		printf("RI: Couldn't create PNG info structure.\n");
		fclose(fh);
		return -1;
	}

	end_info = png_create_info_struct(png_ptr);
	if ( !end_info ) {
		png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
		printf("RI: Couldn't create PNG end info structure.\n");
		fclose(fh);
		return -1;
	}

	if ( setjmp(png_jmpbuf(png_ptr)) ) {
		png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
		fclose(fh);
		printf("RI: PNG read failed.\n");
		return -1;
	}

	png_init_io(png_ptr, fh);
	png_set_sig_bytes(png_ptr, 8);

	/* Read! */
	png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_STRIP_ALPHA, NULL);

	width = png_get_image_width(png_ptr, info_ptr);
	height = png_get_image_height(png_ptr, info_ptr);
	bit_depth = png_get_bit_depth(png_ptr, info_ptr);
	channels = png_get_channels(png_ptr, info_ptr);
	//printf("RI: width=%i, height=%i, depth=%i, channels=%i\n",
	//		width, height, bit_depth, channels);
	if ( (bit_depth != 16) && (bit_depth != 8) ) {
		printf("RI: Whoops! Can't handle images with other"
		       " than 8 or 16 bpp yet...\n");
		png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
		fclose(fh);
		return -1;
	}

	/* Get image data */
	row_pointers = png_get_rows(png_ptr, info_ptr);

	image = malloc(height * width * sizeof(uint16_t));

	for ( y=0; y<height; y++ ) {
		for ( x=0; x<width; x++ ) {

			int val = 0;

			if ( bit_depth == 16 ) {
				int i;
				val = 0;
				for ( i=0; i<channels; i++ ) {
					/* PNG files are big-endian... */
					val += row_pointers[y][(channels*x*2)
								+(2*i)] << 8;
					val += row_pointers[y][(channels*x*2)
								+(2*i)+1];
				}
				val /= channels;
				if ( val > 65535 ) printf("%i\n", val);
			}
			if ( bit_depth == 8 ) {
				int i;
				val = 0;
				for ( i=0; i<channels; i++ ) {
					val += row_pointers[y][(channels*x)+i];
				}
				val /= channels;
			}

			image[x + width*(height-1-y)] = val;

		}
	}

	png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
	fclose(fh);

	image_add(ctx->images, image, width, height, tilt, ctx);

	return 0;

}
