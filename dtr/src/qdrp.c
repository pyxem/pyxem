/*
 * qdrp.c
 *
 * Handle QDRP-style control files
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
#include <string.h>
#include <stdio.h>

#include "readpng.h"
#include "control.h"
#include "reflections.h"
#include "utils.h"

static void qdrp_chomp(char *str) {

	size_t i;
	
	for ( i=0; i<strlen(str); i++ ) {
		if ( str[i] == '\r' ) {
			str[i] = '\0';
			return;
		}
		if ( str[i] == '\n' ) {
			str[i] = '\0';
			return;
		}
	}
}

#define skip_whitespace										\
	while ( (( line[pos] == ' ' ) || ( line[pos] == '\t' )) && ( pos < strlen(line) ) ) {	\
		pos++;										\
	}

#define skip_chars										\
	while ( ( line[pos] != ' ' ) && ( line[pos] != '\t' ) && ( pos < strlen(line) ) ) {	\
		pos++;										\
	}

static int qdrp_parsefileline(ControlContext *ctx, const char *line) {

	size_t pos = 0;
	size_t mark = 0;
	char *tilt_s;
	char *file;
	double tilt;
					
	skip_whitespace;
	mark = pos;
	skip_chars;
	tilt_s = strndup(line+mark, pos-mark);
	
	skip_whitespace;
	mark = pos;
	skip_chars;
	file = strndup(line+mark, pos-mark);
					
	tilt = deg2rad(strtod(tilt_s, NULL));
	free(tilt_s);
	printf("QD: Reading file: Tilt = %f deg, File='%s'\n", rad2deg(tilt), file);
	
	if ( readpng_read(file, tilt, ctx) ) {
		printf("Reconstruction failed.\n");
		return 1;
	}
	
	return 0;
	
}

static int qdrp_parseline(ControlContext *ctx, const char *line) {

	if ( ctx->started ) {
		return qdrp_parsefileline(ctx, line);						
	}
	
	if ( ( line[0] == '-' ) && ( line[1] == '-') ) {
		
		if ( !ctx->camera_length_set ) {
			printf("QD: Parameter 'camera-length' not specified!\n");
			return 1;
		}
					
		if ( !ctx->lambda_set ) {
			printf("QD: Parameter 'lambda' not specified!\n");
			return 1;
		}
					
		if ( !ctx->resolution_set ) {
			printf("QD: Parameter 'resolution' not specified!\n");
			return 1;
		}
					
		if ( !ctx->omega_set ) {
			printf("QD: Parameter 'omega' not specified.\n");
			return 1;
		}
				
		ctx->started = 1;
		ctx->fmode = FORMULATION_CLEN;
						
	}
					
	if ( !ctx->started ) {
						
		size_t pos = 0;
		size_t mark = 0;
						
		skip_whitespace;
						
		if ( strncasecmp(line+pos, "resolution", 10) == 0 ) {
			
			char *resolution_s;
						
			skip_chars;
			skip_whitespace;
			mark = pos;
			skip_chars;
			resolution_s = strndup(line+mark, pos-mark);
			ctx->resolution = strtod(resolution_s, NULL);
			free(resolution_s);
			ctx->resolution_set = 1;
			printf("QD: resolution = %f pixels/m\n", ctx->resolution);	
			
		}
		
		if ( strncasecmp(line+pos, "lambda", 6) == 0 ) {
			
			char *lambda_s;
						
			skip_chars;
			skip_whitespace;
			mark = pos;
			skip_chars;
			lambda_s = strndup(line+mark, pos-mark);
			ctx->lambda = strtod(lambda_s, NULL);
			free(lambda_s);
			ctx->lambda_set = 1;
			printf("QD: lambda = %e m\n", ctx->lambda);	
			
		}
		
		if ( strncasecmp(line+pos, "camera-length", 13) == 0 ) {
			
			char *camera_length_s;
						
			skip_chars;
			skip_whitespace;
			mark = pos;
			skip_chars;
			camera_length_s = strndup(line+mark, pos-mark);
			ctx->camera_length = strtod(camera_length_s, NULL);
			free(camera_length_s);
			ctx->camera_length_set = 1;
			printf("QD: camera-length = %f m\n", ctx->camera_length);	
			
		}
		
		if ( strncasecmp(line+pos, "omega", 5) == 0 ) {
			
			char *omega_s;
			
			skip_chars;
			skip_whitespace;
			mark = pos;
			skip_chars;
			omega_s = strndup(line+mark, pos-mark);
			ctx->omega = deg2rad(strtod(omega_s, NULL));
			free(omega_s);
			ctx->omega_set = 1;
			printf("QD: omega = %f deg\n", rad2deg(ctx->omega));
				
		}
		
		if ( strncasecmp(line+pos, "centre", 6) == 0 ) {
			
			float xc, yc;
			
			if ( sscanf(line+pos, "centre %f %f", &xc, &yc) != 2 ) {
				fprintf(stderr, "Couldn't parse 'centre' line\n");
				return 1;
			}
			ctx->x_centre = xc;
			ctx->y_centre = yc;
			ctx->have_centres = 1;
			printf("QD: centre %.2f, %.2f px\n", ctx->x_centre, ctx->y_centre);

		}
		
	}

	return 0;

}

int qdrp_read(ControlContext *ctx) {
	
	char *line;
	
	line = malloc(256);

	ctx->camera_length_set = 0;
	ctx->omega_set = 0;
	ctx->resolution_set = 0;
	ctx->lambda_set = 0;
	ctx->started = 0;
	
	FILE *fh;
	
	fh = fopen(ctx->filename, "r");
	if ( !fh ) {
		printf("QD: Couldn't open control file '%s'\n", ctx->filename);
		return -1;
	}
	
	while ( !feof(fh) ) {			
		fgets(line, 256, fh);			
		if ( !feof(fh) ) {				
			qdrp_chomp(line);
			if ( strlen(line) == 0 ) continue;			
			if ( line[0] != '#' ) {
				if ( qdrp_parseline(ctx, line) ) {
					fclose(fh);
					free(line);
					return -1;
				}			
			}			
		}
	}
	
	fclose(fh);
	
	free(line);
	
	return 0;	/* Success */
	
}

unsigned int qdrp_is_qdrprc(const char *filename) {

	if ( strstr(filename, "qdrp") ) {
		return 1;
	}

	return 0;

}

