/*
 * dirax.c
 *
 * Invoke the DirAx auto-indexing program
 * also: handle DirAx input files
 *
 * (c) 2007-2009 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <glib.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <pty.h>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <assert.h>
#include <sys/ioctl.h>
#include <termio.h>
#include <sgtty.h>

#include "control.h"
#include "reflections.h"
#include "utils.h"
#include "basis.h"
#include "displaywindow.h"
#include "reproject.h"

typedef enum {
	DIRAX_INPUT_NONE,
	DIRAX_INPUT_LINE,
	DIRAX_INPUT_PROMPT
} DirAxInputType;

static void dirax_parseline(const char *line, ControlContext *ctx)
{
	int i, rf;
	char *copy;

	copy = strdup(line);
	for ( i=0; i<strlen(copy); i++ ) {
		if ( copy[i] == '\r' ) copy[i]='r';
		if ( copy[i] == '\n' ) copy[i]='\0';
	}
	printf("DX: DirAx: %s\n", copy);
	free(copy);

	if ( strstr(line, "reflections from file") ) {
		displaywindow_error("DirAx can't understand this data.",
								ctx->dw);
		return;
	}

	/* Is this the first line of a unit cell specification? */
	rf = 0; i = 0;
	while ( (i<strlen(line)) && ((line[i] == 'R')
				|| (line[i] == 'D') || (line[i] == ' ')) ) {
		if ( line[i] == 'R' ) rf = 1;
		if ( (line[i] == 'D') && rf ) {
			ctx->dirax_read_cell = 1;
			if ( ctx->cell ) {
				free(ctx->cell);
			}
			ctx->cell = malloc(sizeof(Basis));
			ctx->cell->a.x = 0.0;  ctx->cell->a.y = 0.0;
						ctx->cell->a.z = 0.0;
			ctx->cell->b.x = 0.0;  ctx->cell->b.y = 0.0;
						ctx->cell->b.z = 0.0;
			ctx->cell->c.x = 0.0;  ctx->cell->c.y = 0.0;
						ctx->cell->c.z = 0.0;
			return;
		}
		i++;
	}

	/* Parse unit cell vectors as appropriate */
	if ( ctx->dirax_read_cell == 1 ) {
		/* First row of unit cell values */
		float x1, x2, x3;
		sscanf(line, "%f %f %f", &x1, &x2, &x3);
		ctx->cell->a.x = x1*1e10;  ctx->cell->b.x = x2*1e10;
						ctx->cell->c.x = x3*1e10;
		ctx->dirax_read_cell++;
		return;
	} else if ( ctx->dirax_read_cell == 2 ) {
		/* First row of unit cell values */
		float y1, y2, y3;
		sscanf(line, "%f %f %f", &y1, &y2, &y3);
		ctx->cell->a.y = y1*1e10;  ctx->cell->b.y = y2*1e10;
						ctx->cell->c.y = y3*1e10;
		ctx->dirax_read_cell++;
		return;
	} else if ( ctx->dirax_read_cell == 3 ) {
		/* First row of unit cell values */
		float z1, z2, z3;
		sscanf(line, "%f %f %f", &z1, &z2, &z3);
		ctx->cell->a.z = z1*1e10;  ctx->cell->b.z = z2*1e10;
						ctx->cell->c.z = z3*1e10;
		printf("DX: Read a reciprocal unit cell\n");
		displaywindow_update(ctx->dw);
		reproject_lattice_changed(ctx);
		ctx->dirax_read_cell = 0;
		return;
	}

	ctx->dirax_read_cell = 0;
}

static void dirax_sendline(const char *line, ControlContext *ctx)
{
	char *copy;
	int i;

	write(ctx->dirax_pty, line, strlen(line));

	copy = strdup(line);
	for ( i=0; i<strlen(copy); i++ ) {
		if ( copy[i] == '\r' ) copy[i]='\0';
		if ( copy[i] == '\n' ) copy[i]='\0';
	}
	printf("DX: Sent '%s'\n", copy);	/* No newline here */
	free(copy);
}

/* Send a "user" command to DirAx, failing if DirAx is not idle */
static void dirax_sendline_if_idle(const char *line, ControlContext *ctx)
{
	if ( ctx->dirax_step != 0 ) {
		printf("DX: DirAx not idle\n");
		return;
	}

	dirax_sendline(line, ctx);
}

static void dirax_send_next(ControlContext *ctx)
{
	switch ( ctx->dirax_step ) {

		case 1 : {
			dirax_sendline("\\echo off\n", ctx);
			ctx->dirax_step++;
			break;
		}

		case 2 : {
			dirax_sendline("read dtr.drx\n", ctx);
			ctx->dirax_step++;
			break;
		}

		case 3 : {
			dirax_sendline("dmax 10\n", ctx);
			ctx->dirax_step++;
			break;
		}

		case 4 : {
			dirax_sendline("indexfit 2\n", ctx);
			ctx->dirax_step++;
			break;
		}

		case 5 : {
			dirax_sendline("levelfit 200\n", ctx);
			ctx->dirax_step++;
			break;
		}

		case 6 : {
			dirax_sendline("go\n", ctx);
			ctx->dirax_step++;
			break;
		}

		case 7 : {
			dirax_sendline("cell\n", ctx);
			ctx->dirax_step++;
			break;
		}

		default: {
			ctx->dirax_step = 0;
			printf("DX: Prompt.  DirAx is idle\n");
		}

	}
}

static gboolean dirax_readable(GIOChannel *dirax, GIOCondition condition,
				ControlContext *ctx)
{
	int rval;

	rval = read(ctx->dirax_pty, ctx->dirax_rbuffer+ctx->dirax_rbufpos,
			ctx->dirax_rbuflen-ctx->dirax_rbufpos);

	if ( (rval == -1) || (rval == 0) ) {

		printf("DX: Lost connection to DirAx\n");
		waitpid(ctx->dirax_pid, NULL, 0);
		g_io_channel_shutdown(ctx->dirax, FALSE, NULL);
		ctx->dirax = NULL;
		displaywindow_update_dirax(ctx, ctx->dw);
		return FALSE;

	} else {

		int no_string = 0;

		ctx->dirax_rbufpos += rval;
		assert(ctx->dirax_rbufpos <= ctx->dirax_rbuflen);

		while ( (!no_string) && (ctx->dirax_rbufpos > 0) ) {

			int i;
			int block_ready = 0;
			DirAxInputType type = DIRAX_INPUT_NONE;

			/* See if there's a full line in the buffer yet */
			for ( i=0; i<ctx->dirax_rbufpos-1; i++ ) {
				/* Means the last value looked at is rbufpos-2 */

				/* Is there a prompt in the buffer? */
				if ( i+7 <= ctx->dirax_rbufpos ) {
					if ( (strncmp(ctx->dirax_rbuffer+i,
							"Dirax> ", 7) == 0)
					  || (strncmp(ctx->dirax_rbuffer+i,
					  		"PROMPT:", 7) == 0) ) {
						block_ready = 1;
						type = DIRAX_INPUT_PROMPT;
						break;
					}
				}

				if ( (ctx->dirax_rbuffer[i] == '\r')
				  && (ctx->dirax_rbuffer[i+1] == '\n') ) {
					block_ready = 1;
					type = DIRAX_INPUT_LINE;
					break;
				}

			}

			if ( block_ready ) {

				unsigned int new_rbuflen;
				unsigned int endbit_length;

				switch ( type ) {

					case DIRAX_INPUT_LINE : {

						char *block_buffer = NULL;

						block_buffer = malloc(i+1);
						memcpy(block_buffer,
							ctx->dirax_rbuffer, i);
						block_buffer[i] = '\0';

						if ( block_buffer[0] == '\r' ) {
							memmove(block_buffer,
							  block_buffer+1, i);
						}

						dirax_parseline(block_buffer,
									ctx);
						free(block_buffer);
						endbit_length = i+2;

						break;

					}

					case DIRAX_INPUT_PROMPT : {

						dirax_send_next(ctx);
						endbit_length = i+7;
						break;

					}

					default : {
						printf(
			"DX: Unrecognised input mode (this never happens!)\n");
						abort();
					}

				}

				/* Now the block's been parsed, it should be
				 * forgotten about */
				memmove(ctx->dirax_rbuffer,
					ctx->dirax_rbuffer + endbit_length,
					ctx->dirax_rbuflen - endbit_length);

				/* Subtract the number of bytes removed */
				ctx->dirax_rbufpos = ctx->dirax_rbufpos
								- endbit_length;
				new_rbuflen = ctx->dirax_rbuflen
								- endbit_length;
				if ( new_rbuflen == 0 ) {
					new_rbuflen = 256;
				}
				ctx->dirax_rbuffer = realloc(ctx->dirax_rbuffer,
								new_rbuflen);
				ctx->dirax_rbuflen = new_rbuflen;

			} else {

				if ( ctx->dirax_rbufpos==ctx->dirax_rbuflen ) {

					/* More buffer space is needed */
					ctx->dirax_rbuffer = realloc(
						ctx->dirax_rbuffer,
						ctx->dirax_rbuflen + 256);
					ctx->dirax_rbuflen = ctx->dirax_rbuflen
									+ 256;
					/* The new space gets used at the next
					 * read, shortly... */

				}
				no_string = 1;

			}

		}

	}

	return TRUE;
}

void dirax_stop(ControlContext *ctx)
{
	dirax_sendline_if_idle("end\n", ctx);
}

void dirax_rerun(ControlContext *ctx)
{
	dirax_sendline_if_idle("go\n", ctx);
	ctx->dirax_step = 7;
}

static void dirax_send_random_selection(ReflectionList *r, int n, FILE *fh)
{
	char *used;
	int i;

	used = malloc(n*sizeof(char));

	for ( i=0; i<n; i++ ) {
		used[i] = '-';
	}

	i = 0;
	while ( i < 1000 ) {

		Reflection *ref;
		int j;
		long long int ra;

		ra = ((long long int)random() * (long long int)n);
		ra /= RAND_MAX;
		if ( used[ra] == 'U' ) {
			continue;
		}


		/* Dig out the correct reflection. A little faffy
		 * because of the linked list */
		ref = r->reflections;
		for ( j=0; j<ra; j++ ) {
			ref = ref->next;
		}

		/* Limit resolution of reflections */
//		if (	( (ref->x/1e9)*(ref->x/1e9)
//			+ (ref->y/1e9)*(ref->y/1e9)
//			+ (ref->z/1e9)*(ref->z/1e9) ) > (20e9)*(20e9) )
//				continue;

		fprintf(fh, "%10f %10f %10f %8f\n",
					ref->x/1e10, ref->y/1e10,
					ref->z/1e10, ref->intensity);
		used[ra] = 'U';

		i++;

	}
}

void dirax_invoke(ControlContext *ctx)
{
	FILE *fh;
	Reflection *ref;
	unsigned int opts;
	int saved_stderr;
	int n;

	if ( ctx->dirax ) {
		dirax_rerun(ctx);
		return;
	}

	printf("DX: Starting DirAx...\n");

	fh = fopen("dtr.drx", "w");
	if ( !fh ) {
		printf("DX: Couldn't open temporary file dtr.drx\n");
		return;
	}
	fprintf(fh, "%f\n", 0.5);  /* Lie about the wavelength.  */

	n = ctx->reflectionlist->n_reflections;
	printf("DX: There are %i reflections - ", n);
	if ( n > 1000 ) {
		printf("sending a random selection to DirAx\n");
		dirax_send_random_selection(ctx->reflectionlist, n, fh);
	} else {
		printf("sending them all to DirAx\n");
		ref = ctx->reflectionlist->reflections;
		while ( ref ) {
			fprintf(fh, "%10f %10f %10f %8f\n",
				ref->x/1e10, ref->y/1e10,
				ref->z/1e10, ref->intensity);
			ref = ref->next;
		}
	}
	fclose(fh);

	saved_stderr = dup(STDERR_FILENO);
	ctx->dirax_pid = forkpty(&ctx->dirax_pty, NULL, NULL, NULL);
	if ( ctx->dirax_pid == -1 ) {
		printf("DX: Failed to fork.\n");
		return;
	}
	if ( ctx->dirax_pid == 0 ) {

		/* Child process: invoke DirAx */
		struct termios t;

		/* Turn echo off */
		tcgetattr(STDIN_FILENO, &t);
		t.c_lflag &= ~(ECHO | ECHOE | ECHOK | ECHONL);
		tcsetattr(STDIN_FILENO, TCSANOW, &t);

		/* Reconnect stderr */
		dup2(saved_stderr, STDERR_FILENO);

		execlp("dirax", "", (char *)NULL);
		printf("(from the Other Side) Failed to invoke DirAx.\n");
		_exit(0);

	}

	ctx->dirax_rbuffer = malloc(256);
	ctx->dirax_rbuflen = 256;
	ctx->dirax_rbufpos = 0;

	/* Set non-blocking */
	opts = fcntl(ctx->dirax_pty, F_GETFL);
	fcntl(ctx->dirax_pty, F_SETFL, opts | O_NONBLOCK);

	ctx->dirax_step = 1;	/* This starts the "initialisation" procedure */
	ctx->dirax_read_cell = 0;
	ctx->dirax = g_io_channel_unix_new(ctx->dirax_pty);
	g_io_add_watch(ctx->dirax, G_IO_IN | G_IO_HUP, (GIOFunc)dirax_readable,
									ctx);

	displaywindow_update_dirax(ctx, ctx->dw);

	return;
}

/* Despite being part of the same module, this has very little to do with
 * invoking DirAx */
ReflectionList *dirax_load(const char *filename)
{
	FILE *fh;
	char line[256];
	ReflectionList *list;
	int lambda_set = 0;

	fh = fopen(filename, "r");
	if ( !fh ) {
		printf("Couldn't open file '%s'\n", filename);
		return 0;
	}

	list = reflectionlist_new();

	while ( !feof(fh) ) {

		size_t ptr;
		float lambda, theta, chib, phib;

		fgets(line, 255, fh);
		ptr = skipspace(line);
		if ( line[ptr] == '!' ) continue;
		if ( line[ptr] == '\n' ) continue;
		if ( line[ptr] == '\r' ) continue;
		if ( sscanf(line+ptr, "%f %f %f\n", &theta, &phib,
								&chib) == 3 ) {

			double s, x, y, z;
			float blah, intensity;

			/* Try to find an intensity value. Use dummy value
			 * if it fails */
			if ( sscanf(line+ptr, "%f %f %f %f\n", &blah, &blah,
					&blah, &intensity) != 4 ) {
				intensity = 1.0;
			}

			if ( !lambda_set ) {
				printf("DX: Wavelength not specified\n");
				continue;
			}

			chib = deg2rad(chib);
			phib = deg2rad(phib);
			theta = deg2rad(theta);
			s = 2*(sin(theta)/lambda);
			x = -s*cos(chib)*sin(phib);
			y = +s*cos(chib)*cos(phib);
			z = +s*sin(chib);
			reflection_add(list, x, y, z, 1.0, REFLECTION_NORMAL);

			continue;

		}
		if ( sscanf(line+ptr, "%f\n", &lambda) == 1 ) {
			if ( lambda_set ) {
				printf("DX: Warning: Found something which "
					"looks like a second wavelength\n");
			}
			lambda /= 1e10; /* Convert from A to m */
			lambda_set = 1;
		}

	}

	fclose(fh);

	return list;

}

int dirax_is_drxfile(const char *filename)
{
	FILE *fh;
	float lambda;
	char line[256];

	fh = fopen(filename, "r");
	if ( !fh ) {
		printf("Couldn't open file '%s'\n", filename);
		return 0;
	}

	while ( !feof(fh) ) {

		size_t ptr;

		fgets(line, 255, fh);
		ptr = skipspace(line);
		if ( line[ptr] == '!' ) continue;
		if ( line[ptr] == '\n' ) continue;
		if ( line[ptr] == '\r' ) continue;
		fscanf(fh, "%f\n", &lambda);
		fclose(fh);
		if ( lambda > 0.5 ) {
			return 1;
		} else {
			return 0;
		}

	}

	fclose(fh);

	return 0;
}
