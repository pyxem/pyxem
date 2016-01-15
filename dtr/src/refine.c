/*
 * refine.c
 *
 * Refine the reconstruction
 *
 * (c) 2007-2008 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <gtk/gtk.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "control.h"
#include "displaywindow.h"
#include "image.h"
#include "reproject.h"
#include "mapping.h"
#include "refine.h"
#include "gtk-valuegraph.h"
#include "utils.h"

/* Divide numbers by this for display */
#define DISPFACTOR 1.0e9

/* Number of parameters */
#define NUM_PARAMS 9

/* Refine debug */
#define REFINE_DEBUG 1

/* A simplex is an array of ten of these */
typedef struct {
	double dax;	double dbx;	double dcx;
	double day;	double dby;	double dcy;
	double daz;	double dbz;	double dcz;
} SimplexVertex;

typedef struct {
	signed int h;	signed int k;	signed int l;
	double dx;	double dy;	double dz;
} Deviation;

void refine_do_sequence(ControlContext *ctx) {

	double omega_offs;
	int idx;
	double *fit_vals;
	GtkWidget *window_fit;
	GtkWidget *graph_fit;
	double fit_best, omega_offs_best;
	int j;
	
	fit_vals = malloc(401*sizeof(double));
	idx = 0;
	
	fit_best = 1000.0e9;
	omega_offs_best = 0.0;
	for ( omega_offs=-deg2rad(2.0); omega_offs<=deg2rad(2.0); omega_offs+=deg2rad(0.01) ) {

		double fit;
		int i;
		Basis cell_copy;
		
		cell_copy = *ctx->cell;
		
		for ( i=0; i<ctx->images->n_images; i++ ) {
			ctx->images->images[i].omega += omega_offs;
		}
		reproject_lattice_changed(ctx);
		
		fit = refine_do_cell(ctx);
		printf("RF:                           omega_offs=%f deg, fit=%f nm^-1\n", rad2deg(omega_offs),
												fit/DISPFACTOR);
		fit_vals[idx++] = fit;
		if ( fit < fit_best ) {
			fit_best = fit;
			omega_offs_best = omega_offs;
		}
		
		for ( i=0; i<ctx->images->n_images; i++ ) {
			ctx->images->images[i].omega -= omega_offs;
		}
		*ctx->cell = cell_copy;

	}
	
	window_fit = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	gtk_window_set_default_size(GTK_WINDOW(window_fit), 640, 256);
	gtk_window_set_title(GTK_WINDOW(window_fit), "Omega-Search Graph: Fit");
	graph_fit = gtk_value_graph_new();
	gtk_value_graph_set_data(GTK_VALUE_GRAPH(graph_fit), fit_vals, idx);
	gtk_container_add(GTK_CONTAINER(window_fit), graph_fit);
	gtk_widget_show_all(window_fit);
	
	/* Perform final refinement */
	printf("Best omega offset = %f deg (%f nm^-1)\n", rad2deg(omega_offs_best), fit_best/DISPFACTOR);
	for ( j=0; j<ctx->images->n_images; j++ ) {
		ctx->images->images[j].omega += omega_offs_best;
	}
	refine_do_cell(ctx);
	reproject_lattice_changed(ctx);
	mapping_adjust_axis(ctx, omega_offs_best);

}

static double refine_mean_dev(Deviation *d, int nf, SimplexVertex *s, int i) {

	double fom = 0.0;
	int f;
	
	for ( f=0; f<nf; f++ ) {
	
		double xdf, ydf, zdf;
		
		xdf = d[f].h*s[i].dax + d[f].k*s[i].dbx + d[f].l*s[i].dcx;
		ydf = d[f].h*s[i].day + d[f].k*s[i].dby + d[f].l*s[i].dcy;
		zdf = d[f].h*s[i].daz + d[f].k*s[i].dbz + d[f].l*s[i].dcz;
		xdf -= d[f].dx;
		ydf -= d[f].dy;
		zdf -= d[f].dz;
		
		fom += sqrt(xdf*xdf + ydf*ydf + zdf*zdf);
		
	}
	
//	return fabs(s[i].dax-10.0) + fabs(s[i].day-20.0) + fabs(s[i].dbx-30.0) + fabs(s[i].dby-40.0);
	
	return fom/nf;

}

static void refine_display_simplex(SimplexVertex sv) {

	printf("%9.6f %9.6f %9.6f %9.6f %9.6f %9.6f %9.6f %9.6f %9.6f\n",
				sv.dax/DISPFACTOR, sv.day/DISPFACTOR, sv.daz/DISPFACTOR,
				sv.dbx/DISPFACTOR, sv.dby/DISPFACTOR, sv.dbz/DISPFACTOR,
				sv.dcx/DISPFACTOR, sv.dcy/DISPFACTOR, sv.dcz/DISPFACTOR);
	
}

/* Expand the simplex across from vertex v_worst by factor 'fac'.
 *	fac = -1 is a reflection
 *	fac = +n is a 1d expansion
 */
static void refine_simplex_transform(SimplexVertex *s, int v_worst, double fac) {

	SimplexVertex centre;
	int i, nv;
	
	/* Average the coordinates of the non-worst vertices to
	 *  get the centre of the opposite face. */
	centre.dax = 0.0;	centre.day = 0.0;	centre.daz = 0.0;
	centre.dbx = 0.0;	centre.dby = 0.0;	centre.dbz = 0.0;
	centre.dcx = 0.0;	centre.dcy = 0.0;	centre.dcz = 0.0;
	nv = 0;
	for ( i=0; i<=NUM_PARAMS; i++ ) {
		if ( i != v_worst ) {
			centre.dax += s[i].dax;  centre.day += s[i].day;  centre.daz += s[i].daz;
			centre.dbx += s[i].dbx;  centre.dby += s[i].dby;  centre.dbz += s[i].dbz;
			centre.dcx += s[i].dcx;  centre.dcy += s[i].dcy;  centre.dcz += s[i].dcz;
			nv++;
		}
	}
	centre.dax /= nv;	centre.day /= nv;	centre.daz /= nv;
	centre.dbx /= nv;	centre.dby /= nv;	centre.dbz /= nv;
	centre.dcx /= nv;	centre.dcy /= nv;	centre.dcz /= nv;
	
//	printf("Before transformation: ");
//	refine_display_simplex(s[v_worst]);
	
//	printf("             Midpoint: ");
//	refine_display_simplex(centre);
	
	/* Do the transformation */
	s[v_worst].dax = centre.dax + fac * (s[v_worst].dax - centre.dax);
	s[v_worst].day = centre.day + fac * (s[v_worst].day - centre.day);
	s[v_worst].daz = centre.daz + fac * (s[v_worst].daz - centre.daz);
	s[v_worst].dbx = centre.dbx + fac * (s[v_worst].dbx - centre.dbx);
	s[v_worst].dby = centre.dby + fac * (s[v_worst].dby - centre.dby);
	s[v_worst].dbz = centre.dbz + fac * (s[v_worst].dbz - centre.dbz);
	s[v_worst].dcx = centre.dcx + fac * (s[v_worst].dcx - centre.dcx);
	s[v_worst].dcy = centre.dcy + fac * (s[v_worst].dcy - centre.dcy);
	s[v_worst].dcz = centre.dcz + fac * (s[v_worst].dcz - centre.dcz);
	
	if ( REFINE_DEBUG ) {
		printf(" After transformation: ");
		refine_display_simplex(s[v_worst]);
	}
	
}

/* Contract vertex v of simplex s towards vertex v_best */
static void refine_simplex_contract(SimplexVertex *s, int v, int v_best) {
	
	s[v].dax = s[v_best].dax + 0.5 * (s[v].dax - s[v_best].dax);
	s[v].day = s[v_best].day + 0.5 * (s[v].day - s[v_best].day);
	s[v].daz = s[v_best].daz + 0.5 * (s[v].daz - s[v_best].daz);
	s[v].dbx = s[v_best].dbx + 0.5 * (s[v].dbx - s[v_best].dbx);
	s[v].dby = s[v_best].dby + 0.5 * (s[v].dby - s[v_best].dby);
	s[v].dbz = s[v_best].dbz + 0.5 * (s[v].dbz - s[v_best].dbz);
	s[v].dcx = s[v_best].dcx + 0.5 * (s[v].dcx - s[v_best].dcx);
	s[v].dcy = s[v_best].dcy + 0.5 * (s[v].dcy - s[v_best].dcy);
	s[v].dcz = s[v_best].dcz + 0.5 * (s[v].dcz - s[v_best].dcz);
	
}

static double refine_iteration(SimplexVertex *s, Deviation *d, int nf) {

	int v_worst, v_best, v_second_worst, i;
	double fom_worst, fom_new, fom_best, fom_second_worst;
	
	/* Find the least favourable vertex of the simplex */
	v_worst = 0;
	fom_worst = 0.0;
	v_best = 0;
	fom_best = 100e9;
	v_second_worst = 0;
	fom_second_worst = 0.0;
	if ( REFINE_DEBUG ) printf("Vertex    FoM/nm^-1\n");
	for ( i=0; i<=NUM_PARAMS; i++ ) {
	
		double fom;
		
		fom = refine_mean_dev(d, nf, s, i);
		
		if ( REFINE_DEBUG ) printf("%6i     %8f\n", i, fom/DISPFACTOR);
		if ( fom > fom_worst ) {
			v_second_worst = v_worst;
			fom_second_worst = fom_worst;
			fom_worst = fom;
			v_worst = i;
		}
		if ( fom < fom_best ) {
			fom_best = fom;
			v_best = i;
		}
		
	}
	if ( REFINE_DEBUG ) printf("The worst vertex is number %i\n", v_worst);
	
	/* Reflect this vertex across the opposite face */
	refine_simplex_transform(s, v_worst, -1.0);
	
	/* Is the worst vertex any better? */
	fom_new = refine_mean_dev(d, nf, s, v_worst);
	if ( REFINE_DEBUG ) printf("New mean deviation for the worst vertex after reflection is %f nm^-1\n",
				   fom_new/DISPFACTOR);
	if ( fom_new > fom_worst ) {
	
		double fom_new_new;
		
		/* It's worse than before.  Contract in 1D and see if that helps. */
		if ( REFINE_DEBUG ) printf("Worse.  Trying a 1D contraction...\n");
		/* Minus puts it back on the original side of the 'good' face */
		refine_simplex_transform(s, v_worst, -0.5);
		fom_new_new = refine_mean_dev(d, nf, s, v_worst);
		if ( REFINE_DEBUG ) printf("Mean deviation after 1D contraction is %f nm^-1\n",
					   fom_new_new/DISPFACTOR);
		if ( fom_new_new > fom_second_worst ) {
			
			int i;
			
			if ( REFINE_DEBUG ) printf("Not as good as the second worst vertex: contracting around the "
						   "best vertex (%i)\n", v_best);
			for ( i=0; i<=NUM_PARAMS; i++ ) {
				if ( i != v_best ) refine_simplex_contract(s, i, v_best);
			}
		
		}
		
	} else if ( fom_new < fom_worst ) {
	
		/* It's better.  Try to expand in this direction */
		double fom_new_new;
		SimplexVertex save;
		
		if ( REFINE_DEBUG ) printf("This is better.  Trying to expand...\n");
		
		save = s[v_worst];
		refine_simplex_transform(s, v_worst, 2.0);	/* +ve means stay on this side of the 'good' face */
		/* Better? */
		fom_new_new = refine_mean_dev(d, nf, s, v_worst);
		if ( REFINE_DEBUG ) printf("Mean deviation after expansion is %f nm^-1\n", fom_new_new/DISPFACTOR);
		if ( fom_new_new > fom_new ) {
			/* "Got too confident" */
			s[v_worst] = save;
			if ( REFINE_DEBUG ) printf("Got too confident - reverting\n");
		} else {
			if ( REFINE_DEBUG ) printf("Better still.  Great.\n");
		}
	
	} else {
	
		printf("No change!\n");
		
	}
	
	/* Check convergence and return */
	fom_worst = 0.0;
	fom_best = 100e9;
	for ( i=0; i<=NUM_PARAMS; i++ ) {
		double fom;
		fom = refine_mean_dev(d, nf, s, i);
		if ( fom > fom_worst ) {
			fom_worst = fom;
		}
		if ( fom < fom_best ) {
			fom_best = fom;
		}
	}
	
	printf("Simplex size: %e - %e = %e\n", fom_worst/DISPFACTOR, fom_best/DISPFACTOR, (fom_worst - fom_best)/DISPFACTOR);
	
	return fom_best;

}

double refine_do_cell(ControlContext *ctx) {

	SimplexVertex s[10];
	Deviation *d;
	double delta;
	int i, nf, f, it, maxiter;
	const double tol = 0.001e9;	/* Stopping condition */
	//const double tol = 0.001;	/* For testing */
	
	if ( !ctx->cell_lattice ) {
		displaywindow_error("No reciprocal unit cell has been found.", ctx->dw);
		return -1;
	}
	
	if ( ctx->images->n_images == 0 ) {
		displaywindow_error("There are no images to refine against.", ctx->dw);
		return -1;
	}
	
	/* Determine the size of the 'deviation table' */
	nf = 0;
	for ( i=0; i<ctx->images->n_images; i++ ) {
		
		int j;
		
		if ( !ctx->images->images[i].rflist ) {
			ctx->images->images[i].rflist = reproject_get_reflections(&ctx->images->images[i],
										  ctx->cell_lattice);
		}
		
		for ( j=0; j<ctx->images->images[i].rflist->n_features; j++ ) {
			if ( ctx->images->images[i].rflist->features[j].partner != NULL ) nf++;
		}
		
		printf("%i features from image %i\n", nf, i);
		
	}
	if ( REFINE_DEBUG ) printf("RF: There are %i partnered features in total\n", nf);
	
	/* Initialise the 'deviation table' */
	d = malloc(nf*sizeof(Deviation));
	f = 0;
	for ( i=0; i<ctx->images->n_images; i++ ) {
		
		ImageRecord *image;
		int j;
		
		image = &ctx->images->images[i];
		
		for ( j=0; j<ctx->images->images[i].rflist->n_features; j++ ) {
		
			ImageFeature *rf;
			double dix, diy, dx, dy;
			double dlx, dly, dlz;
			double old_x, old_y;
			
			rf = &image->rflist->features[j];
			if ( !rf->partner ) continue;
			
			d[f].h = rf->reflection->h;
			d[f].k = rf->reflection->k;
			d[f].l = rf->reflection->l;
			
			/* Determine the difference vector */
			dix = rf->partner->x - rf->x;
			diy = rf->partner->y - rf->y;
			printf("RF: Feature %3i: %3i %3i %3i dev = %+9.5f %+9.5f px ", j, d[f].h, d[f].k, d[f].l,
													dix, diy);
			
			old_x = rf->partner->x;
			old_y = rf->partner->y;
			rf->partner->x = dix + rf->partner->parent->x_centre;
			rf->partner->y = diy + rf->partner->parent->y_centre;
			mapping_scale(rf->partner, &dx, &dy);
			mapping_rotate(dx, dy, 0.0, &dlx, &dly, &dlz, image->omega, image->tilt);
			rf->partner->x = old_x;
			rf->partner->y = old_y;
			double mod = sqrt(dx*dx + dy*dy)/DISPFACTOR;
			printf("=> %+10.5f %+10.5f %+10.5f nm^-1 (length %10.5f nm^1)\n", dlx/DISPFACTOR, dly/DISPFACTOR, dlz/DISPFACTOR, mod);
			
			d[f].dx = dlx;
			d[f].dy = dly;
			d[f].dz = dlz;
			
			f++;
		
		}
		
	}
	assert( f == nf );
	
	/* Initialise the simplex */
	delta = 0.01e9;
	s[0].dax = 0.0;	s[0].dbx = 0.0;	s[0].dcx = 0.0;
	s[0].day = 0.0;	s[0].dby = 0.0;	s[0].dcy = 0.0;
	s[0].daz = 0.0;	s[0].dbz = 0.0;	s[0].dcz = 0.0;
	memcpy(&s[1], &s[0], sizeof(SimplexVertex));	s[1].dax = delta;
	memcpy(&s[2], &s[0], sizeof(SimplexVertex));	s[2].day = delta;
	memcpy(&s[3], &s[0], sizeof(SimplexVertex));	s[3].dbx = delta;
	memcpy(&s[4], &s[0], sizeof(SimplexVertex));	s[4].dby = delta;  /* 2d vertices first */
	memcpy(&s[5], &s[0], sizeof(SimplexVertex));	s[5].daz = delta;
	memcpy(&s[6], &s[0], sizeof(SimplexVertex));	s[6].dbz = delta;
	memcpy(&s[7], &s[0], sizeof(SimplexVertex));	s[7].dcx = delta;
	memcpy(&s[8], &s[0], sizeof(SimplexVertex));	s[8].dcy = delta;
	memcpy(&s[9], &s[0], sizeof(SimplexVertex));	s[9].dcz = delta;
	
	/* Iterate */
	maxiter = 10000;
	for ( it=0; it<maxiter; it++ ) {
	
		double conv;
		
		//for ( i=0; i<5; i++ ) {
		//	refine_display_simplex(s[i]);
		//}
		
		if ( REFINE_DEBUG ) printf("------------------- Simplex method iteration %i -------------------\n", it);
		conv = refine_iteration(s, d, nf);
		if ( conv < tol ) {
			if ( REFINE_DEBUG ) printf("RF: Converged after %i iterations (%f nm^-1)\n", it,
						   conv/DISPFACTOR);
			break;
		}
		
	}
	if ( it == maxiter ) printf("RF: Did not converge.\n");
	
	/* Apply the final values to the cell */
	ctx->cell->a.x += s[0].dax;	ctx->cell->b.x += s[0].dbx;	ctx->cell->c.x += s[0].dcx;
	ctx->cell->a.y += s[0].day;	ctx->cell->b.y += s[0].dby;	ctx->cell->c.y += s[0].dcy;
	ctx->cell->a.z += s[0].daz;	ctx->cell->b.z += s[0].dbz;	ctx->cell->c.z += s[0].dcz;
	
	ctx->images->images[ctx->dw->cur_image].rflist = NULL;
	reproject_lattice_changed(ctx);
	displaywindow_update(ctx->dw);
	
	return refine_mean_dev(d, nf, s, 0);
		
}

