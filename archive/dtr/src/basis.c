/*
 * basis.c
 *
 * Handle basis structures
 *
 * (c) 2007-2009 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#include "reflections.h"
#include "basis.h"
#include "utils.h"
#include "displaywindow.h"
#include "reproject.h"

static void basis_print(Basis *cell)
{
	printf("%12.8f %12.8f %12.8f\n",
				cell->a.x/1e9, cell->a.y/1e9, cell->a.z/1e9);
	printf("%12.8f %12.8f %12.8f\n",
				cell->b.x/1e9, cell->b.y/1e9, cell->b.z/1e9);
	printf("%12.8f %12.8f %12.8f\n",
				cell->c.x/1e9, cell->c.y/1e9, cell->c.z/1e9);
}

static void cell_print(UnitCell *cell)
{
	printf("%12.8f %12.8f %12.8f nm\n",
	       cell->a*1e9, cell->b*1e9, cell->c*1e9);
	printf("%12.8f %12.8f %12.8f deg\n",
	       rad2deg(cell->alpha), rad2deg(cell->beta), rad2deg(cell->gamma));
}

UnitCell basis_get_cell(Basis *basis)
{
	UnitCell cell;
	gsl_matrix *m;
	gsl_matrix *inv;
	gsl_permutation *perm;
	double ax, ay, az, bx, by, bz, cx, cy, cz;
	int s;

	printf("Reciprocal-space cell (nm^-1):\n");
	basis_print(basis);

	m = gsl_matrix_alloc(3, 3);
	gsl_matrix_set(m, 0, 0, basis->a.x);
	gsl_matrix_set(m, 0, 1, basis->b.x);
	gsl_matrix_set(m, 0, 2, basis->c.x);
	gsl_matrix_set(m, 1, 0, basis->a.y);
	gsl_matrix_set(m, 1, 1, basis->b.y);
	gsl_matrix_set(m, 1, 2, basis->c.y);
	gsl_matrix_set(m, 2, 0, basis->a.z);
	gsl_matrix_set(m, 2, 1, basis->b.z);
	gsl_matrix_set(m, 2, 2, basis->c.z);

	perm = gsl_permutation_alloc(m->size1);
	inv = gsl_matrix_alloc(m->size1, m->size2);
	gsl_linalg_LU_decomp(m, perm, &s);
	gsl_linalg_LU_invert(m, perm, inv);
	gsl_permutation_free(perm);
	gsl_matrix_free(m);

	gsl_matrix_transpose(inv);

	ax = gsl_matrix_get(inv, 0, 0);
	bx = gsl_matrix_get(inv, 0, 1);
	cx = gsl_matrix_get(inv, 0, 2);
	ay = gsl_matrix_get(inv, 1, 0);
	by = gsl_matrix_get(inv, 1, 1);
	cy = gsl_matrix_get(inv, 1, 2);
	az = gsl_matrix_get(inv, 2, 0);
	bz = gsl_matrix_get(inv, 2, 1);
	cz = gsl_matrix_get(inv, 2, 2);

	printf("Real-space cell (nm):\n");
	printf("%12.8f %12.8f %12.8f\n", ax*1e9, ay*1e9, az*1e9);
	printf("%12.8f %12.8f %12.8f\n", bx*1e9, by*1e9, bz*1e9);
	printf("%12.8f %12.8f %12.8f\n", cx*1e9, cy*1e9, cz*1e9);

	cell.a = sqrt(ax*ax + ay*ay + az*az);
	cell.b = sqrt(bx*bx + by*by + bz*bz);
	cell.c = sqrt(cx*cx + cy*cy + cz*cz);
	cell.alpha = acos((bx*cx + by*cy + bz*cz)/(cell.b * cell.c));
	cell.beta = acos((ax*cx + ay*cy + az*cz)/(cell.a * cell.c));
	cell.gamma = acos((bx*ax + by*ay + bz*az)/(cell.b * cell.a));

	gsl_matrix_free(inv);

	printf("Cell parameters:\n");
	cell_print(&cell);

	return cell;
}


static int basis_do_save(Basis *cell, const char *filename)
{
	FILE *fh;
	UnitCell rcell;

	fh = fopen(filename, "w");

	fprintf(fh, "# DTR unit cell description\n");

	/* A "human-readable" form */
	rcell = basis_get_cell(cell);
	fprintf(fh, "# a %12.8f nm\n", rcell.a*1e9);
	fprintf(fh, "# b %12.8f nm\n", rcell.b*1e9);
	fprintf(fh, "# c %12.8f nm\n", rcell.c*1e9);
	fprintf(fh, "# alpha %12.8f deg\n", rad2deg(rcell.alpha));
	fprintf(fh, "# beta %12.8f deg\n", rad2deg(rcell.beta));
	fprintf(fh, "# gamma %12.8f deg\n", rad2deg(rcell.gamma));

	/* The useful form */
	fprintf(fh, "a %12.8f %12.8f %12.8f\n",
					cell->a.x, cell->a.y, cell->a.z);
	fprintf(fh, "b %12.8f %12.8f %12.8f\n",
					cell->b.x, cell->b.y, cell->b.z);
	fprintf(fh, "c %12.8f %12.8f %12.8f\n",
					cell->c.x, cell->c.y, cell->c.z);

	fclose(fh);

	return 0;
}

static gint basis_save_response(GtkWidget *widget, gint response,
					ControlContext *ctx)
{
	if ( response == GTK_RESPONSE_ACCEPT ) {
		char *filename;
		filename = gtk_file_chooser_get_filename(
						GTK_FILE_CHOOSER(widget));
		if ( basis_do_save(ctx->cell, filename) ) {
			displaywindow_error("Failed to save unit cell.",
						ctx->dw);
		}
		g_free(filename);
	}

	gtk_widget_destroy(widget);

	return 0;
}

void basis_save(ControlContext *ctx)
{
	GtkWidget *save;

	save = gtk_file_chooser_dialog_new("Save Unit Cell to File",
					GTK_WINDOW(ctx->dw->window),
					GTK_FILE_CHOOSER_ACTION_SAVE,
					GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
					GTK_STOCK_SAVE, GTK_RESPONSE_ACCEPT,
					NULL);
	g_signal_connect(G_OBJECT(save), "response",
				G_CALLBACK(basis_save_response), ctx);
	gtk_widget_show_all(save);
}

static int basis_do_load(Basis *cell, const char *filename)
{
	FILE *fh;
	float x, y, z;
	int got_a = 0;
	int got_b = 0;
	int got_c = 0;

	fh = fopen(filename, "r");

	while ( !feof(fh) ) {

		char line[256];

		if ( fgets(line, 255, fh) != NULL ) {

			if ( sscanf(line, "a %f %f %f\n", &x, &y, &z) == 3 ) {
				cell->a.x = x;	cell->a.y = y;	cell->a.z = z;
				got_a = 1;
			}
			if ( sscanf(line,  "b %f %f %f\n", &x, &y, &z) == 3 ) {
				cell->b.x = x;	cell->b.y = y;	cell->b.z = z;
				got_b = 1;
			}
			if ( sscanf(line, "c %f %f %f\n", &x, &y, &z) == 3 ) {
				cell->c.x = x;	cell->c.y = y;	cell->c.z = z;
				got_c = 1;
			}

		}

	}

	fclose(fh);

	return !(got_a && got_b && got_c);
}

static gint basis_load_response(GtkWidget *widget, gint response,
					ControlContext *ctx)
{
	if ( response == GTK_RESPONSE_ACCEPT ) {
		char *filename;
		filename = gtk_file_chooser_get_filename(
						GTK_FILE_CHOOSER(widget));
		if ( ctx->cell ) {
			free(ctx->cell);
		}
		ctx->cell = malloc(sizeof(Basis));

		if ( basis_do_load(ctx->cell, filename) ) {
			displaywindow_error("Failed to load unit cell.",
						ctx->dw);
		} else {
			displaywindow_update(ctx->dw);
			reproject_lattice_changed(ctx);
		}
		g_free(filename);
	}

	gtk_widget_destroy(widget);

	return 0;
}

void basis_load(ControlContext *ctx)
{
	GtkWidget *load;

	load = gtk_file_chooser_dialog_new("Load Unit Cell from File",
					GTK_WINDOW(ctx->dw->window),
					GTK_FILE_CHOOSER_ACTION_OPEN,
					GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
					GTK_STOCK_SAVE, GTK_RESPONSE_ACCEPT,
					NULL);
	g_signal_connect(G_OBJECT(load), "response",
				G_CALLBACK(basis_load_response), ctx);
	gtk_widget_show_all(load);
}
