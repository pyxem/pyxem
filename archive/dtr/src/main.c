/*
 * main.c
 *
 * The Top Level Source File
 *
 * (c) 2007-2009 Thomas White <taw27@cam.ac.uk>
 * (c) 2007      Gordon Ball <gfb21@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#define _GNU_SOURCE
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>
#include <gtk/gtk.h>
#include <math.h>
#include <gdk/gdkgl.h>
#include <gtk/gtkgl.h>
#include <sys/stat.h>
#include <glew.h>

#include "displaywindow.h"
#include "reflections.h"
#include "mrc.h"
#include "qdrp.h"
#include "cache.h"
#include "mapping.h"
#include "prealign.h"
#include "control.h"
#include "dirax.h"
#include "itrans.h"

void main_do_reconstruction(ControlContext *ctx)
{
	if ( ctx->inputfiletype != INPUT_DRX ) {

		int val = 0;

		/* Initial centering */
		prealign_sum_stack(ctx->images, ctx->have_centres,
								ctx->sum_stack);
		if ( ctx->finecentering ) {
			prealign_fine_centering(ctx->images, ctx->sum_stack);
		}

		if ( ctx->cache_filename == NULL ) {

			int i;

			/* Find all the features */
			printf("MA: Analysing images..."); fflush(stdout);
			for ( i=0; i<ctx->images->n_images; i++ ) {
				ctx->images->images[i].features =
				   itrans_process_image(&ctx->images->images[i],
				   ctx->psmode);
				itrans_quantify_features(
						       &ctx->images->images[i]);
			}
			printf("done.\n");

		} else {

			int i;

			printf("MA: Loading previous image analysis from %s\n",
							ctx->cache_filename);
			if ( cache_load(ctx->images, ctx->cache_filename) )
									val = 1;

			/* Quantify all the features (unnecessary if the file
			 *  already contains intensities,
			 *  but many of mine don't, and it doesn't hurt. */
			printf("MA: Quantifying features..."); fflush(stdout);
			for ( i=0; i<ctx->images->n_images; i++ ) {
				itrans_quantify_features(
						       &ctx->images->images[i]);
			}
			printf("done.\n");

		}

		if ( ctx->finecentering ) {
			prealign_feature_centering(ctx->images);
		}

		if ( !val ) mapping_map_features(ctx);

	} /* else has already been created by dirax_load() */

	if ( ctx->reflectionlist ) {
		ctx->dw = displaywindow_open(ctx);
	} else {
		fprintf(stderr, "Reconstruction failed.\n");
		gtk_exit(0);
	}
}

static gint main_method_window_response(GtkWidget *method_window, gint response,
					ControlContext *ctx)
{
	if ( response == GTK_RESPONSE_OK ) {

		int val = 0;

		switch ( gtk_combo_box_get_active(
				GTK_COMBO_BOX(ctx->combo_peaksearch)) ) {
			case 0 : ctx->psmode = PEAKSEARCH_NONE; break;
			case 1 : ctx->psmode = PEAKSEARCH_THRESHOLD; break;
			case 2 : ctx->psmode = PEAKSEARCH_ADAPTIVE_THRESHOLD;
									break;
			case 3 : ctx->psmode = PEAKSEARCH_ZAEFFERER; break;
			case 4 : ctx->psmode = PEAKSEARCH_STAT; break;
			case 5 : ctx->psmode = PEAKSEARCH_CACHED; break;
			/* Happens when reading from a cache file */
			default: ctx->psmode = PEAKSEARCH_NONE; break;
		}

		if ( gtk_toggle_button_get_active(
				GTK_TOGGLE_BUTTON(ctx->checkbox_prealign)) ) {
			ctx->prealign = TRUE;
		} else {
			ctx->prealign = FALSE;
		}

		if ( gtk_toggle_button_get_active(
			GTK_TOGGLE_BUTTON(ctx->checkbox_finecentering)) ) {
			ctx->finecentering = TRUE;
		} else {
			ctx->finecentering = FALSE;
		}

		if ( gtk_toggle_button_get_active(
				GTK_TOGGLE_BUTTON(ctx->checkbox_sumstack)) ) {
			ctx->sum_stack = TRUE;
		} else {
			ctx->sum_stack = FALSE;
		}

		if ( ctx->psmode == PEAKSEARCH_CACHED ) {
			ctx->cache_filename = gtk_file_chooser_get_filename(
				GTK_FILE_CHOOSER(ctx->cache_file_selector));
			if ( !ctx->cache_filename ) {
				return 1;
			}
		}

		gtk_widget_destroy(method_window);
		while ( gtk_events_pending() ) gtk_main_iteration();

		/* Load the input */
		if ( ctx->inputfiletype == INPUT_QDRP ) {
			val = qdrp_read(ctx);
		} else if ( ctx->inputfiletype == INPUT_MRC ) {
			val = mrc_read(ctx);
		} else if ( ctx->inputfiletype == INPUT_DRX ) {
			ctx->reflectionlist = dirax_load(ctx->filename);
			if ( !ctx->reflectionlist ) val = 1;
		}

		if ( val ) {
			fprintf(stderr, "Reconstruction failed.\n");
			gtk_exit(0);
		}

		if ( ctx->prealign ) {
			/* this will eventually call main_do_reconstruction() */
			prealign_do_series(ctx);
		} else {
			main_do_reconstruction(ctx);
		}

	} else {
		gtk_exit(0);
	}

	return 0;
}

static gint main_peaksearch_changed(GtkWidget *method_window,
					ControlContext *ctx)
{
	if ( gtk_combo_box_get_active(
				GTK_COMBO_BOX(ctx->combo_peaksearch)) == 5 ) {
		gtk_widget_set_sensitive(GTK_WIDGET(ctx->cache_file_selector),
						TRUE);
	} else {
		gtk_widget_set_sensitive(GTK_WIDGET(ctx->cache_file_selector),
						FALSE);
	}

	return 0;
}

void main_method_dialog_open(ControlContext *ctx)
{
	GtkWidget *method_window;
	GtkWidget *vbox;
	GtkWidget *hbox;
	GtkWidget *table;
	GtkWidget *peaksearch_label;
	GtkWidget *cache_file_selector_label;

	method_window = gtk_dialog_new_with_buttons("Reconstruction Parameters",
					NULL,
					GTK_DIALOG_DESTROY_WITH_PARENT,
					GTK_STOCK_CANCEL, GTK_RESPONSE_CLOSE,
					GTK_STOCK_OK, GTK_RESPONSE_OK, NULL);
	gtk_window_set_default_size(GTK_WINDOW(method_window), 400, -1);

	vbox = gtk_vbox_new(FALSE, 0);
	hbox = gtk_hbox_new(TRUE, 0);
	gtk_box_pack_start(GTK_BOX(GTK_DIALOG(method_window)->vbox),
					GTK_WIDGET(hbox), FALSE, FALSE, 7);
	gtk_box_pack_start(GTK_BOX(hbox), GTK_WIDGET(vbox), FALSE, TRUE, 10);

	table = gtk_table_new(5, 2, FALSE);
	gtk_table_set_row_spacings(GTK_TABLE(table), 5);

	peaksearch_label = gtk_label_new("Peak Search: ");
	gtk_table_attach_defaults(GTK_TABLE(table), peaksearch_label,
								1, 2, 1, 2);
	gtk_misc_set_alignment(GTK_MISC(peaksearch_label), 1, 0.5);
	ctx->combo_peaksearch = gtk_combo_box_new_text();
	gtk_combo_box_append_text(GTK_COMBO_BOX(ctx->combo_peaksearch),
					"None");
	gtk_combo_box_append_text(GTK_COMBO_BOX(ctx->combo_peaksearch),
					"Simple Thresholding");
	gtk_combo_box_append_text(GTK_COMBO_BOX(ctx->combo_peaksearch),
					"Adaptive Thresholding");
	gtk_combo_box_append_text(GTK_COMBO_BOX(ctx->combo_peaksearch),
					"Zaefferer Gradient Search");
	gtk_combo_box_append_text(GTK_COMBO_BOX(ctx->combo_peaksearch),
					"Iterative Statistical Analysis");
	gtk_combo_box_append_text(GTK_COMBO_BOX(ctx->combo_peaksearch),
					"Get From Cache File");
	gtk_combo_box_set_active(GTK_COMBO_BOX(ctx->combo_peaksearch), 3);
	gtk_table_attach_defaults(GTK_TABLE(table), ctx->combo_peaksearch,
								2, 3, 1, 2);
	g_signal_connect(G_OBJECT(ctx->combo_peaksearch), "changed",
				G_CALLBACK(main_peaksearch_changed), ctx);

	cache_file_selector_label = gtk_label_new("Cache File to Load: ");
	gtk_table_attach_defaults(GTK_TABLE(table), cache_file_selector_label,
								1, 2, 2, 3);
	gtk_misc_set_alignment(GTK_MISC(cache_file_selector_label), 1, 0.5);
	ctx->cache_file_selector = gtk_file_chooser_button_new(
						"Select Cache File to Load",
						GTK_FILE_CHOOSER_ACTION_OPEN);
	gtk_table_attach_defaults(GTK_TABLE(table), ctx->cache_file_selector,
						2, 3, 2, 3);
	gtk_widget_set_sensitive(GTK_WIDGET(ctx->cache_file_selector), FALSE);

	ctx->checkbox_prealign = gtk_check_button_new_with_label(
					"Manually pre-align the image stack");
	gtk_table_attach_defaults(GTK_TABLE(table), ctx->checkbox_prealign,
					1, 3, 3, 4);

	ctx->checkbox_finecentering = gtk_check_button_new_with_label(
					"Perform fine pattern centering");
	gtk_table_attach_defaults(GTK_TABLE(table), ctx->checkbox_finecentering,
					1, 3, 4, 5);
	gtk_toggle_button_set_active(
			GTK_TOGGLE_BUTTON(ctx->checkbox_finecentering), TRUE);

	ctx->checkbox_sumstack = gtk_check_button_new_with_label(
					"Show summed image stack");
	gtk_table_attach_defaults(GTK_TABLE(table), ctx->checkbox_sumstack,
					1, 3, 5, 6);

	if ( ctx->inputfiletype == INPUT_DRX ) {
		gtk_combo_box_append_text(GTK_COMBO_BOX(ctx->combo_peaksearch),
					"3D coordinates from DirAx file");
		gtk_widget_set_sensitive(GTK_WIDGET(ctx->combo_peaksearch),
						FALSE);
		gtk_combo_box_set_active(GTK_COMBO_BOX(ctx->combo_peaksearch),
									6);
	}

	gtk_box_pack_start(GTK_BOX(vbox), GTK_WIDGET(table), TRUE, TRUE, 5);

	g_signal_connect(G_OBJECT(method_window), "response",
				G_CALLBACK(main_method_window_response), ctx);

	gtk_widget_show_all(method_window);
}


int main(int argc, char *argv[])
{
	ControlContext *ctx;
	struct stat stat_buffer;
	FILE *fh;

	gtk_init(&argc, &argv);

	if ( gtk_gl_init_check(&argc, &argv) == FALSE ) {
		fprintf(stderr, "Could not initialise gtkglext\n");
		return 1;
	}

	if ( gdk_gl_query_extension() == FALSE ) {
		fprintf(stderr, "OpenGL not supported\n");
		return 1;
	}

	if ( argc != 2 ) {
		fprintf(stderr,
			"Syntax: %s [<MRC file> | qdrp.rc | reflect.cache]\n",
			argv[0]);
		return 1;
	}

	ctx = control_ctx_new();
	ctx->filename = strdup(argv[1]);

	fh = fopen(ctx->filename, "r");
	if ( !fh ) {
		printf("Couldn't open file '%s'\n", ctx->filename);
		return 1;
	}
	fclose(fh);

	if ( qdrp_is_qdrprc(ctx->filename) ) {
		printf("QDRP input file detected.\n");
		ctx->inputfiletype = INPUT_QDRP;
	} else if ( mrc_is_mrcfile(ctx->filename) ) {
		printf("MRC tomography file detected.\n");
		ctx->inputfiletype = INPUT_MRC;
	} else if ( dirax_is_drxfile(ctx->filename) ) {
		printf("Dirax input file detected.\n");
		ctx->inputfiletype = INPUT_DRX;
	} else {
		fprintf(stderr, "Unrecognised input file type\n");
		return 1;
	}

	if ( stat(argv[1], &stat_buffer) == -1 ) {
		fprintf(stderr, "File '%s' not found\n", argv[1]);
		return 1;
	}

	main_method_dialog_open(ctx);

	gtk_main();

	free(ctx->filename);
	free(ctx);

	return 0;
}
