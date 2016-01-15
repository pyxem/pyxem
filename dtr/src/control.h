/*
 * control.h
 *
 * Common control structure
 *
 * (c) 2007-2008 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */
 
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef CONTROL_H
#define CONTROL_H

#include <gtk/gtk.h>
#include <inttypes.h>

typedef enum ift_enum {
	INPUT_NONE,
	INPUT_QDRP,
	INPUT_MRC,
	INPUT_DRX
} InputFileType;

typedef enum {
	FORMULATION_CLEN,
	FORMULATION_PIXELSIZE
} FormulationMode;

typedef enum {
	PEAKSEARCH_NONE,
	PEAKSEARCH_THRESHOLD,
	PEAKSEARCH_ADAPTIVE_THRESHOLD,
	PEAKSEARCH_ZAEFFERER,
	PEAKSEARCH_STAT,
	PEAKSEARCH_CACHED
} PeakSearchMode;

typedef struct cctx_struct {
	
	/* Modes */
	InputFileType			inputfiletype;
	PeakSearchMode 			psmode;
	unsigned int			prealign;
	unsigned int			finecentering;
	unsigned int			have_centres;
	unsigned int			sum_stack;
	
	/* Input filename */
	char 				*filename;
	char				*cache_filename;
	
	/* Basic parameters, stored here solely so they can be copied
	 * into the ImageRecord(s) more easily */
	FormulationMode			fmode;
	double				camera_length;
	double				omega;		/* Degrees */
	double				resolution;
	double				lambda;
	double				pixel_size;
	double				x_centre;
	double				y_centre;
	
	/* QDRP Parser flags */
	unsigned int			started;
	unsigned int			camera_length_set;
	unsigned int			omega_set;
	unsigned int			resolution_set;
	unsigned int			lambda_set;
	
	/* The input images */
	struct imagelist_struct		*images;
	
	/* "Output" */
	struct reflectionlist_struct	*reflectionlist;	/* Measured reflections */
	struct dw_struct		*dw;
	struct basis_struct		*cell;			/* Current estimate of the reciprocal unit cell */
	struct reflectionlist_struct	*cell_lattice;		/* Reflections calculated from 'cell' */
	struct reflectionlist_struct	*integrated;		/* "Final" integrated intensities */
	
	/* GTK bits */
	GtkWidget			*combo_peaksearch;
	GtkWidget			*checkbox_prealign;
	GtkWidget			*checkbox_finecentering;
	GtkWidget			*checkbox_sumstack;
	GtkWidget			*cache_file_selector;
	
	/* DirAx low-level stuff */
	GIOChannel			*dirax;
	int				dirax_pty;
	pid_t				dirax_pid;
	char				*dirax_rbuffer;
	int				dirax_rbufpos;
	int				dirax_rbuflen;
	
	/* DirAx high-level stuff */
	int				dirax_step;
	int				dirax_read_cell;
	
	/* Refinement stuff */
	GtkWidget			*refine_window;
		
} ControlContext;

extern ControlContext *control_ctx_new(void);

#endif	/* CONTROL_H */

