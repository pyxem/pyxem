/*
 * displaywindow.h
 *
 * The display window
 *
 * (c) 2007-2008 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef DISPLAYWINDOW_H
#define DISPLAYWINDOW_H

#include <gtk/gtk.h>
#include <glew.h>

#include "control.h"

typedef enum {
	DW_ORTHO,
	DW_PERSPECTIVE
} DisplayWindowView;

typedef enum {
	DW_MAPPED,	/* Display the features from the images mapped into 3D */
	DW_MEASURED	/* Display the intensities of reflections measured via intensities.c */
} DisplayWindowMode;

typedef struct dw_struct {

	ControlContext		*ctx;
	
	GtkUIManager		*ui;
	GtkActionGroup		*action_group;
	GtkWidget		*window;
	GtkWidget		*bigvbox;
	GtkWidget		*drawing_area;
	GtkWidget		*savecache_window;
	GtkWidget		*messages;
	GtkWidget		*back_plane;
	GtkWidget		*front_plane;
	GtkTextMark		*messages_mark;
	
	/* Low-level OpenGL stuff */
	GLuint			gl_list_id;			/* Display list for "everything else" */
	int			gl_use_buffers;			/* 0=use vertex arrays only, otherwise use VBOs */
	int			gl_use_shaders;			/* 1 = use shaders */
	GLuint			gl_ref_vertex_buffer;		/* "Measured reflection" stuff */
	GLfloat			*gl_ref_vertex_array;
	GLuint			gl_ref_normal_buffer;
	GLfloat			*gl_ref_normal_array;
	GLsizei			gl_ref_num_vertices;
	GLuint			gl_marker_vertex_buffer;	/* Marker "reflection" stuff */
	GLuint			gl_marker_normal_buffer;
	GLfloat			*gl_marker_vertex_array;
	GLfloat			*gl_marker_normal_array;
	GLsizei			gl_marker_num_vertices;
	GLuint			gl_gen_vertex_buffer;		/* Generated reflection stuff */
	GLuint			gl_gen_normal_buffer;
	GLfloat			*gl_gen_vertex_array;
	GLfloat			*gl_gen_normal_array;
	GLsizei			gl_gen_num_vertices;
	GLuint			gl_line_vertex_buffer;		/* Indexing line stuff */
	GLfloat			*gl_line_vertex_array;
	GLsizei			gl_line_num_vertices;
	GLuint			gl_vshader_lightpp;
	GLuint			gl_fshader_lightpp;
	GLuint			gl_program_lightpp;
	int			realised;
	
	/* Display parameters */
	DisplayWindowView	view;
	GLfloat			distance;
	GLfloat			front;
	GLfloat			back;
	GLfloat			x_pos;
	GLfloat			y_pos;
	float			view_quat[4];
	float			theta, phi, psi;
	int				cube;
	int				lines;
	int				background;
	float			x_start;
	float			y_start;
	DisplayWindowMode	mode;
	
	int				cur_image;
	struct imagedisplay_struct	*stack;
	
	/* Tilt axis adjustment window */
	GtkWidget		*tiltaxis_window;
	GtkWidget		*tiltaxis_entry;
	
} DisplayWindow;

extern DisplayWindow *displaywindow_open(ControlContext *ctx);
extern void displaywindow_update_imagestack(DisplayWindow *dw);

extern void displaywindow_update(DisplayWindow *dw);
extern void displaywindow_update_dirax(ControlContext *ctx, DisplayWindow *dw);
extern void displaywindow_error(const char *msg, DisplayWindow *dw);
extern void displaywindow_enable_cell_functions(DisplayWindow *dw, gboolean g);

#endif	/* DISPLAYWINDOW_H */

