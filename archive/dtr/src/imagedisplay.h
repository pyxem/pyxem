/*
 * imagedisplay.h
 *
 * Show raw and processed images
 *
 * (c) 2007-2008 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */
 
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef IMAGEDISPLAY_H
#define IMAGEDISPLAY_H

#include <stdint.h>
#include <gtk/gtk.h>

#include "image.h"

typedef enum {
	IMAGEDISPLAY_NONE		= 0,
	IMAGEDISPLAY_SHOW_CENTRE	= 1<<1,
	IMAGEDISPLAY_SHOW_TILT_AXIS	= 1<<2,
	IMAGEDISPLAY_QUIT_IF_CLOSED	= 1<<3,
	IMAGEDISPLAY_FREE		= 1<<4,
	IMAGEDISPLAY_SCALE_BAR		= 1<<5
} ImageDisplayFlags;

typedef enum {
	IMAGEDISPLAY_MARK_CIRCLE_1,
	IMAGEDISPLAY_MARK_CIRCLE_2,
	IMAGEDISPLAY_MARK_CIRCLE_3,
	IMAGEDISPLAY_MARK_LINE_1,
	IMAGEDISPLAY_MARK_LINE_2
} ImageDisplayMarkType;

typedef struct struct_imagedisplaymark {

	double				x;
	double				y;
	double				x2;
	double				y2;
	ImageDisplayMarkType		type;
	double				weight;	/* The 'intensity' if this is a peak */
	
	struct struct_imagedisplaymark	*next;
	
} ImageDisplayMark;

typedef struct imagedisplay_struct {

	ImageRecord		imagerecord;
	ImageDisplayFlags	flags;
	ImageDisplayMark	*marks;
	const char		*title;
	const char		*message;
	guchar			*data;

	GtkWidget		*window;
	GdkPixbuf		*pixbuf;
	GdkPixbuf		*pixbuf_scaled;
	GtkWidget		*drawingarea;
	GtkWidget		*vbox;
	GCallback		mouse_click_func;
	GdkGC			*gc_centre;
	GdkGC			*gc_tiltaxis;
	GdkGC			*gc_marks_1;
	GdkGC			*gc_marks_2;
	GdkGC			*gc_marks_3;
	GdkGC			*gc_scalebar;
	gboolean		realised;
	
	unsigned int		drawingarea_width;
	unsigned int		drawingarea_height;	/* Size of the drawing area */
	unsigned int		view_width;
	unsigned int		view_height;		/* Size of the picture inside the drawing area */
	
} ImageDisplay;

extern ImageDisplay *imagedisplay_open(ImageRecord image, const char *title, ImageDisplayFlags flags);

extern ImageDisplay *imagedisplay_open_with_message(ImageRecord image, const char *title, const char *message,
						ImageDisplayFlags flags, GCallback mouse_click_func,
						gpointer callback_data);

extern ImageDisplay *imagedisplay_new_nowindow(ImageRecord imagerecord, ImageDisplayFlags flags, const char *message,
						GCallback mouse_click_func, gpointer callback_data);

extern void imagedisplay_add_mark(ImageDisplay *imagedisplay, double x, double y, ImageDisplayMarkType type,
						double weight);

extern void imagedisplay_add_line(ImageDisplay *imagedisplay, double x1, double y1,
						double x2, double y2, ImageDisplayMarkType type);

extern void imagedisplay_force_redraw(ImageDisplay *imagedisplay);
extern void imagedisplay_put_data(ImageDisplay *imagedisplay, ImageRecord imagerecord);
extern void imagedisplay_close(ImageDisplay *imagedisplay);
extern void imagedisplay_clear_marks(ImageDisplay *imagedisplay);

#endif	/* IMAGEDISPLAY_H */

