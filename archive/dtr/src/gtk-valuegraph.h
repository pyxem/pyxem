/*
 * gtk-valuegraph.c
 *
 * A widget to display a graph of a sequence of values
 *
 * (c) 2006-2007 Thomas White <taw27@cam.ac.uk>
 *
 *  synth2d - two-dimensional Fourier synthesis
 *
 */

#ifndef GTKVALUEGRAPH_H
#define GTKVALUEGRAPH_H

#include <gtk/gtk.h>

typedef struct {

	GtkDrawingArea parent;		/* Parent widget */
	
	double *data;			/* Data to be graphed */
	unsigned int n;			/* Number of data points */
	double xmax;			/* Maximum value on x (index) axis */
	double ymax;			/* Maximum value on y (data) axis */
	double ymin;
	
} GtkValueGraph;

typedef struct {
	GtkDrawingAreaClass parent_class;
	void (* changed) (GtkValueGraph *gtkvaluegraph);
} GtkValueGraphClass;

extern guint gtk_value_graph_get_type(void);
extern GtkWidget *gtk_value_graph_new(void);
extern void gtk_value_graph_set_data(GtkValueGraph *vg, double *data, unsigned int n);

#define GTK_VALUE_GRAPH(obj)		GTK_CHECK_CAST(obj, gtk_value_graph_get_type(), GtkValueGraph)
#define GTK_VALUE_GRAPH_CLASS(class)	GTK_CHECK_CLASS_CAST(class, gtk_value_graph_get_type(), GtkValueGraphClass)
#define GTK_IS_VALUE_GRAPH(obj)		GTK_CHECK_TYPE(obj, gtk_value_graph_get_type())

#endif /* GTKVALUEGRAPH_H */

