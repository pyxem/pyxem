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

#include <gtk/gtk.h>
#include <math.h>
#include <stdlib.h>

#include "gtk-valuegraph.h"

static GtkObjectClass *parent_class = NULL;

static void gtk_value_graph_destroy(GtkObject *gtk_value_graph) {
	parent_class->destroy(gtk_value_graph);
}

GtkWidget *gtk_value_graph_new() {

	GtkValueGraph *gtk_value_graph;
	
	gtk_value_graph = GTK_VALUE_GRAPH(gtk_type_new(gtk_value_graph_get_type()));
	gtk_value_graph->data = NULL;
	gtk_value_graph->n = 0;
	
	return GTK_WIDGET(gtk_value_graph);

}

static GObject *gtk_value_graph_constructor(GType type, guint n_construct_properties, GObjectConstructParam *construct_properties) {

	GtkValueGraphClass *class;
	GObjectClass *p_class;
	GObject *obj;
	
	class = GTK_VALUE_GRAPH_CLASS(g_type_class_peek(gtk_value_graph_get_type()));
	p_class = G_OBJECT_CLASS(g_type_class_peek_parent(class));
	
	obj = p_class->constructor(type, n_construct_properties, construct_properties);
	
	return obj;

}

static void gtk_value_graph_class_init(GtkValueGraphClass *class) {

	GtkObjectClass *object_class;
	GObjectClass *g_object_class;
	
	object_class = (GtkObjectClass *) class;
	g_object_class = G_OBJECT_CLASS(class);
	
	object_class->destroy = gtk_value_graph_destroy;
	g_object_class->constructor = gtk_value_graph_constructor;
	
	parent_class = gtk_type_class(gtk_drawing_area_get_type());
	
}

static gint gtk_value_graph_destroyed(GtkWidget *graph, gpointer data) {

	if ( GTK_VALUE_GRAPH(graph)->data ) {
		free(GTK_VALUE_GRAPH(graph)->data);
	}

	return 0;

}

static gint gtk_value_graph_draw(GtkWidget *graph, GdkEventExpose *event, gpointer data) {

	GtkValueGraph *vg;
	double bw_left, bw_right, bw_top, bw_bottom;
	PangoLayout *y0_layout;
	PangoLayout *y1_layout;
	PangoLayout *x0_layout;
	PangoLayout *x1_layout;
	PangoRectangle y0_extent, y1_extent, x0_extent, x1_extent;
	double width, height;
	char tmp[32];
	int i;
	cairo_t *cr;
	PangoFontDescription *desc;
	double scale;
  	
	vg = GTK_VALUE_GRAPH(graph);
	
	cr = gdk_cairo_create(graph->window);
	
	/* Blank white background */
	cairo_rectangle(cr, 0.0, 0.0, graph->allocation.width, graph->allocation.height);
	cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
	cairo_fill(cr);
	
	/* Create PangoLayouts for labels */
	desc = pango_font_description_from_string("Sans, Normal, 10");
	if ( fabs(log(vg->ymax)/log(10)) < 3 ) {
		snprintf(tmp, 31, "%.4f", vg->ymin);
	} else {
		snprintf(tmp, 31, "%1.1e", vg->ymin);
	}
	y0_layout = pango_cairo_create_layout(cr);
	pango_layout_set_text(y0_layout, tmp, -1);
	pango_layout_set_font_description(y0_layout, desc);
  	pango_layout_get_pixel_extents(y0_layout, NULL, &y0_extent);
	
	if ( fabs(log(vg->ymax)/log(10)) < 3 ) {
		snprintf(tmp, 31, "%.4f", vg->ymax);
	} else {
		snprintf(tmp, 31, "%1.1e", vg->ymax);
	}
	y1_layout = pango_cairo_create_layout(cr);
	pango_layout_set_text(y1_layout, tmp, -1);
	pango_layout_set_font_description(y1_layout, desc);
  	pango_layout_get_pixel_extents(y1_layout, NULL, &y1_extent);
	
	x0_layout = pango_cairo_create_layout(cr);
	pango_layout_set_text(x0_layout, "0", -1);
	pango_layout_set_font_description(x0_layout, desc);
  	pango_layout_get_pixel_extents(x0_layout, NULL, &x0_extent);
	
	if ( vg->xmax < 1000 ) {
		snprintf(tmp, 31, "%.0f", vg->xmax);
	} else {
		snprintf(tmp, 31, "%1.1e", (double)vg->xmax);
	}
	x1_layout = pango_cairo_create_layout(cr);
	pango_layout_set_text(x1_layout, tmp, -1);
	pango_layout_set_font_description(x1_layout, desc);
  	pango_layout_get_pixel_extents(x1_layout, NULL, &x1_extent);
	
	/* Determine border widths */
	bw_left = 1+((y1_extent.width > y0_extent.width) ? y1_extent.width : y0_extent.width);
	bw_right = 1+x1_extent.width/2;
	bw_top = 1+y1_extent.height/2;
	bw_bottom = 1+((x1_extent.height > x0_extent.height) ? x1_extent.height : x0_extent.height);
	width = graph->allocation.width;
	height = graph->allocation.height;
	
	/* Draw axis lines */
	cairo_new_path(cr);
	cairo_move_to(cr, bw_left+0.5, height-1-bw_bottom+0.5);
	cairo_line_to(cr, bw_left+0.5, bw_top+0.5);
	cairo_move_to(cr, bw_left+0.5, height-1-bw_bottom+0.5);
	cairo_line_to(cr, width-1-bw_right+0.5, height-1-bw_bottom+0.5);
	cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
	cairo_set_line_width(cr, 1.0);
	cairo_stroke(cr);
	
	/* Label axes */
	cairo_new_path(cr);
	cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
	cairo_move_to(cr, 1+bw_left-x0_extent.width/2, height-1-bw_bottom);
	pango_cairo_layout_path(cr, x0_layout);
	cairo_move_to(cr, width-bw_right-x1_extent.width/2, height-1-bw_bottom);
	pango_cairo_layout_path(cr, x1_layout);
	cairo_move_to(cr, bw_left-y0_extent.width-1, height-1-bw_bottom-y0_extent.height/2);
	pango_cairo_layout_path(cr, y0_layout);
	cairo_move_to(cr, bw_left-y1_extent.width-1, 1);
	pango_cairo_layout_path(cr, y1_layout);
	cairo_fill(cr);
	
	/* Plot data */
	cairo_new_path(cr);
	scale = (height-bw_top-bw_bottom)/(vg->ymax-vg->ymin);
	cairo_move_to(cr, bw_left, height-bw_bottom-1-scale*(vg->data[0]-vg->ymin));
	for ( i=1; i<vg->n; i++ ) {
		
		int p;
		
		p = i-1;
		if ( p < 1 ) p = 1;
		
		if ( !isnan(vg->data[i]) && !isnan(vg->data[p]) ) {
			cairo_line_to(cr, bw_left+((double)i/vg->xmax)*(width-bw_left-bw_right), height-bw_bottom-1-scale*(vg->data[i]-vg->ymin));
		} else {
			cairo_move_to(cr, bw_left+((double)i/vg->xmax)*(width-bw_left-bw_right), height-bw_bottom-1-scale*(vg->data[i]-vg->ymin));
		}
		
	}
	cairo_set_line_width(cr, 0.5);
	cairo_set_source_rgb(cr, 0.0, 0.0, 1.0);
	cairo_stroke(cr);
	
	cairo_destroy(cr);
	
	return 0;

}

static void gtk_value_graph_init(GtkValueGraph *gtk_value_graph) {
	gtk_widget_set_size_request(GTK_WIDGET(gtk_value_graph), 100, 200);
	g_signal_connect(G_OBJECT(gtk_value_graph), "expose_event", G_CALLBACK(gtk_value_graph_draw), NULL);
	g_signal_connect(G_OBJECT(gtk_value_graph), "destroy", G_CALLBACK(gtk_value_graph_destroyed), NULL);
}

guint gtk_value_graph_get_type(void) {

  	static guint gtk_value_graph_type = 0;
	
	if ( !gtk_value_graph_type ) {

		GtkTypeInfo gtk_value_graph_info = {
			"GtkValueGraph",
			sizeof(GtkValueGraph),
			sizeof(GtkValueGraphClass),
			(GtkClassInitFunc) gtk_value_graph_class_init,
			(GtkObjectInitFunc) gtk_value_graph_init,
			NULL,
			NULL,
			(GtkClassInitFunc) NULL,
		};
		gtk_value_graph_type = gtk_type_unique(gtk_drawing_area_get_type(), &gtk_value_graph_info);

	}
	
	return gtk_value_graph_type;

}

static double gtk_value_graph_peak(double *data, unsigned int n) {

	unsigned int i;
	double max;
	
	if ( n == 0 ) return 1;
	
	max = 0;
	for ( i=0; i<n; i++ ) {
		if ( data[i] > max ) max = data[i];
	}
	
	return max;

}

static double gtk_value_graph_min(double *data, unsigned int n) {

	unsigned int i;
	double min;
	
	if ( n == 0 ) return 0;
	
	min = +HUGE_VAL;
	for ( i=0; i<n; i++ ) {
		if ( data[i] < min ) min = data[i];
	}
	
	return min;

}

void gtk_value_graph_set_data(GtkValueGraph *vg, double *data, unsigned int n) {
	
	vg->data = data;
	vg->n = n;
	
	/* Recalculate axes */
	vg->xmax = n;
	vg->ymax = gtk_value_graph_peak(data, n);
	vg->ymin = gtk_value_graph_min(data, n);
	
	//printf("n=%i, dmax=%f => xmax=%i, ymax=%f\n", n, dmax, vg->xmax, vg->ymax);
	
	/* Schedule redraw */
	gtk_widget_queue_draw_area(GTK_WIDGET(vg), 0, 0, GTK_WIDGET(vg)->allocation.width, GTK_WIDGET(vg)->allocation.height);
	
}

