bin_PROGRAMS = dtr refinetest2d refinetest3d1 refinetest3d2 refinetest3d3

dtr_SOURCES = main.c displaywindow.c trackball.c reflections.c readpng.c mrc.c imagedisplay.c utils.c itrans.c qdrp.c \
		cache.c itrans-threshold.c itrans-zaefferer.c itrans-stat.c control.c mapping.c reproject.c prealign.c \
		basis.c dirax.c image.c refine.c gtk-valuegraph.c intensities.c glbits.c
dtr_LDADD = @LIBS@ @GTK_LIBS@ -lm @GTKGLEXT_LIBS@ -lgsl -lgslcblas -lutil

refinetest2d_SOURCES = refinetest2d.c reflections.c basis.c utils.c reproject.c image.c refine.c mapping.c control.c
refinetest2d_LDADD = @LIBS@ @GTK_LIBS@ -lm -lgsl -lgslcblas -lutil

refinetest3d1_SOURCES = refinetest3d1.c reflections.c basis.c utils.c reproject.c image.c refine.c mapping.c control.c
refinetest3d1_LDADD = @LIBS@ @GTK_LIBS@ -lm -lgsl -lgslcblas -lutil

refinetest3d2_SOURCES = refinetest3d2.c reflections.c basis.c utils.c reproject.c image.c refine.c mapping.c control.c
refinetest3d2_LDADD = @LIBS@ @GTK_LIBS@ -lm -lgsl -lgslcblas -lutil

refinetest3d3_SOURCES = refinetest3d3.c reflections.c basis.c utils.c reproject.c image.c refine.c mapping.c control.c
refinetest3d3_LDADD = @LIBS@ @GTK_LIBS@ -lm -lgsl -lgslcblas -lutil

AM_CFLAGS = -Wall -g @CFLAGS@ @GTK_CFLAGS@ @GTKGLEXT_CFLAGS@
AM_CPPFLAGS = -DDATADIR=\""$(datadir)"\" -I/usr/include/GL

