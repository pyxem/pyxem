/*
 * glbits.c
 *
 * OpenGL bits
 *
 * (c) 2007-2008 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <glew.h>
#include <gtk/gtk.h>
#include <gdk/gdkgl.h>
#include <gtk/gtkgl.h>
#include <math.h>
#include <stdlib.h>

#include "displaywindow.h"
#include "trackball.h"
#include "reflections.h"
#include "image.h"
#include "utils.h"

/* Utility function to load and compile a shader, checking the info log */
static GLuint glbits_load_shader(const char *filename, GLenum type) {

	GLuint shader;
	char text[4096];
	size_t len;
	FILE *fh;
	int l;
	GLint status;

	fh = fopen(filename, "r");
	if ( fh == NULL ) {
		fprintf(stderr, "Couldn't load shader '%s'\n", filename);
		return 0;
	}
	len = fread(text, 1, 4095, fh);
	fclose(fh);
	text[len] = '\0';
	const GLchar *source = text;
	shader = glCreateShader(type);
	glShaderSource(shader, 1, &source, NULL);
	glCompileShader(shader);
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	if ( status == GL_FALSE ) {
		glGetShaderInfoLog(shader, 4095, &l, text);
		if ( l > 0 ) {
			printf("%s\n", text); fflush(stdout);
		} else {
			printf("Shader compilation failed.\n");
		}
	}

	return shader;

}

static void glbits_load_shaders(DisplayWindow *dw) {

	/* Lighting-per-fragment */
	dw->gl_vshader_lightpp = glbits_load_shader(DATADIR"/dtr/light-pp.vert", GL_VERTEX_SHADER);
	dw->gl_fshader_lightpp = glbits_load_shader(DATADIR"/dtr/light-pp.frag", GL_FRAGMENT_SHADER);
	dw->gl_program_lightpp = glCreateProgram();
	glAttachShader(dw->gl_program_lightpp, dw->gl_vshader_lightpp);
	glAttachShader(dw->gl_program_lightpp, dw->gl_fshader_lightpp);
	glLinkProgram(dw->gl_program_lightpp);

}

static void glbits_delete_shaders(DisplayWindow *dw) {

	glDetachShader(dw->gl_program_lightpp, dw->gl_fshader_lightpp);
	glDetachShader(dw->gl_program_lightpp, dw->gl_vshader_lightpp);
	glDeleteShader(dw->gl_fshader_lightpp);
	glDeleteShader(dw->gl_program_lightpp);

}

#define BLOB_BITS 7
#define VERTICES_IN_A_BLOB 4*BLOB_BITS*BLOB_BITS*2
#define ADD_VERTEX 						\
	vertices[3*i + 0] = reflection->x/1e9 + size*xv;	\
	vertices[3*i + 1] = reflection->y/1e9 + size*yv;	\
	vertices[3*i + 2] = reflection->z/1e9 + size*zv;	\
	normals[3*i + 0] = xv;					\
	normals[3*i + 1] = yv;					\
	normals[3*i + 2] = zv;					\
	i++;

#define DRAW_BLOB							\
	double step = M_PI/(double)BLOB_BITS;				\
	int is, js;							\
	for ( is=0; is<BLOB_BITS; is++ ) {				\
		for ( js=0; js<BLOB_BITS*2; js++ ) {			\
			double theta, phi;				\
			GLfloat xv, yv, zv;				\
			theta = (M_PI/(double)BLOB_BITS) * (double)js;	\
			phi = (M_PI/(double)BLOB_BITS) * (double)is;	\
			xv = sin(theta)*sin(phi);			\
			yv = cos(phi);					\
			zv = cos(theta)*sin(phi);			\
			ADD_VERTEX					\
			xv = sin(theta)*sin(phi+step);			\
			yv = cos(phi+step);				\
			zv = cos(theta)*sin(phi+step);			\
			ADD_VERTEX					\
			xv = sin(theta+step)*sin(phi+step);		\
			yv = cos(phi+step);				\
			zv = cos(theta+step)*sin(phi+step);		\
			ADD_VERTEX					\
			xv = sin(theta+step)*sin(phi);			\
			yv = cos(phi);					\
			zv = cos(theta+step)*sin(phi);			\
			ADD_VERTEX					\
		}							\
	}

#define DRAW_POINTER_LINE												\
	glBegin(GL_LINES);												\
		glVertex3f(1.0, 0.0, 0.0);										\
		glVertex3f(0.0, 0.0, 0.0);										\
	glEnd();

#define DRAW_POINTER_HEAD												\
	glPushMatrix();													\
	for ( pointer_head_face = 1; pointer_head_face <= 4; pointer_head_face++ ) {					\
		glRotatef(90.0, 1.0, 0.0, 0.0);										\
		glBegin(GL_TRIANGLES);											\
			/* One face */											\
			glNormal3f(0.2, 0.8, 0.0);									\
			glVertex3f(1.0, 0.0, 0.0);									\
			glVertex3f(0.8, 0.2, 0.2);									\
			glVertex3f(0.8, 0.2, -0.2);									\
			/* One quarter of the "bottom square" */							\
			glNormal3f(1.0, 0.0, 0.0);									\
			glVertex3f(0.8, 0.2, 0.2);									\
			glVertex3f(0.8, 0.2, -0.2);									\
			glVertex3f(0.8, 0.0, 0.0);									\
		glEnd();												\
	}														\
	glPopMatrix();

void glbits_prepare(DisplayWindow *dw) {

	GLfloat bblue[] = { 0.0, 0.0, 1.0, 1.0 };
	GLfloat blue[] = { 0.0, 0.0, 0.5, 1.0 };
	GLfloat purple[] = { 0.7, 0.0, 1.0, 1.0 };
	GLfloat red[] = { 1.0, 0.0, 0.0, 1.0 };
	GLfloat green[] = { 0.0, 1.0, 0.0, 1.0 };
	GLfloat yellow[] = { 1.0, 1.0, 0.0, 1.0 };
	GLfloat glass[] = { 0.2, 0.0, 0.8, 000.1 };
	GLfloat black[] = { 0.0, 0.0, 0.0, 1.0 };
	Reflection *reflection;
	int i;
	ControlContext *ctx;
	GLfloat *vertices;
	GLfloat *normals;

	ctx = dw->ctx;

	/* "Measured" reflections */
	if ( dw->gl_use_buffers ) {
		glGenBuffers(1, &dw->gl_ref_vertex_buffer);
		glGenBuffers(1, &dw->gl_ref_normal_buffer);
	}
	reflection = ctx->reflectionlist->reflections;
	i = 0;
	while ( reflection != NULL ) {
		if ( reflection->type == REFLECTION_NORMAL ) i++;
		reflection = reflection->next;
	};
	dw->gl_ref_num_vertices = i;
	if ( dw->gl_ref_num_vertices ) {
		i = 0;
		reflection = ctx->reflectionlist->reflections;
		vertices = malloc(3*dw->gl_ref_num_vertices*sizeof(GLfloat));
		normals = malloc(3*dw->gl_ref_num_vertices*sizeof(GLfloat));
		while ( reflection != NULL ) {
			if ( reflection->type == REFLECTION_NORMAL ) {
				vertices[3*i + 0] = reflection->x/1e9;
				vertices[3*i + 1] = reflection->y/1e9;
				vertices[3*i + 2] = reflection->z/1e9;
				normals[3*i + 0] = reflection->x/1e9;
				normals[3*i + 1] = reflection->y/1e9;
				normals[3*i + 2] = reflection->z/1e9;
				i++;
			}
			reflection = reflection->next;
		};
		if ( dw->gl_use_buffers ) {
			glBindBuffer(GL_ARRAY_BUFFER, dw->gl_ref_vertex_buffer);
			glBufferData(GL_ARRAY_BUFFER, 3*dw->gl_ref_num_vertices*sizeof(GLfloat), vertices, GL_STATIC_DRAW);
			free(vertices);
			glBindBuffer(GL_ARRAY_BUFFER, dw->gl_ref_normal_buffer);
			glBufferData(GL_ARRAY_BUFFER, 3*dw->gl_ref_num_vertices*sizeof(GLfloat), normals, GL_STATIC_DRAW);
			free(normals);
		} else {
			dw->gl_ref_vertex_array = vertices;
			dw->gl_ref_normal_array = normals;
		}
	}

	/* Marker "reflections" */
	if ( dw->gl_use_buffers ) {
		glGenBuffers(1, &dw->gl_marker_vertex_buffer);
		glGenBuffers(1, &dw->gl_marker_normal_buffer);
	}
	reflection = ctx->reflectionlist->reflections;
	i = 0;
	while ( reflection != NULL ) {
		if ( reflection->type == REFLECTION_MARKER ) i++;
		reflection = reflection->next;
	};
	dw->gl_marker_num_vertices = i*VERTICES_IN_A_BLOB;
	if ( dw->gl_marker_num_vertices ) {
		i = 0;
		reflection = ctx->reflectionlist->reflections;
		vertices = malloc(3*dw->gl_marker_num_vertices*sizeof(GLfloat));
		normals = malloc(3*dw->gl_marker_num_vertices*sizeof(GLfloat));
		while ( reflection != NULL ) {
			if ( reflection->type == REFLECTION_MARKER ) {
				double size = 0.15;
				DRAW_BLOB
			}
			reflection = reflection->next;
		};
		if ( dw->gl_use_buffers ) {
			glBindBuffer(GL_ARRAY_BUFFER, dw->gl_marker_vertex_buffer);
			glBufferData(GL_ARRAY_BUFFER, 3*dw->gl_marker_num_vertices*sizeof(GLfloat), vertices, GL_STATIC_DRAW);
			free(vertices);
			glBindBuffer(GL_ARRAY_BUFFER, dw->gl_marker_normal_buffer);
			glBufferData(GL_ARRAY_BUFFER, 3*dw->gl_marker_num_vertices*sizeof(GLfloat), normals, GL_STATIC_DRAW);
			free(normals);
		} else {
			dw->gl_marker_vertex_array = vertices;
			dw->gl_marker_normal_array = normals;
		}
	}

	/* Generated reflections */
	if ( dw->gl_use_buffers ) {
		glGenBuffers(1, &dw->gl_gen_vertex_buffer);
		glGenBuffers(1, &dw->gl_gen_normal_buffer);
	}
	if ( ctx->integrated != NULL ) {
		reflection = ctx->integrated->reflections;
		i = 0;
		while ( reflection != NULL ) {
			if ( reflection->type == REFLECTION_GENERATED ) i++;
			reflection = reflection->next;
		};
		dw->gl_gen_num_vertices = i*VERTICES_IN_A_BLOB;
		if ( dw->gl_gen_num_vertices ) {
			i = 0;
			reflection = ctx->integrated->reflections;
			vertices = malloc(3*dw->gl_gen_num_vertices*sizeof(GLfloat));
			normals = malloc(3*dw->gl_gen_num_vertices*sizeof(GLfloat));
			while ( reflection != NULL ) {
				double size = 5.0 * log(1+0.1*reflection->intensity);
				DRAW_BLOB
				reflection = reflection->next;
			};
			if ( dw->gl_use_buffers ) {
				glBindBuffer(GL_ARRAY_BUFFER, dw->gl_gen_vertex_buffer);
				glBufferData(GL_ARRAY_BUFFER, 3*dw->gl_gen_num_vertices*sizeof(GLfloat), vertices, GL_STATIC_DRAW);
				free(vertices);
				glBindBuffer(GL_ARRAY_BUFFER, dw->gl_gen_normal_buffer);
				glBufferData(GL_ARRAY_BUFFER, 3*dw->gl_gen_num_vertices*sizeof(GLfloat), normals, GL_STATIC_DRAW);
				free(normals);
				glBindBuffer(GL_ARRAY_BUFFER, 0);	/* ************* */
			} else {
				dw->gl_gen_vertex_array = vertices;
				dw->gl_gen_normal_array = normals;
			}
		}
	} else {
		dw->gl_gen_num_vertices = 0;
	}

	/* Indexing lines */
	glLineWidth(2.0);
	if ( ctx->cell && dw->lines ) {

		int max_ind;
		signed int h, k, l;

		max_ind = 10;
		dw->gl_line_num_vertices = 3*2*((2*1+1)*(2*1+1));

		if ( dw->gl_use_buffers ) {
			glGenBuffers(1, &dw->gl_line_vertex_buffer);
		}
		reflection = ctx->reflectionlist->reflections;
		vertices = malloc(3*dw->gl_line_num_vertices*sizeof(GLfloat));

		i=0;

		/* Lines parallel to a */
		for ( k=-1; k<=1; k++ ) {
			for ( l=-1; l<=1; l++ ) {
				vertices[3*i + 0] = (ctx->cell->a.x*(-max_ind) + ctx->cell->b.x*k + ctx->cell->c.x*l)/1e9;
				vertices[3*i + 1] = (ctx->cell->a.y*(-max_ind) + ctx->cell->b.y*k + ctx->cell->c.y*l)/1e9;
				vertices[3*i + 2] = (ctx->cell->a.z*(-max_ind) + ctx->cell->b.z*k + ctx->cell->c.z*l)/1e9;
				i++;
				vertices[3*i + 0] = (ctx->cell->a.x*(max_ind) + ctx->cell->b.x*k + ctx->cell->c.x*l)/1e9;
				vertices[3*i + 1] = (ctx->cell->a.y*(max_ind) + ctx->cell->b.y*k + ctx->cell->c.y*l)/1e9;
				vertices[3*i + 2] = (ctx->cell->a.z*(max_ind) + ctx->cell->b.z*k + ctx->cell->c.z*l)/1e9;
				i++;
			}
		}
		/* Lines parallel to b */
		for ( h=-1; h<=1; h++ ) {
			for ( l=-1; l<=1; l++ ) {
				vertices[3*i + 0] = (ctx->cell->a.x*h + ctx->cell->b.x*(-max_ind) + ctx->cell->c.x*l)/1e9;
				vertices[3*i + 1] = (ctx->cell->a.y*h + ctx->cell->b.y*(-max_ind) + ctx->cell->c.y*l)/1e9;
				vertices[3*i + 2] = (ctx->cell->a.z*h + ctx->cell->b.z*(-max_ind) + ctx->cell->c.z*l)/1e9;
				i++;
				vertices[3*i + 0] = (ctx->cell->a.x*h + ctx->cell->b.x*(max_ind) + ctx->cell->c.x*l)/1e9;
				vertices[3*i + 1] = (ctx->cell->a.y*h + ctx->cell->b.y*(max_ind) + ctx->cell->c.y*l)/1e9;
				vertices[3*i + 2] = (ctx->cell->a.z*h + ctx->cell->b.z*(max_ind) + ctx->cell->c.z*l)/1e9;
				i++;
			}
		}
		/* Lines parallel to c */
		for ( h=-1; h<=1; h++ ) {
			for ( k=-1; k<=1; k++ ) {
				vertices[3*i + 0] = (ctx->cell->a.x*h + ctx->cell->b.x*k + ctx->cell->c.x*(-max_ind))/1e9;
				vertices[3*i + 1] = (ctx->cell->a.y*h + ctx->cell->b.y*k + ctx->cell->c.y*(-max_ind))/1e9;
				vertices[3*i + 2] = (ctx->cell->a.z*h + ctx->cell->b.z*k + ctx->cell->c.z*(-max_ind))/1e9;
				i++;
				vertices[3*i + 0] = (ctx->cell->a.x*h + ctx->cell->b.x*k + ctx->cell->c.x*(max_ind))/1e9;
				vertices[3*i + 1] = (ctx->cell->a.y*h + ctx->cell->b.y*k + ctx->cell->c.y*(max_ind))/1e9;
				vertices[3*i + 2] = (ctx->cell->a.z*h + ctx->cell->b.z*k + ctx->cell->c.z*(max_ind))/1e9;
				i++;
			}
		}

		if ( dw->gl_use_buffers ) {
			glBindBuffer(GL_ARRAY_BUFFER, dw->gl_line_vertex_buffer);
			glBufferData(GL_ARRAY_BUFFER, 3*dw->gl_line_num_vertices*sizeof(GLfloat), vertices, GL_STATIC_DRAW);
			free(vertices);
		} else {
			dw->gl_line_vertex_array = vertices;
		}

	}

	dw->gl_list_id = glGenLists(1);
	glNewList(dw->gl_list_id, GL_COMPILE);

	#if 0
	GLUquadric *quad;
	quad = gluNewQuadric();
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, yellow);
	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, black);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, yellow);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 10.0);
	gluSphere(quad, 10, 32, 32);
	#endif

	/* Bounding cube: 100 nm^-1 side length */
	if ( dw->cube ) {
		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, black);
		glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, blue);
		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);
		glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0);
		glBegin(GL_LINE_LOOP);
			glNormal3f(25.0, 25.0, 25.0);
			glVertex3f(25.0, 25.0, 25.0);
			glNormal3f(-25.0, 25.0, 25.0);
			glVertex3f(-25.0, 25.0, 25.0);
			glNormal3f(-25.0, -25.0, 25.0);
			glVertex3f(-25.0, -25.0, 25.0);
			glNormal3f(25.0, -25.0, 25.0);
			glVertex3f(25.0, -25.0, 25.0);

		glEnd();
		glBegin(GL_LINE_LOOP);
			glNormal3f(25.0, 25.0, -25.0);
			glVertex3f(25.0, 25.0, -25.0);
			glNormal3f(-25.0, 25.0, -25.0);
			glVertex3f(-25.0, 25.0, -25.0);
			glNormal3f(-25.0, -25.0, -25.0);
			glVertex3f(-25.0, -25.0, -25.0);
			glNormal3f(25.0, -25.0, -25.0);
			glVertex3f(25.0, -25.0, -25.0);
		glEnd();
		glBegin(GL_LINES);
			glNormal3f(25.0, 25.0, 25.0);
			glVertex3f(25.0, 25.0, 25.0);
			glNormal3f(25.0, 25.0, -25.0);
			glVertex3f(25.0, 25.0, -25.0);
			glNormal3f(-25.0, 25.0, 25.0);
			glVertex3f(-25.0, 25.0, 25.0);
			glNormal3f(-25.0, 25.0, -25.0);
			glVertex3f(-25.0, 25.0, -25.0);
			glNormal3f(-25.0, -25.0, 25.0);
			glVertex3f(-25.0, -25.0, 25.0);
			glNormal3f(-25.0, -25.0, -25.0);
			glVertex3f(-25.0, -25.0, -25.0);
			glNormal3f(25.0, -25.0, 25.0);
			glVertex3f(25.0, -25.0, 25.0);
			glNormal3f(25.0, -25.0, -25.0);
			glVertex3f(25.0, -25.0, -25.0);
		glEnd();
	}

	/* x, y, z pointers */
	int pointer_head_face;
	/*glPushMatrix();
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);
	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, red);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, black);
	glScalef(10.0, 1.0, 1.0);
	DRAW_POINTER_LINE
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);
	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, black);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, red);
	DRAW_POINTER_HEAD
	glPopMatrix();

	glPushMatrix();
	glRotatef(90.0, 0.0, 0.0, 1.0);
	glScalef(10.0, 1.0, 1.0);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);
	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, green);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, black);
	DRAW_POINTER_LINE
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);
	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, black);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, green);
	DRAW_POINTER_HEAD
	glPopMatrix();

	glPushMatrix();
	glRotatef(-90.0, 0.0, 1.0, 0.0);
	glScalef(10.0, 1.0, 1.0);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);
	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, bblue);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, black);
	DRAW_POINTER_LINE
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);
	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, black);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, bblue);
	DRAW_POINTER_HEAD
	glPopMatrix();*/

	/* Plot the other reflections */
	reflection = ctx->reflectionlist->reflections;
	while ( reflection != NULL ) {

		if ( reflection->type == REFLECTION_VECTOR_MARKER_1 ) {

			glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, black);
			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, red);
			glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);
			glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0);
			glBegin(GL_LINES);
				glNormal3f(0.0, 0.0, 0.0);
				glVertex3f(0.0, 0.0, 0.0);
				glNormal3f(reflection->x/1e9, reflection->y/1e9, reflection->z/1e9);
				glVertex3f(reflection->x/1e9, reflection->y/1e9, reflection->z/1e9);
			glEnd();

		}

		if ( reflection->type == REFLECTION_VECTOR_MARKER_2 ) {

			glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, black);
			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, green);
			glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);
			glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0);
			glBegin(GL_LINES);
				glNormal3f(0.0, 0.0, 0.0);
				glVertex3f(0.0, 0.0, 0.0);
				glNormal3f(reflection->x/1e9, reflection->y/1e9, reflection->z/1e9);
				glVertex3f(reflection->x/1e9, reflection->y/1e9, reflection->z/1e9);
			glEnd();

		}

		if ( reflection->type == REFLECTION_VECTOR_MARKER_3 ) {

			glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, black);
			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, blue);
			glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);
			glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0);
			glBegin(GL_LINES);
				glNormal3f(0.0, 0.0, 0.0);
				glVertex3f(0.0, 0.0, 0.0);
				glNormal3f(reflection->x/1e9, reflection->y/1e9, reflection->z/1e9);
				glVertex3f(reflection->x/1e9, reflection->y/1e9, reflection->z/1e9);
			glEnd();

		}

		reflection = reflection->next;

	};

	/* Draw the reciprocal unit cell if one is available */
	if ( ctx->cell && !dw->lines ) {

		glBegin(GL_LINES);

			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, black);
			glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);
			glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0);
			glNormal3f(1.0, 0.0, 0.0);

			glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, red);
			glVertex3f(0.0, 0.0, 0.0);
			glVertex3f(ctx->cell->a.x/1e9, ctx->cell->a.y/1e9, ctx->cell->a.z/1e9);

			glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, green);
			glVertex3f(0.0, 0.0, 0.0);
			glVertex3f(ctx->cell->b.x/1e9, ctx->cell->b.y/1e9, ctx->cell->b.z/1e9);

			glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, bblue);
			glVertex3f(0.0, 0.0, 0.0);
			glVertex3f(ctx->cell->c.x/1e9, ctx->cell->c.y/1e9, ctx->cell->c.z/1e9);

			glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, purple);
			glVertex3f(ctx->cell->a.x/1e9, ctx->cell->a.y/1e9, ctx->cell->a.z/1e9);
			glVertex3f(ctx->cell->a.x/1e9 + ctx->cell->b.x/1e9,
					ctx->cell->a.y/1e9 + ctx->cell->b.y/1e9,
					ctx->cell->a.z/1e9 + ctx->cell->b.z/1e9);

			glVertex3f(ctx->cell->a.x/1e9 + ctx->cell->b.x/1e9,
					ctx->cell->a.y/1e9 + ctx->cell->b.y/1e9,
					ctx->cell->a.z/1e9 + ctx->cell->b.z/1e9);
			glVertex3f(ctx->cell->b.x/1e9, ctx->cell->b.y/1e9, ctx->cell->b.z/1e9);

			glVertex3f(ctx->cell->c.x/1e9, ctx->cell->c.y/1e9, ctx->cell->c.z/1e9);
			glVertex3f(ctx->cell->c.x/1e9 + ctx->cell->a.x/1e9,
					ctx->cell->c.y/1e9 + ctx->cell->a.y/1e9,
					ctx->cell->c.z/1e9 + ctx->cell->a.z/1e9);

			glVertex3f(ctx->cell->c.x/1e9 + ctx->cell->a.x/1e9,
					ctx->cell->c.y/1e9 + ctx->cell->a.y/1e9,
					ctx->cell->c.z/1e9 + ctx->cell->a.z/1e9);
			glVertex3f(ctx->cell->c.x/1e9 + ctx->cell->a.x/1e9 + ctx->cell->b.x/1e9,
					ctx->cell->c.y/1e9 + ctx->cell->a.y/1e9 + ctx->cell->b.y/1e9,
					ctx->cell->c.z/1e9 + ctx->cell->a.z/1e9 + ctx->cell->b.z/1e9);

			glVertex3f(ctx->cell->c.x/1e9 + ctx->cell->a.x/1e9 + ctx->cell->b.x/1e9,
					ctx->cell->c.y/1e9 + ctx->cell->a.y/1e9 + ctx->cell->b.y/1e9,
					ctx->cell->c.z/1e9 + ctx->cell->a.z/1e9 + ctx->cell->b.z/1e9);
			glVertex3f(ctx->cell->c.x/1e9 + ctx->cell->b.x/1e9,
					ctx->cell->c.y/1e9 + ctx->cell->b.y/1e9,
					ctx->cell->c.z/1e9 + ctx->cell->b.z/1e9);

			glVertex3f(ctx->cell->c.x/1e9 + ctx->cell->b.x/1e9,
					ctx->cell->c.y/1e9 + ctx->cell->b.y/1e9,
					ctx->cell->c.z/1e9 + ctx->cell->b.z/1e9);
			glVertex3f(ctx->cell->c.x/1e9, ctx->cell->c.y/1e9, ctx->cell->c.z/1e9);

			glVertex3f(ctx->cell->c.x/1e9 + ctx->cell->b.x/1e9,
					ctx->cell->c.y/1e9 + ctx->cell->b.y/1e9,
					ctx->cell->c.z/1e9 + ctx->cell->b.z/1e9);
			glVertex3f(ctx->cell->b.x/1e9, ctx->cell->b.y/1e9, ctx->cell->b.z/1e9);

			glVertex3f(ctx->cell->c.x/1e9 + ctx->cell->a.x/1e9 + ctx->cell->b.x/1e9,
					ctx->cell->c.y/1e9 + ctx->cell->a.y/1e9 + ctx->cell->b.y/1e9,
					ctx->cell->c.z/1e9 + ctx->cell->a.z/1e9 + ctx->cell->b.z/1e9);
			glVertex3f(ctx->cell->a.x/1e9 + ctx->cell->b.x/1e9,
					ctx->cell->a.y/1e9 + ctx->cell->b.y/1e9,
					ctx->cell->a.z/1e9 + ctx->cell->b.z/1e9);

			glVertex3f(ctx->cell->c.x/1e9 + ctx->cell->a.x/1e9,
					ctx->cell->c.y/1e9 + ctx->cell->a.y/1e9,
					ctx->cell->c.z/1e9 + ctx->cell->a.z/1e9);
			glVertex3f(ctx->cell->a.x/1e9, ctx->cell->a.y/1e9, ctx->cell->a.z/1e9);

		glEnd();

	}

	/* Tilt axis */
	//if ( ctx->images->n_images > 0 ) {
	//	glPushMatrix();
		/* Images rotate clockwise by omega to put tilt axis at +x,
		 *	so rotate tilt axis anticlockwise by omega.
		 *	Since the rotation is about +z, this is already anticlockwise
		 *	when looking down z. */
	/*	glRotatef(rad2deg(ctx->images->images[0].omega), 0.0, 0.0, 1.0);
		glScalef(50.0, 1.0, 1.0);
		glTranslatef(-0.5, 0.0, 0.0);
		glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, yellow);
		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, black);
		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);
		glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0);
		DRAW_POINTER_LINE
		glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, black);
		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, yellow);
		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);
		glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0);
		DRAW_POINTER_HEAD
		glPopMatrix();
	}*/

	/* Zero plane (must be drawn last for transparency to work) */
//	glBegin(GL_QUADS);
//		glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, black);
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, glass);
//		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);
//		glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0);
//		glNormal3f(0.0, 0.0, 1.0);
//		glVertex3f(50.0, 50.0, 0.0);
//		glVertex3f(50.0, -50.0, 0.0);
//		glVertex3f(-50.0, -50.0, 0.0);
//		glVertex3f(-50.0, 50.0, 0.0);
//	glEnd();

	glEndList();

	//printf("DW: Vertex counts: meas:%i, mark:%i, gen:%i\n", dw->gl_ref_num_vertices, dw->gl_marker_num_vertices, dw->gl_gen_num_vertices);

}

void glbits_set_ortho(DisplayWindow *dw, GLfloat w, GLfloat h) {

	GLfloat aspect = w/h;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-aspect*(dw->distance/2.0), aspect*(dw->distance/2.0), -(dw->distance/2.0), (dw->distance/2.0), 0.001, 400.0);
	//glOrtho(-aspect*(dw->distance/2.0), aspect*(dw->distance/2.0), -(dw->distance/2.0), (dw->distance/2.0), 150.0, 200.0);
	glMatrixMode(GL_MODELVIEW);

}

void glbits_set_perspective(DisplayWindow *dw, GLfloat w, GLfloat h) {
	
	GLfloat aspect = w/h;
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//gluPerspective(50.0, w/h, 0.001, 400.0);
	glOrtho(-aspect*(dw->distance/2.0), aspect*(dw->distance/2.0), -(dw->distance/2.0), (dw->distance/2.0), 150, 152);
		
	glMatrixMode(GL_MODELVIEW);

}

gint glbits_expose(GtkWidget *widget, GdkEventExpose *event, DisplayWindow *dw)
{	
	GdkGLContext *glcontext = gtk_widget_get_gl_context(widget);
	GdkGLDrawable *gldrawable = gtk_widget_get_gl_drawable(widget);
	float m[4][4];
	GLfloat black[] = { 0.0, 0.0, 0.0, 1.0 };
	GLfloat blue[] = { 0.0, 0.0, 0.5, 1.0 };
	GLfloat white[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat gold[] = { 0.7, 0.7, 0.0, 1.0 };
	GLfloat light0_position[] = { 0.0, 1.0, 1.0, 0.0 };
	GLfloat light0_diffuse[]  = { 0.8, 0.8, 0.8, 1.0 };
	GLfloat light0_specular[] = { 0.8, 0.8, 0.8, 1.0 };
	GLfloat grey[] = { 0.6, 0.6, 0.6, 1.0 };
	GLfloat bg_top[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat bg_bot[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat w = dw->drawing_area->allocation.width;
	GLfloat h = dw->drawing_area->allocation.height;
	GLfloat aspect = w/h;

	if ( !gdk_gl_drawable_gl_begin(gldrawable, glcontext) ) {
		return 0;
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if ( dw->background ) {
		bg_top[0] = 0.0; bg_top[1] = 0.3; bg_top[2] = 1.0;
		bg_bot[0] = 0.0; bg_bot[1] = 0.7; bg_bot[2] = 1.0;
	}

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, -20.0);
	/* Draw the background (this is done before setting up rotations) */
	/* Set up "private" projection matrix */
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(-aspect*3.0, aspect*3.0, -3.0, 3.0, 0.001, 21.0);
	/* Draw background plane */
	glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, black);
	glMaterialfv(GL_FRONT, GL_SPECULAR, black);
	glMaterialf(GL_FRONT, GL_SHININESS, 0.0);
	glBegin(GL_QUADS);
		glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, bg_bot);
		glVertex3f(-3.0*aspect, -3.0, 0.0);
		glVertex3f(+3.0*aspect, -3.0, 0.0);
		glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, bg_top);
		glVertex3f(+3.0*aspect, +3.0, 0.0);
		glVertex3f(-3.0*aspect, +3.0, 0.0);
	glEnd();
	/* Restore the old projection matrix */
	glPopMatrix();
	glClear(GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	/* Set up lighting */
	glEnable(GL_LIGHT0);
	glLightfv(GL_LIGHT0, GL_POSITION, light0_position);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light0_specular);

	/* The z component of this makes no difference if the projection is orthographic,
	 *	but controls zoom when perspective is used */
	glTranslatef(0.0, 0.0, -400.0);
	glTranslatef(dw->x_pos, -dw->y_pos, 400.0-dw->distance);
	build_rotmatrix(m, dw->view_quat);
	//dw->theta = atan2(m[1][0], m[1][2])*180.0/M_PI;
	dw->theta =	acos(m[0][0])*180.0/M_PI;										/*angles seem to work, now to output them usefully in the displaywindow!*/
	dw->phi = acos(m[1][1])*180.0/M_PI;
	dw->psi = acos(m[2][2])*180.0/M_PI;
	//dw->psi = atan2(m[0][1], m[2][1])*180.0/M_PI;
	//printf("%f, %f, %f\n", dw->theta, dw->phi, dw->psi);
	printf("%f, %f, %f, %f\n\n", dw->view_quat[0], dw->view_quat[1], dw->view_quat[2], dw->view_quat[3]);
	glMultMatrixf(&m[0][0]);

	/* begin suspect block */
	glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);

	if ( dw->mode == DW_MAPPED ) {

		/* Draw the "measured" reflections */
		if ( dw->gl_ref_num_vertices ) {

			GLfloat att[] = {1.0, 1.0, 0.0};

			glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, att);
			glPointSize(40.0);													/*Changes the 'size' of the rendered reflections*/
			glEnable(GL_POINT_SMOOTH);

			glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, black);
			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, black);
			glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);
			glColor3f(0.0, 0.0, 0.0);
			glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0);

			if ( dw->gl_use_buffers ) {
				glBindBuffer(GL_ARRAY_BUFFER, dw->gl_ref_vertex_buffer);
				glVertexPointer(3, GL_FLOAT, 0, NULL);
				glBindBuffer(GL_ARRAY_BUFFER, dw->gl_ref_normal_buffer);
				glNormalPointer(GL_FLOAT, 0, NULL);
				glDrawArrays(GL_POINTS, 0, dw->gl_ref_num_vertices);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			} else {
				glVertexPointer(3, GL_FLOAT, 0, dw->gl_ref_vertex_array);
				glNormalPointer(GL_FLOAT, 0, dw->gl_ref_normal_array);
				glDrawArrays(GL_POINTS, 0, dw->gl_ref_num_vertices);
			}

			glDisable(GL_POINT_SMOOTH);

		}

		/* Draw marker points */
		if ( dw->gl_marker_num_vertices ) {

			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, blue);
			glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, black);
			glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
			glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 100.0);
			glColor3f(0.0, 0.0, 1.0);

			if ( dw->gl_use_buffers ) {
				glBindBuffer(GL_ARRAY_BUFFER, dw->gl_marker_vertex_buffer);
				glVertexPointer(3, GL_FLOAT, 0, NULL);
				glBindBuffer(GL_ARRAY_BUFFER, dw->gl_marker_normal_buffer);
				glNormalPointer(GL_FLOAT, 0, NULL);
				glDrawArrays(GL_QUADS, 0, dw->gl_marker_num_vertices);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			} else {
				glVertexPointer(3, GL_FLOAT, 0, dw->gl_marker_vertex_array);
				glNormalPointer(GL_FLOAT, 0, dw->gl_marker_normal_array);
				glDrawArrays(GL_QUADS, 0, dw->gl_marker_num_vertices);
			}

		}

	} else {

		/* Draw generated reflections */
		if ( dw->gl_gen_num_vertices ) {

			glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, black);
			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, gold);
			glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
			glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 100.0);
			glColor3f(0.7, 0.7, 0.0);

			if ( dw->gl_use_shaders ) glUseProgram(dw->gl_program_lightpp);
			if ( dw->gl_use_buffers ) {
				glBindBuffer(GL_ARRAY_BUFFER, dw->gl_gen_vertex_buffer);
				glVertexPointer(3, GL_FLOAT, 0, NULL);
				glBindBuffer(GL_ARRAY_BUFFER, dw->gl_gen_normal_buffer);
				glNormalPointer(GL_FLOAT, 0, NULL);
				glDrawArrays(GL_QUADS, 0, dw->gl_gen_num_vertices);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			} else {
				glVertexPointer(3, GL_FLOAT, 0, dw->gl_gen_vertex_array);
				glNormalPointer(GL_FLOAT, 0, dw->gl_gen_normal_array);
				glDrawArrays(GL_QUADS, 0, dw->gl_gen_num_vertices);
			}
			if ( dw->gl_use_shaders ) glUseProgram(0);

		}

	}
	glDisable(GL_NORMAL_ARRAY);

	/* Draw indexing lines */
	if ( dw->lines && dw->gl_line_num_vertices ) {

		glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, grey);
		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, black);
		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);

		if ( dw->gl_use_buffers ) {
			glBindBuffer(GL_ARRAY_BUFFER, dw->gl_line_vertex_buffer);
			glVertexPointer(3, GL_FLOAT, 0, NULL);
			glDrawArrays(GL_LINES, 0, dw->gl_line_num_vertices);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		} else {
			glVertexPointer(3, GL_FLOAT, 0, dw->gl_line_vertex_array);
			glDrawArrays(GL_LINES, 0, dw->gl_line_num_vertices);
		}
	}

	glPopClientAttrib();

	/* Draw everything else */
	glCallList(dw->gl_list_id);

	if ( gdk_gl_drawable_is_double_buffered(gldrawable) ) {
		gdk_gl_drawable_swap_buffers(gldrawable);
	} else {
		glFlush();
	}

	gdk_gl_drawable_gl_end(gldrawable);

	return TRUE;

}

gboolean glbits_configure(GtkWidget *widget, GdkEventConfigure *event, DisplayWindow *dw) {

	GdkGLContext *glcontext = gtk_widget_get_gl_context(widget);
	GdkGLDrawable *gldrawable = gtk_widget_get_gl_drawable(widget);
	GLfloat w = widget->allocation.width;
	GLfloat h = widget->allocation.height;

	/* Set viewport */
	if ( !gdk_gl_drawable_gl_begin(gldrawable, glcontext) ) {
		return FALSE;
	}
	glViewport(0, 0, w, h);

	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_LINE_SMOOTH);

	/* Nudge the projection matrix routines to preserve the aspect ratio */
	if ( dw->view == DW_ORTHO ) {
		glbits_set_ortho(dw, w, h);
	} else {
		glbits_set_perspective(dw, w, h);
	}

	gdk_gl_drawable_gl_end(gldrawable);

	return FALSE;

}

void glbits_free_resources(DisplayWindow *dw) {

	if ( dw->gl_use_buffers ) {
		glDeleteBuffers(1, &dw->gl_ref_vertex_buffer);
		glDeleteBuffers(1, &dw->gl_ref_normal_buffer);
		glDeleteBuffers(1, &dw->gl_marker_vertex_buffer);
		glDeleteBuffers(1, &dw->gl_marker_normal_buffer);
		glDeleteBuffers(1, &dw->gl_gen_vertex_buffer);
		glDeleteBuffers(1, &dw->gl_gen_normal_buffer);
		glDeleteBuffers(1, &dw->gl_line_vertex_buffer);
	} else {
		if ( dw->gl_ref_vertex_array != NULL ) free(dw->gl_ref_vertex_array);
		if ( dw->gl_ref_normal_array != NULL ) free(dw->gl_ref_normal_array);
		if ( dw->gl_marker_vertex_array != NULL ) free(dw->gl_marker_vertex_array);
		if ( dw->gl_marker_normal_array != NULL ) free(dw->gl_marker_normal_array);
		if ( dw->gl_gen_vertex_array != NULL ) free(dw->gl_gen_vertex_array);
		if ( dw->gl_gen_normal_array != NULL ) free(dw->gl_gen_normal_array);
		if ( dw->gl_line_vertex_array != NULL ) free(dw->gl_line_vertex_array);
	}
	glDeleteLists(dw->gl_list_id, 1);

}

void glbits_final_free_resources(DisplayWindow *dw) {
	glbits_free_resources(dw);
	if ( dw->gl_use_shaders ) glbits_delete_shaders(dw);
}

static void glbits_first_prepare(DisplayWindow *dw) {
	
	glbits_prepare(dw);
	if ( dw->gl_use_shaders ) glbits_load_shaders(dw);
}

gint glbits_realise(GtkWidget *widget, DisplayWindow *dw) {

	GdkGLContext *glcontext = gtk_widget_get_gl_context(widget);
	GdkGLDrawable *gldrawable = gtk_widget_get_gl_drawable(widget);
	GLfloat w = widget->allocation.width;
	GLfloat h = widget->allocation.height;

	if ( !gdk_gl_drawable_gl_begin(gldrawable, glcontext) ) {
		return 0;
	}

	/* This has to be done once an OpenGL context has been created */
	GLenum glew_err = glewInit();
	if (glew_err != GLEW_OK) fprintf(stderr, "GLEW initialisation error: %s\n", glewGetErrorString(glew_err));

	glbits_set_ortho(dw, w, h);
	glbits_first_prepare(dw);
	dw->realised = 1;

	gdk_gl_drawable_gl_end(gldrawable);

	return 0;

}
