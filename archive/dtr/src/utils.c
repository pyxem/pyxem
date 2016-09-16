/*
 * utils.c
 *
 * Utility stuff
 *
 * (c) 2007-2008 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#include <math.h>
#include <gsl/gsl_matrix.h>
#include <string.h>

#include "utils.h"

/* Return the MOST POSITIVE of two numbers */
unsigned int biggest(signed int a, signed int b) {
	if ( a>b ) {
		return a;
	}
	return b;
}

/* Return the LEAST POSITIVE of two numbers */
unsigned int smallest(signed int a, signed int b) {
	if ( a<b ) {
		return a;
	}
	return b;
}

double distance(double x1, double y1, double x2, double y2) {
	return sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
}

double modulus(double x, double y, double z) {
	return sqrt(x*x + y*y + z*z);
}

double modulus_squared(double x, double y, double z) {
	return x*x + y*y + z*z;
}

double distance3d(double x1, double y1, double z1, double x2, double y2, double z2) {
	return modulus(x1-x2, y1-y2, z1-z2);
}

/* Angle between two vectors.  Answer in radians */
double angle_between(double x1, double y1, double z1, double x2, double y2, double z2) {

	double mod1 = modulus(x1, y1, z1);
	double mod2 = modulus(x2, y2, z2);
	return acos( (x1*x2 + y1*y2 + z1*z2) / (mod1*mod2) );

}

/* As above, answer in degrees */
double angle_between_d(double x1, double y1, double z1, double x2, double y2, double z2) {
	return rad2deg(angle_between(x1, y1, z1, x2, y2, z2));
}

/* Wavelength of an electron (in m) given accelerating potential (in V) */
double lambda(double V) {

	double m = 9.110E-31;
	double h = 6.625E-34;
	double e = 1.60E-19;
	double c = 2.998E8;
	
	return h / sqrt(2*m*e*V*(1+((e*V) / (2*m*c*c))));

}

size_t skipspace(const char *s) {

	size_t i;
	
	for ( i=0; i<strlen(s); i++ ) {
		if ( (s[i] != ' ') && (s[i] != '\t') ) return i;
	}
	
	return strlen(s);

}

void chomp(char *s) {

	size_t i;
	
	if ( !s ) return;
	
	for ( i=0; i<strlen(s); i++ ) {
		if ( (s[i] == '\n') || (s[i] == '\r') ) {
			s[i] = '\0';
			return;
		}
	}
	
}

void matrix_vector_show(const gsl_matrix *m, const gsl_vector *v, const char *prefix) {

	int i, j;
	
	if ( prefix == NULL ) prefix = "";
	
	for ( i=0; i<m->size1; i++ ) {
		printf("%s[ ", prefix);
		for ( j=0; j<m->size2; j++ ) {
			printf("%12.8f ", gsl_matrix_get(m, i, j));
		}
		printf(" ] [ q%i ]  =  [ %15.2f ]\n", i+1, gsl_vector_get(v, i));
	}

}

int sign(double a) {
	if ( a < 0 ) return -1;
	if ( a > 0 ) return +1;
	return 0;
}

