/*
 * mrc.h
 *
 * Read the MRC tomography format
 *
 * (c) 2007 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef MRC_H
#define MRC_H

#include <stdint.h>

#include "control.h"

typedef struct struct_mrcheader {
	
	int32_t		nx;		/*   0 */
	int32_t		ny;		/*   4 */
	int32_t		nz;		/*   8 */
	int32_t		mode;		/*  12 */
	int32_t		nxstart;	/*  16 */
	int32_t		nystart;	/*  20 */
	int32_t		nzstart;	/*  24 */
	int32_t		mx;		/*  28 */
	int32_t		my;		/*  32 */
	int32_t		mz;		/*  36 */
	float		xlen;		/*  40 */
	float		ylen;		/*  44 */
	float		zlen;		/*  48 */
	float		alpha;		/*  52 */
	float		beta;		/*  56 */
	float		gamma;		/*  60 */
	int32_t		mapc;		/*  64 */
	int32_t		mapr;		/*  68 */
	int32_t		maps;		/*  72 */
	float		amin;		/*  76 */
	float		amax;		/*  80 */
	float		amean;		/*  84 */
	uint16_t	ispg;		/*  88 (4 byte word aligned) Space group number */
	uint16_t	nsymbt;		/*  90 (2 byte word aligned) */
	int32_t		next;		/*  92 (back to 4 byte word aligned) */
	uint16_t	dvid;		/*  96 (4 byte word aligned) */
	char		extra[30];	/*  98 (2 byte aligned, this padding puts it back on 4 byte alignment) */
	uint16_t	numintegers;	/* 128 (4) */
	uint16_t	numfloats;	/* 130 (2) */
	uint16_t	sub;		/* 132 (4) */
	uint16_t	zfac;		/* 134 (2) */
	float		min2;		/* 136 (4 byte word aligned) */
	float		max2;		/* 140 */
	float		min3;		/* 144 */
	float		max3;		/* 148 */
	float		min4;		/* 152 */
	float		max4;		/* 156 */
	uint16_t	idtype;		/* 160 */
	uint16_t	lens;		/* 162 */
	uint16_t	nd1;		/* 164 */
	uint16_t	nd2;		/* 166 */
	uint16_t	vd1;		/* 168 */
	uint16_t	vd2;		/* 170 (back to 4 byte alignment - yay!) */
	float		tiltangles[9];	/* 172 (4 byte alignment) */
	float		zorg;		/* 208 (=172+9*4) */
	float		xorg;		/* 212 */
	float		yorg;		/* 216 */
	int32_t		nlabl;		/* 220 */
	char		data[10][80];	/* 224 (4 byte alignment) */
	
} MRCHeader;	/* 1024 bytes total = 224+10*80 */

typedef struct struct_mrcextheader {

	float		a_tilt;
	float		b_tilt;
	float		x_stage;
	float		y_stage;
	float		z_stage;
	float		x_shift;
	float		y_shift;
	float		defocus;
	float		exp_time;
	float		mean_int;
	float		tilt_axis;
	float		pixel_size;
	float		magnification;
	float		mic_type;
	float		gun_type;
	float		d_number;
	float		voltage;
	float		focus_spread;
	float		mtf;
	float		df_start;
	float		focus_step;
	float		dac_setting;
	float		cs;
	float		semi_conv;
	float		info_limit;
	float		num_images;
	float		num_in_series;
	float		coma1;
	float		coma2;
	float		astig21;
	float		astig22;
	float		astig31;
	float		astig32;
	float		cam_type;
	float		cam_pos;
	float		padding[64];	/* Need to guarantee that reading an extended header (of any size)
						will never write beyond the boundary of this structure... */
	
} MRCExtHeader;

extern int mrc_read(ControlContext *ctx);
extern unsigned int mrc_is_mrcfile(const char *filename);

#endif	/* MRC_H */

