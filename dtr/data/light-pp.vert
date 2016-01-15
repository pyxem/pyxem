/*
 * light-pp.vert
 *
 * Lighting per pixel
 *
 * (c) 2007 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

varying vec3 col_ambi;
varying vec3 col_diff;
varying vec3 col_spec;
varying vec3 col_emit;
varying float shininess;

varying vec3 normal;
varying vec3 lightvc;
varying vec3 lighthvc;

void main() {
	
	normal = normalize(gl_NormalMatrix * gl_Normal);
	lightvc = normalize(vec3(gl_LightSource[0].position));
	lighthvc = normalize(gl_LightSource[0].halfVector.xyz);
	
	col_ambi = gl_Color.rgb;
	col_diff = gl_Color.rgb;
	col_spec = gl_FrontMaterial.specular.rgb;
	col_emit = gl_FrontMaterial.emission.rgb;
	shininess = gl_FrontMaterial.shininess;
	
	gl_Position = ftransform();
	
}

