/*
 * light-pp.frag
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
	
	vec3 ambi;
	vec3 emit;
	vec3 diff;
	vec3 spec;
	
	vec3 norm = normalize(normal);
	
	/* Ambient contribution */
	ambi = col_ambi * gl_LightModel.ambient.rgb;
	
	/* Emission */
	emit = col_emit;
	
	/* Diffuse contribution */
	diff = col_diff * gl_LightSource[0].diffuse.rgb * max(dot(normalize(lightvc).xyz, norm), 0.0);
	
	/* Specular contribution */
	if ( col_spec.r > 0.0 ) {
		float ndothv = max(dot(norm, normalize(lighthvc)), 0.0);
		spec = col_spec * gl_LightSource[0].specular.rgb * pow(ndothv, shininess);
	}
	
	gl_FragColor = vec4(min(emit.r + ambi.r + diff.r + spec.r, 1.0),
			    min(emit.g + ambi.g + diff.g + spec.g, 1.0),
			    min(emit.b + ambi.b + diff.b + spec.b, 1.0),
			    1.0);
	
}

