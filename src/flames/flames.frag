#version 450

in vec2 texCoord;
layout(location = 0) out vec4 fragColour;

uniform usampler2D histogram;
uniform sampler2DArray colour;

uniform uint maximumVal;
uniform float gamma;

uniform vec4 background;

const float PI = 3.14159265358979;

const int kernelRadius = 9;
const float smoothing = 0.4;

float invLogMax = 1 / max(log(float(maximumVal)), 2);

float computeAlpha(uint hits) {
    float alpha = log(max(float(hits), 1.0)) * invLogMax;
	return min(pow(alpha, gamma), 1.0);
}

void main() {
	uint hits = texture(histogram, texCoord).r;
	vec3 result = vec3(0);
	/*for (int x = 0; x<SUPERSAMPLES; x++) {
		for (int y = 0; y<SUPERSAMPLES; y++) {
			result += textureOffset(colour, texCoord, ivec2(x,y)).rgb;
		}
	}*/
	
	// Equivalent to above iteration
	UNROLLED_SUPERSAMPLING

	result *= 1.0 / SUPERSAMPLES;

	float alpha = computeAlpha(hits);
	if (alpha == 0) {
		fragColour = background;
	} else {
		float finalAlpha = alpha + background.a * (1 - alpha);
		fragColour = vec4((result * alpha + background.rgb*background.a*(1-alpha)) / finalAlpha, finalAlpha);
	}
}
