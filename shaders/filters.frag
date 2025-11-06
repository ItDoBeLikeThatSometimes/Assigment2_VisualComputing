#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D texture1;
uniform int   uMode;
uniform vec2  uTexSize;
uniform float uBlock;
uniform float uEdgeThresh;

uniform mat3  uAffine;

float luminance(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }
vec3 posterize(vec3 c, float levels){ return floor(c * levels) / levels; }

float sobelEdge(vec2 uv){
    vec2 t = 1.0 / uTexSize;
    float tl=luminance(texture(texture1, uv+t*vec2(-1, 1)).rgb);
    float  t0=luminance(texture(texture1, uv+t*vec2( 0, 1)).rgb);
    float tr=luminance(texture(texture1, uv+t*vec2( 1, 1)).rgb);
    float l =luminance(texture(texture1, uv+t*vec2(-1, 0)).rgb);
    float r =luminance(texture(texture1, uv+t*vec2( 1, 0)).rgb);
    float bl=luminance(texture(texture1, uv+t*vec2(-1,-1)).rgb);
    float  b=luminance(texture(texture1, uv+t*vec2( 0,-1)).rgb);
    float br=luminance(texture(texture1, uv+t*vec2( 1,-1)).rgb);
    float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
    float gy =  tl + 2.0*t0 + tr - bl - 2.0*b - br;
    return sqrt(gx*gx + gy*gy);
}

void main() {
    if (TexCoord.x < 0.02) { FragColor = vec4(1,0,0,1); return; }


    // Apply affine to destination UV to get source UV
    vec3 src = uAffine * vec3(TexCoord, 1.0);
    vec2 uv  = clamp(src.xy, vec2(0.0), vec2(1.0));

    vec4 base = texture(texture1, uv);

    if (uMode == 1) {
        vec2 stepUV = uBlock / uTexSize;
        vec2 uvq = floor(uv / stepUV) * stepUV + stepUV * 0.5;
        FragColor = texture(texture1, clamp(uvq, vec2(0.0), vec2(1.0)));
    } else if (uMode == 2) {
        vec3 color = posterize(base.rgb, 5.0);
        float e = sobelEdge(uv);
        float edgeMask = (e > uEdgeThresh) ? 0.0 : 1.0;
        FragColor = vec4(color * edgeMask, 1.0);
    } else {
        FragColor = base;
    }
}
