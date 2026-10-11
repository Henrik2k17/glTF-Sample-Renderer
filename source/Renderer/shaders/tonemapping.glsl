uniform float u_Exposure;


const float GAMMA = 2.2;
const float INV_GAMMA = 1.0 / GAMMA;


// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
const mat3 ACESInputMat = mat3
(
    0.59719, 0.07600, 0.02840,
    0.35458, 0.90834, 0.13383,
    0.04823, 0.01566, 0.83777
);


// ODT_SAT => XYZ => D60_2_D65 => sRGB
const mat3 ACESOutputMat = mat3
(
    1.60475, -0.10208, -0.00327,
    -0.53108,  1.10813, -0.07276,
    -0.07367, -0.00605,  1.07602
);


// linear to sRGB approximation
// see http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
vec3 linearTosRGB(vec3 color)
{
    return pow(color, vec3(INV_GAMMA));
}


// sRGB to linear approximation
// see http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
vec3 sRGBToLinear(vec3 srgbIn)
{
    return vec3(pow(srgbIn.xyz, vec3(GAMMA)));
}


vec4 sRGBToLinear(vec4 srgbIn)
{
    return vec4(sRGBToLinear(srgbIn.xyz), srgbIn.w);
}


// ACES tone map (faster approximation)
// see: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 toneMapACES_Narkowicz(vec3 color)
{
    const float A = 2.51;
    const float B = 0.03;
    const float C = 2.43;
    const float D = 0.59;
    const float E = 0.14;
    return clamp((color * (A * color + B)) / (color * (C * color + D) + E), 0.0, 1.0);
}


// ACES filmic tone map approximation
// see https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
vec3 RRTAndODTFit(vec3 color)
{
    vec3 a = color * (color + 0.0245786) - 0.000090537;
    vec3 b = color * (0.983729 * color + 0.4329510) + 0.238081;
    return a / b;
}


// tone mapping
vec3 toneMapACES_Hill(vec3 color)
{
    color = ACESInputMat * color;

    // Apply RRT and ODT
    color = RRTAndODTFit(color);

    color = ACESOutputMat * color;

    // Clamp to [0, 1]
    color = clamp(color, 0.0, 1.0);

    return color;
}

// MSFS 2024 SDR display curve (shader pack, tonemapping): per channel, linear up to 0.22, then
// a soft roll-off towards 1. The sim grades with a 3D LUT and tonemaps locally before it.
#ifdef TONEMAP_MSFS_SDR
const float MSFS_SDR_KNEE = 0.22;
const float MSFS_SDR_SLOPE = 1.849609;

vec3 toneMapMsfsSdr(vec3 color)
{
    vec3 rolled = MSFS_SDR_KNEE + (1.0 - MSFS_SDR_KNEE) * (1.0 - exp(-MSFS_SDR_SLOPE * (color - MSFS_SDR_KNEE)));
    return mix(color, rolled, step(MSFS_SDR_KNEE, color));
}

vec3 toneMapMsfsSdrInverse(vec3 color)
{
    vec3 unrolled = MSFS_SDR_KNEE - log(max(1.0 - (color - MSFS_SDR_KNEE) / (1.0 - MSFS_SDR_KNEE), 1e-6)) / MSFS_SDR_SLOPE;
    return mix(color, unrolled, step(MSFS_SDR_KNEE, color));
}
#endif

// Khronos PBR neutral tone mapping
#ifdef TONEMAP_KHR_PBR_NEUTRAL
vec3 toneMap_KhronosPbrNeutral( vec3 color )
{
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.15;

    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;

    const float d = 1. - startCompression;
    float newPeak = 1. - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    float g = 1. - 1. / (desaturation * (peak - newPeak) + 1.);
    return mix(color, newPeak * vec3(1, 1, 1), g);
}
#endif

vec3 toneMap(vec3 color)
{
    color *= u_Exposure;

#ifdef TONEMAP_ACES_NARKOWICZ
    color = toneMapACES_Narkowicz(color);
#endif

#ifdef TONEMAP_ACES_HILL
    color = toneMapACES_Hill(color);
#endif

#ifdef TONEMAP_ACES_HILL_EXPOSURE_BOOST
    // boost exposure as discussed in https://github.com/mrdoob/three.js/pull/19621
    // this factor is based on the exposure correction of Krzysztof Narkowicz in his
    // implemetation of ACES tone mapping
    color /= 0.6;
    color = toneMapACES_Hill(color);
#endif

#ifdef TONEMAP_KHR_PBR_NEUTRAL
    color = toneMap_KhronosPbrNeutral(color);
#endif

#ifdef TONEMAP_MSFS_SDR
    color = toneMapMsfsSdr(max(color, vec3(0.0)));
#endif

    return linearTosRGB(color);
}

// =============================================================================
// INVERSE TONEMAPPING FUNCTIONS
// =============================================================================

// Inverse ACES tone map (Narkowicz approximation)
// Solves: y = (x*(A*x + B)) / (x*(C*x + D) + E) for x given y
vec3 toneMapACES_NarkowiczInverse(vec3 toneMapped)
{
    const float A = 2.51;
    const float B = 0.03;
    const float C = 2.43;
    const float D = 0.59;
    const float E = 0.14;

    vec3 y = toneMapped;
        
    // Rearrange to: y*(C*x^2 + D*x + E) = x*(A*x + B)
    // Which gives: y*C*x^2 + y*D*x + y*E = A*x^2 + B*x
    // Rearrange to: (y*C - A)*x^2 + (y*D - B)*x + y*E = 0
    
    vec3 a = y * C - A;
    vec3 b = y * D - B;
    vec3 c = y * E;
    
    // Solve quadratic equation: ax^2 + bx + c = 0
    vec3 discriminant = b * b - 4.0 * a * c;
    
    // Take positive root
    vec3 result = (-b - sqrt(discriminant)) / (2.0 * a);
    
    return result;
}

// Inverse RRT and ODT fit
// Solves: y = (x*(x + 0.0245786) - 0.000090537) / (x*(0.983729*x + 0.4329510) + 0.238081) for x
vec3 RRTAndODTFitInverse(vec3 toneMapped)
{
    vec3 y = toneMapped;
        
    // Rearrange: y * (x*(0.983729*x + 0.4329510) + 0.238081) = x*(x + 0.0245786) - 0.000090537
    // y * (0.983729*x^2 + 0.4329510*x + 0.238081) = x^2 + 0.0245786*x - 0.000090537
    // y*0.983729*x^2 + y*0.4329510*x + y*0.238081 = x^2 + 0.0245786*x - 0.000090537
    // (y*0.983729 - 1.0)*x^2 + (y*0.4329510 - 0.0245786)*x + (y*0.238081 + 0.000090537) = 0
    
    vec3 a = y * 0.983729 - 1.0;
    vec3 b = y * 0.4329510 - 0.0245786;
    vec3 c = y * 0.238081 + 0.000090537;
    
    vec3 discriminant = b * b - 4.0 * a * c;
    
    vec3 result = (-b - sqrt(discriminant)) / (2.0 * a);
    
    return result;
}

// Inverse ACES Hill tone mapping
vec3 toneMapACES_HillInverse(vec3 toneMapped)
{
    vec3 color = toneMapped;
    
    // Undo ACES output matrix
    color = inverse(ACESOutputMat) * color;
    
    // Undo RRT and ODT
    color = RRTAndODTFitInverse(color);
    
    // Undo ACES input matrix
    color = inverse(ACESInputMat) * color;
    
    return color;
}

// Inverse Khronos PBR neutral tone mapping
// Note: This is a complex inverse that may not be perfectly accurate due to the conditional logic
vec3 toneMap_KhronosPbrNeutralInverse(vec3 toneMapped)
{
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.15;
    
    // This is an approximation - the forward function has complex conditional logic
    // that makes perfect inversion difficult
    vec3 color = toneMapped;
    
    // Try to undo the desaturation mix
    float peak = max(color.r, max(color.g, color.b));
    
    // Approximate inverse of the compression
    if (peak >= startCompression) {
        const float d = 1.0 - startCompression;
        // Approximate inverse of: newPeak = 1. - d * d / (peak + d - startCompression)
        // This is a rough approximation
        float originalPeak = peak / (1.0 - peak + startCompression);
        color *= originalPeak / peak;
    }
    
    // Try to undo the offset
    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color += offset;
    
    return color;
}

// Complete inverse tone mapping function
vec3 toneMapInverse(vec3 toneMapped)
{
    vec3 color = toneMapped;
    
    // Then undo the specific tonemapping (this would need to match the forward path)
#ifdef TONEMAP_KHR_PBR_NEUTRAL
    color = toneMap_KhronosPbrNeutralInverse(color);
#endif

#ifdef TONEMAP_MSFS_SDR
    color = toneMapMsfsSdrInverse(color);
#endif

#ifdef TONEMAP_ACES_HILL_EXPOSURE_BOOST
    color = toneMapACES_HillInverse(color);
    color *= 0.6;  // Undo the exposure boost
#endif

#ifdef TONEMAP_ACES_HILL
    color = toneMapACES_HillInverse(color);
#endif

#ifdef TONEMAP_ACES_NARKOWICZ
    color = toneMapACES_NarkowiczInverse(color);
#endif

    if (u_Exposure > 0.0)
    {
        color /= u_Exposure; // Undo the exposure
    }
    
    return color;
}
