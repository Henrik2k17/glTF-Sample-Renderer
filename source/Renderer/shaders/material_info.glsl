// Metallic Roughness
uniform float u_MetallicFactor;
uniform float u_RoughnessFactor;
uniform vec4 u_BaseColorFactor;

// Sheen
uniform float u_SheenRoughnessFactor;
uniform vec3 u_SheenColorFactor;

// Clearcoat
uniform float u_ClearcoatFactor;
uniform float u_ClearcoatRoughnessFactor;

// Specular
uniform vec3 u_KHR_materials_specular_specularColorFactor;
uniform float u_KHR_materials_specular_specularFactor;

// Transmission
uniform float u_TransmissionFactor;

// Volume
uniform float u_ThicknessFactor;
uniform vec3 u_AttenuationColor;
uniform float u_AttenuationDistance;

// Iridescence
uniform float u_IridescenceFactor;
uniform float u_IridescenceIor;
uniform float u_IridescenceThicknessMinimum;
uniform float u_IridescenceThicknessMaximum;

// Retroreflection
uniform float u_RetroreflectionFactor;

// Diffuse Transmission
uniform float u_DiffuseTransmissionFactor;
uniform vec3 u_DiffuseTransmissionColorFactor;

// Emissive Strength
uniform float u_EmissiveStrength;

// IOR
uniform float u_Ior;

// Anisotropy
uniform vec3 u_Anisotropy;

// Dispersion
uniform float u_Dispersion;

// Alpha mode
uniform float u_AlphaCutoff;

uniform vec3 u_Camera;

#ifdef MATERIAL_TRANSMISSION
uniform ivec2 u_ScreenSize;
#endif

uniform mat4 u_ModelMatrix;
uniform mat4 u_ViewMatrix;
uniform mat4 u_ProjectionMatrix;

#if DEBUG == DEBUG_TANGENT_W
in float v_TangentWSign;
#endif


struct MaterialInfo
{
    float ior;
    float perceptualRoughness;      // roughness value, as authored by the model creator (input to shader)
    vec3 f0_dielectric;

    float alphaRoughness;           // roughness mapped to a more linear change in the roughness (proposed by [2])

    float fresnel_w;

    vec3 f90;                       // reflectance color at grazing angle
    vec3 f90_dielectric;
    float metallic;

    vec3 baseColor;

    float sheenRoughnessFactor;
    vec3 sheenColorFactor;

    vec3 clearcoatF0;
    vec3 clearcoatF90;
    float clearcoatFactor;
    vec3 clearcoatNormal;
    float clearcoatRoughness;

    // KHR_materials_specular 
    float specularWeight; // product of specularFactor and specularTexture.a

    float transmissionFactor;

    float thickness;
    vec3 attenuationColor;
    float attenuationDistance;

    // KHR_materials_iridescence
    float iridescenceFactor;
    float iridescenceIor;
    float iridescenceThickness;

    // KHR_materials_retroreflection
    float retroreflectionFactor;

    float diffuseTransmissionFactor;
    vec3 diffuseTransmissionColorFactor;

    // KHR_materials_anisotropy
    vec3 anisotropicT;
    vec3 anisotropicB;
    float anisotropyStrength;

    // KHR_materials_dispersion
    float dispersion;
};


// MSFS detail map (ASOBO_material_detail_map), after the 3ds Max viewport shader
// MSFS2024Material_Standard.fx. See msfs_material.js.
#ifdef MSFS_DETAIL_MAP

// Base color texture alpha: with a detail map it is part of the detail mask, not opacity.
float getMsfsBaseTextureAlpha()
{
#if defined(MATERIAL_METALLICROUGHNESS) && defined(HAS_BASE_COLOR_MAP)
    return texture(u_BaseColorSampler, getBaseColorUV()).a;
#else
    return 1.0;
#endif
}

vec2 getMsfsDetailUV(int uvSet)
{
    return getTexcoord(uvSet) * u_MsfsDetailUVScale;
}

// 1 = base material, 0 = detail. Vertex alpha paints the blend; the mask texture and the
// threshold shape the transition (linearstep in the Max shader, a hard step at threshold 0).
float getMsfsBlend(float vertexAlpha)
{
#ifdef MSFS_BLEND_MASK
    float mask = texture(u_MsfsBlendMaskSampler, getTexcoord(u_MsfsBlendMaskUVSet)).r;
    float lo = clamp(mask - u_MsfsBlendThreshold, 0.0, 1.0);
    float hi = clamp(mask + u_MsfsBlendThreshold, 0.0, 1.0);
    if (hi - lo < 1e-5)
    {
        return vertexAlpha > lo ? 1.0 : 0.0;
    }
    return clamp((vertexAlpha - lo) / (hi - lo), 0.0, 1.0);
#else
    return 1.0;
#endif
}

vec4 applyMsfsDetailColor(vec4 albedo, float baseTextureAlpha, float vertexAlpha)
{
#ifdef MSFS_BLEND_MASK
    vec4 blendColor = u_BaseColorFactor;
#ifdef MSFS_DETAIL_COLOR_MAP
    blendColor = sRGBToLinear(texture(u_MsfsDetailColorSampler, getMsfsDetailUV(u_MsfsDetailColorUVSet)));
#endif
    return mix(blendColor, albedo, getMsfsBlend(vertexAlpha));
#elif defined(MSFS_DETAIL_COLOR_MAP)
    // Overlay: 0.5 is neutral. The Max viewport works on gamma-space colors, so the factor is
    // applied there: linear albedo times factor^2.2, clamped like saturate().
    vec4 detail = texture(u_MsfsDetailColorSampler, getMsfsDetailUV(u_MsfsDetailColorUVSet));
    float mask = detail.a * vertexAlpha * baseTextureAlpha;
    vec3 factor = mix(vec3(1.0), detail.rgb * 2.0, mask);
    albedo.rgb = min(albedo.rgb * pow(factor, vec3(GAMMA)), vec3(1.0));
    return albedo;
#else
    return albedo;
#endif
}

#ifdef MSFS_DETAIL_AFFECTS_NORMAL
// Adds the detail normal to a tangent space normal (y already in glTF convention).
vec3 applyMsfsDetailNormal(vec3 n)
{
    vec2 detail = vec2(0.0);
#ifdef MSFS_DETAIL_NORMAL_MAP
    detail = texture(u_MsfsDetailNormalSampler, getMsfsDetailUV(u_MsfsDetailNormalUVSet)).rg * 2.0 - vec2(1.0);
#ifdef NORMAL_MAP_DIRECTX
    detail.y = -detail.y;
#endif
    detail *= u_MsfsDetailNormalScale;
#endif
    float vertexAlpha = getVertexColor().a;
#ifdef MSFS_BLEND_MASK
    n.xy = mix(n.xy, detail, 1.0 - getMsfsBlend(vertexAlpha));
#else
    n.xy += detail * vertexAlpha * getMsfsBaseTextureAlpha();
#endif
    // The Max shader rebuilds z from xy after adding the detail
    n.z = sqrt(max(0.0, 1.0 - dot(n.xy, n.xy)));
    return n;
}
#endif

#endif // MSFS_DETAIL_MAP


// Get normal, tangent and bitangent vectors.
NormalInfo getNormalInfo(vec3 v)
{
    vec2 UV = getNormalUV();
    vec2 uv_dx = dFdx(UV);
    vec2 uv_dy = dFdy(UV);

    if (length(uv_dx) <= 1e-2) {
      uv_dx = vec2(1.0, 0.0);
    }

    if (length(uv_dy) <= 1e-2) {
      uv_dy = vec2(0.0, 1.0);
    }

    vec3 t_ = (uv_dy.t * dFdx(v_Position) - uv_dx.t * dFdy(v_Position)) /
        (uv_dx.s * uv_dy.t - uv_dy.s * uv_dx.t);

    vec3 n, t, b, ng;

    // Compute geometrical TBN:
#ifdef HAS_NORMAL_VEC3
#ifdef HAS_TANGENT_VEC4
    // Trivial TBN computation, present as vertex attribute.
    // Normalize eigenvectors as matrix is linearly interpolated.
    t = normalize(v_TBN[0]);
    b = normalize(v_TBN[1]);
    ng = normalize(v_TBN[2]);
#else
    // Normals are either present as vertex attributes or approximated.
    ng = normalize(v_Normal);
    t = normalize(t_ - ng * dot(ng, t_));
    b = cross(ng, t);
#endif
#else
    ng = normalize(cross(dFdx(v_Position), dFdy(v_Position)));
    t = normalize(t_ - ng * dot(ng, t_));
    b = cross(ng, t);
#endif

#ifndef NOT_TRIANGLE
    // For a back-facing surface, the tangential basis vectors are negated.
    if (gl_FrontFacing == false)
    {
        t *= -1.0;
        b *= -1.0;
        ng *= -1.0;
    }
#endif

    // Compute normals:
    NormalInfo info;
    info.ng = ng;
#if defined(HAS_NORMAL_MAP) || defined(MSFS_DETAIL_AFFECTS_NORMAL)
    info.ntex = vec3(0.0, 0.0, 1.0);
#ifdef HAS_NORMAL_MAP
    info.ntex = texture(u_NormalSampler, UV).rgb * 2.0 - vec3(1.0);
#ifdef NORMAL_MAP_DIRECTX
    info.ntex.y = -info.ntex.y;
#endif
#ifdef NORMAL_MAP_RECONSTRUCT_Z
    info.ntex.z = sqrt(max(0.0, 1.0 - dot(info.ntex.xy, info.ntex.xy)));
#endif
#endif
#ifdef MSFS_DETAIL_AFFECTS_NORMAL
    info.ntex = applyMsfsDetailNormal(info.ntex);
#endif
#ifdef HAS_NORMAL_MAP
    info.ntex *= vec3(u_NormalScale, u_NormalScale, 1.0);
#endif
    info.ntex = normalize(info.ntex);
    info.n = normalize(mat3(t, b, ng) * info.ntex);
#else
    info.n = ng;
#endif
    info.t = t;
    info.b = b;

#if DEBUG == DEBUG_TANGENT_W
    // only run this if debug channel is enabled
    info.tangentWSign = v_TangentWSign;
#endif
    return info;
}


#ifdef MATERIAL_CLEARCOAT
vec3 getClearcoatNormal(NormalInfo normalInfo)
{
#ifdef HAS_CLEARCOAT_NORMAL_MAP
    vec3 n = texture(u_ClearcoatNormalSampler, getClearcoatNormalUV()).rgb * 2.0 - vec3(1.0);
#ifdef NORMAL_MAP_DIRECTX
    n.y = -n.y;
#endif
#ifdef CLEARCOAT_NORMAL_MAP_RECONSTRUCT_Z
    n.z = sqrt(max(0.0, 1.0 - dot(n.xy, n.xy)));
#endif
    n *= vec3(u_ClearcoatNormalScale, u_ClearcoatNormalScale, 1.0);
    n = mat3(normalInfo.t, normalInfo.b, normalInfo.ng) * normalize(n);
    return n;
#else
    return normalInfo.ng;
#endif
}
#endif


vec4 getBaseColor()
{
    vec4 baseColor = u_BaseColorFactor;
    vec4 baseTexel = vec4(1.0);

#if defined(HAS_BASE_COLOR_MAP)
    baseTexel = texture(u_BaseColorSampler, getBaseColorUV());
    baseColor *= baseTexel;
#endif

    vec4 vertexColor = getVertexColor();
#ifdef MSFS_DETAIL_MAP
    // With a detail map, vertex alpha is the detail mask rather than opacity
    baseColor = applyMsfsDetailColor(baseColor, baseTexel.a, vertexColor.a);
    vertexColor.a = 1.0;
#endif
    return baseColor * vertexColor;
}


#ifdef MATERIAL_METALLICROUGHNESS
MaterialInfo getMetallicRoughnessInfo(MaterialInfo info)
{
    info.metallic = u_MetallicFactor;
    info.perceptualRoughness = u_RoughnessFactor;

#ifdef HAS_METALLIC_ROUGHNESS_MAP
    // Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
    // This layout intentionally reserves the 'r' channel for (optional) occlusion map data
    vec4 mrSample = texture(u_MetallicRoughnessSampler, getMetallicRoughnessUV());
#ifndef MSFS_UNIFORM_BASE_ROUGHNESS
    // MSFS clearcoat with "uniform base roughness" uses the map's roughness for the coat only
    info.perceptualRoughness *= mrSample.g;
#endif
    info.metallic *= mrSample.b;
#endif

    return info;
}
#endif


#ifdef MATERIAL_SHEEN
MaterialInfo getSheenInfo(MaterialInfo info)
{
    info.sheenColorFactor = u_SheenColorFactor;
    info.sheenRoughnessFactor = u_SheenRoughnessFactor;

#ifdef HAS_SHEEN_COLOR_MAP
    vec4 sheenColorSample = texture(u_SheenColorSampler, getSheenColorUV());
    info.sheenColorFactor *= sheenColorSample.rgb;
#endif

#ifdef HAS_SHEEN_ROUGHNESS_MAP
    vec4 sheenRoughnessSample = texture(u_SheenRoughnessSampler, getSheenRoughnessUV());
    info.sheenRoughnessFactor *= sheenRoughnessSample.a;
#endif
    return info;
}
#endif


#ifdef MATERIAL_SPECULAR
MaterialInfo getSpecularInfo(MaterialInfo info)
{   
    vec4 specularTexture = vec4(1.0);
#ifdef HAS_SPECULAR_MAP
    specularTexture.a = texture(u_SpecularSampler, getSpecularUV()).a;
#endif
#ifdef HAS_SPECULAR_COLOR_MAP
    specularTexture.rgb = texture(u_SpecularColorSampler, getSpecularColorUV()).rgb;
#endif

    info.f0_dielectric = min(info.f0_dielectric * u_KHR_materials_specular_specularColorFactor * specularTexture.rgb, vec3(1.0));
    info.specularWeight = u_KHR_materials_specular_specularFactor * specularTexture.a;
    info.f90_dielectric = vec3(info.specularWeight);
    return info;
}
#endif


#ifdef MATERIAL_TRANSMISSION
MaterialInfo getTransmissionInfo(MaterialInfo info)
{
    info.transmissionFactor = u_TransmissionFactor;

#ifdef HAS_TRANSMISSION_MAP
    vec4 transmissionSample = texture(u_TransmissionSampler, getTransmissionUV());
    info.transmissionFactor *= transmissionSample.r;
#endif

#ifdef MATERIAL_DISPERSION
    info.dispersion = u_Dispersion;
#else
    info.dispersion = 0.0;
#endif
    return info;
}
#endif

#ifdef MATERIAL_VOLUME
MaterialInfo getVolumeInfo(MaterialInfo info)
{
    info.thickness = u_ThicknessFactor;
    info.attenuationColor = u_AttenuationColor;
    info.attenuationDistance = u_AttenuationDistance;

#ifdef HAS_THICKNESS_MAP
    vec4 thicknessSample = texture(u_ThicknessSampler, getThicknessUV());
    info.thickness *= thicknessSample.g;
#endif
    return info;
}
#endif


#ifdef MATERIAL_IRIDESCENCE
MaterialInfo getIridescenceInfo(MaterialInfo info)
{
    info.iridescenceFactor = u_IridescenceFactor;
    info.iridescenceIor = u_IridescenceIor;
    info.iridescenceThickness = u_IridescenceThicknessMaximum;

    #ifdef HAS_IRIDESCENCE_MAP
        info.iridescenceFactor *= texture(u_IridescenceSampler, getIridescenceUV()).r;
    #endif

    #ifdef HAS_IRIDESCENCE_THICKNESS_MAP
        float thicknessSampled = texture(u_IridescenceThicknessSampler, getIridescenceThicknessUV()).g;
        float thickness = mix(u_IridescenceThicknessMinimum, u_IridescenceThicknessMaximum, thicknessSampled);
        info.iridescenceThickness = thickness;
    #endif

    return info;
}
#endif


#ifdef MATERIAL_RETROREFLECTION
MaterialInfo getRetroreflectionInfo(MaterialInfo info)
{
    info.retroreflectionFactor = u_RetroreflectionFactor;

    #ifdef HAS_RETROREFLECTION_MAP
        info.retroreflectionFactor *= texture(u_RetroreflectionSampler, getRetroreflectionUV()).r;
    #endif

    return info;
}
#endif


#ifdef MATERIAL_DIFFUSE_TRANSMISSION
MaterialInfo getDiffuseTransmissionInfo(MaterialInfo info)
{
    info.diffuseTransmissionFactor = u_DiffuseTransmissionFactor;
    info.diffuseTransmissionColorFactor = u_DiffuseTransmissionColorFactor;

    #ifdef HAS_DIFFUSE_TRANSMISSION_MAP
        info.diffuseTransmissionFactor *= texture(u_DiffuseTransmissionSampler, getDiffuseTransmissionUV()).a;
    #endif

    #ifdef HAS_DIFFUSE_TRANSMISSION_COLOR_MAP
        info.diffuseTransmissionColorFactor *= texture(u_DiffuseTransmissionColorSampler, getDiffuseTransmissionColorUV()).rgb;
    #endif

    return info;
}
#endif


#ifdef MATERIAL_CLEARCOAT
MaterialInfo getClearCoatInfo(MaterialInfo info, NormalInfo normalInfo)
{
    info.clearcoatFactor = u_ClearcoatFactor;
    info.clearcoatRoughness = u_ClearcoatRoughnessFactor;
    info.clearcoatF0 = vec3(pow((info.ior - 1.0) / (info.ior + 1.0), 2.0));
    info.clearcoatF90 = vec3(1.0);

#ifdef HAS_CLEARCOAT_MAP
    vec4 clearcoatSample = texture(u_ClearcoatSampler, getClearcoatUV());
    info.clearcoatFactor *= clearcoatSample.r;
#endif

#ifdef HAS_CLEARCOAT_ROUGHNESS_MAP
    vec4 clearcoatSampleRoughness = texture(u_ClearcoatRoughnessSampler, getClearcoatRoughnessUV());
#ifdef MSFS_CLEARCOAT_ROUGHNESS_ALPHA
    // MSFS clearcoat colour/roughness texture stores the roughness in alpha
    info.clearcoatRoughness *= clearcoatSampleRoughness.a;
#else
    info.clearcoatRoughness *= clearcoatSampleRoughness.g;
#endif
#endif

    info.clearcoatNormal = getClearcoatNormal(normalInfo);
    info.clearcoatRoughness = clamp(info.clearcoatRoughness, 0.0, 1.0);
    return info;
}
#endif


#ifdef MATERIAL_IOR
MaterialInfo getIorInfo(MaterialInfo info)
{
    info.f0_dielectric = vec3(pow(( u_Ior - 1.0) /  (u_Ior + 1.0), 2.0));
    info.ior = u_Ior;
    return info;
}
#endif

#ifdef MATERIAL_ANISOTROPY
MaterialInfo getAnisotropyInfo(MaterialInfo info, NormalInfo normalInfo)
{
    vec2 direction = vec2(1.0, 0.0);
    float strengthFactor = 1.0;
#ifdef HAS_ANISOTROPY_MAP
    vec3 anisotropySample = texture(u_AnisotropySampler, getAnisotropyUV()).xyz;
    direction = anisotropySample.xy * 2.0 - vec2(1.0);
    strengthFactor = anisotropySample.z;
#endif
    vec2 directionRotation = u_Anisotropy.xy; // cos(theta), sin(theta)
    mat2 rotationMatrix = mat2(directionRotation.x, directionRotation.y, -directionRotation.y, directionRotation.x);
    direction = rotationMatrix * direction.xy;

    info.anisotropicT = mat3(normalInfo.t, normalInfo.b, normalInfo.n) * normalize(vec3(direction, 0.0));
    info.anisotropicB = cross(normalInfo.ng, info.anisotropicT);
    info.anisotropyStrength = clamp(u_Anisotropy.z * strengthFactor, 0.0, 1.0);
    return info;
}
#endif


float albedoSheenScalingLUT(float NdotV, float sheenRoughnessFactor)
{
    return texture(u_SheenELUT, vec2(NdotV, sheenRoughnessFactor)).r;
}
