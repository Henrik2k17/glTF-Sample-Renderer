import { mat3 } from "gl-matrix";
import { GltfObject } from "./gltf_object.js";
import { gltfTextureInfo } from "./texture.js";

// MSFS (ASOBO_*) material features that need their own shader paths. The maths follows Asobo's
// 3ds Max viewport shaders (MSFS 2024 SDK, MSFS2024_Material/Fx: core.fx and
// MSFS2024Material_Standard.fx), i.e. what artists see in 3ds Max, not the sim renderer itself.
// Field names and defaults follow the exporter (FlightSimMaterialExporter.cs).

/**
 * ASOBO_material_UV_options, animatable through ASOBO_property_animation. The exporter omits
 * floats that are 0, so a missing tiling really is a tiling of 0 (3ds Max defaults to 1).
 */
class ASOBO_material_UV_options extends GltfObject {
    static animatedProperties = ["UVOffsetU", "UVOffsetV", "UVTilingU", "UVTilingV", "UVRotation"];
    constructor() {
        super();
        this.clampUVX = false;
        this.clampUVY = false;
        this.clampUVZ = false;
        this.UVOffsetU = 0;
        this.UVOffsetV = 0;
        this.UVTilingU = 0;
        this.UVTilingV = 0;
        this.UVRotation = 0;
    }

    // Plain values for display (e.g. an inspector), without the animation bookkeeping.
    toJSON() {
        const json = {};
        for (const key of ["clampUVX", "clampUVY", "clampUVZ", ...ASOBO_material_UV_options.animatedProperties]) {
            json[key] = this[key];
        }
        return json;
    }
}

/**
 * The UV0 transform of core.fx transformUV, in glTF UV space (V down; 3ds Max adds 1 to its
 * V-up coordinates for the same). Rotation is in degrees around the texture centre and is
 * applied before tiling; offset U is subtracted and offset V added.
 * @returns {mat3} Column-major matrix for vec3(uv, 1).
 */
function getMsfsUV0Transform(options) {
    const angle = (options.UVRotation * Math.PI) / 180;
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const tilingU = options.UVTilingU;
    const tilingV = options.UVTilingV;
    // uv' = A * (uv - 0.5) + 0.5 + offset, with A = tiling * rotation (row-vector rotation in HLSL)
    const a00 = tilingU * c;
    const a01 = tilingU * s;
    const a10 = -tilingV * s;
    const a11 = tilingV * c;
    const tx = 0.5 - (a00 + a01) * 0.5 - options.UVOffsetU;
    const ty = 0.5 - (a10 + a11) * 0.5 + options.UVOffsetV;
    // prettier-ignore
    return mat3.fromValues(
        a00, a10, 0,
        a01, a11, 0,
        tx, ty, 1
    );
}

function isLoaded(gltf, textureInfo) {
    const texture = gltf.textures[textureInfo?.index];
    return gltf.images[texture?.source]?.isLoaded() === true;
}

function addTexture(material, gltf, json, samplerName, linear, define) {
    if (json === undefined || !isLoaded(gltf, json)) {
        return undefined;
    }
    const textureInfo = new gltfTextureInfo(undefined, 0, linear);
    textureInfo.fromJson(json);
    textureInfo.samplerName = samplerName;
    material.textures.push(textureInfo);
    material.defines.push(define);
    return textureInfo;
}

/**
 * Registers the textures and defines of the MSFS material features. Call from
 * gltfMaterial.initGl. Sets material.msfs (uniform values) and material.msfsVertDefines.
 */
function initMsfsMaterial(material, gltf) {
    const ext = material.extensions ?? {};
    const msfs = {};
    material.msfs = msfs;
    material.msfsVertDefines = [];

    const uvOptions = ext.ASOBO_material_UV_options;
    if (uvOptions instanceof ASOBO_material_UV_options) {
        msfs.uvOptions = uvOptions;
        material.msfsVertDefines.push("MSFS_UV0_TRANSFORM 1");
        // Not part of the 3ds Max viewport shader: clamp-to-edge addressing of the UV0
        // textures, as the names describe (see getMsfsClampedTextures).
        msfs.clampU = uvOptions.clampUVX === true;
        msfs.clampV = uvOptions.clampUVY === true;
    }

    // Detail map. Any detail texture turns vertex alpha (times base color alpha) into the
    // detail mask instead of opacity. With a blend mask the detail replaces the base instead
    // of modulating it, painted in by vertex alpha.
    const detail = ext.ASOBO_material_detail_map;
    if (detail !== undefined) {
        const hasDetail =
            detail.detailColorTexture !== undefined ||
            detail.detailNormalTexture !== undefined ||
            detail.detailMetalRoughAOTexture !== undefined;
        if (hasDetail) {
            material.defines.push("MSFS_DETAIL_MAP 1");
        }
        const blendMaskMissing =
            detail.blendMaskTexture !== undefined && !isLoaded(gltf, detail.blendMaskTexture);
        // Without its mask a blend detail would wrongly show as an overlay, so leave it out.
        if (!blendMaskMissing) {
            msfs.blendMask = addTexture(material, gltf, detail.blendMaskTexture, "u_MsfsBlendMaskSampler", true, "MSFS_BLEND_MASK 1");
            // Sampled raw: as an overlay 0.5 is neutral; as a blend color it is decoded in the shader.
            msfs.detailColor = addTexture(material, gltf, detail.detailColorTexture, "u_MsfsDetailColorSampler", true, "MSFS_DETAIL_COLOR_MAP 1");
            msfs.detailNormal = addTexture(material, gltf, detail.detailNormalTexture, "u_MsfsDetailNormalSampler", true, "MSFS_DETAIL_NORMAL_MAP 1");
        }
        // A blend mask also flattens the base normal map where the detail shows.
        if (msfs.detailNormal !== undefined || (msfs.blendMask !== undefined && material.normalTexture !== undefined)) {
            material.defines.push("MSFS_DETAIL_AFFECTS_NORMAL 1");
        }
        msfs.detailUVScale = detail.UVScale ?? 1;
        msfs.blendThreshold = detail.blendThreshold ?? 0;
        // detailMetalRoughAOTexture is not used by the 3ds Max viewport shader, so not here either.
    }

    msfs.extraOcclusion = addTexture(
        material,
        gltf,
        ext.ASOBO_extra_occlusion?.extraOcclusionTexture,
        "u_MsfsExtraOcclusionSampler",
        true,
        "MSFS_EXTRA_OCCLUSION_MAP 1"
    );

    // 0..1 fades the occlusion in, 1..2 darkens towards black.
    const occlusionStrength = ext.ASOBO_occlusion_strength?.strength;
    if (occlusionStrength !== undefined) {
        msfs.occlusionStrength = occlusionStrength;
        material.defines.push("MSFS_OCCLUSION_STRENGTH 1");
    }
}

/**
 * For ASOBO_material_UV_options clamping: the material's UV0 textures that should be sampled
 * with clamp-to-edge, and on which axes. Detail map textures keep repeating, they tile by design.
 * @returns {{clampU: boolean, clampV: boolean, textures: Set<object>} | undefined}
 */
function getMsfsClampedTextures(material) {
    const msfs = material.msfs;
    if (msfs === undefined || !(msfs.clampU || msfs.clampV)) {
        return undefined;
    }
    const detailTextures = [msfs.detailColor, msfs.detailNormal, msfs.blendMask];
    const textures = new Set(
        material.textures.filter((info) => (info.texCoord ?? 0) === 0 && !detailTextures.includes(info))
    );
    return { clampU: msfs.clampU, clampV: msfs.clampV, textures };
}

/** Uploads the uniforms of initMsfsMaterial's features. Call after the core material uniforms. */
function updateMsfsMaterialUniforms(shader, material) {
    const msfs = material.msfs;
    if (msfs === undefined) {
        return;
    }
    if (msfs.uvOptions !== undefined) {
        shader.updateUniform("u_MsfsUV0Transform", getMsfsUV0Transform(msfs.uvOptions), false);
    }
    if (msfs.detailUVScale !== undefined) {
        shader.updateUniform("u_MsfsDetailUVScale", msfs.detailUVScale, false);
        shader.updateUniform("u_MsfsBlendThreshold", msfs.blendThreshold, false);
    }
    shader.updateUniform("u_MsfsDetailColorUVSet", msfs.detailColor?.texCoord, false);
    shader.updateUniform("u_MsfsDetailNormalUVSet", msfs.detailNormal?.texCoord, false);
    shader.updateUniform("u_MsfsDetailNormalScale", msfs.detailNormal?.scale, false);
    shader.updateUniform("u_MsfsBlendMaskUVSet", msfs.blendMask?.texCoord, false);
    shader.updateUniform("u_MsfsExtraOcclusionUVSet", msfs.extraOcclusion?.texCoord, false);
    if (msfs.occlusionStrength !== undefined) {
        shader.updateUniform("u_OcclusionStrength", msfs.occlusionStrength, false);
    }
}

export {
    ASOBO_material_UV_options,
    getMsfsUV0Transform,
    getMsfsClampedTextures,
    initMsfsMaterial,
    updateMsfsMaterialUniforms
};
