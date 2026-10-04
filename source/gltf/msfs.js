// Helpers for Microsoft Flight Simulator (ASOBO_*) glTF extensions.

/**
 * Returns true for materials flagged with ASOBO_material_invisible. MSFS uses these for
 * collision shells, LOD bounding boxes and other helper geometry that is never drawn in the sim.
 * @param {object} material - A gltfMaterial.
 * @returns {boolean}
 */
function isMsfsInvisibleMaterial(material) {
    const invisible = material?.extensions?.ASOBO_material_invisible;
    return invisible !== undefined && invisible.enabled !== false;
}

/**
 * Returns true if the primitive should not be rendered with the given rendering parameters.
 * @param {object} gltf - The glTF.
 * @param {object} primitive - A gltfPrimitive.
 * @param {object} renderingParameters - GltfState.renderingParameters.
 * @returns {boolean}
 */
function isMsfsHiddenPrimitive(gltf, primitive, renderingParameters) {
    return (
        !renderingParameters.showMsfsInvisibleMaterials &&
        isMsfsInvisibleMaterial(gltf.materials[primitive.material])
    );
}

/**
 * Returns the ASOBO_material_draw_order offset of a material (0 if not set).
 * Within the transparent pass, MSFS draws lower offsets first.
 * @param {object} material - A gltfMaterial.
 * @returns {number}
 */
function getMsfsDrawOrder(material) {
    return material?.extensions?.ASOBO_material_draw_order?.drawOrderOffset ?? 0;
}

/**
 * Returns the polygon offset units for a material, or 0 if none should be applied.
 * Geometry decals (ASOBO_material_geometry_decal) are modelled as separate meshes lying
 * on top of the surface they decorate; without a depth bias they z-fight with it.
 * A higher draw order pulls the decal further towards the camera, so stacked decals resolve
 * in the same order as in the sim.
 * @param {object} material - A gltfMaterial.
 * @returns {number}
 */
function getMsfsDepthBias(material) {
    if (material?.extensions?.ASOBO_material_geometry_decal === undefined) {
        return 0;
    }
    return -(1 + Math.max(0, getMsfsDrawOrder(material)));
}

/**
 * Stable sort of depth-sorted transparent drawables by MSFS draw order.
 * Depth order is kept within the same draw order.
 * @param {object} gltf - The glTF.
 * @param {Array} drawables - Drawables sorted back to front.
 * @returns {Array}
 */
function sortByMsfsDrawOrder(gltf, drawables) {
    return drawables.sort(
        (a, b) =>
            getMsfsDrawOrder(gltf.materials[a.primitive.material]) -
            getMsfsDrawOrder(gltf.materials[b.primitive.material])
    );
}

function withTiling(textureInfo, tiling) {
    if (tiling === undefined || tiling === 1) {
        return textureInfo;
    }
    return {
        ...textureInfo,
        extensions: {
            ...textureInfo.extensions,
            KHR_texture_transform: { scale: [tiling, tiling] }
        }
    };
}

/**
 * Translates MSFS material extensions into the KHR equivalents the renderer already supports.
 * Works on the material JSON before it is parsed and returns the shader defines needed for the
 * parts that have no KHR equivalent.
 * - ASOBO_material_clear_coat_v2 -> KHR_materials_clearcoat. With "uniform base roughness"
 *   (clearcoatInverseRoughness) the coat takes its roughness from the base comp texture and the
 *   layer below gets clearcoatBaseRoughness. Otherwise the coat roughness is the alpha channel of
 *   clearcoatColorRoughnessTexture. The coat colour (RGB) is not supported.
 * - ASOBO_material_emissive -> KHR_materials_emissive_strength, so the day/night multipliers can
 *   be applied as emissive strength (see getMsfsEmissiveMultiplier).
 * - ASOBO_material_alphamode_dither -> BLEND, approximating dithered transparency.
 * @param {object} json - The material JSON.
 * @returns {{json: object, defines: string[]}}
 */
function translateMsfsMaterialJson(json) {
    const ext = json.extensions;
    if (ext === undefined) {
        return { json, defines: [] };
    }
    const out = { ...json, extensions: { ...ext } };
    const defines = [];

    const clearcoat = ext.ASOBO_material_clear_coat_v2;
    if (clearcoat !== undefined && ext.KHR_materials_clearcoat === undefined) {
        const pbr = { ...json.pbrMetallicRoughness };
        out.pbrMetallicRoughness = pbr;
        const khr = {
            clearcoatFactor: 1,
            clearcoatRoughnessFactor: clearcoat.clearcoatRoughnessFactor ?? 1
        };
        if (clearcoat.clearcoatInverseRoughness) {
            if (pbr.metallicRoughnessTexture !== undefined) {
                khr.clearcoatRoughnessTexture = { ...pbr.metallicRoughnessTexture };
            } else {
                khr.clearcoatRoughnessFactor *= pbr.roughnessFactor ?? 1;
            }
            pbr.roughnessFactor = clearcoat.clearcoatBaseRoughness ?? 0.5;
            defines.push("MSFS_UNIFORM_BASE_ROUGHNESS 1");
        } else if (clearcoat.clearcoatColorRoughnessTexture !== undefined) {
            khr.clearcoatRoughnessTexture = withTiling(
                clearcoat.clearcoatColorRoughnessTexture,
                clearcoat.clearcoatColorRoughnessTiling
            );
            defines.push("MSFS_CLEARCOAT_ROUGHNESS_ALPHA 1");
        }
        if (clearcoat.clearcoatNormalTexture !== undefined) {
            khr.clearcoatNormalTexture = withTiling(
                {
                    ...clearcoat.clearcoatNormalTexture,
                    scale:
                        (clearcoat.clearcoatNormalTexture.scale ?? 1) *
                        (clearcoat.clearcoatNormalFactor ?? 1)
                },
                clearcoat.clearcoatNormalTiling
            );
        } else if (clearcoat.clearcoatBaseAffectCoat !== false && json.normalTexture) {
            khr.clearcoatNormalTexture = { ...json.normalTexture };
        }
        out.extensions.KHR_materials_clearcoat = khr;
    }

    const emissive = ext.ASOBO_material_emissive;
    if (
        emissive !== undefined &&
        ext.KHR_materials_emissive_strength === undefined &&
        ((emissive.emissiveDayMultiplier ?? 1) !== 1 ||
            (emissive.emissiveNightMultiplier ?? 1) !== 1)
    ) {
        out.extensions.KHR_materials_emissive_strength = { emissiveStrength: 1 };
    }

    if (
        ext.ASOBO_material_alphamode_dither !== undefined &&
        (json.alphaMode ?? "OPAQUE") === "OPAQUE"
    ) {
        out.alphaMode = "BLEND";
    }

    return { json: out, defines };
}

/**
 * Returns the ASOBO_material_emissive multiplier for the current time of day.
 * @param {object} material - A gltfMaterial.
 * @param {object} renderingParameters - GltfState.renderingParameters.
 * @returns {number}
 */
function getMsfsEmissiveMultiplier(material, renderingParameters) {
    const emissive = material?.extensions?.ASOBO_material_emissive;
    if (emissive === undefined) {
        return 1;
    }
    return renderingParameters.msfsNightLighting
        ? (emissive.emissiveNightMultiplier ?? 1)
        : (emissive.emissiveDayMultiplier ?? 1);
}

// Property animation targets that map onto animatable renderer properties.
const supportedPropertyAnimationTargets = [
    /^materials\/\d+\/emissiveFactor$/,
    /^materials\/\d+\/pbrMetallicRoughness\/(baseColorFactor|metallicFactor|roughnessFactor)$/,
    /^cameras\/\d+\/perspective\/yfov$/,
    /^materials\/\d+\/extensions\/ASOBO_material_UV_options\/(UVOffsetU|UVOffsetV|UVTilingU|UVTilingV|UVRotation)$/
];

/**
 * Translates ASOBO_property_animation channels (material and camera property animations) into
 * KHR_animation_pointer channels. Targets of MSFS-only properties the renderer has no
 * equivalent for (e.g. wiper or dirt states) are skipped.
 * @param {object} json - The animation JSON.
 * @returns {object} The animation JSON with the translated channels appended.
 */
function translateMsfsAnimationJson(json) {
    const propertyChannels = json.extensions?.ASOBO_property_animation?.channels;
    if (propertyChannels === undefined) {
        return json;
    }
    const channels = [...(json.channels ?? [])];
    const skipped = [];
    for (const channel of propertyChannels) {
        const target = channel.target?.replace(/^\//, "");
        if (!supportedPropertyAnimationTargets.some((pattern) => pattern.test(target))) {
            skipped.push(target);
            continue;
        }
        channels.push({
            sampler: channel.sampler,
            target: {
                path: "pointer",
                extensions: { KHR_animation_pointer: { pointer: "/" + target } }
            }
        });
    }
    if (skipped.length > 0) {
        console.info(
            `Animation "${json.name}": unsupported MSFS property animation targets: ${skipped.join(", ")}`
        );
    }
    return { ...json, channels };
}

/**
 * Returns true if the glTF was produced by an MSFS exporter (uses ASOBO extensions).
 * @param {object} gltf - The glTF.
 * @returns {boolean}
 */
function isMsfsAsset(gltf) {
    return (
        gltf?.extensionsUsed?.some((name) => name.startsWith("ASOBO_")) === true ||
        gltf?.asset?.extensions?.ASOBO_normal_map_convention !== undefined
    );
}

/**
 * MSFS animations are authored in frames, but glTF stores seconds without the frame rate.
 * Detects the frame rate at which all keyframe times fall on whole frames (30 if none fits).
 * @param {object} gltf - The glTF, with buffers loaded.
 * @returns {number}
 */
function detectMsfsFrameRate(gltf) {
    const times = [];
    for (const animation of gltf.animations ?? []) {
        for (const sampler of animation.samplers ?? []) {
            const input = gltf.accessors[sampler.input]?.getNormalizedDeinterlacedView(gltf);
            if (input !== undefined) {
                times.push(...input);
            }
        }
    }
    for (const fps of [30, 24, 25, 60, 50, 120]) {
        if (times.every((time) => Math.abs(time * fps - Math.round(time * fps)) < 0.01)) {
            return fps;
        }
    }
    return 30;
}

export {
    translateMsfsAnimationJson,
    isMsfsAsset,
    detectMsfsFrameRate,
    translateMsfsMaterialJson,
    getMsfsEmissiveMultiplier,
    isMsfsInvisibleMaterial,
    isMsfsHiddenPrimitive,
    getMsfsDrawOrder,
    getMsfsDepthBias,
    sortByMsfsDrawOrder
};
