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

export {
    isMsfsInvisibleMaterial,
    isMsfsHiddenPrimitive,
    getMsfsDrawOrder,
    getMsfsDepthBias,
    sortByMsfsDrawOrder
};
