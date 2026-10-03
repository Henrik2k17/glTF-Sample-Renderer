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

export { isMsfsInvisibleMaterial, isMsfsHiddenPrimitive };
