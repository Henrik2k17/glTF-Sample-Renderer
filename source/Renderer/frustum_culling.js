import { gltfAccessor } from "../gltf/accessor.js";

// Frustum culling: a drawable whose world-space bounding box lies completely outside the view
// frustum is not drawn. Large MSFS packages have thousands of primitives, most of them off
// screen in close-up views.

/**
 * The six clip planes (left, right, bottom, top, near, far) of a view-projection matrix, as
 * (a, b, c, d) with a*x + b*y + c*z + d >= 0 inside. Not normalized; the box test doesn't need it.
 * @param {mat4} m - Column-major view-projection matrix (WebGL clip space, z in -w..w).
 * @param {Float32Array} out - 24 floats.
 */
function getFrustumPlanes(m, out) {
    for (let axis = 0; axis < 3; axis++) {
        for (let side = 0; side < 2; side++) {
            const sign = side === 0 ? 1 : -1;
            const o = (axis * 2 + side) * 4;
            // row 3 +/- row axis
            out[o] = m[3] + sign * m[axis];
            out[o + 1] = m[7] + sign * m[4 + axis];
            out[o + 2] = m[11] + sign * m[8 + axis];
            out[o + 3] = m[15] + sign * m[12 + axis];
        }
    }
    return out;
}

/**
 * Local bounding box of a primitive from its POSITION accessor's min/max, as
 * { center, extent } (half sizes), or null if the accessor has no bounds.
 */
function getPrimitiveBounds(gltf, primitive) {
    if (primitive.cullingBounds !== undefined) {
        return primitive.cullingBounds;
    }
    let bounds = null;
    const attribute = primitive.glAttributes?.find((a) => a.attribute === "POSITION");
    const accessor = gltf.accessors[attribute?.accessor];
    let min = accessor?.min;
    let max = accessor?.max;
    if (min?.length >= 3 && max?.length >= 3 && min.every(Number.isFinite) && max.every(Number.isFinite)) {
        if (accessor.normalized) {
            min = gltfAccessor.dequantize(min, accessor.componentType);
            max = gltfAccessor.dequantize(max, accessor.componentType);
        }
        bounds = {
            center: [0, 1, 2].map((i) => (min[i] + max[i]) / 2),
            extent: [0, 1, 2].map((i) => (max[i] - min[i]) / 2)
        };
    }
    primitive.cullingBounds = bounds;
    return bounds;
}

/**
 * Whether a drawable may be visible. Skinned, morphed and GPU-instanced primitives move
 * outside their accessor bounds, so they always count as visible.
 */
function isDrawableInFrustum(gltf, drawable, planes) {
    const { node, primitive } = drawable;
    if (node.skin !== undefined || primitive.targets?.length > 0 || node.instanceMatrices !== undefined) {
        return true;
    }
    const bounds = getPrimitiveBounds(gltf, primitive);
    if (bounds === null) {
        return true;
    }
    // World-space box around the transformed local box (Arvo): centre transformed, half sizes
    // summed over the absolute matrix entries.
    const m = node.getRenderedWorldTransform();
    const [lx, ly, lz] = bounds.center;
    const [ex, ey, ez] = bounds.extent;
    const cx = m[0] * lx + m[4] * ly + m[8] * lz + m[12];
    const cy = m[1] * lx + m[5] * ly + m[9] * lz + m[13];
    const cz = m[2] * lx + m[6] * ly + m[10] * lz + m[14];
    const wx = Math.abs(m[0]) * ex + Math.abs(m[4]) * ey + Math.abs(m[8]) * ez;
    const wy = Math.abs(m[1]) * ex + Math.abs(m[5]) * ey + Math.abs(m[9]) * ez;
    const wz = Math.abs(m[2]) * ex + Math.abs(m[6]) * ey + Math.abs(m[10]) * ez;
    for (let o = 0; o < 24; o += 4) {
        const a = planes[o];
        const b = planes[o + 1];
        const c = planes[o + 2];
        const distance = a * cx + b * cy + c * cz + planes[o + 3];
        const radius = Math.abs(a) * wx + Math.abs(b) * wy + Math.abs(c) * wz;
        if (distance < -radius) {
            return false;
        }
    }
    return true;
}

export { getFrustumPlanes, isDrawableInFrustum };
