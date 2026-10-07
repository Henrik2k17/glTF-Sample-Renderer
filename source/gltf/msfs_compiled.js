import { mat3, mat4, vec3 } from "gl-matrix";
import { pushAccessor } from "./tangent_workers.js";
// WebGL constants as numbers: the lookup tables below are built when the module loads, which can
// be before webgl.js (GL) is initialized because of circular imports.
const BYTE = 5120;
const UNSIGNED_BYTE = 5121;
const SHORT = 5122;
const UNSIGNED_SHORT = 5123;
const FLOAT = 5126;
const UNSIGNED_INT = 5125;
const ARRAY_BUFFER = 34962;
const ELEMENT_ARRAY_BUFFER = 34963;
const TRIANGLES = 4;

// glTFs compiled by the MSFS 2024 package builder (asset extension ASOBO_asset_optimized)
// store their geometry for the sim's renderer, not as plain glTF:
// - All primitives of a mesh share one vertex and one index accessor covering the whole mesh.
//   Each primitive is a draw range given in extras.ASOBO_primitive: indices
//   [StartIndex, StartIndex + PrimitiveCount * 3), each offset by BaseVertexIndex.
// - Vertices are interleaved (VertexType "VTX"): normals and tangents as signed bytes that are
//   not flagged as normalized, texture coordinates and colors as 16 bit floats that are
//   declared as (UNSIGNED_)SHORT. Skinned vertices ("BLEND4", "BLEND1") store colors as
//   unsigned bytes declared as BYTE; BLEND1 has a single float weight.
// - Triangles wind clockwise (Direct3D front faces), glTF expects counter-clockwise.
// decodeMsfsCompiledGeometry turns this into standard accessors, one set per primitive.

const HalfToFloat = (() => {
    const table = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
        const sign = h & 0x8000 ? -1 : 1;
        const exponent = (h >> 10) & 0x1f;
        const mantissa = h & 0x3ff;
        if (exponent === 0) {
            table[h] = sign * mantissa * 2 ** -24;
        } else if (exponent === 31) {
            table[h] = mantissa === 0 ? sign * Infinity : NaN;
        } else {
            table[h] = sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
        }
    }
    return table;
})();

function isMsfsCompiled(gltf) {
    return (
        gltf.asset?.extensions?.ASOBO_asset_optimized !== undefined ||
        gltf.meshes?.some((mesh) => mesh.extras?.msfsCompiled === true) === true
    );
}

/**
 * glTFs merged from several files (MSFS packages) can mix compiled and source exports; their
 * compiled meshes are marked with extras.msfsCompiled.
 */
function isMsfsCompiledMesh(gltf, mesh) {
    return (
        gltf.asset?.extensions?.ASOBO_asset_optimized !== undefined ||
        mesh.extras?.msfsCompiled === true
    );
}

// Attributes stored as 16 bit floats behind a SHORT or UNSIGNED_SHORT component type
function isHalfFloatAttribute(name, accessor) {
    return (
        /^(TEXCOORD|COLOR|WEIGHTS)_\d+$/.test(name) &&
        !accessor.normalized &&
        (accessor.componentType === SHORT || accessor.componentType === UNSIGNED_SHORT)
    );
}

/**
 * Copies `count` elements starting at element `first` of an (interleaved) accessor into a
 * tightly packed Float32Array, converting each component with `convert(view, byteOffset)`.
 */
function readElements(gltf, accessor, first, count, components, componentBytes, convert) {
    const bufferView = gltf.bufferViews[accessor.bufferView];
    const buffer = gltf.buffers[bufferView.buffer].buffer;
    const stride = bufferView.byteStride || components * componentBytes;
    const start = (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0) + first * stride;
    const view = new DataView(
        buffer,
        start,
        Math.max(0, (count - 1) * stride + components * componentBytes)
    );
    const out = new Float32Array(count * components);
    for (let i = 0; i < count; i++) {
        for (let c = 0; c < components; c++) {
            out[i * components + c] = convert(view, i * stride + c * componentBytes);
        }
    }
    return out;
}

const ComponentReaders = {
    [FLOAT]: [4, (view, offset) => view.getFloat32(offset, true)],
    [BYTE]: [1, (view, offset) => Math.max(view.getInt8(offset) / 127, -1)],
    [UNSIGNED_BYTE]: [1, (view, offset) => view.getUint8(offset) / 255],
    [SHORT]: [2, (view, offset) => Math.max(view.getInt16(offset, true) / 32767, -1)],
    [UNSIGNED_SHORT]: [2, (view, offset) => view.getUint16(offset, true) / 65535]
};
const UnsignedByteReader = [1, (view, offset) => view.getUint8(offset) / 255];
const HalfReader = [2, (view, offset) => HalfToFloat[view.getUint16(offset, true)]];
const IntegerReaders = {
    [BYTE]: [1, (view, offset) => view.getInt8(offset)],
    [UNSIGNED_BYTE]: [1, (view, offset) => view.getUint8(offset)],
    [SHORT]: [2, (view, offset) => view.getInt16(offset, true)],
    [UNSIGNED_SHORT]: [2, (view, offset) => view.getUint16(offset, true)]
};

/** One attribute of one draw range as a standard, tightly packed accessor. */
function decodeAttribute(gltf, name, accessor, first, count) {
    let components = accessor.getComponentCount(accessor.type);
    let type = accessor.type;
    if (name === "NORMAL" && components === 4) {
        components = 3; // the fourth byte is padding
        type = "VEC3";
    }

    if (/^JOINTS_\d+$/.test(name)) {
        // Joint indices stay integers
        const [bytes, read] = IntegerReaders[accessor.componentType];
        const values = readElements(gltf, accessor, first, count, components, bytes, read);
        const data =
            accessor.componentType === UNSIGNED_BYTE
                ? Uint8Array.from(values)
                : Uint16Array.from(values);
        const componentType = data instanceof Uint8Array ? UNSIGNED_BYTE : UNSIGNED_SHORT;
        return pushAccessor(gltf, data, type, componentType, ARRAY_BUFFER);
    }

    let reader = ComponentReaders[accessor.componentType];
    if (isHalfFloatAttribute(name, accessor)) {
        reader = HalfReader;
    } else if (/^COLOR_\d+$/.test(name) && accessor.componentType === BYTE) {
        // Skinned vertices (BLEND1/BLEND4) store colors as unsigned bytes behind BYTE
        reader = UnsignedByteReader;
    }
    // Tangent and normal bytes are normalized in the sim regardless of the accessor flag.
    let data = readElements(gltf, accessor, first, count, components, ...reader);
    if (/^WEIGHTS_\d+$/.test(name) && components === 1) {
        // BLEND1: a single weight for the first joint; glTF needs four
        data = Float32Array.from({ length: count * 4 }, (_, i) => (i % 4 === 0 ? data[i / 4] : 0));
        type = "VEC4";
    }
    const index = pushAccessor(gltf, data, type, FLOAT, ARRAY_BUFFER);
    if (name === "POSITION") {
        const result = gltf.accessors[index];
        result.min = [Infinity, Infinity, Infinity];
        result.max = [-Infinity, -Infinity, -Infinity];
        for (let i = 0; i < data.length; i += 3) {
            for (let c = 0; c < 3; c++) {
                result.min[c] = Math.min(result.min[c], data[i + c]);
                result.max[c] = Math.max(result.max[c], data[i + c]);
            }
        }
    }
    return index;
}

/**
 * Copies part of an index accessor, swapping two corners per triangle (see above).
 * Draw ranges of 65536 vertices use index 0xFFFF, which WebGL 2 always treats as primitive
 * restart for 16 bit indices; such ranges are widened to 32 bit indices.
 */
function copyTriangles(gltf, accessor, firstIndex, count) {
    const bufferView = gltf.bufferViews[accessor.bufferView];
    const buffer = gltf.buffers[bufferView.buffer].buffer;
    const Type = accessor.componentType === UNSIGNED_INT ? Uint32Array : Uint16Array;
    const start =
        (bufferView.byteOffset ?? 0) +
        (accessor.byteOffset ?? 0) +
        firstIndex * Type.BYTES_PER_ELEMENT;
    // Index data is not necessarily aligned within the buffer, so copy it bytewise first.
    const source = new Type(buffer.slice(start, start + count * Type.BYTES_PER_ELEMENT));
    const restart = Type === Uint16Array && source.includes(0xffff);
    const OutType = restart ? Uint32Array : Type;
    const data = new OutType(count - (count % 3));
    for (let i = 0; i < data.length; i += 3) {
        data[i] = source[i];
        data[i + 1] = source[i + 2];
        data[i + 2] = source[i + 1];
    }
    const componentType = restart ? UNSIGNED_INT : accessor.componentType;
    return pushAccessor(gltf, data, "SCALAR", componentType, ELEMENT_ARRAY_BUFFER);
}

/** Rest pose world transforms of all nodes, from their TRS. */
function restWorldTransforms(gltf) {
    const parents = new Map();
    gltf.nodes.forEach((node, index) =>
        node.children.forEach((child) => parents.set(child, index))
    );
    const worlds = new Map();
    const world = (index) => {
        if (!worlds.has(index)) {
            const node = gltf.nodes[index];
            const local = mat4.fromRotationTranslationScale(
                mat4.create(),
                node.rotation,
                node.translation,
                node.scale
            );
            const parent = parents.get(index);
            worlds.set(
                index,
                parent === undefined ? local : mat4.multiply(mat4.create(), world(parent), local)
            );
        }
        return worlds.get(index);
    };
    return { world, parents };
}

/**
 * The Float32Array of a tightly packed accessor created by decodeAttribute, to change in place.
 * (getDeinterlacedView returns a copy, which would not reach the GPU.)
 */
function floatData(gltf, accessor) {
    accessor.filteredView = undefined;
    accessor.normalizedFilteredView = undefined;
    const buffer = gltf.buffers[gltf.bufferViews[accessor.bufferView].buffer].buffer;
    return new Float32Array(buffer);
}

/**
 * Skinned vertices are compiled in the space of the skeleton root's parent node (rest pose),
 * i.e. with inverse(world(parent of skin.skeleton)) * world(mesh node) applied, while glTF
 * skinning (and the unchanged inverse bind matrices) expects the mesh's own space. Moves
 * positions, normals and tangents back with the inverse of that transform.
 */
function toMeshSpace(gltf, primitive, inverseWorld, done) {
    const normalMatrix = mat3.normalFromMat4(mat3.create(), inverseWorld);
    const linear = mat3.fromMat4(mat3.create(), inverseWorld);
    const transform = (name, components, apply) => {
        const accessor = gltf.accessors[primitive.attributes[name]];
        // primitives with the same draw range share decoded accessors; move them once
        if (accessor === undefined || done.has(accessor)) {
            return;
        }
        done.add(accessor);
        const data = floatData(gltf, accessor);
        const v = vec3.create();
        for (let i = 0; i < data.length; i += components) {
            vec3.set(v, data[i], data[i + 1], data[i + 2]);
            apply(v);
            data.set(v, i);
        }
    };
    transform("POSITION", 3, (v) => vec3.transformMat4(v, v, inverseWorld));
    transform("NORMAL", 3, (v) => vec3.normalize(v, vec3.transformMat3(v, v, normalMatrix)));
    transform("TANGENT", 4, (v) => vec3.normalize(v, vec3.transformMat3(v, v, linear)));
    const position = gltf.accessors[primitive.attributes.POSITION];
    const data = floatData(gltf, position);
    position.min = [Infinity, Infinity, Infinity];
    position.max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < data.length; i += 3) {
        for (let c = 0; c < 3; c++) {
            position.min[c] = Math.min(position.min[c], data[i + c]);
            position.max[c] = Math.max(position.max[c], data[i + c]);
        }
    }
}

/**
 * Converts the geometry of an MSFS 2024 compiled glTF into standard accessors (see above).
 * Does nothing for other glTFs. Call after the buffers are loaded and before initGl.
 * @returns {boolean} true if the glTF was a compiled MSFS asset
 */
function decodeMsfsCompiledGeometry(gltf) {
    if (!isMsfsCompiled(gltf)) {
        return false;
    }
    // Primitives drawing the same range share the decoded accessors.
    const decoded = new Map();
    const movedToMeshSpace = new Set();
    const decode = (name, accessorIndex, first, count) => {
        const key = `${name}:${accessorIndex}:${first}:${count}`;
        if (!decoded.has(key)) {
            decoded.set(
                key,
                decodeAttribute(gltf, name, gltf.accessors[accessorIndex], first, count)
            );
        }
        return decoded.get(key);
    };

    // Skinned meshes and the node that uses them (see toMeshSpace)
    const { world, parents } = restWorldTransforms(gltf);
    const skinnedMeshNodes = new Map();
    gltf.nodes.forEach((node, index) => {
        if (
            node.skin !== undefined &&
            node.mesh !== undefined &&
            !skinnedMeshNodes.has(node.mesh)
        ) {
            skinnedMeshNodes.set(node.mesh, index);
        }
    });

    gltf.meshes.forEach((mesh, meshIndex) => {
        if (!isMsfsCompiledMesh(gltf, mesh)) {
            return;
        }
        const skinnedNode = skinnedMeshNodes.get(meshIndex);
        let inverseWorld = undefined;
        if (skinnedNode !== undefined) {
            // compiled space -> mesh space: inverse(world(mesh node)) * world(skeleton root's parent)
            const skin = gltf.skins[gltf.nodes[skinnedNode].skin];
            const skeletonParent = parents.get(skin?.skeleton ?? skin?.joints?.[0]);
            inverseWorld = mat4.invert(mat4.create(), world(skinnedNode));
            if (skeletonParent !== undefined) {
                mat4.multiply(inverseWorld, inverseWorld, world(skeletonParent));
            }
        }
        for (const primitive of mesh.primitives) {
            const range = primitive.extras?.ASOBO_primitive;
            const position = gltf.accessors[primitive.attributes.POSITION];
            if (position === undefined) {
                continue;
            }
            const baseVertex = range?.BaseVertexIndex ?? 0;
            const vertexCount = range?.VertexCount ?? position.count - baseVertex;
            for (const [name, accessorIndex] of Object.entries(primitive.attributes)) {
                primitive.attributes[name] = decode(name, accessorIndex, baseVertex, vertexCount);
            }
            if (inverseWorld !== undefined && primitive.attributes.JOINTS_0 !== undefined) {
                toMeshSpace(gltf, primitive, inverseWorld, movedToMeshSpace);
            }
            const indices = gltf.accessors[primitive.indices];
            if (indices !== undefined && (primitive.mode ?? TRIANGLES) === TRIANGLES) {
                primitive.indices = copyTriangles(
                    gltf,
                    indices,
                    range?.StartIndex ?? 0,
                    range?.PrimitiveCount !== undefined ? range.PrimitiveCount * 3 : indices.count
                );
            }
        }
    });
    return true;
}

/**
 * Skins of MSFS exports bind to the rest pose: the package builder compiles skinned vertices from
 * the unskinned rest geometry and ignores the inverse bind matrices. Those don't always match:
 * the Babylon.js 3ds Max exporter writes a wrong inverse bind matrix for the skeleton root (the
 * first joint), which moves its vertices away from the mesh (e.g. 3.9 m on the A32X flaps).
 * Rebuilds every skin's inverse bind matrices as inverse(rest world(joint)) * rest world(mesh
 * node), which leaves the other joints unchanged. Call after decodeMsfsCompiledGeometry.
 * @returns {number} the number of joints whose inverse bind matrix changed
 */
function rebindMsfsSkins(gltf) {
    if (!gltf.skins?.length) {
        return 0;
    }
    const { world } = restWorldTransforms(gltf);
    const meshNodes = new Map(); // skin index -> first mesh node using it
    gltf.nodes.forEach((node, index) => {
        if (node.skin !== undefined && node.mesh !== undefined && !meshNodes.has(node.skin)) {
            meshNodes.set(node.skin, index);
        }
    });
    let changed = 0;
    gltf.skins.forEach((skin, skinIndex) => {
        const meshNode = meshNodes.get(skinIndex);
        if (meshNode === undefined || skin.joints.length === 0) {
            return;
        }
        const old =
            skin.inverseBindMatrices !== undefined
                ? gltf.accessors[skin.inverseBindMatrices].getDeinterlacedView(gltf)
                : undefined;
        const data = new Float32Array(skin.joints.length * 16);
        const matrix = mat4.create();
        skin.joints.forEach((joint, i) => {
            mat4.invert(matrix, world(joint));
            mat4.multiply(matrix, matrix, world(meshNode));
            data.set(matrix, i * 16);
            const before = old?.subarray(i * 16, i * 16 + 16);
            if (before === undefined || matrix.some((value, k) => Math.abs(value - before[k]) > 1e-3)) {
                changed++;
            }
        });
        skin.inverseBindMatrices = pushAccessor(gltf, data, "MAT4", FLOAT, undefined);
    });
    return changed;
}

export { decodeMsfsCompiledGeometry, isMsfsCompiled, rebindMsfsSkins };
