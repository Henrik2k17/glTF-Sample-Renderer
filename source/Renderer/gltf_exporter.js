import { mat3 } from "gl-matrix";
import { gltfTextureInfo } from "../gltf/texture.js";
import { getMsfsUV0Transform, getMsfsClampedTextures } from "../gltf/msfs_material.js";

// Writes the shown part of a loaded glTF as a standard GLB (glTF 2.0 core plus KHR material
// extensions), e.g. to look at or edit an MSFS model in Blender. Everything is read from the
// loaded objects, so it includes what the loader did: MSFS compiled geometry decoded, skins
// rebound to the rest pose, packages merged into one glTF, edited materials. MSFS extensions
// are dropped; what has a KHR equivalent (clearcoat, iridescence, UV options as texture
// transforms) is written as that. Textures are read back from the GPU (the CPU copies are
// released after upload) and stored as PNG, normal maps converted to OpenGL convention.
// Nodes are written in the pose shown (animations and behaviors applied), materials with their
// rest values; animations are written as they are, for Blender to play.

const BYTE = 5120;
const UNSIGNED_BYTE = 5121;
const SHORT = 5122;
const UNSIGNED_SHORT = 5123;
const UNSIGNED_INT = 5125;
const FLOAT = 5126;
const ARRAY_BUFFER = 34962;
const ELEMENT_ARRAY_BUFFER = 34963;
const CLAMP_TO_EDGE = 33071;

const ComponentCount = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 };
const TypedArrayComponentType = new Map([
    [Int8Array, BYTE],
    [Uint8Array, UNSIGNED_BYTE],
    [Int16Array, SHORT],
    [Uint16Array, UNSIGNED_SHORT],
    [Uint32Array, UNSIGNED_INT],
    [Float32Array, FLOAT]
]);
const NodePaths = ["translation", "rotation", "scale", "weights"];
// GLB chunk lengths are 32 bit
const MaxGlbLength = 2 ** 32 - 1;

/** Rest (not animated) value of a renderer object's property, arrays as plain arrays. */
function restValue(object, key) {
    const property = object.animatedPropertyObjects?.[key];
    const value = property !== undefined ? property.restValue : object[key];
    return ArrayBuffer.isView(value) ? Array.from(value) : value;
}

/**
 * An accessor's elements, tightly packed in its component type (normalized: as floats). The
 * accessor's cached views are left as they were, so exporting doesn't keep a copy of all
 * geometry in memory.
 */
function readAccessor(gltf, accessor, normalized) {
    const saved = [accessor.filteredView, accessor.normalizedFilteredView];
    try {
        return normalized
            ? accessor.getNormalizedDeinterlacedView(gltf)
            : accessor.getDeinterlacedView(gltf);
    } finally {
        [accessor.filteredView, accessor.normalizedFilteredView] = saved;
    }
}

/** Whether core glTF allows the accessor's component type for a vertex attribute. */
function isCoreAttributeType(semantic, accessor) {
    const small =
        accessor.componentType === UNSIGNED_BYTE || accessor.componentType === UNSIGNED_SHORT;
    switch (semantic.replace(/_\d+$/, "")) {
        case "TEXCOORD":
        case "COLOR":
        case "WEIGHTS":
            return accessor.componentType === FLOAT || (small && accessor.normalized);
        case "JOINTS":
            return small && !accessor.normalized;
        default:
            return accessor.componentType === FLOAT;
    }
}

function minMax(data, components) {
    const min = new Array(components).fill(Infinity);
    const max = new Array(components).fill(-Infinity);
    for (let i = 0; i < data.length; i++) {
        const c = i % components;
        min[c] = Math.min(min[c], data[i]);
        max[c] = Math.max(max[c], data[i]);
    }
    return { min, max };
}

const CrcTable = (() => {
    const table = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
        let c = n;
        for (let k = 0; k < 8; k++) {
            c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        }
        table[n] = c >>> 0;
    }
    return table;
})();

function crc32(bytes) {
    let crc = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) {
        crc = CrcTable[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
    }
    return (crc ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data) {
    const chunk = new Uint8Array(12 + data.length);
    const view = new DataView(chunk.buffer);
    view.setUint32(0, data.length);
    for (let i = 0; i < 4; i++) {
        chunk[4 + i] = type.charCodeAt(i);
    }
    chunk.set(data, 8);
    view.setUint32(8 + data.length, crc32(chunk.subarray(4, 8 + data.length)));
    return chunk;
}

/**
 * Encodes 8-bit RGBA pixels (row 0 first) as PNG, RGB when every pixel is opaque. The pixels
 * are stored exactly; a canvas would premultiply alpha and lose the colour of transparent texels.
 * @returns {Promise<Blob>}
 */
async function encodePng(width, height, rgba) {
    let opaque = true;
    for (let i = 3; i < rgba.length; i += 4) {
        if (rgba[i] !== 255) {
            opaque = false;
            break;
        }
    }
    const channels = opaque ? 3 : 4;
    const rowLength = width * channels;
    // filter type 1 (Sub) on every row: each byte minus the same channel of the pixel before
    const filtered = new Uint8Array(height * (rowLength + 1));
    for (let y = 0; y < height; y++) {
        const out = y * (rowLength + 1);
        filtered[out] = 1;
        for (let x = 0; x < width; x++) {
            for (let c = 0; c < channels; c++) {
                const i = (y * width + x) * 4 + c;
                const left = x > 0 ? rgba[i - 4] : 0;
                filtered[out + 1 + x * channels + c] = rgba[i] - left;
            }
        }
    }
    // "deflate" is the zlib format PNG's IDAT chunk uses
    const compressed = new Uint8Array(
        await new Response(
            new Blob([filtered]).stream().pipeThrough(new CompressionStream("deflate"))
        ).arrayBuffer()
    );
    const header = new Uint8Array(13);
    const headerView = new DataView(header.buffer);
    headerView.setUint32(0, width);
    headerView.setUint32(4, height);
    header[8] = 8; // bit depth
    header[9] = opaque ? 2 : 6; // RGB : RGBA
    const signature = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
    return new Blob(
        [
            signature,
            pngChunk("IHDR", header),
            pngChunk("IDAT", compressed),
            pngChunk("IEND", new Uint8Array(0))
        ],
        { type: "image/png" }
    );
}

/**
 * Splits a 2D affine UV transform (column-major mat3) into KHR_texture_transform's
 * offset * rotation * scale (see gltfMaterial.updateTextureTransforms). Shear, which tiling
 * after rotation with different U and V tiling gives, can't be expressed and is dropped.
 */
function toTextureTransform(m) {
    const scaleX = Math.hypot(m[0], m[1]);
    const rotation = Math.atan2(-m[1], m[0]);
    const s = Math.sin(rotation);
    const c = Math.cos(rotation);
    const scaleY = m[3] * s + m[4] * c;
    return { offset: [m[6], m[7]], rotation, scale: [scaleX, scaleY] };
}

function fromTextureTransform(transform) {
    const offset = restValue(transform, "offset") ?? [0, 0];
    const rotation = restValue(transform, "rotation") ?? 0;
    const scale = restValue(transform, "scale") ?? [1, 1];
    const s = Math.sin(rotation);
    const c = Math.cos(rotation);
    // prettier-ignore
    return mat3.fromValues(
        c * scale[0], -s * scale[0], 0,
        s * scale[1], c * scale[1], 0,
        offset[0], offset[1], 1
    );
}

/** Binary chunk of the GLB, collected as Blobs so the data can leave the JS heap. */
class BinaryChunk {
    constructor() {
        this.parts = [];
        this.byteLength = 0;
        this.bufferViews = [];
    }

    /** Appends data (TypedArray or Blob) 4-byte aligned; returns the buffer view index. */
    addBufferView(data, extra = {}) {
        const byteLength = data instanceof Blob ? data.size : data.byteLength;
        this.parts.push(data instanceof Blob ? data : new Blob([data]));
        this.bufferViews.push({ buffer: 0, byteOffset: this.byteLength, byteLength, ...extra });
        this.byteLength += byteLength;
        const padding = (4 - (this.byteLength % 4)) % 4;
        if (padding > 0) {
            this.parts.push(new Uint8Array(padding));
            this.byteLength += padding;
        }
        return this.bufferViews.length - 1;
    }
}

class GlbExporter {
    /**
     * @param {object} gltf The loaded glTF.
     * @param {object} options
     * @param {number} options.sceneIndex The scene to export.
     * @param {Set<object>} options.shownNodes Nodes to export with their meshes (others are
     *   left out, unless a skin needs them as joints).
     * @param {(primitive) => boolean} options.isPrimitiveShown
     * @param {(textureIndex, maxSize, normalMap) => {width, height, pixels} | undefined} options.readTexture
     * @param {number} [options.maxTextureSize] Largest exported texture side (0: as loaded).
     * @param {(material) => number} [options.emissiveStrength] Extra emissive factor per material.
     * @param {({done, total}) => void} [options.onProgress] Called per written texture.
     */
    constructor(gltf, options) {
        this.gltf = gltf;
        this.options = options;
        this.binary = new BinaryChunk();
        this.json = {
            asset: { version: "2.0", generator: "glTF-Sample-Viewer (MSFS 2024 export)" },
            scene: 0,
            scenes: [],
            nodes: [],
            meshes: [],
            materials: [],
            textures: [],
            images: [],
            samplers: [],
            accessors: [],
            bufferViews: this.binary.bufferViews,
            buffers: [],
            skins: [],
            animations: []
        };
        this.extensionsUsed = new Set();
        this.accessorMap = new Map(); // `${index}:${float}` -> exported accessor
        this.materialMap = new Map();
        this.meshMap = new Map();
        this.skinMap = new Map();
        this.textureMap = new Map(); // `${texture}:${normalMap}:${clamp}` -> exported texture
        this.imageMap = new Map(); // `${texture}:${normalMap}` -> exported image
        this.samplerMap = new Map();
        this.pendingImages = []; // { index, textureIndex, normalMap }
        this.warnings = [];
    }

    async export() {
        const gltf = this.gltf;
        const scene = gltf.scenes[this.options.sceneIndex];
        const exported = this.collectNodes(scene);
        const nodeMap = new Map([...exported].map((node, i) => [node, i]));
        this.json.nodes = [...exported].map((node) => this.writeNode(node, nodeMap));
        this.json.scenes.push({
            name: scene.name,
            nodes: scene.nodes
                .map((index) => nodeMap.get(gltf.nodes[index]))
                .filter((i) => i !== undefined)
        });
        this.writeAnimations(nodeMap);
        await this.writeImages();
        return this.toGlb();
    }

    /** Shown nodes plus the joints (and their ancestors) of the skins of shown meshes, in glTF order. */
    collectNodes(scene) {
        const gltf = this.gltf;
        const include = new Set();
        const addWithAncestors = (node) => {
            for (let n = node; n !== undefined && !include.has(n); n = n.parentNode) {
                include.add(n);
            }
        };
        for (const node of this.options.shownNodes) {
            if (node.scene !== scene) {
                continue;
            }
            addWithAncestors(node);
            if (node.mesh !== undefined && node.skin !== undefined) {
                for (const joint of gltf.skins[node.skin].joints) {
                    addWithAncestors(gltf.nodes[joint]);
                }
            }
        }
        return new Set(gltf.nodes.filter((node) => include.has(node)));
    }

    writeNode(node, nodeMap) {
        const gltf = this.gltf;
        const json = { name: node.name };
        for (const key of ["translation", "rotation", "scale"]) {
            json[key] = Array.from(node[key]); // as shown
        }
        const children = node.children
            .map((i) => nodeMap.get(gltf.nodes[i]))
            .filter((i) => i !== undefined);
        if (children.length > 0) {
            json.children = children;
        }
        if (node.mesh !== undefined && this.options.shownNodes.has(node)) {
            const mesh = this.writeMesh(node.mesh);
            if (mesh !== undefined) {
                json.mesh = mesh;
                const weights = node.weights === undefined ? undefined : Array.from(node.weights);
                if (weights?.length > 0) {
                    json.weights = weights;
                }
                if (node.skin !== undefined) {
                    json.skin = this.writeSkin(node.skin, nodeMap);
                }
            }
        }
        return json;
    }

    writeMesh(meshIndex) {
        if (this.meshMap.has(meshIndex)) {
            return this.meshMap.get(meshIndex);
        }
        const mesh = this.gltf.meshes[meshIndex];
        const primitives = mesh.primitives
            .filter((primitive) => this.options.isPrimitiveShown(primitive))
            .map((primitive) => this.writePrimitive(primitive))
            .filter((primitive) => primitive !== undefined);
        let index = undefined;
        if (primitives.length > 0) {
            const json = { name: mesh.name, primitives };
            const weights = restValue(mesh, "weights");
            if (weights?.length > 0) {
                json.weights = weights;
            }
            this.json.meshes.push(json);
            index = this.json.meshes.length - 1;
        }
        this.meshMap.set(meshIndex, index);
        return index;
    }

    writePrimitive(primitive) {
        if (
            primitive.attributes.POSITION === undefined ||
            primitive.extensions?.KHR_gaussian_splatting !== undefined
        ) {
            return undefined;
        }
        const writeAttributes = (attributes) =>
            Object.fromEntries(
                Object.entries(attributes).map(([semantic, index]) => [
                    semantic,
                    this.writeAttribute(semantic, index)
                ])
            );
        const json = { attributes: writeAttributes(primitive.attributes), mode: primitive.mode };
        if (primitive.indices !== undefined) {
            json.indices = this.writeAccessor(primitive.indices, { target: ELEMENT_ARRAY_BUFFER });
        }
        if (primitive.targets?.length > 0) {
            json.targets = primitive.targets.map((target) => writeAttributes(target));
        }
        const material = this.writeMaterial(primitive.material);
        if (material !== undefined) {
            json.material = material;
        }
        return json;
    }

    writeAttribute(semantic, index) {
        const accessor = this.gltf.accessors[index];
        const float = !isCoreAttributeType(semantic, accessor);
        return this.writeAccessor(index, {
            target: ARRAY_BUFFER,
            float,
            joints: semantic.startsWith("JOINTS_"),
            minMax: semantic === "POSITION"
        });
    }

    /**
     * @param {object} flags float: write as (dequantized) floats, or with joints as unsigned shorts;
     *   minMax: write min and max; target: buffer view target.
     */
    writeAccessor(
        index,
        { float = false, joints = false, minMax: withMinMax = false, target = undefined } = {}
    ) {
        const key = `${index}:${float}:${joints}`;
        if (this.accessorMap.has(key)) {
            return this.accessorMap.get(key);
        }
        const accessor = this.gltf.accessors[index];
        const components = ComponentCount[accessor.type];
        let data = readAccessor(this.gltf, accessor, float && !joints);
        if (joints && float) {
            data = Uint16Array.from(data); // joint indices in a type core glTF doesn't allow
        } else if (float && !(data instanceof Float32Array)) {
            data = Float32Array.from(data);
        }
        const componentType = TypedArrayComponentType.get(data.constructor);
        const json = {
            componentType,
            count: accessor.count,
            type: accessor.type
        };
        if (!float && accessor.normalized) {
            json.normalized = true;
        }
        if (withMinMax) {
            Object.assign(json, minMax(data, components));
        }
        // Vertex attribute elements must start on 4 byte boundaries
        const elementSize = components * data.BYTES_PER_ELEMENT;
        const extra = target === undefined ? {} : { target };
        if (target === ARRAY_BUFFER && elementSize % 4 !== 0) {
            const stride = Math.ceil(elementSize / 4) * 4;
            const padded = new Uint8Array(stride * accessor.count);
            const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
            for (let i = 0; i < accessor.count; i++) {
                padded.set(bytes.subarray(i * elementSize, (i + 1) * elementSize), i * stride);
            }
            json.bufferView = this.binary.addBufferView(padded, { ...extra, byteStride: stride });
        } else {
            json.bufferView = this.binary.addBufferView(data, extra);
        }
        this.json.accessors.push(json);
        const result = this.json.accessors.length - 1;
        this.accessorMap.set(key, result);
        return result;
    }

    writeSkin(skinIndex, nodeMap) {
        if (this.skinMap.has(skinIndex)) {
            return this.skinMap.get(skinIndex);
        }
        const gltf = this.gltf;
        const skin = gltf.skins[skinIndex];
        const json = {
            name: skin.name,
            joints: skin.joints.map((joint) => nodeMap.get(gltf.nodes[joint]))
        };
        if (skin.inverseBindMatrices !== undefined) {
            json.inverseBindMatrices = this.writeAccessor(skin.inverseBindMatrices, {
                float: true
            });
        }
        const skeleton = nodeMap.get(gltf.nodes[skin.skeleton]);
        if (skeleton !== undefined) {
            json.skeleton = skeleton;
        }
        this.json.skins.push(json);
        const index = this.json.skins.length - 1;
        this.skinMap.set(skinIndex, index);
        return index;
    }

    writeAnimations(nodeMap) {
        const gltf = this.gltf;
        for (const animation of gltf.animations) {
            const samplers = [];
            const samplerMap = new Map();
            const channels = [];
            for (const channel of animation.channels) {
                const node = nodeMap.get(gltf.nodes[channel.target.node]);
                // material and other property animations (KHR_animation_pointer) are left out
                if (node === undefined || !NodePaths.includes(channel.target.path)) {
                    continue;
                }
                if (channel.target.path === "weights" && this.json.nodes[node].mesh === undefined) {
                    continue;
                }
                if (!samplerMap.has(channel.sampler)) {
                    const sampler = animation.samplers[channel.sampler];
                    samplers.push({
                        input: this.writeAccessor(sampler.input, { float: true, minMax: true }),
                        output: this.writeAccessor(sampler.output, { float: true }),
                        interpolation: sampler.interpolation
                    });
                    samplerMap.set(channel.sampler, samplers.length - 1);
                }
                channels.push({
                    sampler: samplerMap.get(channel.sampler),
                    target: { node, path: channel.target.path }
                });
            }
            if (channels.length > 0) {
                this.json.animations.push({ name: animation.name, channels, samplers });
            }
        }
    }

    writeMaterial(materialIndex) {
        if (this.materialMap.has(materialIndex)) {
            return this.materialMap.get(materialIndex);
        }
        const material = this.gltf.materials[materialIndex];
        // the renderer's default material (appended at load) stays the importer's default
        const isDefault =
            materialIndex === this.gltf.materials.length - 1 &&
            material.name === "Default Material";
        const index = isDefault ? undefined : this.json.materials.length;
        this.materialMap.set(materialIndex, index);
        if (isDefault) {
            return undefined;
        }
        this.json.materials.push(this.materialJson(material));
        return index;
    }

    materialJson(material) {
        const json = { name: material.name };
        const normalMap = material.defines.includes("NORMAL_MAP_DIRECTX 1") ? "directx" : "opengl";
        // ASOBO_material_UV_options: one transform of UV0 for all its textures
        const uvOptions = material.msfs?.uvOptions;
        const uv0Transform =
            uvOptions === undefined
                ? undefined
                : getMsfsUV0Transform(
                      Object.fromEntries(
                          ["UVOffsetU", "UVOffsetV", "UVTilingU", "UVTilingV", "UVRotation"].map(
                              (key) => [key, restValue(uvOptions, key)]
                          )
                      )
                  );
        const clamped = getMsfsClampedTextures(material);
        const texture = (info, normal = false) =>
            this.textureInfoJson(info, normal ? normalMap : undefined, uv0Transform, clamped);

        const pbr = material.pbrMetallicRoughness;
        if (material.type !== "SG") {
            json.pbrMetallicRoughness = {
                baseColorFactor: restValue(pbr, "baseColorFactor"),
                metallicFactor: restValue(pbr, "metallicFactor"),
                roughnessFactor: restValue(pbr, "roughnessFactor"),
                baseColorTexture: texture(pbr.baseColorTexture),
                metallicRoughnessTexture: texture(pbr.metallicRoughnessTexture)
            };
        }
        json.normalTexture = texture(material.normalTexture, true);
        json.occlusionTexture = texture(material.occlusionTexture);
        json.emissiveTexture = texture(material.emissiveTexture);
        json.emissiveFactor = restValue(material, "emissiveFactor");
        json.alphaMode = material.alphaMode;
        if (material.alphaMode === "MASK") {
            json.alphaCutoff = restValue(material, "alphaCutoff");
        }
        json.doubleSided = material.doubleSided;

        const extensions = {};
        for (const [name, extension] of Object.entries(material.extensions ?? {})) {
            if (!name.startsWith("KHR_materials_") || name === "KHR_materials_variants") {
                continue;
            }
            const out = {};
            for (const key of new Set([
                ...Object.keys(extension),
                ...(extension.constructor.animatedProperties ?? [])
            ])) {
                if (
                    ["extensions", "extras", "gltfObjectIndex", "animatedPropertyObjects"].includes(
                        key
                    )
                ) {
                    continue;
                }
                const value = restValue(extension, key);
                if (key.endsWith("Texture")) {
                    out[key] = texture(value, key === "clearcoatNormalTexture");
                } else if (
                    ["number", "boolean", "string"].includes(typeof value) ||
                    Array.isArray(value)
                ) {
                    out[key] = value;
                }
            }
            extensions[name] = out;
        }
        // glTF limits emissiveFactor to 0..1 (MSFS exports can go higher): the rest becomes strength
        let emissiveStrength = this.options.emissiveStrength?.(material) ?? 1;
        const brightest = Math.max(...json.emissiveFactor);
        if (brightest > 1) {
            json.emissiveFactor = json.emissiveFactor.map((value) => value / brightest);
            emissiveStrength *= brightest;
        }
        if (emissiveStrength !== 1 || extensions.KHR_materials_emissive_strength !== undefined) {
            const strength =
                (extensions.KHR_materials_emissive_strength?.emissiveStrength ?? 1) *
                emissiveStrength;
            extensions.KHR_materials_emissive_strength = { emissiveStrength: strength };
        }
        if (Object.keys(extensions).length > 0) {
            json.extensions = extensions;
            Object.keys(extensions).forEach((name) => this.extensionsUsed.add(name));
        }
        return JSON.parse(JSON.stringify(json)); // drops undefined members
    }

    /** Texture info JSON (index remapped, MSFS UV0 transform folded in), undefined if not readable. */
    textureInfoJson(info, normalMap, uv0Transform, clamped) {
        if (info === undefined || info.index === undefined) {
            return undefined;
        }
        const clamp = clamped?.textures.has(info) ? clamped : undefined;
        const index = this.writeTexture(info.index, normalMap, clamp);
        if (index === undefined) {
            return undefined;
        }
        const json = { index };
        const transform = info.extensions?.KHR_texture_transform;
        const texCoord = transform?.texCoord ?? info.texCoord ?? 0;
        if (texCoord !== 0) {
            json.texCoord = texCoord;
        }
        if (info instanceof gltfTextureInfo) {
            if (normalMap !== undefined) {
                json.scale = restValue(info, "scale");
            } else if (restValue(info, "strength") !== 1) {
                json.strength = restValue(info, "strength");
            }
        } else {
            Object.assign(json, { scale: info.scale, strength: info.strength });
        }
        let matrix = transform === undefined ? undefined : fromTextureTransform(transform);
        if (uv0Transform !== undefined && texCoord === 0) {
            matrix =
                matrix === undefined
                    ? uv0Transform
                    : mat3.multiply(mat3.create(), matrix, uv0Transform);
        }
        if (matrix !== undefined) {
            json.extensions = { KHR_texture_transform: toTextureTransform(matrix) };
            this.extensionsUsed.add("KHR_texture_transform");
        }
        return json;
    }

    writeTexture(textureIndex, normalMap, clamp) {
        const key = `${textureIndex}:${normalMap}:${clamp?.clampU}:${clamp?.clampV}`;
        if (this.textureMap.has(key)) {
            return this.textureMap.get(key);
        }
        const texture = this.gltf.textures[textureIndex];
        const image = this.gltf.images[texture?.source];
        let index = undefined;
        if (image?.image !== undefined) {
            const imageKey = `${textureIndex}:${normalMap}`;
            if (!this.imageMap.has(imageKey)) {
                // without the image extensions (MSFS: x.png.dds, x.png.ktx2)
                const name = (
                    image.name ??
                    image.uri?.split(/[\\/]/).pop() ??
                    `texture${textureIndex}`
                ).replace(/(\.(png|jpe?g|webp|dds|ktx2|tga|tiff?))+$/i, "");
                this.json.images.push({
                    name: normalMap === "directx" ? `${name}_opengl` : name,
                    mimeType: "image/png"
                });
                this.imageMap.set(imageKey, this.json.images.length - 1);
                this.pendingImages.push({
                    index: this.json.images.length - 1,
                    textureIndex,
                    normalMap
                });
            }
            const json = { source: this.imageMap.get(imageKey) };
            const sampler = this.writeSampler(this.gltf.samplers[texture.sampler], clamp);
            if (sampler !== undefined) {
                json.sampler = sampler;
            }
            this.json.textures.push(json);
            index = this.json.textures.length - 1;
        }
        this.textureMap.set(key, index);
        return index;
    }

    writeSampler(sampler, clamp) {
        if (sampler === undefined) {
            return undefined;
        }
        const json = {
            magFilter: sampler.magFilter,
            minFilter: sampler.minFilter,
            wrapS: clamp?.clampU ? CLAMP_TO_EDGE : sampler.wrapS,
            wrapT: clamp?.clampV ? CLAMP_TO_EDGE : sampler.wrapT
        };
        const key = JSON.stringify(json);
        if (!this.samplerMap.has(key)) {
            this.json.samplers.push(json);
            this.samplerMap.set(key, this.json.samplers.length - 1);
        }
        return this.samplerMap.get(key);
    }

    /** Reads the textures back from the GPU and stores them as PNG; unreadable ones are left out. */
    async writeImages() {
        const total = this.pendingImages.length;
        const maxSize = this.options.maxTextureSize > 0 ? this.options.maxTextureSize : Infinity;
        const failed = new Set();
        for (let done = 0; done < total; done++) {
            const { index, textureIndex, normalMap } = this.pendingImages[done];
            this.options.onProgress?.({ done, total });
            const result = this.options.readTexture(textureIndex, maxSize, normalMap);
            if (result === undefined) {
                failed.add(index);
                continue;
            }
            const png = await encodePng(result.width, result.height, result.pixels);
            this.json.images[index].bufferView = this.binary.addBufferView(png);
            if (this.binary.byteLength > MaxGlbLength) {
                throw new Error(
                    `The textures don't fit into the 4 GB a GLB file can hold (${done + 1} of ${total} written). Choose a smaller texture size.`
                );
            }
        }
        this.options.onProgress?.({ done: total, total });
        if (failed.size > 0) {
            this.warnings.push(`${failed.size} texture(s) could not be read and were left out.`);
            this.dropImages(failed);
        }
    }

    /** Removes images (and the textures and texture infos using them) after the fact. */
    dropImages(failed) {
        const imageMap = new Map();
        this.json.images = this.json.images.filter((image, i) => {
            if (!failed.has(i)) {
                imageMap.set(i, imageMap.size);
            }
            return !failed.has(i);
        });
        const textureMap = new Map();
        this.json.textures = this.json.textures.filter((texture, i) => {
            const keep = imageMap.has(texture.source);
            if (keep) {
                texture.source = imageMap.get(texture.source);
                textureMap.set(i, textureMap.size);
            }
            return keep;
        });
        const remap = (object) => {
            for (const [key, value] of Object.entries(object)) {
                if (key.endsWith("Texture") && typeof value?.index === "number") {
                    if (textureMap.has(value.index)) {
                        value.index = textureMap.get(value.index);
                    } else {
                        delete object[key];
                    }
                } else if (typeof value === "object" && value !== null) {
                    remap(value);
                }
            }
        };
        this.json.materials.forEach(remap);
    }

    /** @returns {{blob: Blob, warnings: string[]}} */
    toGlb() {
        const json = this.json;
        if (this.binary.byteLength > 0) {
            json.buffers.push({ byteLength: this.binary.byteLength });
        }
        if (this.extensionsUsed.size > 0) {
            json.extensionsUsed = [...this.extensionsUsed];
        }
        for (const key of Object.keys(json)) {
            if (Array.isArray(json[key]) && json[key].length === 0) {
                delete json[key];
            }
        }
        const jsonBytes = new TextEncoder().encode(JSON.stringify(json));
        const jsonPadding = (4 - (jsonBytes.length % 4)) % 4;
        const jsonLength = jsonBytes.length + jsonPadding;
        const binLength = this.binary.byteLength;
        const totalLength = 12 + 8 + jsonLength + (binLength > 0 ? 8 + binLength : 0);
        if (totalLength > MaxGlbLength) {
            throw new Error(
                `The export would be ${(totalLength / 2 ** 30).toFixed(1)} GB, more than the 4 GB a GLB file can hold. Choose a smaller texture size.`
            );
        }
        const header = new DataView(new ArrayBuffer(20));
        header.setUint32(0, 0x46546c67, true); // "glTF"
        header.setUint32(4, 2, true);
        header.setUint32(8, totalLength, true);
        header.setUint32(12, jsonLength, true);
        header.setUint32(16, 0x4e4f534a, true); // "JSON"
        const parts = [header, jsonBytes, new Uint8Array(jsonPadding).fill(0x20)];
        if (binLength > 0) {
            const binHeader = new DataView(new ArrayBuffer(8));
            binHeader.setUint32(0, binLength, true);
            binHeader.setUint32(4, 0x004e4942, true); // "BIN"
            parts.push(binHeader, ...this.binary.parts);
        }
        return { blob: new Blob(parts, { type: "model/gltf-binary" }), warnings: this.warnings };
    }
}

/** See GlbExporter. @returns {Promise<{blob: Blob, warnings: string[]}>} */
async function exportGlb(gltf, options) {
    return await new GlbExporter(gltf, options).export();
}

export { exportGlb, encodePng };
