import { gltfBuffer } from "./buffer.js";
import { gltfAccessor } from "./accessor.js";
import { gltfBufferView } from "./buffer_view.js";
import { GL } from "../Renderer/webgl.js";

/**
 * Generates MikkTSpace tangents for all primitives that need them in a pool of web workers,
 * before gltf.initGl runs. Primitives that are handled here get a TANGENT attribute and keep
 * an (re-welded) index buffer; everything else falls back to the synchronous path in
 * gltfPrimitive.initGl.
 * @param {*} gltf The glTF document; buffers must be loaded.
 * @param {String} workerUrl URL of tangent.worker.js.
 */
async function generateTangentsInWorkers(gltf, workerUrl) {
    if (typeof Worker === "undefined") {
        return;
    }

    // Primitives with the same source data share one job.
    const jobs = new Map();
    for (const mesh of gltf.meshes) {
        for (const primitive of mesh.primitives) {
            if (!needsTangents(primitive)) {
                continue;
            }
            const hash = `${primitive.indices}_${primitive.attributes.POSITION}_${primitive.attributes.NORMAL}_${primitive.attributes.TEXCOORD_0}`;
            let job = jobs.get(hash);
            if (job === undefined) {
                job = createJob(gltf, primitive);
                if (job === undefined) {
                    continue;
                }
                jobs.set(hash, job);
            }
            job.primitives.push(primitive);
        }
    }
    if (jobs.size === 0) {
        return;
    }

    // Largest first, so a big primitive does not start last.
    const queue = [...jobs.values()].sort((a, b) => b.size - a.size);
    const workerCount = Math.max(
        1,
        Math.min(queue.length, (navigator.hardwareConcurrency ?? 4) - 1, 8)
    );

    let workers;
    try {
        workers = Array.from(
            { length: workerCount },
            () => new Worker(workerUrl, { type: "module" })
        );
    } catch (error) {
        console.warn(
            "Failed to spawn tangent workers, generating tangents on the main thread:",
            error
        );
        return;
    }

    try {
        await Promise.all(workers.map((worker) => runWorker(gltf, worker, queue)));
    } finally {
        for (const worker of workers) {
            worker.terminate();
        }
    }
}

function needsTangents(primitive) {
    return (
        primitive.attributes.TANGENT === undefined &&
        primitive.attributes.NORMAL !== undefined &&
        primitive.attributes.TEXCOORD_0 !== undefined &&
        primitive.attributes.POSITION !== undefined &&
        primitive.mode === GL.TRIANGLES &&
        primitive.extensions?.KHR_draco_mesh_compression === undefined
    );
}

function createJob(gltf, primitive) {
    let positions =
        gltf.accessors[primitive.attributes.POSITION].getNormalizedDeinterlacedView(gltf);
    const normals = gltf.accessors[primitive.attributes.NORMAL].getNormalizedDeinterlacedView(gltf);
    let texcoords =
        gltf.accessors[primitive.attributes.TEXCOORD_0].getNormalizedDeinterlacedView(gltf);
    if (
        positions instanceof Float64Array ||
        normals instanceof Float32Array === false ||
        texcoords instanceof Float64Array
    ) {
        // Leave these to the synchronous path, which reports the problem.
        return undefined;
    }

    // Copies, because the views may point into the shared glTF buffer.
    const message = {
        positions: new Float32Array(positions),
        normals: new Float32Array(normals),
        texcoords: new Float32Array(texcoords),
        indices: undefined
    };
    if (primitive.indices !== undefined) {
        message.indices = new Uint32Array(
            gltf.accessors[primitive.indices].getDeinterlacedView(gltf)
        );
    }
    return {
        message,
        size: message.indices?.length ?? message.positions.length / 3,
        primitives: []
    };
}

function runWorker(gltf, worker, queue) {
    return new Promise((resolve) => {
        const next = () => {
            const job = queue.shift();
            if (job === undefined) {
                resolve();
                return;
            }
            const { message } = job;
            const transfer = [
                message.positions.buffer,
                message.normals.buffer,
                message.texcoords.buffer
            ];
            if (message.indices !== undefined) {
                transfer.push(message.indices.buffer);
            }
            worker.onmessage = (event) => {
                if (event.data.error !== undefined) {
                    console.warn(
                        "Tangent worker failed, falling back to the main thread:",
                        event.data.error
                    );
                } else {
                    applyResult(gltf, job.primitives, event.data);
                }
                next();
            };
            worker.onerror = (event) => {
                event.preventDefault();
                console.warn(
                    "Tangent worker failed, falling back to the main thread:",
                    event.message
                );
                // The worker is unusable; the remaining jobs go to the other workers or the sync path.
                resolve();
            };
            worker.postMessage({ id: 0, ...message }, transfer);
            job.message = undefined;
        };
        next();
    });
}

function applyResult(gltf, primitives, result) {
    const tangent = pushAccessor(gltf, result.tangents, "VEC4", GL.FLOAT, GL.ARRAY_BUFFER);
    if (result.remap === undefined) {
        for (const primitive of primitives) {
            primitive.attributes.TANGENT = tangent;
        }
        return;
    }

    const indices = pushAccessor(
        gltf,
        result.indices,
        "SCALAR",
        GL.UNSIGNED_INT,
        GL.ELEMENT_ARRAY_BUFFER
    );
    // Primitives of one job can still differ in their other attributes; share what is equal.
    const remapped = new Map();
    const remap = (accessorIndex) => {
        let index = remapped.get(accessorIndex);
        if (index === undefined) {
            index = remapAccessor(gltf, gltf.accessors[accessorIndex], result.remap);
            remapped.set(accessorIndex, index);
        }
        return index;
    };
    for (const primitive of primitives) {
        for (const [attribute, accessorIndex] of Object.entries(primitive.attributes)) {
            primitive.attributes[attribute] = remap(accessorIndex);
        }
        for (const target of primitive.targets) {
            for (const [attribute, accessorIndex] of Object.entries(target)) {
                target[attribute] = remap(accessorIndex);
            }
        }
        primitive.indices = indices;
        primitive.attributes.TANGENT = tangent;
    }
}

function remapAccessor(gltf, accessor, remap) {
    const componentCount = accessor.getComponentCount(accessor.type);
    const source = accessor.getDeinterlacedView(gltf);
    const data = new source.constructor(remap.length * componentCount);
    for (let i = 0; i < remap.length; i++) {
        const from = remap[i] * componentCount;
        const to = i * componentCount;
        for (let j = 0; j < componentCount; j++) {
            data[to + j] = source[from + j];
        }
    }
    const index = pushAccessor(gltf, data, accessor.type, accessor.componentType, GL.ARRAY_BUFFER);
    const result = gltf.accessors[index];
    result.min = accessor.min;
    result.max = accessor.max;
    result.normalized = accessor.normalized;
    return index;
}

function pushAccessor(gltf, data, type, componentType, target) {
    const buffer = new gltfBuffer();
    buffer.byteLength = data.byteLength;
    buffer.buffer = data.buffer;
    gltf.buffers.push(buffer);

    const bufferView = new gltfBufferView();
    bufferView.buffer = gltf.buffers.length - 1;
    bufferView.byteLength = data.byteLength;
    bufferView.target = target;
    gltf.bufferViews.push(bufferView);

    const accessor = new gltfAccessor();
    accessor.bufferView = gltf.bufferViews.length - 1;
    accessor.byteOffset = 0;
    accessor.type = type;
    accessor.count = data.length / accessor.getComponentCount(type);
    accessor.componentType = componentType;
    gltf.accessors.push(accessor);
    return gltf.accessors.length - 1;
}

export { generateTangentsInWorkers, pushAccessor };
