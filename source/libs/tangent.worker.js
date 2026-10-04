/**
 * tangent.worker.js
 *
 * Web Worker that generates MikkTSpace tangents for one indexed triangle list and welds
 * the result again, so the primitive keeps an index buffer.
 *
 * Messages in:
 *   { id, positions: Float32Array, normals: Float32Array, texcoords: Float32Array,
 *     indices: Uint32Array | undefined }   (welded vertex data)
 *
 * Messages out:
 *   { id, tangents: Float32Array }                       (non-indexed input: one per vertex)
 *   { id, tangents: Float32Array, indices: Uint32Array,
 *     remap: Uint32Array }                               (indexed input: remap[newVertex] = oldVertex)
 *   { id, error: string }
 */

import init, { generateTangents } from "./mikktspace.js";

const ready = init(new URL("./mikktspace_bg.wasm", import.meta.url));

function unweld(data, indices, componentCount) {
    const out = new Float32Array(indices.length * componentCount);
    for (let i = 0; i < indices.length; i++) {
        const src = indices[i] * componentCount;
        const dst = i * componentCount;
        for (let j = 0; j < componentCount; j++) {
            out[dst + j] = data[src + j];
        }
    }
    return out;
}

/**
 * Merges corners that come from the same source vertex and received the same tangent.
 * MikkTSpace writes identical values for corners in one smoothing group, so an exact
 * bit comparison is enough.
 */
function weld(indices, cornerTangents, vertexCount) {
    const cornerBits = new Uint32Array(cornerTangents.buffer);
    const firstNew = new Int32Array(vertexCount).fill(-1); // source vertex -> first new vertex
    const nextNew = new Int32Array(indices.length); // new vertex -> next new vertex with the same source
    const remap = new Uint32Array(indices.length);
    const tangentBits = new Uint32Array(indices.length * 4);
    const newIndices = new Uint32Array(indices.length);
    let count = 0;

    for (let i = 0; i < indices.length; i++) {
        const source = indices[i];
        const c = i * 4;
        let v = firstNew[source];
        let last = -1;
        while (v !== -1) {
            const t = v * 4;
            if (
                tangentBits[t] === cornerBits[c] &&
                tangentBits[t + 1] === cornerBits[c + 1] &&
                tangentBits[t + 2] === cornerBits[c + 2] &&
                tangentBits[t + 3] === cornerBits[c + 3]
            ) {
                break;
            }
            last = v;
            v = nextNew[v];
        }
        if (v === -1) {
            v = count++;
            remap[v] = source;
            nextNew[v] = -1;
            tangentBits.set(cornerBits.subarray(c, c + 4), v * 4);
            if (last === -1) {
                firstNew[source] = v;
            } else {
                nextNew[last] = v;
            }
        }
        newIndices[i] = v;
    }

    return {
        indices: newIndices,
        remap: remap.slice(0, count),
        tangents: new Float32Array(tangentBits.buffer.slice(0, count * 16))
    };
}

self.onmessage = async (event) => {
    const { id, positions, normals, texcoords, indices } = event.data;
    try {
        await ready;
        let tangents;
        if (indices === undefined) {
            tangents = generateTangents(positions, normals, texcoords);
        } else {
            tangents = generateTangents(
                unweld(positions, indices, 3),
                unweld(normals, indices, 3),
                unweld(texcoords, indices, 2)
            );
        }

        // convert coordinate system handedness to respect output format of MikkTSpace
        for (let i = 3; i < tangents.length; i += 4) {
            tangents[i] = -tangents[i];
        }

        if (indices === undefined) {
            self.postMessage({ id, tangents }, [tangents.buffer]);
            return;
        }
        const welded = weld(indices, tangents, positions.length / 3);
        self.postMessage({ id, ...welded }, [
            welded.indices.buffer,
            welded.remap.buffer,
            welded.tangents.buffer
        ]);
    } catch (error) {
        self.postMessage({ id, error: String(error) });
    }
};
