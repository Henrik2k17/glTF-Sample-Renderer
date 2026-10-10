import { GltfState } from "../GltfState/gltf_state.js";
import { gltfRenderer } from "../Renderer/renderer.js";
import { GL } from "../Renderer/webgl.js";
import { ResourceLoader } from "../ResourceLoader/resource_loader.js";
import { TextureReader } from "../Renderer/texture_reader.js";
import { FrameProfiler } from "../Renderer/frame_profiler.js";
import { gltfMaterial } from "../gltf/material.js";
import { addAnimatedMsfsMaterialExtensions } from "../gltf/msfs_material.js";
import { gltfLoader } from "../ResourceLoader/loader.js";

/**
 * GltfView represents a view on a gltf, e.g. in a canvas
 */
class GltfView {
    /**
     * GltfView representing one WebGl 2.0 context or in other words one
     * 3D rendering of the Gltf.
     * You can create multiple views for example when multiple canvases should
     * be shown on the same webpage.
     * @param {*} context WebGl 2.0 context. Get it from a canvas with `canvas.getContext("webgl2")`
     */
    constructor(context) {
        this.context = context;
        this.renderer = new gltfRenderer(this.context);
        this.lastFrameTime = undefined;
        /** Frame statistics for a performance overlay; set profiler.enabled to collect them. */
        this.profiler = new FrameProfiler(this.context);
        this.renderer.profiler = this.profiler;
    }

    /**
     * createState constructs a new GltfState for the GltfView. The resources
     * referenced in a gltf state can directly be stored as resources on the WebGL
     * context of GltfView, therefore GltfStates cannot not be shared between
     * GltfViews.
     * @returns {GltfState} GltfState
     */
    createState() {
        return new GltfState(this);
    }

    /**
     * createResourceLoader creates a resource loader with which glTFs and
     * environments can be loaded for the view
     * @param {Object} [externalDracoLib] optional object of an external Draco library, e.g. from a CDN
     * @param {Object} [externalKtxLib] optional object of an external KTX library, e.g. from a CDN
     * @param {string} [libPath] optional path to the libraries. Used to define the path to the WASM files on repackaging
     * @returns {ResourceLoader} ResourceLoader
     */
    createResourceLoader(
        externalDracoLib = undefined,
        externalKtxLib = undefined,
        libPath = undefined
    ) {
        let resourceLoader = new ResourceLoader(this, libPath);
        resourceLoader.initKtxLib(externalKtxLib);
        resourceLoader.initDracoLib(externalDracoLib);
        return resourceLoader;
    }

    /**
     * Reads a texture of the state's glTF back from the GPU as 8-bit RGBA, scaled down so its
     * longer side is at most maxSize pixels. Values are as stored in the image (sRGB textures
     * are not linearized). Row 0 is the first row of the image.
     * @param {GltfState} state
     * @param {number} textureIndex index into state.gltf.textures
     * @param {number} [maxSize]
     * @returns {{width, height, pixels: Uint8ClampedArray, sourceWidth, sourceHeight} | undefined}
     */
    readTexture(state, textureIndex, maxSize = 256) {
        if (state.gltf === undefined) {
            return undefined;
        }
        this.textureReader ??= new TextureReader(this.renderer.webGl);
        return this.textureReader.read(state.gltf, textureIndex, maxSize);
    }

    /**
     * Replaces a material of the loaded glTF with one built from glTF material JSON, e.g. for a
     * material editor. Textures the new material uses that were not uploaded yet are uploaded;
     * the draw lists are rebuilt (alpha mode or MSFS material type may have changed).
     * A texture whose image data was already released after upload can only be used again in
     * the same color space (linear / sRGB) it was uploaded in.
     * @param {GltfState} state
     * @param {number} materialIndex
     * @param {object} json glTF material JSON (texture indices into state.gltf.textures)
     */
    async replaceMaterial(state, materialIndex, json) {
        const gltf = state.gltf;
        if (gltf?.materials[materialIndex] === undefined) {
            return;
        }
        const material = new gltfMaterial();
        material.fromJson(structuredClone(json));
        material.gltfObjectIndex = materialIndex;
        gltf.materials[materialIndex] = material;
        // materials animated through MSFS extensions they lack (see the loader)
        addAnimatedMsfsMaterialExtensions(gltf);
        material.initGl(gltf, this.context);
        await gltfLoader.uploadDeferredImages(gltf, this.context);
        state.materialsVersion = (state.materialsVersion ?? 0) + 1;
    }

    /**
     * renderFrame to the context's default frame buffer
     * Call this function in the javascript animation update loop for continuous rendering to a canvas
     * @param {*} state GltfState that is be used for rendering
     * @param {*} width of the viewport
     * @param {*} height of the viewport
     */
    renderFrame(state, width, height) {
        this.profiler.beginFrame();
        try {
            this._renderFrame(state, width, height);
        } finally {
            this.profiler.endFrame();
        }
    }

    _renderFrame(state, width, height) {
        const profiler = this.profiler;
        const lastFrameTime =
            this.lastFrameTime === undefined ? performance.now() : this.lastFrameTime;
        this.lastFrameTime = performance.now();
        const currentFrameTime = performance.now();
        this.renderer.init(state);
        let start = profiler.now();
        this._animate(state);
        profiler.time("animation", start);

        this.renderer.resize(width, height);

        this.renderer.clearFrame(state.renderingParameters.clearColor);

        if (state.gltf === undefined) {
            return;
        }

        const scene = state.gltf.scenes[state.sceneIndex];

        if (scene === undefined) {
            return;
        }

        if (state.graphController?.playing) {
            state.graphController.simulateTick();
        }

        start = profiler.now();
        scene.applyTransformHierarchy(state.gltf);
        profiler.time("transforms", start);
        if (state.physicsController.playing && state.physicsController.enabled) {
            start = profiler.now();
            state.physicsController.simulateStep(state, (currentFrameTime - lastFrameTime) / 1000);
            profiler.time("physics", start);
        }

        this.renderer.drawScene(state, scene);

        // We do not want to reset the dirty flags when the physics simulation is paused, since changes from interactivity would not be applied after resuming the simulation.
        if (
            (state.physicsController.playing && state.physicsController.enabled) ||
            !state.physicsController.enabled
        ) {
            state.gltf.resetAllDirtyFlags();
        }
    }

    /**
     * gatherStatistics collects information about the GltfState such as the number of
     * rendered meshes or triangles
     * @param {*} state GltfState about which the statistics should be collected
     * @returns {Object} an object containing statistics information
     */
    gatherStatistics(state) {
        if (state.gltf === undefined) {
            return {
                meshCount: 0,
                faceCount: 0,
                opaqueMaterialsCount: 0,
                transparentMaterialsCount: 0
            };
        }

        // gather information from the active scene
        const scene = state.gltf.scenes[state.sceneIndex];
        if (scene === undefined) {
            return {
                meshCount: 0,
                faceCount: 0,
                opaqueMaterialsCount: 0,
                transparentMaterialsCount: 0
            };
        }
        const nodes = scene.gatherNodes(
            state.gltf,
            state.renderingParameters.enabledExtensions
        ).nodes;
        const activeMeshes = nodes
            .filter((node) => node.mesh !== undefined)
            .map((node) => state.gltf.meshes[node.mesh]);
        const activePrimitives = activeMeshes
            .reduce((acc, mesh) => acc.concat(mesh.primitives), [])
            .filter((primitive) => primitive.material !== undefined);
        const activeMaterials = [
            ...new Set(
                activePrimitives.map((primitive) => state.gltf.materials[primitive.material])
            )
        ];
        const opaqueMaterials = activeMaterials.filter(
            (material) => material.alphaMode !== "BLEND"
        );
        const transparentMaterials = activeMaterials.filter(
            (material) => material.alphaMode === "BLEND"
        );
        const faceCount = activePrimitives
            .map((primitive) => {
                let vertexCount = 0;
                if (primitive.indices !== undefined) {
                    vertexCount = state.gltf.accessors[primitive.indices].count;
                } else {
                    vertexCount = state.gltf.accessors[primitive.attributes["POSITION"]].count;
                }
                if (vertexCount === 0) {
                    return 0;
                }

                // convert vertex count to point, line or triangle count
                switch (primitive.mode) {
                    case GL.POINTS:
                        return vertexCount;
                    case GL.LINES:
                        return vertexCount / 2;
                    case GL.LINE_LOOP:
                        return vertexCount;
                    case GL.LINE_STRIP:
                        return vertexCount - 1;
                    case GL.TRIANGLES:
                        return vertexCount / 3;
                    case GL.TRIANGLE_STRIP:
                    case GL.TRIANGLE_FAN:
                        return vertexCount - 2;
                }
            })
            .reduce((acc, faceCount) => acc + faceCount, 0);

        // assemble statistics object
        return {
            meshCount: activeMeshes.length,
            faceCount: faceCount,
            opaqueMaterialsCount: opaqueMaterials.length,
            transparentMaterialsCount: transparentMaterials.length
        };
    }

    _animate(state) {
        if (state.gltf === undefined || state.gltf.animations === undefined) {
            return;
        }
        let disabledAnimations = [];
        let enabledAnimations = [];

        if (
            state.gltf?.extensions?.KHR_interactivity !== undefined &&
            state.renderingParameters.enabledExtensions.KHR_interactivity
        ) {
            if (state.graphController.playing) {
                for (const animation of state.gltf.animations) {
                    if (animation.createdTimestamp !== undefined) {
                        enabledAnimations.push(animation);
                    }
                }
            }
        } else if (state.animationIndices !== undefined) {
            disabledAnimations = state.gltf.animations.filter((anim, index) => {
                return false === state.animationIndices.includes(index);
            });
            enabledAnimations = state.animationIndices
                .map((index) => {
                    return state.gltf.animations[index];
                })
                .filter((animation) => animation !== undefined);
            for (const animation of enabledAnimations) {
                if (animation.createdTimestamp !== undefined) {
                    animation.reset();
                }
            }
        }

        for (const disabledAnimation of disabledAnimations) {
            disabledAnimation.advance(state.gltf, undefined);
        }

        const t = state.animationTimer.elapsedSec();

        for (const animation of enabledAnimations) {
            let time = state.animationTimeOverrides?.get(animation.gltfObjectIndex);
            if (time === undefined) {
                time = t;
            } else {
                // An override is an exact position, not playback: clamp instead of looping.
                // Otherwise a time a hair past the end (e.g. frame 100 / 30 fps vs. the float32
                // max of the input accessor) wraps around to the first keyframe.
                animation.computeMinMaxTime(state.gltf);
                if (animation.minTime !== undefined && animation.maxTime !== undefined) {
                    time = Math.min(Math.max(time, animation.minTime), animation.maxTime);
                }
            }
            animation.advance(state.gltf, time);
        }
    }
}

export { GltfView };
