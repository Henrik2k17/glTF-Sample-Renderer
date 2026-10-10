import { generateTangentsInWorkers } from "../gltf/tangent_workers.js";
import { decodeMsfsCompiledGeometry, rebindMsfsSkins } from "../gltf/msfs_compiled.js";
import { isMsfsAsset } from "../gltf/msfs.js";
import { addAnimatedMsfsMaterialExtensions } from "../gltf/msfs_material.js";
import { gltfWebGl } from "../Renderer/webgl.js";

class gltfLoader {
    /**
     * @param {String} [tangentWorkerUrl] URL of tangent.worker.js; when set, missing tangents
     * are generated in web workers instead of on the main thread during initGl.
     */
    static async load(
        gltf,
        webGlContext,
        appendix = undefined,
        allowResourceAbsolutePath = true,
        tangentWorkerUrl = undefined
    ) {
        const buffers = gltfLoader.getBuffers(appendix);
        const additionalFiles = gltfLoader.getAdditionalFiles(appendix);

        const buffersPromise = gltfLoader.loadBuffers(
            gltf,
            buffers,
            additionalFiles,
            allowResourceAbsolutePath
        );

        await buffersPromise; // images might be stored in the buffers
        // MSFS 2024 compiled glTFs: shared draw ranges and packed vertices to standard glTF
        decodeMsfsCompiledGeometry(gltf);
        if (isMsfsAsset(gltf)) {
            // MSFS skins bind to the rest pose (see rebindMsfsSkins)
            rebindMsfsSkins(gltf);
            // materials animated through MSFS extensions they lack
            addAnimatedMsfsMaterialExtensions(gltf);
        }
        // Texture files wait as Blobs until uploadDeferredImages (see gltfImage.setImageFromBlob)
        gltf.deferImageData = typeof Blob !== "undefined" && webGlContext !== undefined;
        const imagesPromise = gltfLoader.loadImages(
            gltf,
            additionalFiles,
            allowResourceAbsolutePath
        );
        const tangentsPromise =
            tangentWorkerUrl === undefined
                ? Promise.resolve()
                : generateTangentsInWorkers(gltf, tangentWorkerUrl).catch((error) =>
                      console.warn("Tangent generation in workers failed:", error)
                  );

        await Promise.all([buffersPromise, imagesPromise, tangentsPromise]);
        const result = gltf.initGl(webGlContext);
        if (gltf.deferImageData) {
            await gltfLoader.uploadDeferredImages(gltf, webGlContext);
        }
        return result;
    }

    /**
     * Uploads the textures the materials use, one image at a time: each image is decoded from
     * its Blob, uploaded in every variant (linear / sRGB) its materials need, and released
     * (gltfWebGl.releaseUploadedImage). Images no material uses stay deferred.
     */
    static async uploadDeferredImages(gltf, webGlContext) {
        const webGl = new gltfWebGl(webGlContext);
        const infosByImage = new Map();
        for (const material of gltf.materials ?? []) {
            for (const info of material.textures ?? []) {
                const source = gltf.textures[info.index]?.source;
                if (source !== undefined && gltf.images[source]?.deferredSource !== undefined) {
                    if (!infosByImage.has(source)) {
                        infosByImage.set(source, []);
                    }
                    infosByImage.get(source).push(info);
                }
            }
        }
        // a few images are read and decoded at the same time; uploads run between the awaits
        const queue = [...infosByImage];
        const worker = async () => {
            for (let entry = queue.shift(); entry !== undefined; entry = queue.shift()) {
                const [index, infos] = entry;
                const image = gltf.images[index];
                try {
                    await image.restoreDeferredImage(gltf);
                } catch (error) {
                    console.info(`Image "${image.uri ?? image.name}" could not be decoded: ${error?.message ?? error}`);
                    continue;
                }
                for (const info of infos) {
                    webGl.uploadTexture(gltf, info);
                }
            }
        };
        await Promise.all(Array.from({ length: Math.min(4, queue.length) }, worker));
        webGlContext.bindTexture(webGlContext.TEXTURE_2D, null);
    }

    /**
     * Frees the GPU objects and the CPU copies of a glTF's textures and buffers. The glTF can't
     * be drawn afterwards. Large models (MSFS packages: ~2 GB of textures) otherwise stay in
     * memory after the next model is loaded, until the browser runs out of memory.
     */
    static unload(gltf, webGlContext) {
        const isGlTexture = (object) =>
            typeof WebGLTexture !== "undefined" && object instanceof WebGLTexture;
        for (const image of gltf.images ?? []) {
            const content = image.image;
            if (isGlTexture(content)) {
                // KTX2 textures, joint and morph target textures live on the GPU only
                webGlContext.deleteTexture(content);
            } else {
                content?.close?.(); // ImageBitmap
            }
            image.image = undefined;
        }
        gltf.images = [];

        for (const texture of gltf.textures ?? []) {
            for (const glTexture of [texture.glTexture, texture.glTextureSRGB]) {
                if (isGlTexture(glTexture)) {
                    webGlContext.deleteTexture(glTexture); // deleting twice is allowed
                }
            }
            texture.glTexture = undefined;
            texture.glTextureSRGB = undefined;
        }
        gltf.textures = [];

        for (const accessor of gltf.accessors ?? []) {
            if (accessor.glBuffer !== undefined) {
                webGlContext.deleteBuffer(accessor.glBuffer);
                accessor.glBuffer = undefined;
            }
        }
        gltf.accessors = [];

        for (const buffer of gltf.buffers ?? []) {
            buffer.buffer = undefined;
        }
    }

    static getBuffers(appendix) {
        return gltfLoader.getTypedAppendix(appendix, ArrayBuffer);
    }

    static getAdditionalFiles(appendix) {
        if (typeof File !== "undefined") {
            return gltfLoader.getTypedAppendix(appendix, File);
        } else {
            return;
        }
    }

    static getTypedAppendix(appendix, Type) {
        if (appendix && appendix.length > 0) {
            if (appendix[0] instanceof Type || appendix[0][1] instanceof Type) {
                return appendix;
            }
        }
    }

    static loadBuffers(gltf, buffers, additionalFiles, allowResourceAbsolutePath) {
        const promises = [];

        if (buffers !== undefined && buffers[0] !== undefined) {
            //GLB
            //There is only one buffer for the glb binary data
            //see https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#glb-file-format-specification
            if (buffers.length > 1) {
                console.warn("Too many buffer chunks in GLB file. Only one or zero allowed");
            }

            gltf.buffers[0].buffer = buffers[0];
            for (let i = 1; i < gltf.buffers.length; ++i) {
                promises.push(
                    gltf.buffers[i].load(gltf, additionalFiles, allowResourceAbsolutePath)
                );
            }
        } else {
            for (const buffer of gltf.buffers) {
                promises.push(buffer.load(gltf, additionalFiles, allowResourceAbsolutePath));
            }
        }
        return Promise.all(promises);
    }

    static loadImages(gltf, additionalFiles, allowResourceAbsolutePath) {
        // A bounded number at a time: an image can try several URLs (MSFS texture fallbacks),
        // and thousands of parallel requests (MSFS packages) make the browser fail some with
        // ERR_INSUFFICIENT_RESOURCES, which looks like a missing file.
        const queue = [...gltf.images];
        const worker = async () => {
            for (let image = queue.shift(); image !== undefined; image = queue.shift()) {
                await image.load(gltf, additionalFiles, allowResourceAbsolutePath);
            }
        };
        return Promise.all(Array.from({ length: Math.min(16, queue.length) }, worker));
    }
}

export { gltfLoader };
