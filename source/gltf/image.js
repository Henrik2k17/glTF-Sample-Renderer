import { GltfObject } from "./gltf_object.js";
import { GL } from "../Renderer/webgl";
import { ImageMimeType } from "./image_mime_type.js";
import * as jpeg from "jpeg-js";
import * as png from "fast-png";
import { ResourceLoaderUtils } from "../ResourceLoader/loader_utils.js";
import {
    isDecodedImageType,
    decodeImageBytes,
    isBlockCompressedKtx2,
    decodeBlockCompressedKtx2,
    describeBlockCompressedKtx2,
    Ktx2HeaderBytes,
    skippedMipLevels,
    limitedImageSize,
    downsampleRgba
} from "../ResourceLoader/image_decoders.js";

class gltfImage extends GltfObject {
    static animatedProperties = [];
    constructor(
        uri = undefined,
        type = GL.TEXTURE_2D,
        miplevel = 0,
        bufferView = undefined,
        name = undefined,
        mimeType = undefined,
        image = undefined
    ) {
        super();
        this.uri = uri;
        this.bufferView = bufferView;
        this.mimeType = mimeType;
        this.image = image; // javascript image
        this.name = name;
        this.type = type; // nonstandard
        this.miplevel = miplevel; // nonstandard
    }

    async load(gltf, additionalFiles = undefined, allowResourceAbsolutePath) {
        if (this.image !== undefined) {
            if (this.mimeType !== ImageMimeType.GLTEXTURE) {
                console.error("image has already been loaded");
            }
            return;
        }

        // A missing or undecodable image must not abort loading the whole model. Textures that
        // reference an unloaded image are skipped by the material (see gltfMaterial.initGl).
        // Logged as info: the sample viewer turns warnings into UI toasts, and a model with many
        // missing textures would otherwise flood the UI. Callers can check isLoaded() instead.
        let reason = "";
        try {
            await this.loadFromSources(gltf, additionalFiles, allowResourceAbsolutePath);
        } catch (error) {
            reason = `: ${error?.message ?? error}`;
        }
        if (this.image === undefined) {
            console.info(`Image "${this.uri ?? this.name}" could not be loaded${reason}`);
        }
    }

    async loadFromSources(gltf, additionalFiles, allowResourceAbsolutePath) {
        // For dropped models, a relative URI that is not among the dropped files would only
        // resolve against the viewer's own server, so don't fetch it.
        const isMissingDroppedFile =
            additionalFiles !== undefined &&
            this.uri !== undefined &&
            !this.uri.startsWith("data:") &&
            !ResourceLoaderUtils.isAbsoluteUrl(this.uri);
        if (
            (await this.setImageFromBufferView(gltf)) ||
            (await this.setImageFromFiles(gltf, additionalFiles))
        ) {
            return true;
        }
        // A URI that cannot be fetched may still be found by the texture file resolver.
        let uriError = undefined;
        if (!isMissingDroppedFile) {
            try {
                if (await this.setImageFromUri(gltf, allowResourceAbsolutePath)) {
                    return true;
                }
            } catch (error) {
                uriError = error;
            }
        }
        if (!isMissingDroppedFile && (await this.setImageFromTextureCfg(gltf))) {
            return true;
        }
        if (await this.setImageFromResolver(gltf)) {
            return true;
        }
        if (uriError !== undefined) {
            throw uriError;
        }
        return await this.setImageFromBase64(gltf);
    }

    /**
     * Asks gltf.textureFileResolver (see ResourceLoader.textureFileResolver) for a file.
     * The MIME type follows the resolved file, which may differ from the URI (e.g. x.png.dds).
     */
    async setImageFromResolver(gltf) {
        if (
            gltf.textureFileResolver === undefined ||
            typeof this.uri !== "string" ||
            this.uri.startsWith("data:")
        ) {
            return false;
        }
        const resolved = await gltf.textureFileResolver(this.uri);
        if (resolved === undefined) {
            return false;
        }
        const [name, file] = resolved;
        const uriMimeType = this.mimeType;
        this.mimeType = undefined;
        if (await this.setImageFromFile(gltf, name, file)) {
            return true;
        }
        this.mimeType = uriMimeType;
        return false;
    }

    isLoaded() {
        return this.image !== undefined;
    }

    /**
     * Decodes a PNG, JPEG or WebP image from a URL or Blob. Uses createImageBitmap where
     * available, which decodes off the main thread; an HTMLImageElement would only be decoded
     * synchronously during the first texture upload.
     * @param {String | Blob} source
     * @returns {Promise<ImageBitmap | HTMLImageElement>}
     */
    static async loadBrowserImage(source) {
        if (typeof createImageBitmap === "undefined") {
            if (source instanceof Blob) {
                const objectURL = URL.createObjectURL(source);
                try {
                    return await gltfImage.loadHTMLImage(objectURL);
                } finally {
                    URL.revokeObjectURL(objectURL);
                }
            }
            return await gltfImage.loadHTMLImage(source);
        }
        let blob = source;
        if (!(source instanceof Blob)) {
            const response = await fetch(source);
            if (!response.ok) {
                throw new Error(`Could not load image from ${source}`);
            }
            blob = await response.blob();
        }
        // Match the WebGL unpack defaults that applied to HTMLImageElement uploads.
        return await createImageBitmap(blob, {
            premultiplyAlpha: "none",
            colorSpaceConversion: "default"
        });
    }

    static loadHTMLImage(url) {
        return new Promise((resolve, reject) => {
            const image = new Image();
            image.addEventListener("load", () => resolve(image));
            image.addEventListener("error", () => reject());
            image.src = url;
            image.crossOrigin = "";
        });
    }

    setMimetypeFromFilename(filename) {
        let extension = ResourceLoaderUtils.getExtension(filename);
        if (extension == "ktx2" || extension == "ktx") {
            this.mimeType = ImageMimeType.KTX2;
        } else if (extension == "jpg" || extension == "jpeg") {
            this.mimeType = ImageMimeType.JPEG;
        } else if (extension == "png") {
            this.mimeType = ImageMimeType.PNG;
        } else if (extension == "webp") {
            this.mimeType = ImageMimeType.WEBP;
        } else if (extension == "dds") {
            this.mimeType = ImageMimeType.DDS;
        } else if (extension == "tga") {
            this.mimeType = ImageMimeType.TGA;
        } else if (extension == "tif" || extension == "tiff") {
            this.mimeType = ImageMimeType.TIFF;
        } else {
            console.warn("MimeType not defined");
            // assume jpeg encoding as best guess
            this.mimeType = ImageMimeType.JPEG;
        }
    }

    /** Image types setImageFromBlob handles (browser images only where they can be decoded). */
    static isBlobDecodable(mimeType) {
        return (
            isDecodedImageType(mimeType) ||
            mimeType === ImageMimeType.KTX2 ||
            (typeof Image !== "undefined" &&
                (mimeType === ImageMimeType.JPEG ||
                    mimeType === ImageMimeType.PNG ||
                    mimeType === ImageMimeType.WEBP))
        );
    }

    /** Decodes a texture file (this.mimeType) into this.image. */
    async decodeBlob(gltf, blob) {
        if (isDecodedImageType(this.mimeType)) {
            this.image = decodeImageBytes(this.mimeType, new Uint8Array(await blob.arrayBuffer()));
        } else if (this.mimeType === ImageMimeType.KTX2) {
            await this.setKtx2FromBytes(gltf, new Uint8Array(await blob.arrayBuffer()));
        } else {
            this.image = await gltfImage.loadBrowserImage(blob);
        }
        await this.limitSize(gltf);
    }

    /**
     * Shrinks this.image to gltf.maxTextureSize (largest width or height; 0 or undefined: no
     * limit), e.g. to fit an MSFS package's textures (~4 GB) into GPU memory. Block-compressed
     * images lose their larger mip levels, so they look like the full texture seen from further
     * away; browser images are resized (Chrome keeps the colour of transparent texels), RGBA8
     * data is box filtered. Compressed images without mip levels stay as they are.
     */
    async limitSize(gltf) {
        const maxSize = gltf.maxTextureSize ?? 0;
        const image = this.image;
        if (
            !(maxSize > 0) ||
            image === undefined ||
            image.released ||
            Math.max(image.width, image.height) <= maxSize
        ) {
            return;
        }
        if (image.compressed?.levels !== undefined) {
            const levels = image.compressed.levels;
            const skip = skippedMipLevels(levels[0].width, levels[0].height, levels.length, maxSize);
            if (skip > 0) {
                image.compressed.levels = levels.slice(skip);
                image.width = levels[skip].width;
                image.height = levels[skip].height;
            }
        } else if (
            typeof createImageBitmap !== "undefined" &&
            ((typeof ImageBitmap !== "undefined" && image instanceof ImageBitmap) ||
                (typeof HTMLImageElement !== "undefined" && image instanceof HTMLImageElement))
        ) {
            const [width, height] = limitedImageSize(image.width, image.height, maxSize);
            this.image = await createImageBitmap(image, {
                resizeWidth: width,
                resizeHeight: height,
                resizeQuality: "high",
                premultiplyAlpha: "none",
                colorSpaceConversion: "default"
            });
            image.close?.();
        } else if (image.data instanceof Uint8Array && image.data.length === image.width * image.height * 4) {
            this.image = downsampleRgba(image, maxSize);
        }
    }

    /**
     * Loads a texture file. With gltf.deferImageData, only its size and format stay in memory
     * (a placeholder with released: true) and the file is kept as a Blob, which the browser
     * may keep outside the page's memory, until gltfLoader.uploadDeferredImages decodes and
     * uploads one image at a time. Otherwise all of a model's decoded textures (MSFS
     * packages: ~4 GB) are in memory at once before the first draw uploads them.
     */
    async setImageFromBlob(gltf, blob) {
        if (!gltf.deferImageData) {
            await this.decodeBlob(gltf, blob);
            return true;
        }
        if (this.mimeType === ImageMimeType.KTX2) {
            // MSFS textures: size and format are in the header; no need to read the file twice
            const header = new Uint8Array(await blob.slice(0, Ktx2HeaderBytes).arrayBuffer());
            const { byteCount, ...description } =
                describeBlockCompressedKtx2(header, blob.size, gltf.maxTextureSize) ?? {};
            if (byteCount !== undefined) {
                this.image = { ...description, released: true };
                // levels above the size limit are at the end of the file
                this.deferredSource = byteCount < blob.size ? blob.slice(0, byteCount) : blob;
                return true;
            }
        }
        if (this.mimeType === ImageMimeType.PNG) {
            // the size is in the header (IHDR); no need to decode the image twice
            const header = new DataView(await blob.slice(0, 24).arrayBuffer());
            if (header.byteLength === 24 && header.getUint32(12) === 0x49484452) {
                const [width, height] = limitedImageSize(header.getUint32(16), header.getUint32(20), gltf.maxTextureSize);
                this.image = { width, height, released: true };
                this.deferredSource = blob;
                return true;
            }
        }
        await this.decodeBlob(gltf, blob);
        const content = this.image;
        let placeholder = undefined;
        if (content?.compressed?.levels !== undefined) {
            const { format, levels } = content.compressed;
            const byteLength = levels.reduce((sum, level) => sum + level.data.byteLength, 0);
            placeholder = { width: content.width, height: content.height, compressed: { format, byteLength } };
        } else if (typeof ImageBitmap !== "undefined" && content instanceof ImageBitmap) {
            placeholder = { width: content.width, height: content.height };
            content.close();
        } else if (isDecodedImageType(this.mimeType) && content?.data !== undefined) {
            placeholder = { width: content.width, height: content.height };
        }
        if (placeholder !== undefined) {
            placeholder.released = true;
            this.image = placeholder;
            this.deferredSource = blob;
        }
        return true;
    }

    /**
     * For gltf.deferImageData: describes a KTX2 (block-compressed) or PNG image from the start
     * of the file, requested with a Range header; the upload fetches the whole file. Keeping
     * whole responses as Blobs instead made the browser write gigabytes to its blob storage
     * (MSFS packages), which was slow.
     * @returns {Promise<boolean>} whether the image was described; throws if the URL fails
     */
    async describeFromUrl(gltf, url) {
        const headerBytes =
            this.mimeType === ImageMimeType.KTX2 ? Ktx2HeaderBytes : this.mimeType === ImageMimeType.PNG ? 24 : 0;
        if (headerBytes === 0) {
            return false;
        }
        const response = await fetch(url, { headers: { Range: `bytes=0-${headerBytes - 1}` } });
        if (!response.ok) {
            throw new Error(`Could not load image from ${url}`);
        }
        if (response.body === null || response.body === undefined) {
            return false;
        }
        // 206: just the range, its Content-Range has the file size; 200: the server sent it all
        const header = await gltfImage.readPrefix(response, headerBytes);
        const range = /\/(\d+)$/.exec(response.headers.get("content-range") ?? "");
        const contentLength = Number(response.headers.get("content-length"));
        const fileSize = range
            ? Number(range[1])
            : response.status === 200 && contentLength > 0
              ? contentLength
              : Number.MAX_SAFE_INTEGER;
        let description = undefined;
        let byteCount = undefined;
        if (this.mimeType === ImageMimeType.KTX2) {
            description = describeBlockCompressedKtx2(header, fileSize, gltf.maxTextureSize);
            if (description !== undefined) {
                ({ byteCount, ...description } = description);
            }
        } else if (header.byteLength === 24) {
            const view = new DataView(header.buffer, header.byteOffset, header.byteLength);
            if (view.getUint32(12) === 0x49484452) {
                const [width, height] = limitedImageSize(view.getUint32(16), view.getUint32(20), gltf.maxTextureSize);
                description = { width, height };
            }
        }
        if (description === undefined) {
            return false;
        }
        this.image = { ...description, released: true };
        // KTX2 levels above the size limit are at the end of the file: no need to download them
        this.deferredSource = { url, byteCount: byteCount < fileSize ? byteCount : undefined };
        return true;
    }

    /** The first byteCount bytes of a response body (fewer if it is shorter); cancels the rest. */
    static async readPrefix(response, byteCount) {
        const reader = response.body.getReader();
        const chunks = [];
        let length = 0;
        while (length < byteCount) {
            const { done, value } = await reader.read();
            if (done) {
                break;
            }
            chunks.push(value);
            length += value.byteLength;
        }
        reader.cancel().catch(() => {});
        const prefix = new Uint8Array(Math.min(length, byteCount));
        let offset = 0;
        for (const chunk of chunks) {
            const part = chunk.subarray(0, prefix.byteLength - offset);
            prefix.set(part, offset);
            offset += part.byteLength;
            if (offset >= prefix.byteLength) {
                break;
            }
        }
        return prefix;
    }

    /** Decodes an image loaded with gltf.deferImageData again, for its upload. */
    async restoreDeferredImage(gltf) {
        const source = this.deferredSource;
        if (source === undefined) {
            return false;
        }
        this.deferredSource = undefined;
        if (source.url === undefined) {
            await this.decodeBlob(gltf, source);
            return true;
        }
        const headers = source.byteCount === undefined ? undefined : { Range: `bytes=0-${source.byteCount - 1}` };
        const response = await fetch(source.url, { headers });
        if (!response.ok) {
            throw new Error(`Could not load image from ${source.url}`);
        }
        if (this.mimeType === ImageMimeType.KTX2) {
            await this.setKtx2FromBytes(gltf, new Uint8Array(await response.arrayBuffer()));
        } else {
            await this.decodeBlob(gltf, await response.blob());
        }
        return true;
    }

    /**
     * KTX2 with plain block-compressed data (e.g. textures compiled by MSFS 2024) is uploaded
     * as is; other KTX2 files (Basis Universal) are transcoded by libktx.
     */
    async setKtx2FromBytes(gltf, array) {
        if (isBlockCompressedKtx2(array)) {
            this.image = decodeBlockCompressedKtx2(array, gltf.maxTextureSize);
        } else if (gltf.ktxDecoder !== undefined) {
            this.image = await gltf.ktxDecoder.loadKtxFromBuffer(array);
        } else {
            console.warn("Loading of ktx images failed: KtxDecoder not initalized");
        }
    }

    async setImageFromBytes(gltf, array) {
        if (isDecodedImageType(this.mimeType)) {
            this.image = decodeImageBytes(this.mimeType, array);
        } else if (this.mimeType === ImageMimeType.KTX2) {
            await this.setKtx2FromBytes(gltf, array);
        } else if (
            typeof Image !== "undefined" &&
            (this.mimeType === ImageMimeType.JPEG ||
                this.mimeType === ImageMimeType.PNG ||
                this.mimeType === ImageMimeType.WEBP)
        ) {
            const blob = new Blob([array], { type: this.mimeType });
            try {
                this.image = await gltfImage.loadBrowserImage(blob);
            } catch {
                throw new Error(`Could not load image "${this.name}" from buffer`);
            }
        } else if (this.mimeType === ImageMimeType.JPEG) {
            this.image = jpeg.decode(array, { useTArray: true });
        } else if (this.mimeType === ImageMimeType.PNG) {
            this.image = png.decode(array);
        } else {
            console.error("Unsupported image type " + this.mimeType);
            return false;
        }

        await this.limitSize(gltf);
        return true;
    }

    async setImageFromBase64(gltf) {
        if (this.uri === undefined || !this.uri.startsWith("data:")) {
            return false;
        }
        const parts = this.uri.split(",");
        if (this.mimeType === undefined) {
            switch (parts[0]) {
                case "data:image/jpeg;base64":
                    this.mimeType = ImageMimeType.JPEG;
                    break;
                case "data:image/png;base64":
                    this.mimeType = ImageMimeType.PNG;
                    break;
                case "data:image/webp;base64":
                    this.mimeType = ImageMimeType.WEBP;
                    break;
                case "data:image/ktx2;base64":
                    this.mimeType = ImageMimeType.KTX2;
                    break;
                default:
                    console.warn(`Data URI ${parts[0]} not supported`);
                    return false;
            }
        }
        const res = await fetch(this.uri);
        const buffer = await res.arrayBuffer();
        return await this.setImageFromBytes(gltf, new Uint8Array(buffer));
    }

    /**
     * MSFS packages list shared texture folders in texture.cfg next to a model's textures
     * ([fltsim] fallback.1=..\..\Assets\texture, ...). The sim looks for a texture by file name
     * in the model's texture folder (model/../texture) and then in these fallbacks; compiled
     * glTFs even declare that only the file name of an image URI counts
     * (ASOBO_asset_optimized.UseOnlyFilenameForImageURI). Done here for models loaded from a URL,
     * after the URI itself; the URI's own folder and its texture.cfg are tried last.
     */
    async setImageFromTextureCfg(gltf) {
        const modelPath = this.sourcePath(gltf);
        if (typeof this.uri !== "string" || this.uri.startsWith("data:") || !modelPath) {
            return false;
        }
        const uri = this.uri.replace(/\\/g, "/");
        const fileName = uri.substring(uri.lastIndexOf("/") + 1);
        let textureFolders;
        try {
            const modelUrl = new URL(modelPath, globalThis.location?.href);
            // An attachment can select texture variants (texture.<name>, searched first)
            textureFolders = [
                ...(this.extras?.textureFolders ?? ["../texture/"]).map(
                    (folder) => new URL(folder, modelUrl)
                ),
                new URL(uri.substring(0, uri.lastIndexOf("/") + 1), modelUrl)
            ];
        } catch {
            return false;
        }
        gltf.msfsTextureFallbacks ??= new Map();
        const folders = [];
        for (const textureFolder of textureFolders) {
            const key = textureFolder.href;
            if (!gltf.msfsTextureFallbacks.has(key)) {
                gltf.msfsTextureFallbacks.set(key, gltfImage.readTextureCfg(textureFolder));
            }
            const fallbacks = await gltf.msfsTextureFallbacks.get(key);
            if (fallbacks === null) {
                // the request failed (not a missing file): ask again next time
                gltf.msfsTextureFallbacks.delete(key);
            }
            folders.push(textureFolder, ...(fallbacks ?? []));
        }
        const tried = new Set([new URL(uri, new URL(modelPath, globalThis.location?.href)).href]);
        for (const folder of folders) {
            const candidate = new URL(fileName, folder).href;
            if (tried.has(candidate)) {
                continue;
            }
            tried.add(candidate);
            try {
                if (await this.setImageFromUrl(gltf, candidate)) {
                    return true;
                }
            } catch {
                // not in this folder
            }
        }
        return false;
    }

    /**
     * @returns {Promise<URL[] | null>} the fallback folders of the texture.cfg in a folder, in
     *   order; null if the request failed
     */
    static async readTextureCfg(textureFolder) {
        let response;
        try {
            response = await fetch(new URL("texture.cfg", textureFolder));
        } catch {
            return textureFolder.protocol.startsWith("http") ? null : [];
        }
        try {
            if (!response.ok) {
                return [];
            }
            return (await response.text())
                .split(/\r?\n/)
                .map((line) => /^\s*fallback\.(\d+)\s*=\s*(.+?)\s*$/i.exec(line))
                .filter((match) => match !== null)
                .sort((a, b) => Number(a[1]) - Number(b[1]))
                .map(
                    (match) =>
                        new URL(match[2].replace(/\\/g, "/").replace(/\/?$/, "/"), textureFolder)
                );
        } catch {
            return [];
        }
    }

    async setImageFromUri(gltf, allowResourceAbsolutePath) {
        if (this.uri === undefined || this.uri.startsWith("data:")) {
            return false;
        }
        if (!allowResourceAbsolutePath && ResourceLoaderUtils.isAbsoluteUrl(this.uri)) {
            throw new Error("Absolute URLs are not allowed for security reasons: " + this.uri);
        }
        const parentPath = ResourceLoaderUtils.getContainingFolder(this.sourcePath(gltf) ?? "");
        return await this.setImageFromUrl(gltf, parentPath + this.uri);
    }

    /**
     * Path the URI is relative to: the glTF's own, or for glTFs merged into one (MSFS
     * packages) the path of the file the image came from (extras.sourcePath).
     */
    sourcePath(gltf) {
        return this.extras?.sourcePath ?? gltf.path;
    }

    async setImageFromUrl(gltf, fullPath) {
        if (this.mimeType === undefined) {
            this.setMimetypeFromFilename(this.uri);
        }

        if (gltfImage.isBlobDecodable(this.mimeType) && typeof Blob !== "undefined") {
            if (gltf.deferImageData && (await this.describeFromUrl(gltf, fullPath))) {
                return true;
            }
            const response = await fetch(fullPath);
            if (!response.ok) {
                throw new Error(`Could not load image from ${fullPath}`);
            }
            try {
                return await this.setImageFromBlob(gltf, await response.blob());
            } catch (error) {
                throw new Error(`Could not load image from ${fullPath}: ${error?.message ?? error}`);
            }
        } else if (this.mimeType === ImageMimeType.JPEG && this.uri instanceof ArrayBuffer) {
            this.image = jpeg.decode(this.uri, { useTArray: true });
        } else if (this.mimeType === ImageMimeType.PNG && this.uri instanceof ArrayBuffer) {
            this.image = png.decode(this.uri);
        } else {
            console.error("Unsupported image type " + this.mimeType);
            return false;
        }

        return true;
    }

    async setImageFromBufferView(gltf) {
        const view = gltf.bufferViews[this.bufferView];
        if (view === undefined) {
            return false;
        }

        const buffer = gltf.buffers[view.buffer].buffer;
        const array = new Uint8Array(buffer, view.byteOffset, view.byteLength);
        return await this.setImageFromBytes(gltf, array);
    }

    async setImageFromFiles(gltf, files) {
        const foundFile = ResourceLoaderUtils.findFile(files, this.uri, gltf.path);
        if (foundFile === undefined) {
            return false;
        }
        return await this.setImageFromFile(gltf, foundFile[0], foundFile[1]);
    }

    async setImageFromFile(gltf, name, file) {
        if (this.mimeType === undefined) {
            this.setMimetypeFromFilename(name);
        }

        if (gltfImage.isBlobDecodable(this.mimeType)) {
            try {
                return await this.setImageFromBlob(gltf, file);
            } catch {
                console.error("Error while reading image from file " + name);
            }
        } else {
            console.error("Unsupported image type " + this.mimeType);
            return false;
        }

        return true;
    }
}

export { gltfImage, ImageMimeType };
