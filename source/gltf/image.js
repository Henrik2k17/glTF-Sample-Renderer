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
    decodeBlockCompressedKtx2
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

    /**
     * KTX2 with plain block-compressed data (e.g. textures compiled by MSFS 2024) is uploaded
     * as is; other KTX2 files (Basis Universal) are transcoded by libktx.
     */
    async setKtx2FromBytes(gltf, array) {
        if (isBlockCompressedKtx2(array)) {
            this.image = decodeBlockCompressedKtx2(array);
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
        if (typeof this.uri !== "string" || this.uri.startsWith("data:") || !gltf.path) {
            return false;
        }
        const uri = this.uri.replace(/\\/g, "/");
        const fileName = uri.substring(uri.lastIndexOf("/") + 1);
        let textureFolders;
        try {
            const modelUrl = new URL(gltf.path, globalThis.location?.href);
            textureFolders = [
                new URL("../texture/", modelUrl),
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
            folders.push(textureFolder, ...(await gltf.msfsTextureFallbacks.get(key)));
        }
        const tried = new Set([new URL(uri, new URL(gltf.path, globalThis.location?.href)).href]);
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

    /** @returns {Promise<URL[]>} the fallback folders of the texture.cfg in a folder, in order */
    static async readTextureCfg(textureFolder) {
        try {
            const response = await fetch(new URL("texture.cfg", textureFolder));
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
        const parentPath = ResourceLoaderUtils.getContainingFolder(gltf.path ?? "");
        return await this.setImageFromUrl(gltf, parentPath + this.uri);
    }

    async setImageFromUrl(gltf, fullPath) {
        if (this.mimeType === undefined) {
            this.setMimetypeFromFilename(this.uri);
        }

        if (isDecodedImageType(this.mimeType)) {
            const response = await fetch(fullPath);
            if (!response.ok) {
                throw new Error(`Could not load image from ${fullPath}`);
            }
            this.image = decodeImageBytes(
                this.mimeType,
                new Uint8Array(await response.arrayBuffer())
            );
        } else if (this.mimeType === ImageMimeType.KTX2) {
            const response = await fetch(fullPath);
            if (!response.ok) {
                throw new Error(`Could not load image from ${fullPath}`);
            }
            await this.setKtx2FromBytes(gltf, new Uint8Array(await response.arrayBuffer()));
        } else if (
            typeof Image !== "undefined" &&
            (this.mimeType === ImageMimeType.JPEG ||
                this.mimeType === ImageMimeType.PNG ||
                this.mimeType === ImageMimeType.WEBP)
        ) {
            try {
                this.image = await gltfImage.loadBrowserImage(fullPath);
            } catch {
                throw new Error(`Could not load image from ${fullPath}`);
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

        if (isDecodedImageType(this.mimeType)) {
            const data = new Uint8Array(await file.arrayBuffer());
            this.image = decodeImageBytes(this.mimeType, data);
        } else if (this.mimeType === ImageMimeType.KTX2) {
            await this.setKtx2FromBytes(gltf, new Uint8Array(await file.arrayBuffer()));
        } else if (
            typeof Image !== "undefined" &&
            (this.mimeType === ImageMimeType.JPEG ||
                this.mimeType === ImageMimeType.PNG ||
                this.mimeType === ImageMimeType.WEBP)
        ) {
            try {
                this.image = await gltfImage.loadBrowserImage(file);
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
