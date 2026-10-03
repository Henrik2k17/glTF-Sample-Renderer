import { GltfObject } from "./gltf_object.js";
import { AsyncFileReader } from "../ResourceLoader/async_file_reader.js";
import { GL } from "../Renderer/webgl";
import { ImageMimeType } from "./image_mime_type.js";
import * as jpeg from "jpeg-js";
import * as png from "fast-png";
import { ResourceLoaderUtils } from "../ResourceLoader/loader_utils.js";
import { isDecodedImageType, decodeImageBytes } from "../ResourceLoader/image_decoders.js";

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
        return (
            (await this.setImageFromBufferView(gltf)) ||
            (await this.setImageFromFiles(gltf, additionalFiles)) ||
            (!isMissingDroppedFile &&
                (await this.setImageFromUri(gltf, allowResourceAbsolutePath))) ||
            (await this.setImageFromBase64(gltf))
        );
    }

    isLoaded() {
        return this.image !== undefined;
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

    async setImageFromBytes(gltf, array) {
        if (isDecodedImageType(this.mimeType)) {
            this.image = decodeImageBytes(this.mimeType, array);
        } else if (this.mimeType === ImageMimeType.KTX2) {
            if (gltf.ktxDecoder !== undefined) {
                this.image = await gltf.ktxDecoder.loadKtxFromBuffer(array);
            } else {
                console.warn("Loading of ktx images failed: KtxDecoder not initalized");
            }
        } else if (
            typeof Image !== "undefined" &&
            (this.mimeType === ImageMimeType.JPEG ||
                this.mimeType === ImageMimeType.PNG ||
                this.mimeType === ImageMimeType.WEBP)
        ) {
            const blob = new Blob([array], { type: this.mimeType });
            const objectURL = URL.createObjectURL(blob);
            try {
                this.image = await gltfImage.loadHTMLImage(objectURL);
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

    async setImageFromUri(gltf, allowResourceAbsolutePath) {
        if (this.uri === undefined || this.uri.startsWith("data:")) {
            return false;
        }
        if (!allowResourceAbsolutePath && ResourceLoaderUtils.isAbsoluteUrl(this.uri)) {
            throw new Error("Absolute URLs are not allowed for security reasons: " + this.uri);
        }
        const parentPath = ResourceLoaderUtils.getContainingFolder(gltf.path ?? "");
        const fullPath = parentPath + this.uri;
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
            if (gltf.ktxDecoder !== undefined) {
                this.image = await gltf.ktxDecoder.loadKtxFromUri(fullPath);
            } else {
                console.warn("Loading of ktx images failed: KtxDecoder not initalized");
            }
        } else if (
            typeof Image !== "undefined" &&
            (this.mimeType === ImageMimeType.JPEG ||
                this.mimeType === ImageMimeType.PNG ||
                this.mimeType === ImageMimeType.WEBP)
        ) {
            try {
                this.image = await gltfImage.loadHTMLImage(fullPath);
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

        if (this.mimeType === undefined) {
            this.setMimetypeFromFilename(foundFile[0]);
        }

        if (isDecodedImageType(this.mimeType)) {
            const data = new Uint8Array(await foundFile[1].arrayBuffer());
            this.image = decodeImageBytes(this.mimeType, data);
        } else if (this.mimeType === ImageMimeType.KTX2) {
            if (gltf.ktxDecoder !== undefined) {
                const data = new Uint8Array(await foundFile[1].arrayBuffer());
                this.image = await gltf.ktxDecoder.loadKtxFromBuffer(data);
            } else {
                console.warn("Loading of ktx images failed: KtxDecoder not initalized");
            }
        } else if (
            typeof Image !== "undefined" &&
            (this.mimeType === ImageMimeType.JPEG ||
                this.mimeType === ImageMimeType.PNG ||
                this.mimeType === ImageMimeType.WEBP)
        ) {
            const imageData = await AsyncFileReader.readAsDataURL(foundFile[1]).catch(() => {
                console.error("Could not load image with FileReader");
            });
            try {
                this.image = await gltfImage.loadHTMLImage(imageData);
            } catch {
                console.error("Error while reading image from file " + this.uri);
            }
        } else {
            console.error("Unsupported image type " + this.mimeType);
            return false;
        }

        return true;
    }
}

export { gltfImage, ImageMimeType };
