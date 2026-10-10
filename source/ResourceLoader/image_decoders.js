// Decoders for image formats browsers cannot load natively: DDS, TGA and TIFF, and KTX2 files
// that hold plain block-compressed data. MSFS exporters reference source textures in these
// formats (DDS via MSFT_texture_dds); MSFS 2024 package builds compile textures to such KTX2.
//
// Decoded images are plain objects:
// - { width, height, data: Uint8Array } with tightly packed RGBA8 rows, top row first, or
// - { width, height, compressed: { format, levels: [{ width, height, data }] } } for
//   block-compressed DDS data that is uploaded to the GPU as is.

import UTIF from "utif";
import { ImageMimeType } from "../gltf/image_mime_type.js";

/**
 * Block-compressed formats, keyed by the name used in compressed.format.
 * blockBytes: bytes per 4x4 block. channels: number of meaningful channels.
 */
const CompressedFormats = {
    BC1: { blockBytes: 8, channels: 4 },
    BC2: { blockBytes: 16, channels: 4 },
    BC3: { blockBytes: 16, channels: 4 },
    BC4: { blockBytes: 8, channels: 1 },
    BC5: { blockBytes: 16, channels: 2 },
    BC7: { blockBytes: 16, channels: 4 },
    // signed variants: sampled as -1..1 (e.g. MSFS normal maps)
    BC4_SNORM: { blockBytes: 8, channels: 1 },
    BC5_SNORM: { blockBytes: 16, channels: 2 }
};

// KTX2 vkFormat values of the block-compressed formats above
const VkFormats = {
    131: "BC1", // BC1_RGB_UNORM
    132: "BC1", // BC1_RGB_SRGB
    133: "BC1", // BC1_RGBA_UNORM
    134: "BC1", // BC1_RGBA_SRGB
    135: "BC2",
    136: "BC2",
    137: "BC3",
    138: "BC3",
    139: "BC4",
    140: "BC4_SNORM",
    141: "BC5",
    142: "BC5_SNORM",
    145: "BC7",
    146: "BC7"
};

const Ktx2Identifier = [0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a];

/**
 * True for a KTX2 file with block-compressed data and no supercompression, which can be
 * uploaded without transcoding (Basis Universal files need libktx instead).
 * @param {Uint8Array} bytes
 */
function isBlockCompressedKtx2(bytes) {
    if (bytes.byteLength < 80 || Ktx2Identifier.some((value, i) => bytes[i] !== value)) {
        return false;
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    return VkFormats[view.getUint32(12, true)] !== undefined && view.getUint32(44, true) === 0;
}

/**
 * Number of leading mip levels to leave out so that the largest remaining one fits maxSize
 * (0: no limit). At least the last level stays.
 */
function skippedMipLevels(width, height, levelCount, maxSize) {
    let skip = 0;
    if (maxSize > 0) {
        while (skip < levelCount - 1 && Math.max(width >> skip, height >> skip) > maxSize) {
            skip++;
        }
    }
    return skip;
}

/** Size of the largest mip level (halving, as mip levels do) that fits maxSize (0: no limit). */
function limitedImageSize(width, height, maxSize) {
    let level = 0;
    if (maxSize > 0) {
        while (Math.max(width >> level, height >> level) > maxSize) {
            level++;
        }
    }
    return [Math.max(1, width >> level), Math.max(1, height >> level)];
}

/**
 * Halves an RGBA8 image (see top of file) with a 2x2 box filter until it fits maxSize.
 * @returns {object} The image itself if it fits, else a new one.
 */
function downsampleRgba(image, maxSize) {
    let { width, height, data } = image;
    while (Math.max(width, height) > maxSize) {
        const w = Math.max(1, width >> 1);
        const h = Math.max(1, height >> 1);
        const out = new Uint8Array(w * h * 4);
        for (let y = 0; y < h; y++) {
            const y0 = Math.min(2 * y, height - 1);
            const y1 = Math.min(2 * y + 1, height - 1);
            for (let x = 0; x < w; x++) {
                const x0 = Math.min(2 * x, width - 1);
                const x1 = Math.min(2 * x + 1, width - 1);
                const a = (y0 * width + x0) * 4;
                const b = (y0 * width + x1) * 4;
                const c = (y1 * width + x0) * 4;
                const d = (y1 * width + x1) * 4;
                const o = (y * w + x) * 4;
                for (let i = 0; i < 4; i++) {
                    out[o + i] = (data[a + i] + data[b + i] + data[c + i] + data[d + i] + 2) >> 2;
                }
            }
        }
        width = w;
        height = h;
        data = out;
    }
    return data === image.data ? image : { width, height, data };
}

/**
 * Decodes a block-compressed KTX2 file (see isBlockCompressedKtx2). 2D textures only.
 * Mip levels larger than maxSize are left out; KTX2 stores the smallest level first, so
 * bytes may end after the largest level kept (see describeBlockCompressedKtx2's byteCount).
 * @param {Uint8Array} bytes
 * @param {number} [maxSize] - Largest width or height to keep (0: all levels).
 * @returns {object} A decoded image (see top of file).
 */
function decodeBlockCompressedKtx2(bytes, maxSize = 0) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const format = VkFormats[view.getUint32(12, true)];
    const width = view.getUint32(20, true);
    const height = Math.max(1, view.getUint32(24, true));
    const depth = view.getUint32(28, true);
    const layers = view.getUint32(32, true);
    const faces = view.getUint32(36, true);
    if (depth > 1 || layers > 1 || faces > 1) {
        throw new Error("Only 2D KTX2 textures are supported");
    }
    const levelCount = Math.max(1, view.getUint32(40, true));
    const levels = [];
    const skip = skippedMipLevels(width, height, levelCount, maxSize);
    for (let level = skip; level < levelCount; level++) {
        // Level index: byteOffset, byteLength, uncompressedByteLength (uint64 each), level 0 first
        const entry = 80 + level * 24;
        const offset = Number(view.getBigUint64(entry, true));
        const length = Number(view.getBigUint64(entry + 8, true));
        if (offset + length > bytes.byteLength) {
            break; // truncated file, keep the levels that are there
        }
        levels.push({
            width: Math.max(1, width >> level),
            height: Math.max(1, height >> level),
            data: bytes.subarray(offset, offset + length)
        });
    }
    if (levels.length === 0) {
        throw new Error("KTX2 file contains no image data");
    }
    return { width: levels[0].width, height: levels[0].height, compressed: { format, levels } };
}

/** Bytes describeBlockCompressedKtx2 needs: the header and the index of up to 16 levels. */
const Ktx2HeaderBytes = 80 + 16 * 24;

/**
 * Size, format and data size of a block-compressed KTX2 file (see isBlockCompressedKtx2) from
 * its first Ktx2HeaderBytes bytes, without reading the image data.
 * @param {Uint8Array} header
 * @param {number} fileSize
 * @param {number} [maxSize] - As in decodeBlockCompressedKtx2: levels larger than this are left out.
 * @returns {object | undefined} { width, height, compressed: { format, byteLength }, byteCount }
 *   with byteCount = the bytes from the start of the file decodeBlockCompressedKtx2 needs, or
 *   undefined if the file isn't a block-compressed 2D KTX2 file
 */
function describeBlockCompressedKtx2(header, fileSize, maxSize = 0) {
    if (!isBlockCompressedKtx2(header)) {
        return undefined;
    }
    const view = new DataView(header.buffer, header.byteOffset, header.byteLength);
    if (view.getUint32(28, true) > 1 || view.getUint32(32, true) > 1 || view.getUint32(36, true) > 1) {
        return undefined; // not 2D: decodeBlockCompressedKtx2 reports the error
    }
    const width = view.getUint32(20, true);
    const height = Math.max(1, view.getUint32(24, true));
    const levelCount = Math.max(1, view.getUint32(40, true));
    const skip = skippedMipLevels(width, height, levelCount, maxSize);
    let byteLength = 0;
    let byteCount = 0;
    for (let level = skip; level < levelCount; level++) {
        const entry = 80 + level * 24;
        if (entry + 16 > header.byteLength) {
            return undefined;
        }
        const offset = Number(view.getBigUint64(entry, true));
        const length = Number(view.getBigUint64(entry + 8, true));
        if (offset + length > fileSize) {
            break; // truncated file, as in decodeBlockCompressedKtx2
        }
        byteLength += length;
        byteCount = Math.max(byteCount, offset + length);
    }
    if (byteLength === 0) {
        return undefined;
    }
    return {
        width: Math.max(1, width >> skip),
        height: Math.max(1, height >> skip),
        compressed: { format: VkFormats[view.getUint32(12, true)], byteLength },
        byteCount
    };
}

function fourCC(value) {
    return String.fromCharCode(
        value & 0xff,
        (value >> 8) & 0xff,
        (value >> 16) & 0xff,
        (value >> 24) & 0xff
    );
}

const FourCCFormats = {
    DXT1: "BC1",
    DXT2: "BC2",
    DXT3: "BC2",
    DXT4: "BC3",
    DXT5: "BC3",
    ATI1: "BC4",
    BC4U: "BC4",
    ATI2: "BC5",
    BC5U: "BC5"
};

const DxgiFormats = {
    70: "BC1",
    71: "BC1",
    72: "BC1",
    73: "BC2",
    74: "BC2",
    75: "BC2",
    76: "BC3",
    77: "BC3",
    78: "BC3",
    79: "BC4",
    80: "BC4",
    82: "BC5",
    83: "BC5",
    97: "BC7",
    98: "BC7",
    99: "BC7",
    27: "RGBA8",
    28: "RGBA8",
    29: "RGBA8",
    87: "BGRA8",
    88: "BGRX8",
    90: "BGRA8",
    91: "BGRA8",
    92: "BGRX8",
    93: "BGRX8"
};

/**
 * Decodes a DDS file. Only 2D textures are supported (no cube maps or arrays).
 * @param {Uint8Array} bytes
 * @returns {object} A decoded image (see top of file).
 */
function decodeDds(bytes) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (view.getUint32(0, true) !== 0x20534444) {
        throw new Error("Not a DDS file");
    }
    const height = view.getUint32(12, true);
    const width = view.getUint32(16, true);
    const mipCount = Math.max(1, view.getUint32(28, true));
    const pfFlags = view.getUint32(80, true);
    const pfFourCC = fourCC(view.getUint32(84, true));
    const caps2 = view.getUint32(112, true);
    if (caps2 & 0x200) {
        throw new Error("DDS cube maps are not supported");
    }

    let offset = 128;
    let format = undefined;
    let masks = undefined;
    if (pfFlags & 0x4) {
        if (pfFourCC === "DX10") {
            const dxgiFormat = view.getUint32(128, true);
            const arraySize = view.getUint32(140, true);
            if (arraySize > 1) {
                throw new Error("DDS texture arrays are not supported");
            }
            format = DxgiFormats[dxgiFormat];
            offset = 148;
            if (format === undefined) {
                throw new Error(`Unsupported DDS DXGI format ${dxgiFormat}`);
            }
        } else {
            format = FourCCFormats[pfFourCC];
            if (format === undefined) {
                throw new Error(`Unsupported DDS FourCC ${pfFourCC}`);
            }
        }
    } else if (pfFlags & 0x40 && view.getUint32(88, true) === 32) {
        format = "MASKED32";
        masks = [
            view.getUint32(92, true),
            view.getUint32(96, true),
            view.getUint32(100, true),
            pfFlags & 0x1 ? view.getUint32(104, true) : 0
        ];
    } else {
        throw new Error("Unsupported uncompressed DDS pixel format");
    }

    const compressedFormat = CompressedFormats[format];
    if (compressedFormat !== undefined) {
        const levels = [];
        for (let level = 0; level < mipCount; level++) {
            const levelWidth = Math.max(1, width >> level);
            const levelHeight = Math.max(1, height >> level);
            const size =
                Math.max(1, Math.ceil(levelWidth / 4)) *
                Math.max(1, Math.ceil(levelHeight / 4)) *
                compressedFormat.blockBytes;
            if (offset + size > bytes.byteLength) {
                break; // truncated mip chain, keep what is there
            }
            levels.push({
                width: levelWidth,
                height: levelHeight,
                data: bytes.subarray(offset, offset + size)
            });
            offset += size;
        }
        if (levels.length === 0) {
            throw new Error("DDS file contains no image data");
        }
        return { width, height, compressed: { format, levels } };
    }

    // Uncompressed 32 bit: convert the top level to RGBA8.
    if (format === "RGBA8") {
        masks = [0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000];
    } else if (format === "BGRA8") {
        masks = [0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000];
    } else if (format === "BGRX8") {
        masks = [0x00ff0000, 0x0000ff00, 0x000000ff, 0];
    }
    const pixelCount = width * height;
    if (offset + pixelCount * 4 > bytes.byteLength) {
        throw new Error("DDS file is truncated");
    }
    const shifts = masks.map((mask) => (mask === 0 ? 0 : Math.log2(mask & -mask)));
    const data = new Uint8Array(pixelCount * 4);
    for (let i = 0; i < pixelCount; i++) {
        const pixel = view.getUint32(offset + i * 4, true);
        for (let c = 0; c < 4; c++) {
            data[i * 4 + c] = masks[c] === 0 ? 255 : ((pixel & masks[c]) >>> shifts[c]) & 0xff;
        }
    }
    return { width, height, data };
}

/**
 * Decodes a TGA file (true color, grayscale and color mapped, raw or RLE).
 * @param {Uint8Array} bytes
 * @returns {object} A decoded image (see top of file).
 */
function decodeTga(bytes) {
    const idLength = bytes[0];
    const colorMapType = bytes[1];
    const imageType = bytes[2];
    const colorMapStart = bytes[3] | (bytes[4] << 8);
    const colorMapLength = bytes[5] | (bytes[6] << 8);
    const colorMapDepth = bytes[7];
    const width = bytes[12] | (bytes[13] << 8);
    const height = bytes[14] | (bytes[15] << 8);
    const pixelDepth = bytes[16];
    const descriptor = bytes[17];
    const rle = imageType >= 9;
    const baseType = rle ? imageType - 8 : imageType;
    if (![1, 2, 3].includes(baseType)) {
        throw new Error(`Unsupported TGA image type ${imageType}`);
    }

    let offset = 18 + idLength;
    const readColor = (src, at, depth, out, outAt) => {
        if (depth === 8) {
            out[outAt] = out[outAt + 1] = out[outAt + 2] = src[at];
            out[outAt + 3] = 255;
        } else if (depth === 16 || depth === 15) {
            const value = src[at] | (src[at + 1] << 8);
            out[outAt] = (((value >> 10) & 0x1f) * 255) / 31;
            out[outAt + 1] = (((value >> 5) & 0x1f) * 255) / 31;
            out[outAt + 2] = ((value & 0x1f) * 255) / 31;
            out[outAt + 3] = depth === 16 && !(value & 0x8000) ? 0 : 255;
        } else {
            out[outAt] = src[at + 2];
            out[outAt + 1] = src[at + 1];
            out[outAt + 2] = src[at];
            out[outAt + 3] = depth === 32 ? src[at + 3] : 255;
        }
    };

    let colorMap = undefined;
    if (colorMapType === 1) {
        const entryBytes = Math.ceil(colorMapDepth / 8);
        colorMap = new Uint8Array((colorMapStart + colorMapLength) * 4);
        for (let i = 0; i < colorMapLength; i++) {
            readColor(
                bytes,
                offset + i * entryBytes,
                colorMapDepth,
                colorMap,
                (colorMapStart + i) * 4
            );
        }
        offset += colorMapLength * entryBytes;
    }

    // Decode the pixel stream (RLE or raw) into one value per pixel.
    const pixelBytes = Math.ceil(pixelDepth / 8);
    const pixelCount = width * height;
    let stream = bytes.subarray(offset);
    if (rle) {
        const unpacked = new Uint8Array(pixelCount * pixelBytes);
        let src = 0;
        let dst = 0;
        while (dst < unpacked.length && src < stream.length) {
            const header = stream[src++];
            const count = (header & 0x7f) + 1;
            if (header & 0x80) {
                for (let i = 0; i < count; i++) {
                    unpacked.set(stream.subarray(src, src + pixelBytes), dst);
                    dst += pixelBytes;
                }
                src += pixelBytes;
            } else {
                unpacked.set(stream.subarray(src, src + count * pixelBytes), dst);
                src += count * pixelBytes;
                dst += count * pixelBytes;
            }
        }
        stream = unpacked;
    }
    if (stream.length < pixelCount * pixelBytes) {
        throw new Error("TGA file is truncated");
    }

    const data = new Uint8Array(pixelCount * 4);
    const rightToLeft = (descriptor & 0x10) !== 0;
    const topToBottom = (descriptor & 0x20) !== 0;
    for (let y = 0; y < height; y++) {
        const outRow = topToBottom ? y : height - 1 - y;
        for (let x = 0; x < width; x++) {
            const outCol = rightToLeft ? width - 1 - x : x;
            const src = (y * width + x) * pixelBytes;
            const dst = (outRow * width + outCol) * 4;
            if (baseType === 1) {
                const index = pixelBytes === 2 ? stream[src] | (stream[src + 1] << 8) : stream[src];
                data.set(colorMap.subarray(index * 4, index * 4 + 4), dst);
            } else {
                readColor(stream, src, baseType === 3 ? 8 : pixelDepth, data, dst);
            }
        }
    }

    // No alpha bits declared: some writers still emit an unused, all zero alpha channel.
    if ((pixelDepth === 32 || pixelDepth === 16) && (descriptor & 0x0f) === 0) {
        for (let i = 3; i < data.length; i += 4) {
            data[i] = 255;
        }
    }
    return { width, height, data };
}

/**
 * Decodes the first image of a TIFF file.
 * @param {Uint8Array} bytes
 * @returns {object} A decoded image (see top of file).
 */
function decodeTiff(bytes) {
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const ifds = UTIF.decode(buffer);
    if (ifds.length === 0) {
        throw new Error("TIFF file contains no images");
    }
    UTIF.decodeImage(buffer, ifds[0]);
    const data = new Uint8Array(UTIF.toRGBA8(ifds[0]));
    return { width: ifds[0].width, height: ifds[0].height, data };
}

/**
 * Returns true for mime types decoded by decodeImageBytes instead of the browser.
 * @param {string} mimeType
 * @returns {boolean}
 */
function isDecodedImageType(mimeType) {
    return (
        mimeType === ImageMimeType.DDS ||
        mimeType === ImageMimeType.TGA ||
        mimeType === ImageMimeType.TIFF
    );
}

/**
 * Decodes DDS, TGA or TIFF bytes.
 * @param {string} mimeType - One of ImageMimeType.DDS, TGA or TIFF.
 * @param {Uint8Array} bytes
 * @returns {object} A decoded image (see top of file).
 */
function decodeImageBytes(mimeType, bytes) {
    switch (mimeType) {
        case ImageMimeType.DDS:
            return decodeDds(bytes);
        case ImageMimeType.TGA:
            return decodeTga(bytes);
        case ImageMimeType.TIFF:
            return decodeTiff(bytes);
        default:
            throw new Error(`No decoder for ${mimeType}`);
    }
}

export {
    CompressedFormats,
    isBlockCompressedKtx2,
    decodeBlockCompressedKtx2,
    describeBlockCompressedKtx2,
    Ktx2HeaderBytes,
    skippedMipLevels,
    limitedImageSize,
    downsampleRgba,
    isDecodedImageType,
    decodeImageBytes,
    decodeDds,
    decodeTga,
    decodeTiff
};
