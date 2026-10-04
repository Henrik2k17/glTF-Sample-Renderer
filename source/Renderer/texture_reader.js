import { GL } from "./webgl.js";
import { gltfTextureInfo } from "../gltf/texture.js";

const vertexSource = `#version 300 es
void main() {
    // Full screen triangle
    vec2 position = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}`;

const fragmentSource = `#version 300 es
precision highp float;
uniform sampler2D u_Texture;
uniform vec2 u_Size;
uniform bool u_EncodeSRGB;
uniform bool u_Signed;
uniform bool u_TwoChannel;
out vec4 color;
void main() {
    // Row 0 of the read back pixels is v = 0, the first row of the image.
    vec4 texel = texture(u_Texture, gl_FragCoord.xy / u_Size);
    if (u_Signed) {
        // signed formats (BC4/BC5 SNORM) sample as -1..1; show them as stored in unsigned maps
        texel.rgb = texel.rgb * 0.5 + 0.5;
    }
    if (u_TwoChannel) {
        // BC5 normal maps store X and Y only; show them like a full normal map
        texel.b = 1.0;
    }
    if (u_EncodeSRGB) {
        vec3 c = clamp(texel.rgb, 0.0, 1.0);
        texel.rgb = mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
    }
    color = texel;
}`;

/**
 * Reads glTF textures back from the GPU as 8-bit RGBA, e.g. for thumbnails in a material
 * debugger. Going through the GPU texture works for every image format the renderer supports
 * (including block compressed DDS and KTX2) and shows what the shaders actually sample.
 */
class TextureReader {
    constructor(webGl) {
        this.webGl = webGl;
        this.program = undefined;
    }

    createProgram() {
        const gl = this.webGl.context;
        const compile = (type, source) => {
            const shader = gl.createShader(type);
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
                throw new Error(gl.getShaderInfoLog(shader));
            }
            return shader;
        };
        const program = gl.createProgram();
        gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSource));
        gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            throw new Error(gl.getProgramInfoLog(program));
        }
        this.program = program;
        this.locations = {
            texture: gl.getUniformLocation(program, "u_Texture"),
            size: gl.getUniformLocation(program, "u_Size"),
            encodeSRGB: gl.getUniformLocation(program, "u_EncodeSRGB"),
            signed: gl.getUniformLocation(program, "u_Signed"),
            twoChannel: gl.getUniformLocation(program, "u_TwoChannel")
        };
        this.vertexArray = gl.createVertexArray();
    }

    /**
     * @param {*} gltf The glTF the texture belongs to.
     * @param {number} textureIndex Index into gltf.textures.
     * @param {number} maxSize The longer side of the result is at most this many pixels.
     * @returns {{width, height, pixels: Uint8ClampedArray, sourceWidth, sourceHeight} | undefined}
     *   undefined if the texture's image is not loaded.
     */
    read(gltf, textureIndex, maxSize) {
        const texture = gltf.textures[textureIndex];
        const image = gltf.images[texture?.source];
        if (texture === undefined || image?.image === undefined || texture.type !== GL.TEXTURE_2D) {
            return undefined;
        }
        const gl = this.webGl.context;
        if (this.program === undefined) {
            this.createProgram();
        }

        const sourceWidth = image.image.width ?? maxSize;
        const sourceHeight = image.image.height ?? maxSize;
        const scale = Math.min(1, maxSize / Math.max(sourceWidth, sourceHeight));
        const width = Math.max(1, Math.round(sourceWidth * scale));
        const height = Math.max(1, Math.round(sourceHeight * scale));

        const saved = {
            framebuffer: gl.getParameter(gl.FRAMEBUFFER_BINDING),
            viewport: gl.getParameter(gl.VIEWPORT),
            program: gl.getParameter(gl.CURRENT_PROGRAM),
            activeTexture: gl.getParameter(gl.ACTIVE_TEXTURE),
            blend: gl.isEnabled(gl.BLEND),
            depthTest: gl.isEnabled(gl.DEPTH_TEST),
            cullFace: gl.isEnabled(gl.CULL_FACE),
            scissorTest: gl.isEnabled(gl.SCISSOR_TEST)
        };

        const target = gl.createTexture();
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, target);
        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, width, height);
        const framebuffer = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target, 0);

        gl.useProgram(this.program);
        // Reuse the texture as uploaded for rendering; sRGB textures are encoded back so the
        // result shows the stored values. Textures not drawn yet are uploaded linear (raw).
        const srgb = texture.initializedSRGB === true && texture.initialized !== true;
        const textureInfo = new gltfTextureInfo(textureIndex, 0, !srgb);
        const pixels = new Uint8ClampedArray(width * height * 4);
        let bound = false;
        try {
            bound = this.webGl.setTexture(this.locations.texture, gltf, textureInfo, 0);
            if (bound) {
                gl.bindSampler(0, null);
                gl.uniform2f(this.locations.size, width, height);
                gl.uniform1i(this.locations.encodeSRGB, srgb ? 1 : 0);
                const format = image.image.compressed?.format ?? "";
                gl.uniform1i(this.locations.signed, format.endsWith("_SNORM") ? 1 : 0);
                gl.uniform1i(this.locations.twoChannel, format.startsWith("BC5") ? 1 : 0);
                gl.bindVertexArray(this.vertexArray);
                gl.disable(gl.BLEND);
                gl.disable(gl.DEPTH_TEST);
                gl.disable(gl.CULL_FACE);
                gl.disable(gl.SCISSOR_TEST);
                gl.viewport(0, 0, width, height);
                gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
                gl.drawArrays(gl.TRIANGLES, 0, 3);
                gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
            }
        } finally {
            gl.bindVertexArray(null);
            gl.bindFramebuffer(gl.FRAMEBUFFER, saved.framebuffer);
            gl.deleteFramebuffer(framebuffer);
            gl.deleteTexture(target);
            gl.viewport(...saved.viewport);
            gl.useProgram(saved.program);
            gl.activeTexture(saved.activeTexture);
            const restore = (capability, enabled) =>
                enabled ? gl.enable(capability) : gl.disable(capability);
            restore(gl.BLEND, saved.blend);
            restore(gl.DEPTH_TEST, saved.depthTest);
            restore(gl.CULL_FACE, saved.cullFace);
            restore(gl.SCISSOR_TEST, saved.scissorTest);
        }
        if (!bound) {
            return undefined;
        }
        return { width, height, pixels, sourceWidth, sourceHeight };
    }
}

export { TextureReader };
