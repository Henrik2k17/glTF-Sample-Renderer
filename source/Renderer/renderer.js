import { mat4, mat3, vec3, quat, vec4 } from "gl-matrix";
import { ShaderCache } from "./shader_cache.js";
import { GltfState } from "../GltfState/gltf_state.js";
import { gltfWebGl, GL } from "./webgl.js";
import { EnvironmentRenderer } from "./environment_renderer.js";

import pbrShader from "./shaders/pbr.frag";
import pickingShader from "./shaders/picking.frag";
import pickingVertShader from "./shaders/picking.vert";
import brdfShader from "./shaders/brdf.glsl";
import iridescenceShader from "./shaders/iridescence.glsl";
import materialInfoShader from "./shaders/material_info.glsl";
import iblShader from "./shaders/ibl.glsl";
import punctualShader from "./shaders/punctual.glsl";
import primitiveShader from "./shaders/primitive.vert";
import texturesShader from "./shaders/textures.glsl";
import tonemappingShader from "./shaders/tonemapping.glsl";
import shaderFunctions from "./shaders/functions.glsl";
import animationShader from "./shaders/animation.glsl";
import cubemapVertShader from "./shaders/cubemap.vert";
import cubemapFragShader from "./shaders/cubemap.frag";
import scatterShader from "./shaders/scatter.frag";
import simpleFragShader from "./shaders/simple.frag";
import specularGlossinesShader from "./shaders/specular_glossiness.frag";
import splatVertexShader from "./shaders/splat.vert";
import splatFragmentShader from "./shaders/splat.frag";
import fullscreenVertShader from "./shaders/fullscreen.vert";
import tonemapMainFragShader from "./shaders/tonemap_main.frag";
import splatCompositeFragShader from "./shaders/splat_composite.frag";
import { gltfLight } from "../gltf/light.js";
import {
    isMsfsHiddenPrimitive,
    getMsfsDepthBias,
    sortByMsfsDrawOrder,
    getMsfsEmissiveMultiplier
} from "../gltf/msfs.js";
import { jsToGl } from "../gltf/utils.js";
import { getMsfsClampedTextures, updateMsfsMaterialUniforms } from "../gltf/msfs_material.js";
import { buildMsfsHelperLines } from "../gltf/msfs_helpers.js";
import { gltfMaterial } from "../gltf/material.js";

const NoHighlightColor = vec4.fromValues(0, 0, 0, 0);
// Picking ids hold the node's picking color in the low 24 bits (see drawPrimitive).
const PickingNodeMask = 0xffffff;

// Texel that makes a texture slot behave as if the texture was absent, for textures switched
// off in a material debugger (textureInfo.debugDisabled). White suits factors that multiply.
const NeutralTexels = {
    u_NormalSampler: [128, 128, 255, 255],
    u_ClearcoatNormalSampler: [128, 128, 255, 255],
    u_MsfsDetailNormalSampler: [128, 128, 255, 255],
    u_AnisotropySampler: [255, 128, 255, 255],
    u_MsfsDetailColorSampler: [128, 128, 128, 255], // overlay neutral
    u_MsfsDirtSampler: [0, 0, 0, 0], // no dirt
    u_MsfsTireDetailsSampler: [255, 0, 0, 255] // no mud, no dust
};
const WhiteTexel = [255, 255, 255, 255];

class gltfRenderer {
    constructor(context) {
        this.shader = undefined; // current shader

        this.currentWidth = 0;
        this.currentHeight = 0;

        this.webGl = new gltfWebGl(context);
        this.initialized = false;
        this.samples = 4;

        // create render target for non transmission materials
        this.opaqueRenderTexture = 0;
        this.opaqueFramebuffer = 0;
        this.opaqueDepthTexture = 0;
        this.pickingIDTexture = 0;
        this.msfsSamplers = new Map(); // ASOBO_material_UV_options clamp samplers by parameters
        this.msfsSamplerUnits = []; // texture units with such a sampler bound for the current draw
        this.pickingPositionTexture = 0;
        this.pickingDepthTexture = 0;
        this.hoverIDTexture = 0;
        this.hoverDepthTexture = 0;
        this.opaqueFramebufferWidth = 1024;
        this.opaqueFramebufferHeight = 1024;

        // create render target for subsurface scattering
        this.scatterFrontTexture = 0;
        this.scatterDepthTexture = 0;

        const shaderSources = new Map();
        shaderSources.set("primitive.vert", primitiveShader);
        shaderSources.set("pbr.frag", pbrShader);
        shaderSources.set("picking.frag", pickingShader);
        shaderSources.set("picking.vert", pickingVertShader);
        shaderSources.set("material_info.glsl", materialInfoShader);
        shaderSources.set("brdf.glsl", brdfShader);
        shaderSources.set("iridescence.glsl", iridescenceShader);
        shaderSources.set("ibl.glsl", iblShader);
        shaderSources.set("punctual.glsl", punctualShader);
        shaderSources.set("tonemapping.glsl", tonemappingShader);
        shaderSources.set("textures.glsl", texturesShader);
        shaderSources.set("functions.glsl", shaderFunctions);
        shaderSources.set("animation.glsl", animationShader);
        shaderSources.set("scatter.frag", scatterShader);
        shaderSources.set("cubemap.vert", cubemapVertShader);
        shaderSources.set("cubemap.frag", cubemapFragShader);
        shaderSources.set("specular_glossiness.frag", specularGlossinesShader);
        shaderSources.set("simple.frag", simpleFragShader);
        shaderSources.set("splat.vert", splatVertexShader);
        shaderSources.set("splat.frag", splatFragmentShader);
        shaderSources.set("fullscreen.vert", fullscreenVertShader);
        shaderSources.set("tonemap_main.frag", tonemapMainFragShader);
        shaderSources.set("splat_composite.frag", splatCompositeFragShader);

        this.shaderCache = new ShaderCache(shaderSources, this.webGl);

        this.webGl.loadWebGlExtensions();

        this.visibleLights = [];

        this.viewMatrix = mat4.create();
        this.projMatrix = mat4.create();
        this.viewProjectionMatrix = mat4.create();

        this.currentCameraPosition = vec3.create();

        this.lightKey = new gltfLight();
        this.lightFill = new gltfLight();
        this.lightFill.intensity = 0.5;
        const quatKey = quat.fromValues(-0.3535534, -0.353553385, -0.146446586, 0.8535534);
        const quatFill = quat.fromValues(-0.8535534, 0.146446645, -0.353553325, -0.353553444);
        this.lightKey.direction = vec3.create();
        this.lightFill.direction = vec3.create();
        vec3.transformQuat(this.lightKey.direction, [0, 0, -1], quatKey);
        vec3.transformQuat(this.lightFill.direction, [0, 0, -1], quatFill);

        this.maxVertAttributes = undefined;
        this.instanceBuffer = undefined;

        this.splatVBO = undefined;
        this.currentSortBuffer = undefined;

        this.mainTexture = undefined;
        this.mainTextureHDR = undefined;
        this.mainDepthTexture = undefined;
        this.mainTonemapTexture = undefined;
        this.mainFramebuffer = undefined;

        this.framebufferFormat = undefined;
        this.framebufferType = undefined;

        // Tracks last value of state.renderingParameters.floatingPointFramebuffer so we
        // only re-attach when the setting actually changes.
        this._floatingPointFramebuffer = undefined;

        // Splat isolation framebuffer: same colour format as mainFramebuffer,
        // shares mainDepthTexture for depth-test (no depth writes).
        this.splatColorTexture = undefined;
        this.splatColorTextureHDR = undefined;
        this.splatFramebuffer = undefined;
    }

    /////////////////////////////////////////////////////////////////////
    // Render glTF scene graph
    /////////////////////////////////////////////////////////////////////

    // app state
    // prettier-ignore
    init(state)
    {
        const context = this.webGl.context;
        const maxSamples = context.getParameter(context.MAX_SAMPLES);
        const samples = state.internalMSAA < maxSamples ? state.internalMSAA : maxSamples;
        if (!this.initialized){

            context.pixelStorei(GL.UNPACK_COLORSPACE_CONVERSION_WEBGL, GL.NONE);
            context.enable(GL.DEPTH_TEST);
            context.depthFunc(GL.LEQUAL);
            context.colorMask(true, true, true, true);
            context.clearDepth(1.0);

            this.opaqueRenderTexture = context.createTexture();
            context.bindTexture(context.TEXTURE_2D, this.opaqueRenderTexture);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MIN_FILTER, context.LINEAR_MIPMAP_LINEAR);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_S, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_T, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MAG_FILTER, context.NEAREST);
            context.texImage2D(context.TEXTURE_2D, 0, context.RGBA, this.opaqueFramebufferWidth, this.opaqueFramebufferHeight, 0, context.RGBA, context.UNSIGNED_BYTE, null);
            context.bindTexture(context.TEXTURE_2D, null);

            this.opaqueDepthTexture = context.createTexture();
            context.bindTexture(context.TEXTURE_2D, this.opaqueDepthTexture);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MIN_FILTER, context.NEAREST);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_S, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_T, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MAG_FILTER, context.NEAREST);
            context.texImage2D( context.TEXTURE_2D, 0, context.DEPTH_COMPONENT24, this.opaqueFramebufferWidth, this.opaqueFramebufferHeight, 0, context.DEPTH_COMPONENT, context.UNSIGNED_INT, null);
            context.bindTexture(context.TEXTURE_2D, null);

            this.scatterDepthTexture = context.createTexture();
            context.bindTexture(context.TEXTURE_2D, this.scatterDepthTexture);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MIN_FILTER, context.NEAREST);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_S, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_T, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MAG_FILTER, context.NEAREST);
            context.texImage2D( context.TEXTURE_2D, 0, context.DEPTH_COMPONENT24, this.opaqueFramebufferWidth, this.opaqueFramebufferHeight, 0, context.DEPTH_COMPONENT, context.UNSIGNED_INT, null);
            context.bindTexture(context.TEXTURE_2D, null);

            this.scatterInternalFormat = context.supports_EXT_color_buffer_half_float ? context.RGBA16F : context.RGBA;
            this.scatterType = context.supports_EXT_color_buffer_half_float ? context.HALF_FLOAT : context.UNSIGNED_BYTE;
            
            this.scatterFrontTexture = context.createTexture();
            context.bindTexture(context.TEXTURE_2D, this.scatterFrontTexture);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MIN_FILTER, context.NEAREST);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MAG_FILTER, context.NEAREST);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_S, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_T, context.CLAMP_TO_EDGE);
            context.texImage2D(context.TEXTURE_2D, 0, this.scatterInternalFormat, this.opaqueFramebufferWidth, this.opaqueFramebufferHeight, 0, context.RGBA, this.scatterType, null);
            context.bindTexture(context.TEXTURE_2D, null);

            this.scatterFramebuffer = context.createFramebuffer();
            context.bindFramebuffer(context.FRAMEBUFFER, this.scatterFramebuffer);
            context.framebufferTexture2D(context.FRAMEBUFFER, context.COLOR_ATTACHMENT0, context.TEXTURE_2D, this.scatterFrontTexture, 0);
            context.framebufferTexture2D(context.FRAMEBUFFER, context.DEPTH_ATTACHMENT, context.TEXTURE_2D, this.scatterDepthTexture, 0);
            context.drawBuffers([context.COLOR_ATTACHMENT0]);
            
            this.pickingIDTexture = context.createTexture();
            context.bindTexture(context.TEXTURE_2D, this.pickingIDTexture);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MIN_FILTER, context.NEAREST);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_S, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_T, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MAG_FILTER, context.NEAREST);
            context.texImage2D(context.TEXTURE_2D, 0, context.R32UI, 1, 1, 0, context.RED_INTEGER, context.UNSIGNED_INT, null);
            context.bindTexture(context.TEXTURE_2D, null);

            this.pickingPositionTexture = context.createTexture();
            context.bindTexture(context.TEXTURE_2D, this.pickingPositionTexture);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MIN_FILTER, context.NEAREST);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_S, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_T, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MAG_FILTER, context.NEAREST);
            context.texImage2D(context.TEXTURE_2D, 0, context.R32UI, 1, 1, 0, context.RED_INTEGER, context.UNSIGNED_INT, null);
            context.bindTexture(context.TEXTURE_2D, null);

            this.pickingDepthTexture = context.createTexture();
            context.bindTexture(context.TEXTURE_2D, this.pickingDepthTexture);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MIN_FILTER, context.NEAREST);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_S, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_T, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MAG_FILTER, context.NEAREST);
            context.texImage2D( context.TEXTURE_2D, 0, context.DEPTH_COMPONENT24, 1, 1, 0, context.DEPTH_COMPONENT, context.UNSIGNED_INT, null);
            context.bindTexture(context.TEXTURE_2D, null);

            this.hoverIDTexture = context.createTexture();
            context.bindTexture(context.TEXTURE_2D, this.hoverIDTexture);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MIN_FILTER, context.NEAREST);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_S, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_T, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MAG_FILTER, context.NEAREST);
            context.texImage2D(context.TEXTURE_2D, 0, context.R32UI, 1, 1, 0, context.RED_INTEGER, context.UNSIGNED_INT, null);
            context.bindTexture(context.TEXTURE_2D, null);

            this.hoverDepthTexture = context.createTexture();
            context.bindTexture(context.TEXTURE_2D, this.hoverDepthTexture);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MIN_FILTER, context.NEAREST);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_S, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_T, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MAG_FILTER, context.NEAREST);
            context.texImage2D( context.TEXTURE_2D, 0, context.DEPTH_COMPONENT24, 1, 1, 0, context.DEPTH_COMPONENT, context.UNSIGNED_INT, null);
            context.bindTexture(context.TEXTURE_2D, null);

            this.colorRenderBuffer = context.createRenderbuffer();
            context.bindRenderbuffer(context.RENDERBUFFER, this.colorRenderBuffer);
            context.renderbufferStorageMultisample( context.RENDERBUFFER, samples, context.RGBA8,  this.opaqueFramebufferWidth, this.opaqueFramebufferHeight);

            this.depthRenderBuffer = context.createRenderbuffer();
            context.bindRenderbuffer(context.RENDERBUFFER, this.depthRenderBuffer);
            context.renderbufferStorageMultisample( context.RENDERBUFFER,
                samples,
                context.DEPTH_COMPONENT24, 
                this.opaqueFramebufferWidth,
                this.opaqueFramebufferHeight);
            
            this.pickingFramebuffer = context.createFramebuffer();
            context.bindFramebuffer(context.FRAMEBUFFER, this.pickingFramebuffer);
            context.framebufferTexture2D(context.FRAMEBUFFER, context.COLOR_ATTACHMENT0, context.TEXTURE_2D, this.pickingIDTexture, 0);
            context.framebufferTexture2D(context.FRAMEBUFFER, context.DEPTH_ATTACHMENT, context.TEXTURE_2D, this.pickingDepthTexture, 0);
            context.framebufferTexture2D(context.FRAMEBUFFER, context.COLOR_ATTACHMENT1, context.TEXTURE_2D, this.pickingPositionTexture, 0);
            context.drawBuffers([context.COLOR_ATTACHMENT0, context.COLOR_ATTACHMENT1]);

            this.hoverFramebuffer = context.createFramebuffer();
            context.bindFramebuffer(context.FRAMEBUFFER, this.hoverFramebuffer);
            context.framebufferTexture2D(context.FRAMEBUFFER, context.COLOR_ATTACHMENT0, context.TEXTURE_2D, this.hoverIDTexture, 0);
            context.framebufferTexture2D(context.FRAMEBUFFER, context.DEPTH_ATTACHMENT, context.TEXTURE_2D, this.hoverDepthTexture, 0);

            this.samples = samples;

            this.opaqueFramebufferMSAA = context.createFramebuffer();
            context.bindFramebuffer(context.FRAMEBUFFER, this.opaqueFramebufferMSAA);
            context.framebufferRenderbuffer(context.FRAMEBUFFER, context.COLOR_ATTACHMENT0, context.RENDERBUFFER, this.colorRenderBuffer);
            context.framebufferRenderbuffer(context.FRAMEBUFFER, context.DEPTH_ATTACHMENT, context.RENDERBUFFER, this.depthRenderBuffer);


            this.opaqueFramebuffer = context.createFramebuffer();
            context.bindFramebuffer(context.FRAMEBUFFER, this.opaqueFramebuffer);
            context.framebufferTexture2D(context.FRAMEBUFFER, context.COLOR_ATTACHMENT0, context.TEXTURE_2D, this.opaqueRenderTexture, 0);
            context.framebufferTexture2D(context.FRAMEBUFFER, context.DEPTH_ATTACHMENT, context.TEXTURE_2D, this.opaqueDepthTexture, 0);
            context.viewport(0, 0, this.opaqueFramebufferWidth, this.opaqueFramebufferHeight);
            context.bindFramebuffer(context.FRAMEBUFFER, null);

            const quatVertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
            this.splatVBO = context.createBuffer();
            context.bindBuffer(context.ARRAY_BUFFER, this.splatVBO);
            context.bufferData(context.ARRAY_BUFFER, quatVertices, context.STATIC_DRAW);
            context.bindBuffer(context.ARRAY_BUFFER, null);

            this.currentSortBuffer = context.createBuffer();

            this.framebufferFormat = context.supports_EXT_color_buffer_half_float
                ? context.RGBA16F : context.RGBA;
            this.framebufferType = context.supports_EXT_color_buffer_half_float
                ? context.HALF_FLOAT : context.UNSIGNED_BYTE;

            const hdrW = Math.max(this.currentWidth,  1);
            const hdrH = Math.max(this.currentHeight, 1);

            // Main color texture
            this.mainTexture = context.createTexture();
            context.bindTexture(context.TEXTURE_2D, this.mainTexture);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MIN_FILTER, context.NEAREST);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MAG_FILTER, context.NEAREST);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_S, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_T, context.CLAMP_TO_EDGE);
            context.texImage2D(context.TEXTURE_2D, 0, context.RGBA,
                hdrW, hdrH, 0, context.RGBA, context.UNSIGNED_BYTE, null);
            context.bindTexture(context.TEXTURE_2D, null);

            if (context.supports_EXT_color_buffer_half_float) {
                this.mainTextureHDR = context.createTexture();
                context.bindTexture(context.TEXTURE_2D, this.mainTextureHDR);
                context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MIN_FILTER, context.NEAREST);
                context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MAG_FILTER, context.NEAREST);
                context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_S, context.CLAMP_TO_EDGE);
                context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_T, context.CLAMP_TO_EDGE);
                context.texImage2D(context.TEXTURE_2D, 0, context.RGBA16F,
                    hdrW, hdrH, 0, context.RGBA, context.HALF_FLOAT, null);
                context.bindTexture(context.TEXTURE_2D, null);
            }

            // Main depth texture
            this.mainDepthTexture = context.createTexture();
            context.bindTexture(context.TEXTURE_2D, this.mainDepthTexture);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MIN_FILTER, context.NEAREST);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MAG_FILTER, context.NEAREST);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_S, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_T, context.CLAMP_TO_EDGE);
            context.texImage2D(context.TEXTURE_2D, 0, context.DEPTH_COMPONENT24,
                hdrW, hdrH, 0, context.DEPTH_COMPONENT, context.UNSIGNED_INT, null);
            context.bindTexture(context.TEXTURE_2D, null);

            this.mainFramebuffer = context.createFramebuffer();
            context.bindFramebuffer(context.FRAMEBUFFER, this.mainFramebuffer);
            context.framebufferTexture2D(context.FRAMEBUFFER, context.COLOR_ATTACHMENT0,
                context.TEXTURE_2D, this.mainTexture, 0);
            context.framebufferTexture2D(context.FRAMEBUFFER, context.DEPTH_ATTACHMENT,
                context.TEXTURE_2D, this.mainDepthTexture, 0);

            // Tonemap flag texture: R8UI, one flag per pixel.
            // 0 = linear only, 1 = sRGB gamma only, 2 = tonemap + sRGB.
            this.mainTonemapTexture = context.createTexture();
            context.bindTexture(context.TEXTURE_2D, this.mainTonemapTexture);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MIN_FILTER, context.NEAREST);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MAG_FILTER, context.NEAREST);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_S, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_T, context.CLAMP_TO_EDGE);
            context.texImage2D(context.TEXTURE_2D, 0, context.R8UI,
                hdrW, hdrH, 0, context.RED_INTEGER, context.UNSIGNED_BYTE, null);
            context.bindTexture(context.TEXTURE_2D, null);

            context.framebufferTexture2D(context.FRAMEBUFFER, context.COLOR_ATTACHMENT1,
                context.TEXTURE_2D, this.mainTonemapTexture, 0);
            context.drawBuffers([context.COLOR_ATTACHMENT0, context.COLOR_ATTACHMENT1]);
            context.bindFramebuffer(context.FRAMEBUFFER, null);

            this.splatColorTexture = context.createTexture();
            context.bindTexture(context.TEXTURE_2D, this.splatColorTexture);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MIN_FILTER, context.NEAREST);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MAG_FILTER, context.NEAREST);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_S, context.CLAMP_TO_EDGE);
            context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_T, context.CLAMP_TO_EDGE);
            context.texImage2D(context.TEXTURE_2D, 0, context.RGBA,
                hdrW, hdrH, 0, context.RGBA, context.UNSIGNED_BYTE, null);
            context.bindTexture(context.TEXTURE_2D, null);

            if (context.supports_EXT_color_buffer_half_float) {
                this.splatColorTextureHDR = context.createTexture();
                context.bindTexture(context.TEXTURE_2D, this.splatColorTextureHDR);
                context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MIN_FILTER, context.NEAREST);
                context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MAG_FILTER, context.NEAREST);
                context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_S, context.CLAMP_TO_EDGE);
                context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_T, context.CLAMP_TO_EDGE);
                context.texImage2D(context.TEXTURE_2D, 0, context.RGBA16F,
                    hdrW, hdrH, 0, context.RGBA, context.HALF_FLOAT, null);
                context.bindTexture(context.TEXTURE_2D, null);
            }

            // Initially attach the LDR colour texture; re-attached together with
            // mainFramebuffer when _floatingPointFramebuffer is resolved.
            this.splatFramebuffer = context.createFramebuffer();
            context.bindFramebuffer(context.FRAMEBUFFER, this.splatFramebuffer);
            context.framebufferTexture2D(context.FRAMEBUFFER, context.COLOR_ATTACHMENT0,
                context.TEXTURE_2D, this.splatColorTexture, 0);
            context.framebufferTexture2D(context.FRAMEBUFFER, context.DEPTH_ATTACHMENT,
                context.TEXTURE_2D, this.mainDepthTexture, 0);
            context.drawBuffers([context.COLOR_ATTACHMENT0]);
            context.bindFramebuffer(context.FRAMEBUFFER, null);

            this.maxVertAttributes = context.getParameter(context.MAX_VERTEX_ATTRIBS);

            this.initialized = true;

            this.environmentRenderer = new EnvironmentRenderer(this.webGl);
        }
        else {
            if (this.samples != samples)
            {
                this.samples = samples;
                context.bindRenderbuffer(context.RENDERBUFFER, this.colorRenderBuffer);
                context.renderbufferStorageMultisample( context.RENDERBUFFER,
                    samples,
                    context.RGBA8, 
                    this.opaqueFramebufferWidth,
                    this.opaqueFramebufferHeight);
                
                context.bindRenderbuffer(context.RENDERBUFFER, this.depthRenderBuffer);
                context.renderbufferStorageMultisample( context.RENDERBUFFER,
                    samples,
                    context.DEPTH_COMPONENT16, 
                    this.opaqueFramebufferWidth,
                    this.opaqueFramebufferHeight);
            }
        }
    }

    resize(width, height) {
        if (this.currentWidth !== width || this.currentHeight !== height) {
            this.currentHeight = height;
            this.currentWidth = width;
            this.webGl.context.viewport(0, 0, width, height);
            if (this.initialized) {
                this.webGl.context.bindFramebuffer(
                    this.webGl.context.FRAMEBUFFER,
                    this.scatterFramebuffer
                );
                this.webGl.context.bindTexture(
                    this.webGl.context.TEXTURE_2D,
                    this.scatterFrontTexture
                );
                this.webGl.context.texImage2D(
                    this.webGl.context.TEXTURE_2D,
                    0,
                    this.scatterInternalFormat,
                    this.currentWidth,
                    this.currentHeight,
                    0,
                    this.webGl.context.RGBA,
                    this.scatterType,
                    null
                );
                this.webGl.context.bindTexture(
                    this.webGl.context.TEXTURE_2D,
                    this.scatterDepthTexture
                );
                this.webGl.context.texImage2D(
                    this.webGl.context.TEXTURE_2D,
                    0,
                    this.webGl.context.DEPTH_COMPONENT24,
                    this.currentWidth,
                    this.currentHeight,
                    0,
                    this.webGl.context.DEPTH_COMPONENT,
                    this.webGl.context.UNSIGNED_INT,
                    null
                );
                this.webGl.context.bindTexture(this.webGl.context.TEXTURE_2D, null);
                this.webGl.context.bindFramebuffer(this.webGl.context.FRAMEBUFFER, null);

                // Resize main color texture (always RGBA/UNSIGNED_BYTE).
                this.webGl.context.bindTexture(this.webGl.context.TEXTURE_2D, this.mainTexture);
                this.webGl.context.texImage2D(
                    this.webGl.context.TEXTURE_2D,
                    0,
                    this.webGl.context.RGBA,
                    this.currentWidth,
                    this.currentHeight,
                    0,
                    this.webGl.context.RGBA,
                    this.webGl.context.UNSIGNED_BYTE,
                    null
                );

                // Resize HDR color texture (RGBA16F/HALF_FLOAT, only when available).
                if (this.mainTextureHDR) {
                    this.webGl.context.bindTexture(
                        this.webGl.context.TEXTURE_2D,
                        this.mainTextureHDR
                    );
                    this.webGl.context.texImage2D(
                        this.webGl.context.TEXTURE_2D,
                        0,
                        this.webGl.context.RGBA16F,
                        this.currentWidth,
                        this.currentHeight,
                        0,
                        this.webGl.context.RGBA,
                        this.webGl.context.HALF_FLOAT,
                        null
                    );
                }

                this.webGl.context.bindTexture(
                    this.webGl.context.TEXTURE_2D,
                    this.mainDepthTexture
                );
                this.webGl.context.texImage2D(
                    this.webGl.context.TEXTURE_2D,
                    0,
                    this.webGl.context.DEPTH_COMPONENT24,
                    this.currentWidth,
                    this.currentHeight,
                    0,
                    this.webGl.context.DEPTH_COMPONENT,
                    this.webGl.context.UNSIGNED_INT,
                    null
                );
                this.webGl.context.bindTexture(this.webGl.context.TEXTURE_2D, null);

                // Resize tonemap flag texture (R8UI).
                this.webGl.context.bindTexture(
                    this.webGl.context.TEXTURE_2D,
                    this.mainTonemapTexture
                );
                this.webGl.context.texImage2D(
                    this.webGl.context.TEXTURE_2D,
                    0,
                    this.webGl.context.R8UI,
                    this.currentWidth,
                    this.currentHeight,
                    0,
                    this.webGl.context.RED_INTEGER,
                    this.webGl.context.UNSIGNED_BYTE,
                    null
                );
                this.webGl.context.bindTexture(this.webGl.context.TEXTURE_2D, null);

                // Resize splat isolation colour textures.
                this.webGl.context.bindTexture(
                    this.webGl.context.TEXTURE_2D,
                    this.splatColorTexture
                );
                this.webGl.context.texImage2D(
                    this.webGl.context.TEXTURE_2D,
                    0,
                    this.webGl.context.RGBA,
                    this.currentWidth,
                    this.currentHeight,
                    0,
                    this.webGl.context.RGBA,
                    this.webGl.context.UNSIGNED_BYTE,
                    null
                );
                if (this.splatColorTextureHDR) {
                    this.webGl.context.bindTexture(
                        this.webGl.context.TEXTURE_2D,
                        this.splatColorTextureHDR
                    );
                    this.webGl.context.texImage2D(
                        this.webGl.context.TEXTURE_2D,
                        0,
                        this.webGl.context.RGBA16F,
                        this.currentWidth,
                        this.currentHeight,
                        0,
                        this.webGl.context.RGBA,
                        this.webGl.context.HALF_FLOAT,
                        null
                    );
                }
                this.webGl.context.bindTexture(this.webGl.context.TEXTURE_2D, null);
            }
        }
    }

    // frame state
    clearFrame(clearColor) {
        // Ensure depth writes are enabled before clearing any depth buffers.
        // Transparent drawables leave depthMask(false); unlike the canvas whose depth is
        // auto-cleared by the browser, mainDepthTexture must be cleared explicitly and
        // clearBufferfv(DEPTH) is masked by depthMask.
        this.webGl.context.depthMask(true);
        // Convert clear color from sRGB to linear since we will always transfer to sRGB by default
        const linearClearColor = clearColor.map((c) => Math.pow(c, 2.2));
        this.webGl.context.bindFramebuffer(this.webGl.context.FRAMEBUFFER, this.opaqueFramebuffer);
        this.webGl.context.clearColor(...linearClearColor);
        this.webGl.context.clear(GL.COLOR_BUFFER_BIT | GL.DEPTH_BUFFER_BIT);
        this.webGl.context.bindFramebuffer(
            this.webGl.context.FRAMEBUFFER,
            this.opaqueFramebufferMSAA
        );
        this.webGl.context.clearColor(...linearClearColor);
        this.webGl.context.clear(GL.COLOR_BUFFER_BIT | GL.DEPTH_BUFFER_BIT);
        this.webGl.context.bindFramebuffer(this.webGl.context.FRAMEBUFFER, this.scatterFramebuffer);
        this.webGl.context.clearColor(0, 0, 0, 0);
        this.webGl.context.clear(GL.COLOR_BUFFER_BIT | GL.DEPTH_BUFFER_BIT);
        // Main framebuffer: clear color (attachment 0) and depth separately,
        // then zero the tonemap flag attachment (attachment 1) via clearBufferuiv.
        if (this.mainFramebuffer) {
            const gl = this.webGl.context;
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.mainFramebuffer);
            // Clear color attachment 0 to the scene background colour.
            gl.clearBufferfv(gl.COLOR, 0, linearClearColor);
            // Clear tonemap flag attachment to 1 (linear to sRGB by default).
            gl.clearBufferuiv(gl.COLOR, 1, new Uint32Array([1, 1, 1, 1]));
            // Clear depth.
            gl.clearBufferfv(gl.DEPTH, 0, new Float32Array([1.0]));
        }
        this.webGl.context.bindFramebuffer(this.webGl.context.FRAMEBUFFER, null);
        this.webGl.context.clearColor(...linearClearColor);
        this.webGl.context.clear(GL.COLOR_BUFFER_BIT | GL.DEPTH_BUFFER_BIT);
        this.webGl.context.bindFramebuffer(this.webGl.context.FRAMEBUFFER, null);
        this.webGl.context.bindFramebuffer(this.webGl.context.FRAMEBUFFER, this.pickingFramebuffer);
        this.webGl.context.clearBufferuiv(GL.COLOR, 0, new Uint32Array([0, 0, 0, 0]));
        this.webGl.context.clearBufferuiv(GL.COLOR, 1, new Uint32Array([0, 0, 0, 0]));
        this.webGl.context.clearBufferfv(GL.DEPTH, 0, new Float32Array([1.0]));
        this.webGl.context.bindFramebuffer(this.webGl.context.FRAMEBUFFER, null);
        this.webGl.context.bindFramebuffer(this.webGl.context.FRAMEBUFFER, this.hoverFramebuffer);
        this.webGl.context.clearBufferuiv(GL.COLOR, 0, new Uint32Array([0, 0, 0, 0]));
        this.webGl.context.clearBufferfv(GL.DEPTH, 0, new Float32Array([1.0]));
        this.webGl.context.bindFramebuffer(this.webGl.context.FRAMEBUFFER, null);
    }

    // ASOBO_material_UV_options clamping: a sampler object overrides the texture's own wrap
    // mode on one texture unit, so the shared texture is not changed for other materials.
    bindMsfsClampSampler(gltf, textureInfo, clamp, unit) {
        const gl = this.webGl.context;
        const texture = gltf.textures[textureInfo.index];
        const gltfSampler = gltf.samplers[texture?.sampler];
        if (gltfSampler === undefined) {
            return;
        }
        const wrapS = clamp.clampU ? GL.CLAMP_TO_EDGE : gltfSampler.wrapS;
        const wrapT = clamp.clampV ? GL.CLAMP_TO_EDGE : gltfSampler.wrapT;
        const key = [wrapS, wrapT, gltfSampler.minFilter, gltfSampler.magFilter].join(",");
        let sampler = this.msfsSamplers.get(key);
        if (sampler === undefined) {
            sampler = gl.createSampler();
            gl.samplerParameteri(sampler, GL.TEXTURE_WRAP_S, wrapS);
            gl.samplerParameteri(sampler, GL.TEXTURE_WRAP_T, wrapT);
            gl.samplerParameteri(sampler, GL.TEXTURE_MIN_FILTER, gltfSampler.minFilter);
            gl.samplerParameteri(sampler, GL.TEXTURE_MAG_FILTER, gltfSampler.magFilter);
            this.msfsSamplers.set(key, sampler);
        }
        gl.bindSampler(unit, sampler);
        this.msfsSamplerUnits.push(unit);
    }

    // MSFS collision gizmos, fade volumes and lights as wireframes (see msfs_helpers.js): faint
    // where hidden behind geometry, solid where visible.
    drawMsfsHelpers(state) {
        const params = state.renderingParameters;
        if (!params.showMsfsColliders && !params.showMsfsLights) {
            return;
        }
        const extents = state.userCamera.sceneExtents;
        const sceneSize = vec3.distance(extents.min, extents.max);
        const groups = buildMsfsHelperLines(state.gltf, state.sceneIndex, {
            colliders: params.showMsfsColliders,
            lights: params.showMsfsLights,
            size: Number.isFinite(sceneSize) && sceneSize > 0 ? Math.min(Math.max(sceneSize * 0.05, 0.25), 2) : 0.5
        });
        if (groups.length === 0) {
            return;
        }
        const gl = this.webGl.context;
        const fragmentHash = this.shaderCache.selectShader("simple.frag", []);
        const vertexHash = this.shaderCache.selectShader("picking.vert", []);
        this.shader = this.shaderCache.getShaderProgram(fragmentHash, vertexHash);
        gl.useProgram(this.shader.program);
        this.shader.updateUniform("u_ViewProjectionMatrix", this.viewProjectionMatrix);
        this.shader.updateUniform("u_ModelMatrix", mat4.create());
        const location = this.shader.getAttributeLocation("a_position");
        if (location === null) {
            return;
        }
        if (this.msfsHelperBuffer === undefined) {
            this.msfsHelperBuffer = gl.createBuffer();
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, this.msfsHelperBuffer);
        // Earlier passes can leave attribute arrays enabled without data for this draw.
        for (let i = 0; i < gl.getParameter(GL.MAX_VERTEX_ATTRIBS); i++) {
            if (gl.getVertexAttrib(i, GL.VERTEX_ATTRIB_ARRAY_ENABLED)) {
                gl.disableVertexAttribArray(i);
            }
        }
        gl.enableVertexAttribArray(location);
        gl.enable(GL.BLEND);
        gl.blendFuncSeparate(GL.SRC_ALPHA, GL.ONE_MINUS_SRC_ALPHA, GL.ONE, GL.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(false);
        // simple.frag has no output for the integer tone map flag attachment, which WebGL
        // rejects while that draw buffer is enabled; the flag of the surface below is kept.
        gl.drawBuffers([GL.COLOR_ATTACHMENT0, GL.NONE]);
        for (const [depthFunc, alpha] of [[GL.GREATER, 0.3], [GL.LEQUAL, 1.0]]) {
            gl.depthFunc(depthFunc);
            for (const { color, positions } of groups) {
                this.shader.updateUniform("u_Color", vec4.fromValues(color[0], color[1], color[2], alpha));
                gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STREAM_DRAW);
                gl.vertexAttribPointer(location, 3, GL.FLOAT, false, 0, 0);
                gl.drawArrays(GL.LINES, 0, positions.length / 3);
            }
        }
        gl.drawBuffers([GL.COLOR_ATTACHMENT0, GL.COLOR_ATTACHMENT1]);
        gl.depthFunc(GL.LEQUAL);
        gl.depthMask(true);
        gl.disable(GL.BLEND);
        gl.disableVertexAttribArray(location);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
    }

    /** Binds a 1x1 texture that stands in for a switched off texture, see NeutralTexels. */
    bindNeutralTexture(location, samplerName, textureUnit) {
        const gl = this.webGl.context;
        const texel = NeutralTexels[samplerName] ?? WhiteTexel;
        const key = texel.join(",");
        this.neutralTextures ??= new Map();
        let texture = this.neutralTextures.get(key);
        if (texture === undefined) {
            texture = gl.createTexture();
            gl.bindTexture(GL.TEXTURE_2D, texture);
            gl.texImage2D(GL.TEXTURE_2D, 0, GL.RGBA, 1, 1, 0, GL.RGBA, GL.UNSIGNED_BYTE, new Uint8Array(texel));
            gl.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.NEAREST);
            gl.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.NEAREST);
            this.neutralTextures.set(key, texture);
        }
        gl.activeTexture(GL.TEXTURE0 + textureUnit);
        gl.bindTexture(GL.TEXTURE_2D, texture);
        gl.bindSampler(textureUnit, null);
        gl.uniform1i(location, textureUnit);
    }

    /** MSFS helper geometry, hidden materials and, while materials are isolated, all other materials are hidden. */
    isHiddenPrimitive(state, primitive) {
        if (state.hiddenMaterialIndices?.has(primitive.material)) {
            return true;
        }
        if (
            state.isolatedMaterialIndices !== undefined &&
            !state.isolatedMaterialIndices.has(primitive.material)
        ) {
            return true;
        }
        return isMsfsHiddenPrimitive(state.gltf, primitive, state.renderingParameters);
    }

    prepareScene(state, scene) {
        const params = state.renderingParameters;
        const newNodes = scene.gatherNodes(state.gltf, state.renderingParameters.enabledExtensions);
        this.selectionDrawables = newNodes.selectableNodes
            .filter((node) => node.mesh !== undefined)
            .reduce(
                (accumulator, node) =>
                    accumulator.concat(
                        state.gltf.meshes[node.mesh].primitives.map((primitive, index) => {
                            return { node: node, primitive: primitive, primitiveIndex: index };
                        })
                    ),
                []
            )
            .filter(({ primitive }) => !this.isHiddenPrimitive(state, primitive));
        this.hoverDrawables = newNodes.hoverableNodes
            .filter((node) => node.mesh !== undefined)
            .reduce(
                (accumulator, node) =>
                    accumulator.concat(
                        state.gltf.meshes[node.mesh].primitives.map((primitive, index) => {
                            return { node: node, primitive: primitive, primitiveIndex: index };
                        })
                    ),
                []
            )
            .filter(({ primitive }) => !this.isHiddenPrimitive(state, primitive));

        // check if nodes have changed since previous frame to avoid unnecessary updates
        if (
            newNodes.nodes.length === this.nodes?.length &&
            newNodes.nodes.every((element, i) => element === this.nodes[i]) &&
            params.showMsfsInvisibleMaterials === this.showMsfsInvisibleMaterials &&
            state.isolatedMaterialIndices === this.isolatedMaterialIndices &&
            state.hiddenMaterialIndices === this.hiddenMaterialIndices
        ) {
            return;
        }
        this.nodes = newNodes.nodes;
        this.showMsfsInvisibleMaterials = params.showMsfsInvisibleMaterials;
        this.isolatedMaterialIndices = state.isolatedMaterialIndices;
        this.hiddenMaterialIndices = state.hiddenMaterialIndices;

        // collect drawables by essentially zipping primitives (for geometry and material)
        // and nodes for the transform
        const drawables = this.nodes
            .filter((node) => node.mesh !== undefined)
            .reduce(
                (accumulator, node) =>
                    accumulator.concat(
                        state.gltf.meshes[node.mesh].primitives.map((primitive, index) => {
                            return { node: node, primitive: primitive, primitiveIndex: index };
                        })
                    ),
                []
            )
            .filter(
                ({ primitive }) =>
                    primitive.material !== undefined && !this.isHiddenPrimitive(state, primitive)
            );
        this.drawables = drawables;

        // opaque drawables don't need sorting
        this.opaqueDrawables = drawables.filter(
            ({ primitive }) =>
                state.gltf.materials[primitive.material].alphaMode !== "BLEND" &&
                (state.gltf.materials[primitive.material].extensions === undefined ||
                    state.gltf.materials[primitive.material].extensions
                        .KHR_materials_transmission === undefined) &&
                primitive.extensions?.KHR_gaussian_splatting === undefined
        );

        let counter = 0;
        this.opaqueDrawables = Object.groupBy(this.opaqueDrawables, (a) => {
            const winding = Math.sign(mat4.determinant(a.node.getRenderedWorldTransform()));
            const id = `${a.node.mesh}_${winding}_${a.primitiveIndex}`;
            // Disable instancing for skins, morph targets and if the GPU attributes limit is reached.
            // Additionally we define a new id for each instance of the EXT_mesh_gpu_instancing extension.
            if (
                a.node.skin ||
                a.primitive.targets.length > 0 ||
                a.primitive.glAttributes.length + 4 > this.maxVertAttributes ||
                a.node.instanceMatrices
            ) {
                if (
                    a.node.instanceMatrices &&
                    a.primitive.glAttributes.length + 4 > this.maxVertAttributes
                ) {
                    console.warn(
                        `EXT_mesh_gpu_instancing disabled for mesh ${a.node.mesh} because the GPU vertex attribute limit is reached.`
                    );
                }
                counter++;
                return id + "_" + counter;
            }
            return id;
        });

        // transparent drawables need sorting before they can be drawn
        this.transparentDrawables = drawables.filter(
            ({ primitive }) =>
                (state.gltf.materials[primitive.material].alphaMode === "BLEND" &&
                    (state.gltf.materials[primitive.material].extensions === undefined ||
                        state.gltf.materials[primitive.material].extensions
                            .KHR_materials_transmission === undefined)) ||
                primitive.extensions?.KHR_gaussian_splatting !== undefined
        );

        this.transmissionDrawables = drawables.filter(
            ({ primitive }) =>
                state.gltf.materials[primitive.material].extensions !== undefined &&
                state.gltf.materials[primitive.material].extensions.KHR_materials_transmission !==
                    undefined &&
                primitive.extensions?.KHR_gaussian_splatting === undefined
        );

        this.scatterDrawables = drawables.filter(
            ({ primitive }) =>
                state.gltf.materials[primitive.material].extensions !== undefined &&
                state.gltf.materials[primitive.material].extensions.KHR_materials_volume_scatter !==
                    undefined &&
                state.gltf.materials[primitive.material].extensions.KHR_materials_volume !==
                    undefined &&
                primitive.extensions?.KHR_gaussian_splatting === undefined
        );
    }

    // render complete gltf scene with given camera
    drawScene(state, scene) {
        this.prepareScene(state, scene);

        let currentCamera = undefined;

        if (state.cameraNodeIndex === undefined) {
            currentCamera = state.userCamera;
            currentCamera.perspective.aspectRatio = this.currentWidth / this.currentHeight;
        } else {
            currentCamera = state.gltf.cameras[state.gltf.nodes[state.cameraNodeIndex].camera];
            if (currentCamera === undefined) {
                throw new Error("Camera is misconfigured.");
            }
            currentCamera.setNode(state.gltf, state.cameraNodeIndex);
        }

        let aspectHeight = this.currentHeight;
        let aspectWidth = this.currentWidth;
        let aspectOffsetX = 0;
        let aspectOffsetY = 0;
        const currentAspectRatio = aspectWidth / aspectHeight;
        if (currentCamera.type === "perspective") {
            if (currentCamera.perspective.aspectRatio) {
                if (currentCamera.perspective.aspectRatio > currentAspectRatio) {
                    aspectHeight = (aspectWidth * 1) / currentCamera.perspective.aspectRatio;
                } else {
                    aspectWidth = aspectHeight * currentCamera.perspective.aspectRatio;
                }
            }
        } else {
            const orthoAspect = currentCamera.orthographic.xmag / currentCamera.orthographic.ymag;
            if (orthoAspect > currentAspectRatio) {
                aspectHeight = (aspectWidth * 1) / orthoAspect;
            } else {
                aspectWidth = aspectHeight * orthoAspect;
            }
        }
        if (aspectHeight < this.currentHeight) {
            aspectOffsetY = (this.currentHeight - aspectHeight) / 2;
        }
        if (aspectWidth < this.currentWidth) {
            aspectOffsetX = (this.currentWidth - aspectWidth) / 2;
        }

        this.projMatrix = currentCamera.getProjectionMatrix(currentAspectRatio);
        this.viewMatrix = currentCamera.getViewMatrix(state.gltf);
        this.currentCameraPosition = currentCamera.getPosition(state.gltf);

        this.visibleLights = this.getVisibleLights(state.gltf, this.nodes);
        if (
            this.visibleLights.length === 0 &&
            !state.renderingParameters.useIBL &&
            state.renderingParameters.useDirectionalLightsWithDisabledIBL
        ) {
            this.visibleLights.push([null, this.lightKey]);
            this.visibleLights.push([null, this.lightFill]);
        }

        mat4.multiply(this.viewProjectionMatrix, this.projMatrix, this.viewMatrix);

        // Update skins.
        for (const node of state.gltf.nodes) {
            if (node.mesh !== undefined && node.skin !== undefined) {
                this.updateSkin(state, node);
            }
        }

        const instanceWorldTransforms = [];
        for (const instance of Object.values(this.opaqueDrawables)) {
            let instanceOffset = undefined;
            if (instance.length > 1) {
                instanceOffset = [];
                for (const iDrawable of instance) {
                    instanceOffset.push(iDrawable.node.getRenderedWorldTransform());
                }
            } else if (instance[0].node.instanceMatrices !== undefined) {
                // Set instance matrices for EXT_mesh_gpu_instancing extension
                if (instance[0].primitive.glAttributes.length + 4 <= this.maxVertAttributes) {
                    instanceOffset = instance[0].node.instanceWorldTransforms;
                }
            }
            instanceWorldTransforms.push(instanceOffset);
        }

        const scatterEnabled =
            this.scatterDrawables.length > 0 &&
            state.renderingParameters.enabledExtensions.KHR_materials_volume_scatter &&
            state.renderingParameters.enabledExtensions.KHR_materials_volume;

        if (scatterEnabled) {
            this.webGl.context.bindFramebuffer(
                this.webGl.context.FRAMEBUFFER,
                this.scatterFramebuffer
            );
            if (
                state.renderingParameters.debugOutput ===
                GltfState.DebugOutput.volumeScatter.PRE_SCATTER_PASS
            ) {
                this.webGl.context.bindFramebuffer(this.webGl.context.FRAMEBUFFER, null);
            }
            this.webGl.context.viewport(aspectOffsetX, aspectOffsetY, aspectWidth, aspectHeight);

            let counter = 1;
            for (const drawable of this.scatterDrawables) {
                let renderpassConfiguration = {};
                renderpassConfiguration.linearOutput = true;
                renderpassConfiguration.scatter = true;
                renderpassConfiguration.drawID = counter;
                renderpassConfiguration.frameBufferSize = [this.currentWidth, this.currentHeight];
                this.drawPrimitive(
                    state,
                    renderpassConfiguration,
                    drawable.primitive,
                    drawable.node,
                    this.viewProjectionMatrix
                );
                ++counter;
            }
            this.webGl.context.bindFramebuffer(this.webGl.context.FRAMEBUFFER, null);
        }
        if (
            state.renderingParameters.debugOutput ===
            GltfState.DebugOutput.volumeScatter.PRE_SCATTER_PASS
        ) {
            return;
        }

        let pickingProjection = undefined;
        let pickingViewProjection = mat4.create();

        let pickingX = state.selectionPositions[0].x;
        let pickingY = state.selectionPositions[0].y;

        // Draw a 1x1 texture for picking
        if (state.triggerSelection && pickingX !== undefined && pickingY !== undefined) {
            pickingProjection = currentCamera.getProjectionMatrixForPixel(
                pickingX - aspectOffsetX,
                this.currentHeight - pickingY - aspectOffsetY,
                aspectWidth,
                aspectHeight
            );
            mat4.multiply(pickingViewProjection, pickingProjection, this.viewMatrix);
            this.webGl.context.bindFramebuffer(
                this.webGl.context.FRAMEBUFFER,
                this.pickingFramebuffer
            );
            this.webGl.context.viewport(0, 0, 1, 1);

            for (const drawable of this.selectionDrawables) {
                let renderpassConfiguration = {};
                renderpassConfiguration.picking = true;
                this.drawPrimitive(
                    state,
                    renderpassConfiguration,
                    drawable.primitive,
                    drawable.node,
                    pickingViewProjection
                );
            }
        }

        pickingX = state.hoverPositions[0].x;
        pickingY = state.hoverPositions[0].y;

        const needsHover = state.graphController.needsHover();
        const calcHoverInfo =
            (state.enableHover || needsHover) && pickingX !== undefined && pickingY !== undefined;

        // Draw a 1x1 texture for hover
        if (calcHoverInfo) {
            // We do not need to recalculate the picking projection matrix if selection and hover use the same position
            if (
                pickingProjection === undefined ||
                pickingX !== state.selectionPositions[0].x ||
                pickingY !== state.selectionPositions[0].y
            ) {
                pickingProjection = currentCamera.getProjectionMatrixForPixel(
                    pickingX - aspectOffsetX,
                    this.currentHeight - pickingY - aspectOffsetY,
                    aspectWidth,
                    aspectHeight
                );
                mat4.multiply(pickingViewProjection, pickingProjection, this.viewMatrix);
            }
            this.webGl.context.bindFramebuffer(
                this.webGl.context.FRAMEBUFFER,
                this.hoverFramebuffer
            );
            this.webGl.context.viewport(0, 0, 1, 1);

            for (const drawable of this.hoverDrawables) {
                let renderpassConfiguration = {};
                renderpassConfiguration.picking = true;
                this.drawPrimitive(
                    state,
                    renderpassConfiguration,
                    drawable.primitive,
                    drawable.node,
                    pickingViewProjection
                );
            }
        }

        // If any transmissive drawables are present, render all opaque and transparent drawables into a separate framebuffer.
        if (this.transmissionDrawables.length > 0) {
            // Render transmission sample texture
            this.webGl.context.bindFramebuffer(
                this.webGl.context.FRAMEBUFFER,
                this.opaqueFramebufferMSAA
            );
            this.webGl.context.viewport(
                0,
                0,
                this.opaqueFramebufferWidth,
                this.opaqueFramebufferHeight
            );

            // Render environment for the transmission background
            this.environmentRenderer.drawEnvironmentMap(
                this.webGl,
                this.viewProjectionMatrix,
                state,
                this.shaderCache,
                ["TRANSMISSION_PASS 1"]
            );

            let drawableCounter = 0;
            for (const instance of Object.values(this.opaqueDrawables)) {
                const drawable = instance[0];
                let renderpassConfiguration = {};
                renderpassConfiguration.linearOutput = true;
                renderpassConfiguration.transmission = true;
                renderpassConfiguration.frameBufferSize = [
                    this.opaqueFramebufferWidth,
                    this.opaqueFramebufferHeight
                ];
                const instanceOffset = instanceWorldTransforms[drawableCounter];
                drawableCounter++;

                let sampledTextures = {};
                if (scatterEnabled) {
                    sampledTextures.scatterSampleTexture = this.scatterFrontTexture;
                    sampledTextures.scatterDepthSampleTexture = this.scatterDepthTexture;
                }
                this.drawPrimitive(
                    state,
                    renderpassConfiguration,
                    drawable.primitive,
                    drawable.node,
                    this.viewProjectionMatrix,
                    sampledTextures,
                    instanceOffset
                );
            }

            this.transparentDrawables = sortByMsfsDrawOrder(
                state.gltf,
                currentCamera.sortPrimitivesByDepth(state.gltf, this.transparentDrawables)
            );
            for (const drawable of this.transparentDrawables) {
                let renderpassConfiguration = {};
                renderpassConfiguration.linearOutput = true;
                renderpassConfiguration.transmission = true;
                renderpassConfiguration.frameBufferSize = [
                    this.opaqueFramebufferWidth,
                    this.opaqueFramebufferHeight
                ];
                this.drawPrimitive(
                    state,
                    renderpassConfiguration,
                    drawable.primitive,
                    drawable.node,
                    this.viewProjectionMatrix
                );
            }

            // "blit" the multisampled opaque texture into the color buffer, which adds antialiasing
            this.webGl.context.bindFramebuffer(
                this.webGl.context.READ_FRAMEBUFFER,
                this.opaqueFramebufferMSAA
            );
            this.webGl.context.bindFramebuffer(
                this.webGl.context.DRAW_FRAMEBUFFER,
                this.opaqueFramebuffer
            );
            this.webGl.context.blitFramebuffer(
                0,
                0,
                this.opaqueFramebufferWidth,
                this.opaqueFramebufferHeight,
                0,
                0,
                this.opaqueFramebufferWidth,
                this.opaqueFramebufferHeight,
                this.webGl.context.COLOR_BUFFER_BIT,
                this.webGl.context.NEAREST
            );

            // Create Framebuffer Mipmaps
            this.webGl.context.bindTexture(this.webGl.context.TEXTURE_2D, this.opaqueRenderTexture);

            this.webGl.context.generateMipmap(this.webGl.context.TEXTURE_2D);
        }

        // Re-attach mainFramebuffer and splatFramebuffer COLOR_ATTACHMENT0 if
        // the floating-point toggle changed.
        const wantFP =
            state.renderingParameters.floatingPointFramebuffer !== false &&
            this.mainTextureHDR !== undefined;
        if (wantFP !== this._floatingPointFramebuffer) {
            this._floatingPointFramebuffer = wantFP;
            const gl = this.webGl.context;
            const colorTex = wantFP ? this.mainTextureHDR : this.mainTexture;
            const splatColorTex = wantFP ? this.splatColorTextureHDR : this.splatColorTexture;
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.mainFramebuffer);
            gl.framebufferTexture2D(
                gl.FRAMEBUFFER,
                gl.COLOR_ATTACHMENT0,
                gl.TEXTURE_2D,
                colorTex,
                0
            );
            if (this.splatFramebuffer) {
                gl.bindFramebuffer(gl.FRAMEBUFFER, this.splatFramebuffer);
                gl.framebufferTexture2D(
                    gl.FRAMEBUFFER,
                    gl.COLOR_ATTACHMENT0,
                    gl.TEXTURE_2D,
                    splatColorTex ?? this.splatColorTexture,
                    0
                );
            }
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        }

        this.webGl.context.bindFramebuffer(this.webGl.context.FRAMEBUFFER, this.mainFramebuffer);
        this.webGl.context.viewport(aspectOffsetX, aspectOffsetY, aspectWidth, aspectHeight);

        // Environment (always linear in this pass)
        this.environmentRenderer.drawEnvironmentMap(
            this.webGl,
            this.viewProjectionMatrix,
            state,
            this.shaderCache,
            ["LINEAR_OUTPUT 1"]
        );

        // Physics debug view
        if (state.physicsController.enabled) {
            const lines = state.physicsController.getDebugLineData();
            if (lines.length !== 0) {
                const vertexShader = "picking.vert";
                const fragmentShader = "simple.frag";
                const fragmentHash = this.shaderCache.selectShader(fragmentShader, []);
                const vertexHash = this.shaderCache.selectShader(vertexShader, []);
                this.shader = this.shaderCache.getShaderProgram(fragmentHash, vertexHash);
                this.webGl.context.useProgram(this.shader.program);
                this.shader.updateUniform("u_ViewProjectionMatrix", this.viewProjectionMatrix);
                this.shader.updateUniform("u_ModelMatrix", mat4.create());
                this.shader.updateUniform("u_Color", vec4.fromValues(1.0, 0.0, 0.0, 1.0));
                const location = this.shader.getAttributeLocation("a_position");
                if (location !== null) {
                    if (this.physicsDebugBuffer === undefined) {
                        this.physicsDebugBuffer = this.webGl.context.createBuffer();
                    }
                    this.webGl.context.bindBuffer(
                        this.webGl.context.ARRAY_BUFFER,
                        this.physicsDebugBuffer
                    );
                    this.webGl.context.bufferData(
                        this.webGl.context.ARRAY_BUFFER,
                        new Float32Array(lines),
                        this.webGl.context.STATIC_DRAW
                    );
                    this.webGl.context.vertexAttribPointer(
                        location,
                        3,
                        this.webGl.context.FLOAT,
                        false,
                        0,
                        0
                    );
                    this.webGl.context.enableVertexAttribArray(location);
                    this.webGl.context.drawArrays(this.webGl.context.LINES, 0, lines.length / 3);
                    this.webGl.context.disableVertexAttribArray(location);
                    this.webGl.context.bindBuffer(this.webGl.context.ARRAY_BUFFER, null);
                }
            }
        }

        let drawableCounter = 0;
        for (const instance of Object.values(this.opaqueDrawables)) {
            const drawable = instance[0];
            let renderpassConfiguration = {};
            renderpassConfiguration.linearOutput = true;
            renderpassConfiguration.frameBufferSize = [this.currentWidth, this.currentHeight];
            const instanceOffset = instanceWorldTransforms[drawableCounter];
            drawableCounter++;
            let sampledTextures = {};
            if (scatterEnabled) {
                sampledTextures.scatterSampleTexture = this.scatterFrontTexture;
                sampledTextures.scatterDepthSampleTexture = this.scatterDepthTexture;
            }
            this.drawPrimitive(
                state,
                renderpassConfiguration,
                drawable.primitive,
                drawable.node,
                this.viewProjectionMatrix,
                sampledTextures,
                instanceOffset
            );
        }

        // filter materials with transmission extension
        this.transmissionDrawables = currentCamera.sortPrimitivesByDepth(
            state.gltf,
            this.transmissionDrawables
        );
        for (const drawable of this.transmissionDrawables.filter((a) => a.depth <= 0)) {
            let renderpassConfiguration = {};
            renderpassConfiguration.linearOutput = true;
            renderpassConfiguration.frameBufferSize = [this.currentWidth, this.currentHeight];
            let sampledTextures = {};
            sampledTextures.transmissionSampleTexture = this.opaqueRenderTexture;
            if (scatterEnabled) {
                sampledTextures.scatterSampleTexture = this.scatterFrontTexture;
                sampledTextures.scatterDepthSampleTexture = this.scatterDepthTexture;
            }
            this.drawPrimitive(
                state,
                renderpassConfiguration,
                drawable.primitive,
                drawable.node,
                this.viewProjectionMatrix,
                sampledTextures
            );
        }

        this.transparentDrawables = sortByMsfsDrawOrder(
            state.gltf,
            currentCamera.sortPrimitivesByDepth(state.gltf, this.transparentDrawables)
        );
        this.needsRedraw = false;

        // ── Transparent drawables (non-splat and splat) in depth-sorted order ───
        // Splats are rendered one at a time into the dedicated splatFramebuffer
        // (which shares mainDepthTexture for depth-testing) and
        // immediately composited back into mainFramebuffer before the next
        // drawable is processed.  This preserves correct depth-sorted blending
        // order between splats and regular transparent geometry.
        for (const drawable of this.transparentDrawables.filter((a) => a.depth <= 0)) {
            if (
                drawable.primitive.extensions?.KHR_gaussian_splatting !== undefined &&
                state.renderingParameters.enabledExtensions.KHR_gaussian_splatting
            ) {
                if (currentCamera.type !== "perspective") {
                    console.warn("Splat rendering with non-perspective projection is undefined");
                }
                // Isolate this splat in its own framebuffer pass.
                if (this.splatFramebuffer) {
                    const gl = this.webGl.context;

                    // Clear only the colour attachment – depth is shared with
                    // mainFramebuffer and must not be touched here.
                    gl.bindFramebuffer(gl.FRAMEBUFFER, this.splatFramebuffer);
                    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
                    gl.viewport(aspectOffsetX, aspectOffsetY, aspectWidth, aspectHeight);

                    this.drawSplat(
                        state,
                        drawable.primitive,
                        drawable.node,
                        this.projMatrix,
                        this.viewMatrix
                    );

                    // Composite into mainFramebuffer.
                    gl.bindFramebuffer(gl.FRAMEBUFFER, this.mainFramebuffer);
                    gl.viewport(aspectOffsetX, aspectOffsetY, aspectWidth, aspectHeight);
                    this.splatCompositePass(state, drawable.primitive.linear);
                }
            } else {
                let renderpassConfiguration = {};
                renderpassConfiguration.linearOutput = true;
                renderpassConfiguration.frameBufferSize = [this.currentWidth, this.currentHeight];
                this.drawPrimitive(
                    state,
                    renderpassConfiguration,
                    drawable.primitive,
                    drawable.node,
                    this.viewProjectionMatrix
                );
            }
        }

        this.drawMsfsHelpers(state);

        // Handle selection
        if (state.triggerSelection) {
            this.webGl.context.bindFramebuffer(
                this.webGl.context.FRAMEBUFFER,
                this.pickingFramebuffer
            );
            this.webGl.context.viewport(0, 0, 1, 1);
            state.triggerSelection = false;
            this.webGl.context.readBuffer(this.webGl.context.COLOR_ATTACHMENT0);

            // Read pixel under controller (e.g. mouse cursor), which contains the picking ID
            const pixels = new Uint32Array(1);
            this.webGl.context.readPixels(
                0,
                0,
                1,
                1,
                this.webGl.context.RED_INTEGER,
                this.webGl.context.UNSIGNED_INT,
                pixels
            );

            // Compute ray origin in world space. This is the near plane position of the current pixel.
            pickingX = state.selectionPositions[0].x;
            pickingY = state.selectionPositions[0].y;
            const x = pickingX - aspectOffsetX;
            const y = this.currentHeight - pickingY - aspectOffsetY;
            const nearPlane = currentCamera.getNearPlaneForPixel(
                x + 0.5,
                y + 0.5,
                aspectWidth,
                aspectHeight
            );
            const zNear = currentCamera.perspective?.znear
                ? currentCamera.perspective.znear
                : currentCamera.orthographic.znear;
            let rayOrigin = vec3.fromValues(nearPlane.left, nearPlane.bottom, -zNear);
            vec3.transformMat4(rayOrigin, rayOrigin, currentCamera.getTransformMatrix(state.gltf));

            let pickingResult = {
                node: undefined,
                position: undefined,
                rayOrigin: rayOrigin,
                controller: 0
            };

            // Search for node with matching picking ID
            let found = false;
            for (const node of state.gltf.nodes) {
                if (node.pickingColor === (pixels[0] & PickingNodeMask)) {
                    found = true;
                    pickingResult.node = node;
                    pickingResult.primitiveIndex = pixels[0] >>> 24;
                    break;
                }
            }

            // If a node was found, we need to calculate the ray intersection position
            if (found) {
                // WebGL does not allow reading from depth buffer
                this.webGl.context.readBuffer(this.webGl.context.COLOR_ATTACHMENT1);
                const position = new Uint32Array(1);
                this.webGl.context.readPixels(
                    0,
                    0,
                    1,
                    1,
                    this.webGl.context.RED_INTEGER,
                    this.webGl.context.UNSIGNED_INT,
                    position
                );

                // Transform uint to float [-1, 1] in clip space
                const z = (position[0] / 4294967295) * 2.0 - 1.0;

                // Get view space position
                const clipSpacePosition = vec4.fromValues(0, 0, z, 1);
                vec4.transformMat4(
                    clipSpacePosition,
                    clipSpacePosition,
                    mat4.invert(mat4.create(), pickingProjection)
                );

                // Divide by w to get normalized device coordinates
                vec4.divide(
                    clipSpacePosition,
                    clipSpacePosition,
                    vec4.fromValues(
                        clipSpacePosition[3],
                        clipSpacePosition[3],
                        clipSpacePosition[3],
                        clipSpacePosition[3]
                    )
                );
                const worldPos = vec4.transformMat4(
                    vec4.create(),
                    clipSpacePosition,
                    mat4.invert(mat4.create(), this.viewMatrix)
                );
                pickingResult.position = vec3.fromValues(worldPos[0], worldPos[1], worldPos[2]);
            }

            // Send picking result to Interactivity engine
            state.graphController.receiveSelection(pickingResult);

            if (state.selectionCallback) {
                state.selectionCallback(pickingResult);
            }
        }

        if (calcHoverInfo) {
            this.webGl.context.bindFramebuffer(
                this.webGl.context.FRAMEBUFFER,
                this.hoverFramebuffer
            );
            this.webGl.context.viewport(0, 0, 1, 1);
            this.webGl.context.readBuffer(this.webGl.context.COLOR_ATTACHMENT0);

            // Read pixel under controller (e.g. mouse cursor), which contains the picking ID
            const pixels = new Uint32Array(1);
            this.webGl.context.readPixels(
                0,
                0,
                1,
                1,
                this.webGl.context.RED_INTEGER,
                this.webGl.context.UNSIGNED_INT,
                pixels
            );

            let pickingResult = {
                node: undefined,
                controller: 0
            };

            // Search for node with matching picking ID
            for (const node of state.gltf.nodes) {
                if (node.pickingColor === (pixels[0] & PickingNodeMask)) {
                    pickingResult.node = node;
                    break;
                }
            }

            // Send picking result to Interactivity engine
            state.graphController.receiveHover(pickingResult);

            if (state.enableHover && state.hoverCallback) {
                state.hoverCallback(pickingResult);
            }
        }

        // ── Final tonemapping pass → canvas ───────────────────────────────────
        this.tonemapPass(state, aspectOffsetX, aspectOffsetY, aspectWidth, aspectHeight);

        state.needsRedraw = this.needsRedraw;
    }

    drawSplat(state, primitive, node, projectionMatrix, viewMatrix) {
        if (primitive.skip) return;
        // Request an async worker sort each frame (no-op if the previous sort
        // has not yet finished or no worker is available).
        const modelViewMatrix = mat4.multiply(mat4.create(), viewMatrix, node.worldTransform);
        primitive.requestSort(modelViewMatrix);
        if (primitive.sortPending) {
            this.needsRedraw = true;
        }
        let defines = primitive.defines.slice();
        if (primitive.linear === true) {
            defines.push("LINEAR_OUTPUT 1");
        }

        // Debug views
        if (state.renderingParameters.debugOutput !== GltfState.DebugOutput.NONE) {
            if (
                state.renderingParameters.debugOutput ===
                GltfState.DebugOutput.gaussianSplatting.SH_DEGREE_0
            ) {
                defines = defines.filter(
                    (define) => !define.startsWith("HAS_GAUSSIAN_SPLATTING_DEGREE")
                );
            } else if (
                state.renderingParameters.debugOutput ===
                GltfState.DebugOutput.gaussianSplatting.SH_DEGREE_1
            ) {
                defines = defines.filter(
                    (define) =>
                        define !== "HAS_GAUSSIAN_SPLATTING_DEGREE_2 1" &&
                        define !== "HAS_GAUSSIAN_SPLATTING_DEGREE_3 1"
                );
            } else if (
                state.renderingParameters.debugOutput ===
                GltfState.DebugOutput.gaussianSplatting.SH_DEGREE_2
            ) {
                defines = defines.filter(
                    (define) => define !== "HAS_GAUSSIAN_SPLATTING_DEGREE_3 1"
                );
            }
        }

        const fragmentHash = this.shaderCache.selectShader("splat.frag", defines);
        const vertexHash = this.shaderCache.selectShader("splat.vert", defines);
        if (fragmentHash && vertexHash) {
            this.shader = this.shaderCache.getShaderProgram(fragmentHash, vertexHash);
        }

        if (this.shader === undefined) {
            return;
        }

        this.webGl.context.useProgram(this.shader.program);

        this.webGl.context.uniformMatrix4fv(
            this.shader.getUniformLocation("u_ModelMatrix"),
            false,
            node.worldTransform
        );

        const rotationMatrixInv = mat3.create();
        mat3.fromQuat(rotationMatrixInv, node.worldQuaternion);
        mat3.transpose(rotationMatrixInv, rotationMatrixInv);
        this.shader.updateUniform("u_ModelRotationInv", rotationMatrixInv);

        this.webGl.context.uniformMatrix4fv(
            this.shader.getUniformLocation("u_ViewMatrix"),
            false,
            viewMatrix
        );
        this.webGl.context.uniformMatrix4fv(
            this.shader.getUniformLocation("u_ProjectionMatrix"),
            false,
            projectionMatrix
        );
        this.webGl.context.uniform2i(
            this.shader.getUniformLocation("u_FramebufferSize"),
            this.currentWidth,
            this.currentHeight
        );
        // The projection matrix stores the focal length in the first and second element of the diagonal.
        // We need to convert from NDC space to screen space, which is done by multiplying with the framebuffer dimensions and dividing by 2, since NDC goes from -1 to 1.
        this.webGl.context.uniform2f(
            this.shader.getUniformLocation("u_FocalLength"),
            this.projMatrix[0] * this.currentWidth * 0.5,
            this.projMatrix[5] * this.currentHeight * 0.5
        );
        this.shader.updateUniform("u_TextureWidth", primitive.splatTextureWidth);
        this.shader.updateUniform("u_Camera", this.currentCameraPosition);

        this.webGl.context.enable(GL.BLEND);
        this.webGl.context.blendFuncSeparate(
            GL.ONE,
            GL.ONE_MINUS_SRC_ALPHA,
            GL.ONE,
            GL.ONE_MINUS_SRC_ALPHA
        );
        this.webGl.context.blendEquation(GL.FUNC_ADD);
        this.webGl.context.depthMask(false);

        let textureIndex = 0;

        let location = this.shader.getUniformLocation(primitive.positionTextureInfo.samplerName);
        this.webGl.setTexture(location, state.gltf, primitive.positionTextureInfo, textureIndex++);

        location = this.shader.getUniformLocation(primitive.rotationTextureInfo.samplerName);
        this.webGl.setTexture(location, state.gltf, primitive.rotationTextureInfo, textureIndex++);

        location = this.shader.getUniformLocation(primitive.scaleTextureInfo.samplerName);
        this.webGl.setTexture(location, state.gltf, primitive.scaleTextureInfo, textureIndex++);

        location = this.shader.getUniformLocation(primitive.opacityTextureInfo.samplerName);
        this.webGl.setTexture(location, state.gltf, primitive.opacityTextureInfo, textureIndex++);

        location = this.shader.getUniformLocation(primitive.shArray.samplerName);
        this.webGl.setTexture(location, state.gltf, primitive.shArray, textureIndex++);

        this.webGl.context.bindBuffer(this.webGl.context.ARRAY_BUFFER, this.splatVBO);
        location = this.shader.getAttributeLocation("a_position");
        this.webGl.context.enableVertexAttribArray(location);
        this.webGl.context.vertexAttribPointer(location, 2, GL.FLOAT, false, 0, 0);
        this.webGl.context.bindBuffer(this.webGl.context.ARRAY_BUFFER, null);

        this.webGl.context.bindBuffer(this.webGl.context.ARRAY_BUFFER, this.currentSortBuffer);
        this.webGl.context.bufferData(
            this.webGl.context.ARRAY_BUFFER,
            primitive.sortOrder,
            this.webGl.context.DYNAMIC_DRAW
        );
        location = this.shader.getAttributeLocation("a_instance_sort_index");
        this.webGl.context.enableVertexAttribArray(location);
        this.webGl.context.vertexAttribIPointer(location, 1, GL.UNSIGNED_INT, 0, 0);
        this.webGl.context.vertexAttribDivisor(location, 1);
        this.webGl.context.bindBuffer(this.webGl.context.ARRAY_BUFFER, null);

        this.webGl.context.drawArraysInstanced(GL.TRIANGLE_STRIP, 0, 4, primitive.sortOrder.length);

        this.webGl.context.vertexAttribDivisor(location, 0);
        this.webGl.context.disableVertexAttribArray(location);
        this.webGl.context.depthMask(true);
    }

    tonemapPass(state, aspectOffsetX, aspectOffsetY, aspectWidth, aspectHeight) {
        const gl = this.webGl.context;

        const fragDefines = [];
        this.pushFragParameterDefines(fragDefines, state);

        const fragHash = this.shaderCache.selectShader("tonemap_main.frag", fragDefines);
        const vertHash = this.shaderCache.selectShader("fullscreen.vert", []);
        if (!fragHash || !vertHash) return;
        const shader = this.shaderCache.getShaderProgram(fragHash, vertHash);
        if (!shader) return;

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(aspectOffsetX, aspectOffsetY, aspectWidth, aspectHeight);

        gl.useProgram(shader.program);
        shader.updateUniform("u_Exposure", state.renderingParameters.exposure, false);

        const hdrLoc = shader.getUniformLocation("u_MainSampler");
        gl.activeTexture(GL.TEXTURE0);
        gl.bindTexture(
            gl.TEXTURE_2D,
            this._floatingPointFramebuffer ? this.mainTextureHDR : this.mainTexture
        );
        gl.uniform1i(hdrLoc, 0);

        const tonemapLoc = shader.getUniformLocation("u_TonemapSampler");
        gl.activeTexture(GL.TEXTURE1);
        gl.bindTexture(GL.TEXTURE_2D, this.mainTonemapTexture);
        gl.uniform1i(tonemapLoc, 1);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.splatVBO);
        const posLoc = shader.getAttributeLocation("a_position");
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, GL.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);

        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.BLEND);
        gl.drawArrays(GL.TRIANGLE_STRIP, 0, 4);

        gl.enable(gl.DEPTH_TEST);
        gl.disableVertexAttribArray(posLoc);
    }

    // Composites the splat isolation framebuffer onto the currently-bound mainFramebuffer.
    splatCompositePass(state, isLinear) {
        const gl = this.webGl.context;

        const fragDefines = [];
        this.pushFragParameterDefines(fragDefines, state);
        if (isLinear) {
            fragDefines.push("LINEAR_OUTPUT 1");
        }
        const fragHash = this.shaderCache.selectShader("splat_composite.frag", fragDefines);
        const vertHash = this.shaderCache.selectShader("fullscreen.vert", []);
        if (!fragHash || !vertHash) return;
        const shader = this.shaderCache.getShaderProgram(fragHash, vertHash);
        if (!shader) return;

        gl.useProgram(shader.program);

        const splatTex = this._floatingPointFramebuffer
            ? this.splatColorTextureHDR
            : this.splatColorTexture;
        const samplerLoc = shader.getUniformLocation("u_SplatSampler");
        gl.activeTexture(GL.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, splatTex ?? this.splatColorTexture);
        gl.uniform1i(samplerLoc, 0);

        shader.updateUniform("u_Exposure", state.renderingParameters.exposure, false);

        // Bind the fullscreen quad VBO (a_position attribute expected by fullscreen.vert).
        gl.bindBuffer(gl.ARRAY_BUFFER, this.splatVBO);
        const posLoc = shader.getAttributeLocation("a_position");
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, GL.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);

        gl.enable(gl.BLEND);
        this.webGl.context.blendFuncSeparate(
            GL.SRC_ALPHA,
            GL.ONE_MINUS_SRC_ALPHA,
            GL.ONE,
            GL.ONE_MINUS_SRC_ALPHA
        );
        this.webGl.context.blendEquation(GL.FUNC_ADD);

        // No depth test or depth write – this is a pure compositing pass.
        gl.disable(gl.DEPTH_TEST);
        gl.depthMask(false);

        gl.drawArrays(GL.TRIANGLE_STRIP, 0, 4);

        // Restore state.
        gl.disable(gl.BLEND);
        gl.enable(gl.DEPTH_TEST);
        gl.depthMask(true);
        gl.disableVertexAttribArray(posLoc);
    }

    // vertices with given material
    // prettier-ignore
    drawPrimitive(state, renderpassConfiguration, primitive, node, viewProjectionMatrix, sampledTextures, instanceOffset = undefined)
    {
        if (primitive.skip) return;

        let material;
        if(primitive.mappings !== undefined && state.variant != "default" && state.gltf.extensions?.KHR_materials_variants.variants !== undefined)
        {
            const names = state.gltf.extensions.KHR_materials_variants.variants.map(obj => obj.name);
            const idx = names.indexOf(state.variant);
            let materialIdx = primitive.material;
            primitive.mappings.forEach(element => {
                if(element.variants.indexOf(idx) >= 0)
                {
                    materialIdx = element.material;
                }
            });
            material = state.gltf.materials[materialIdx];
        }
        else
        {
            material = state.gltf.materials[primitive.material];
        }

        //select shader permutation, compile and link program.

        let vertDefines = [];
        this.pushVertParameterDefines(
            vertDefines,
            state.renderingParameters,
            state.gltf,
            node,
            primitive,
            state.renderingParameters.debugOutput
        );
        vertDefines = primitive.defines.concat(vertDefines);
        vertDefines.push(...(material.msfsVertDefines ?? []));
        if (instanceOffset !== undefined) {
            vertDefines.push("USE_INSTANCING 1");
        }
        if (material.textureTransforms.length > 0) {
            for (let i = 0; i < material.textureTransforms.length; i++) {
                if (material.textureTransforms[i] !== undefined && material.textureTransforms[i].key === "Normal") {
                    vertDefines.push("HAS_VERT_NORMAL_UV_TRANSFORM 1");
                    break;
                }
            }
        }

        let fragDefines = material.getDefines(state.renderingParameters).concat(vertDefines);
        if (renderpassConfiguration.linearOutput)
        {
            fragDefines.push("LINEAR_OUTPUT 1");
        }
        if (renderpassConfiguration.transmission)
        {
            fragDefines.push("TRANSMISSION_PASS 1");
        }

        // POINTS, LINES, LINE_LOOP, LINE_STRIP
        if (primitive.mode < 4) {
            fragDefines.push("NOT_TRIANGLE 1");
            if (primitive.attributes?.NORMAL !== undefined && primitive.attributes?.TANGENT === undefined) {
                //Points or Lines with NORMAL but without TANGENT attributes SHOULD be rendered with standard lighting but ignoring any normal textures on the material.
                fragDefines = fragDefines.filter(e => e !== "HAS_NORMAL_MAP 1" && e !== "HAS_CLEARCOAT_NORMAL_MAP 1");
            }
        }

        this.pushFragParameterDefines(fragDefines, state);
        
        const vertexShader = renderpassConfiguration.picking ? "picking.vert" : "primitive.vert";
        let fragmentShader = "pbr.frag";
        if (material.type === "SG") {
            fragmentShader = "specular_glossiness.frag";
        } else if (renderpassConfiguration.scatter) {
            fragmentShader = "scatter.frag";
        } else if (renderpassConfiguration.picking) {
        	fragmentShader = "picking.frag";
        }

        const fragmentHash = this.shaderCache.selectShader(fragmentShader, fragDefines);
        const vertexHash = this.shaderCache.selectShader(vertexShader, vertDefines);

        if (fragmentHash && vertexHash)
        {
            this.shader = this.shaderCache.getShaderProgram(fragmentHash, vertexHash);
        }

        if (this.shader === undefined)
        {
            return;
        }

        this.webGl.context.useProgram(this.shader.program);

        if (state.renderingParameters.usePunctual && !renderpassConfiguration.picking)
        {
            this.applyLights();
        }

        if (renderpassConfiguration.scatter) {
            this.webGl.context.uniform1i(this.shader.getUniformLocation("u_MaterialID"), renderpassConfiguration.drawID);
        }

        const worldTransform = node.getRenderedWorldTransform();
      
        const normalMatrix = mat4.create();
        mat4.invert(normalMatrix, node.getRenderedWorldTransform());
        mat4.transpose(normalMatrix, normalMatrix);

        // update model dependant matrices once per node
        this.shader.updateUniform("u_ViewProjectionMatrix", viewProjectionMatrix);
        this.shader.updateUniform("u_ModelMatrix", worldTransform);
        this.shader.updateUniform("u_NormalMatrix", normalMatrix, false);
        this.shader.updateUniform("u_Exposure", state.renderingParameters.exposure, false);
        this.shader.updateUniform("u_Camera", this.currentCameraPosition, false);
        if (renderpassConfiguration.picking) {
            // Node picking id in the low 24 bits, primitive index within the mesh in the high 8.
            const primitiveIndex = Math.min(
                state.gltf.meshes[node.mesh]?.primitives.indexOf(primitive) ?? 0,
                255
            );
            this.shader.updateUniform(
                "u_PickingColor",
                (node.pickingColor | (primitiveIndex << 24)) >>> 0,
                false
            );
        } else {
            // Uniforms persist per program, so reset the tint for every non-highlighted draw.
            const highlighted =
                state.highlightedNodeIndices.has(node.gltfObjectIndex) ||
                state.highlightedMaterialIndices.has(primitive.material);
            const params = state.renderingParameters;
            this.shader.updateUniform(
                "u_HighlightColor",
                highlighted
                    ? vec4.fromValues(...params.highlightColor, params.highlightStrength)
                    : NoHighlightColor,
                false
            );
        }

        this.updateAnimationUniforms(state, node, primitive);

        if (mat4.determinant(worldTransform) < 0.0)
        {
            this.webGl.context.frontFace(GL.CW);
        }
        else
        {
            this.webGl.context.frontFace(GL.CCW);
        }

        if (material.doubleSided || renderpassConfiguration.picking)
        {
            this.webGl.context.disable(GL.CULL_FACE);
        }
        else
        {
            this.webGl.context.enable(GL.CULL_FACE);
        }
    
        if (material.alphaMode === 'BLEND' && !renderpassConfiguration.picking)
        {
            this.webGl.context.enable(GL.BLEND);
            if (material.msfs?.decalFactors !== undefined && state.renderingParameters.debugOutput === GltfState.DebugOutput.NONE)
            {
                // MSFS geometry decal: premultiplied color plus a relighting factor in alpha
                this.webGl.context.blendFuncSeparate(GL.ONE, GL.SRC_ALPHA, GL.ZERO, GL.ONE);
            }
            else
            {
                this.webGl.context.blendFuncSeparate(GL.SRC_ALPHA, GL.ONE_MINUS_SRC_ALPHA, GL.ONE, GL.ONE_MINUS_SRC_ALPHA);
            }
            this.webGl.context.blendEquation(GL.FUNC_ADD);
            this.webGl.context.depthMask(false);
        }
        else
        {
            this.webGl.context.disable(GL.BLEND);
            this.webGl.context.depthMask(true);
        }
        

        const drawIndexed = primitive.indices !== undefined;
        if (drawIndexed)
        {
            if (!this.webGl.setIndices(state.gltf, primitive.indices))
            {
                return;
            }
        }

        let vertexCount = 0;
        for (const attribute of primitive.glAttributes)
        {
            if (renderpassConfiguration.picking && (attribute.attribute !== "POSITION" || attribute.attribute.startsWith("JOINTS") || attribute.attribute.startsWith("WEIGHTS"))) {
                continue;
            }
            const gltfAccessor = state.gltf.accessors[attribute.accessor];
            vertexCount = gltfAccessor.count;

            const location = this.shader.getAttributeLocation(attribute.name);
            if (location === null)
            {
                continue; // only skip this attribute
            }
            if (!this.webGl.enableAttribute(state.gltf, location, gltfAccessor))
            {
                return; // skip this primitive
            }
        }

        if (instanceOffset !== undefined) {
            const location = this.shader.getAttributeLocation("a_instance_model_matrix");
            const location2 = location + 1;
            const location3 = location2 + 1;
            const location4 = location3 + 1;
            if (this.instanceBuffer === undefined) {
                this.instanceBuffer = this.webGl.context.createBuffer();
            }
            this.webGl.context.enableVertexAttribArray(location);
            this.webGl.context.enableVertexAttribArray(location2);
            this.webGl.context.enableVertexAttribArray(location3);
            this.webGl.context.enableVertexAttribArray(location4);

            this.webGl.context.bindBuffer(GL.ARRAY_BUFFER, this.instanceBuffer);
            const data = new Float32Array(instanceOffset.length * 16);
            instanceOffset.forEach((element, index) => {
                data.set(element, 16 * index);
            });
            this.webGl.context.bufferData(GL.ARRAY_BUFFER, data, GL.DYNAMIC_DRAW);
            this.webGl.context.vertexAttribPointer(location, 4, GL.FLOAT, GL.FALSE, 4 * 16, 0);
            this.webGl.context.vertexAttribPointer(location2, 4, GL.FLOAT, GL.FALSE, 4 * 16, 4 * 4);
            this.webGl.context.vertexAttribPointer(location3, 4, GL.FLOAT, GL.FALSE, 4 * 16, 4 * 8);
            this.webGl.context.vertexAttribPointer(location4, 4, GL.FLOAT, GL.FALSE, 4 * 16, 4 * 12);
            
            this.webGl.context.vertexAttribDivisor(location, 1);
            this.webGl.context.vertexAttribDivisor(location2, 1);
            this.webGl.context.vertexAttribDivisor(location3, 1);
            this.webGl.context.vertexAttribDivisor(location4, 1);
        }

        // Update material uniforms
        material.updateTextureTransforms(this.shader);

        this.shader.updateUniform("u_EmissiveFactor", jsToGl(material.emissiveFactor));
        this.shader.updateUniform("u_AlphaCutoff", material.alphaCutoff);

        this.shader.updateUniform("u_NormalScale", material.normalTexture?.scale);
        this.shader.updateUniform("u_NormalUVSet", material.normalTexture?.texCoord);

        this.shader.updateUniform("u_OcclusionStrength", material.occlusionTexture?.strength);
        this.shader.updateUniform("u_OcclusionUVSet", material.occlusionTexture?.texCoord);

        this.shader.updateUniform("u_EmissiveUVSet", material.emissiveTexture?.texCoord);

        this.shader.updateUniform("u_BaseColorUVSet", material.pbrMetallicRoughness?.baseColorTexture?.texCoord);
        
        this.shader.updateUniform("u_MetallicRoughnessUVSet", material.pbrMetallicRoughness?.metallicRoughnessTexture?.texCoord);
        this.shader.updateUniform("u_MetallicFactor", material.pbrMetallicRoughness?.metallicFactor);
        this.shader.updateUniform("u_RoughnessFactor", material.pbrMetallicRoughness?.roughnessFactor);
        this.shader.updateUniform("u_BaseColorFactor", jsToGl(material.pbrMetallicRoughness?.baseColorFactor));

        this.shader.updateUniform("u_AnisotropyUVSet", material.extensions?.KHR_materials_anisotropy?.anisotropyTexture?.texCoord);

        const factor = material.extensions?.KHR_materials_anisotropy?.anisotropyStrength;
        const rotation = material.extensions?.KHR_materials_anisotropy?.anisotropyRotation;
        const anisotropy =  vec3.fromValues(Math.cos(rotation ?? 0), Math.sin(rotation ?? 0), factor ?? 0.0);
        this.shader.updateUniform("u_Anisotropy", anisotropy);

        this.shader.updateUniform("u_ClearcoatFactor", material.extensions?.KHR_materials_clearcoat?.clearcoatFactor);
        this.shader.updateUniform("u_ClearcoatRoughnessFactor", material.extensions?.KHR_materials_clearcoat?.clearcoatRoughnessFactor);
        this.shader.updateUniform("u_ClearcoatUVSet", material.extensions?.KHR_materials_clearcoat?.clearcoatTexture?.texCoord);
        this.shader.updateUniform("u_ClearcoatRoughnessUVSet", material.extensions?.KHR_materials_clearcoat?.clearcoatRoughnessTexture?.texCoord);
        this.shader.updateUniform("u_ClearcoatNormalUVSet", material.extensions?.KHR_materials_clearcoat?.clearcoatNormalTexture?.texCoord);
        this.shader.updateUniform("u_ClearcoatNormalScale", material.extensions?.KHR_materials_clearcoat?.clearcoatNormalTexture?.scale);

        this.shader.updateUniform("u_Dispersion", material.extensions?.KHR_materials_dispersion?.dispersion);

        const emissiveStrength = material.extensions?.KHR_materials_emissive_strength?.emissiveStrength;
        this.shader.updateUniform(
            "u_EmissiveStrength",
            emissiveStrength === undefined ? undefined : emissiveStrength * getMsfsEmissiveMultiplier(material, state.renderingParameters)
        );

        this.shader.updateUniform("u_Ior", material.extensions?.KHR_materials_ior?.ior);

        this.shader.updateUniform("u_IridescenceFactor", material.extensions?.KHR_materials_iridescence?.iridescenceFactor);
        this.shader.updateUniform("u_IridescenceIor", material.extensions?.KHR_materials_iridescence?.iridescenceIor);
        this.shader.updateUniform("u_IridescenceThicknessMaximum", material.extensions?.KHR_materials_iridescence?.iridescenceThicknessMaximum);
        this.shader.updateUniform("u_IridescenceUVSet", material.extensions?.KHR_materials_iridescence?.iridescenceTexture?.texCoord);
        this.shader.updateUniform("u_IridescenceThicknessUVSet", material.extensions?.KHR_materials_iridescence?.iridescenceThicknessTexture?.texCoord);
        this.shader.updateUniform("u_IridescenceThicknessMinimum", material.extensions?.KHR_materials_iridescence?.iridescenceThicknessMinimum);

        this.shader.updateUniform("u_RetroreflectionFactor", material.extensions?.KHR_materials_retroreflection?.retroreflectionFactor);
        this.shader.updateUniform("u_RetroreflectionUVSet", material.extensions?.KHR_materials_retroreflection?.retroreflectionTexture?.texCoord);

        this.shader.updateUniform("u_SheenRoughnessFactor", material.extensions?.KHR_materials_sheen?.sheenRoughnessFactor);
        this.shader.updateUniform("u_SheenColorFactor", jsToGl(material.extensions?.KHR_materials_sheen?.sheenColorFactor));
        this.shader.updateUniform("u_SheenRoughnessUVSet", material.extensions?.KHR_materials_sheen?.sheenRoughnessTexture?.texCoord);
        this.shader.updateUniform("u_SheenColorUVSet", material.extensions?.KHR_materials_sheen?.sheenColorTexture?.texCoord);
        
        this.shader.updateUniform("u_KHR_materials_specular_specularColorFactor", jsToGl(material.extensions?.KHR_materials_specular?.specularColorFactor));
        this.shader.updateUniform("u_KHR_materials_specular_specularFactor", material.extensions?.KHR_materials_specular?.specularFactor);
        this.shader.updateUniform("u_SpecularUVSet", material.extensions?.KHR_materials_specular?.specularTexture?.texCoord);
        this.shader.updateUniform("u_SpecularColorUVSet", material.extensions?.KHR_materials_specular?.specularColorTexture?.texCoord);

        this.shader.updateUniform("u_TransmissionFactor", material.extensions?.KHR_materials_transmission?.transmissionFactor);
        this.shader.updateUniform("u_TransmissionUVSet", material.extensions?.KHR_materials_transmission?.transmissionTexture?.texCoord);

        this.shader.updateUniform("u_AttenuationColor", jsToGl(material.extensions?.KHR_materials_volume?.attenuationColor));
        this.shader.updateUniform("u_AttenuationDistance", material.extensions?.KHR_materials_volume?.attenuationDistance);
        this.shader.updateUniform("u_ThicknessFactor", material.extensions?.KHR_materials_volume?.thicknessFactor);
        this.shader.updateUniform("u_ThicknessUVSet", material.extensions?.KHR_materials_volume?.thicknessTexture?.texCoord);

        this.shader.updateUniform("u_DiffuseTransmissionFactor", material.extensions?.KHR_materials_diffuse_transmission?.diffuseTransmissionFactor);
        this.shader.updateUniform("u_DiffuseTransmissionColorFactor", jsToGl(material.extensions?.KHR_materials_diffuse_transmission?.diffuseTransmissionColorFactor));
        this.shader.updateUniform("u_DiffuseTransmissionUVSet", material.extensions?.KHR_materials_diffuse_transmission?.diffuseTransmissionTexture?.texCoord);
        this.shader.updateUniform("u_DiffuseTransmissionColorUVSet", material.extensions?.KHR_materials_diffuse_transmission?.diffuseTransmissionColorTexture?.texCoord);

        this.shader.updateUniform("u_DiffuseFactor", jsToGl(material.extensions?.KHR_materials_pbrSpecularGlossiness?.diffuseFactor));
        this.shader.updateUniform("u_SpecularFactor", jsToGl(material.extensions?.KHR_materials_pbrSpecularGlossiness?.specularFactor));
        this.shader.updateUniform("u_GlossinessFactor", material.extensions?.KHR_materials_pbrSpecularGlossiness?.glossinessFactor);
        this.shader.updateUniform("u_SpecularGlossinessUVSet", material.extensions?.KHR_materials_pbrSpecularGlossiness?.specularGlossinessTexture?.texCoord);
        this.shader.updateUniform("u_DiffuseUVSet", material.extensions?.KHR_materials_pbrSpecularGlossiness?.diffuseTexture?.texCoord);

        this.shader.updateUniform("u_MultiScatterColor", jsToGl(material.extensions?.KHR_materials_volume_scatter?.multiscatterColor));
    
        updateMsfsMaterialUniforms(this.shader, material);

        const msfsClamp = renderpassConfiguration.picking ? undefined : getMsfsClampedTextures(material);
        let textureIndex = 0;
        for (; textureIndex < material.textures.length; ++textureIndex)
        {
            let info = material.textures[textureIndex];
            const location = this.shader.getUniformLocation(info.samplerName);
            if (info.debugDisabled && !renderpassConfiguration.picking)
            {
                this.bindNeutralTexture(location, info.samplerName, textureIndex);
                continue;
            }
            if (!this.webGl.setTexture(location, state.gltf, info, textureIndex))
            {
                continue;
            }
            if (msfsClamp?.textures.has(info))
            {
                this.bindMsfsClampSampler(state.gltf, info, msfsClamp, textureIndex);
            }
        }


        // set the morph target texture
        if (primitive.morphTargetTextureInfo !== undefined) 
        {
            const location = this.shader.getUniformLocation(primitive.morphTargetTextureInfo.samplerName);
            this.webGl.setTexture(location, state.gltf, primitive.morphTargetTextureInfo, textureIndex); // binds texture and sampler
            textureIndex++;
        }

        // set the joints texture
        if (state.renderingParameters.skinning && node.skin !== undefined && primitive.hasWeights && primitive.hasJoints) 
        {
            const skin = state.gltf.skins[node.skin];
            const location = this.shader.getUniformLocation(skin.jointTextureInfo.samplerName);
            this.webGl.setTexture(location, state.gltf, skin.jointTextureInfo, textureIndex); // binds texture and sampler
            textureIndex++;
        }

        if (!renderpassConfiguration.picking) {
            let textureCount = textureIndex;

            textureCount = this.applyEnvironmentMap(state, textureCount);


            if (state.environment !== undefined)
            {
                this.webGl.setTexture(this.shader.getUniformLocation("u_SheenELUT"), state.environment, state.environment.sheenELUT, textureCount++);
            }

	        if (material.hasVolumeScatter && sampledTextures?.scatterSampleTexture !== undefined)
	        {
	            this.webGl.context.activeTexture(GL.TEXTURE0 + textureCount);
	            this.webGl.context.bindTexture(this.webGl.context.TEXTURE_2D, sampledTextures.scatterSampleTexture);
	            this.webGl.context.uniform1i(this.shader.getUniformLocation("u_ScatterFramebufferSampler"), textureCount);
	            textureCount++;
	
	            this.webGl.context.activeTexture(GL.TEXTURE0 + textureCount);
	            this.webGl.context.bindTexture(this.webGl.context.TEXTURE_2D, sampledTextures.scatterDepthSampleTexture);
	            this.webGl.context.uniform1i(this.shader.getUniformLocation("u_ScatterDepthFramebufferSampler"), textureCount);
	            textureCount++;
	
	            this.webGl.context.uniform1f(this.shader.getUniformLocation("u_MinRadius"), gltfMaterial.scatterMinRadius);
	            this.webGl.context.uniform2i(this.shader.getUniformLocation("u_FramebufferSize"), renderpassConfiguration.frameBufferSize[0], renderpassConfiguration.frameBufferSize[1]);
	            this.webGl.context.uniformMatrix4fv(this.shader.getUniformLocation("u_ProjectionMatrix"),false, this.projMatrix);
	
	            this.shader.updateUniformArray("u_ScatterSamples", gltfMaterial.scatterSamples);
	        }
	
	        if(sampledTextures?.transmissionSampleTexture !== undefined &&
	            state.environment &&
	            state.renderingParameters.enabledExtensions.KHR_materials_transmission)
	        {
	            this.webGl.context.activeTexture(GL.TEXTURE0 + textureCount);
	            this.webGl.context.bindTexture(this.webGl.context.TEXTURE_2D, this.opaqueRenderTexture);
	            this.webGl.context.uniform1i(this.shader.getUniformLocation("u_TransmissionFramebufferSampler"), textureCount);
	            textureCount++;
	
	            this.webGl.context.uniform2i(this.shader.getUniformLocation("u_TransmissionFramebufferSize"), this.opaqueFramebufferWidth, this.opaqueFramebufferHeight);
	
	            this.webGl.context.uniformMatrix4fv(this.shader.getUniformLocation("u_ModelMatrix"),false, worldTransform);
	            this.webGl.context.uniformMatrix4fv(this.shader.getUniformLocation("u_ViewMatrix"),false, this.viewMatrix);
	            this.webGl.context.uniformMatrix4fv(this.shader.getUniformLocation("u_ProjectionMatrix"),false, this.projMatrix);
	        }
	    }
	    
        // MSFS geometry decals need a depth bias to win against the surface they lie on.
        const depthBias = getMsfsDepthBias(material);
        if (depthBias !== 0)
        {
            this.webGl.context.enable(GL.POLYGON_OFFSET_FILL);
            this.webGl.context.polygonOffset(depthBias, depthBias);
        }

        if (drawIndexed)
        {
            const indexAccessor = state.gltf.accessors[primitive.indices];
            if (instanceOffset !== undefined) {
                this.webGl.context.drawElementsInstanced(primitive.mode, indexAccessor.count, indexAccessor.componentType, 0, instanceOffset.length);
            } else {
                this.webGl.context.drawElements(primitive.mode, indexAccessor.count, indexAccessor.componentType, 0);
            }
        }
        else
        {
            if (instanceOffset !== undefined) {
                this.webGl.context.drawArraysInstanced(primitive.mode, 0, vertexCount, instanceOffset.length);
            } else {
                this.webGl.context.drawArrays(primitive.mode, 0, vertexCount);
            }
        }

        if (depthBias !== 0)
        {
            this.webGl.context.disable(GL.POLYGON_OFFSET_FILL);
        }

        // Sampler objects stay bound to their unit; release them for later draws.
        for (const unit of this.msfsSamplerUnits)
        {
            this.webGl.context.bindSampler(unit, null);
        }
        this.msfsSamplerUnits.length = 0;

        for (const attribute of primitive.glAttributes)
        {
            if (renderpassConfiguration.picking && (attribute.attribute !== "POSITION" || attribute.attribute.startsWith("JOINTS") || attribute.attribute.startsWith("WEIGHTS"))) {
                continue;
            }
            const location = this.shader.getAttributeLocation(attribute.name);
            if (location === null)
            {
                continue; // skip this attribute
            }
            this.webGl.context.disableVertexAttribArray(location);
        }
        if (instanceOffset !== undefined) {
            const location = this.shader.getAttributeLocation("a_instance_model_matrix");
            this.webGl.context.vertexAttribDivisor(location, 0);
            this.webGl.context.vertexAttribDivisor(location + 1, 0);
            this.webGl.context.vertexAttribDivisor(location + 2, 0);
            this.webGl.context.vertexAttribDivisor(location + 3, 0);
            this.webGl.context.disableVertexAttribArray(location);
            this.webGl.context.disableVertexAttribArray(location + 1);
            this.webGl.context.disableVertexAttribArray(location + 2);
            this.webGl.context.disableVertexAttribArray(location + 3);
        }
    }

    /// Compute a list of lights instantiated by one or more nodes as a list of node-light tuples.
    getVisibleLights(gltf, nodes) {
        let nodeLights = [];

        for (const node of nodes) {
            const lightIndex = node.extensions?.KHR_lights_punctual?.light;
            if (lightIndex === undefined) {
                continue;
            }
            const light = gltf.extensions?.KHR_lights_punctual?.lights[lightIndex];
            nodeLights.push([node, light]);
        }

        return nodeLights;
    }

    updateSkin(state, node) {
        if (state.renderingParameters.skinning && state.gltf.skins !== undefined) {
            const skin = state.gltf.skins[node.skin];
            skin.computeJoints(state.gltf, this.webGl.context);
        }
    }

    pushVertParameterDefines(vertDefines, parameters, gltf, node, primitive, debugOutput) {
        // skinning
        if (
            parameters.skinning &&
            node.skin !== undefined &&
            primitive.hasWeights &&
            primitive.hasJoints
        ) {
            vertDefines.push("USE_SKINNING 1");
        }

        // morphing
        if (parameters.morphing && node.mesh !== undefined && primitive.targets.length > 0) {
            const weights = node.getWeights(gltf);
            if (weights !== undefined && weights.length > 0) {
                vertDefines.push("USE_MORPHING 1");
                vertDefines.push("WEIGHT_COUNT " + weights.length);
            }
        }

        vertDefines.push("DEBUG_VERT_NONE 0");
        vertDefines.push("DEBUG_VERT_TANGENT_W 1");
        if (debugOutput == GltfState.DebugOutput.generic.TANGENTW) {
            vertDefines.push("DEBUG_VERT DEBUG_VERT_TANGENT_W");
        } else {
            vertDefines.push("DEBUG_VERT DEBUG_VERT_NONE");
        }
    }

    updateAnimationUniforms(state, node, primitive) {
        if (
            state.renderingParameters.morphing &&
            node.mesh !== undefined &&
            primitive.targets.length > 0
        ) {
            const weights = node.getWeights(state.gltf);
            if (weights !== undefined && weights.length > 0) {
                this.shader.updateUniformArray("u_morphWeights", weights);
            }
        }
    }

    pushFragParameterDefines(fragDefines, state) {
        if (state.renderingParameters.usePunctual) {
            fragDefines.push("USE_PUNCTUAL 1");
            fragDefines.push(`LIGHT_COUNT ${this.visibleLights.length}`);
        }

        if (state.renderingParameters.useIBL && state.environment) {
            fragDefines.push("USE_IBL 1");
        }

        switch (state.renderingParameters.toneMap) {
            case GltfState.ToneMaps.KHR_PBR_NEUTRAL:
                fragDefines.push("TONEMAP_KHR_PBR_NEUTRAL 1");
                break;
            case GltfState.ToneMaps.ACES_NARKOWICZ:
                fragDefines.push("TONEMAP_ACES_NARKOWICZ 1");
                break;
            case GltfState.ToneMaps.ACES_HILL:
                fragDefines.push("TONEMAP_ACES_HILL 1");
                break;
            case GltfState.ToneMaps.ACES_HILL_EXPOSURE_BOOST:
                fragDefines.push("TONEMAP_ACES_HILL_EXPOSURE_BOOST 1");
                break;
            case GltfState.ToneMaps.NONE:
            default:
                fragDefines.push("TONEMAP_NONE 1");
                break;
        }

        let debugOutputMapping = [
            { debugOutput: GltfState.DebugOutput.NONE, shaderDefine: "DEBUG_NONE" },

            {
                debugOutput: GltfState.DebugOutput.generic.WORLDSPACENORMAL,
                shaderDefine: "DEBUG_NORMAL_SHADING"
            },
            {
                debugOutput: GltfState.DebugOutput.generic.NORMAL,
                shaderDefine: "DEBUG_NORMAL_TEXTURE"
            },
            {
                debugOutput: GltfState.DebugOutput.generic.GEOMETRYNORMAL,
                shaderDefine: "DEBUG_NORMAL_GEOMETRY"
            },
            { debugOutput: GltfState.DebugOutput.generic.TANGENT, shaderDefine: "DEBUG_TANGENT" },
            {
                debugOutput: GltfState.DebugOutput.generic.TANGENTW,
                shaderDefine: "DEBUG_TANGENT_W"
            },
            {
                debugOutput: GltfState.DebugOutput.generic.BITANGENT,
                shaderDefine: "DEBUG_BITANGENT"
            },
            { debugOutput: GltfState.DebugOutput.generic.ALPHA, shaderDefine: "DEBUG_ALPHA" },
            { debugOutput: GltfState.DebugOutput.generic.UV_COORDS_0, shaderDefine: "DEBUG_UV_0" },
            { debugOutput: GltfState.DebugOutput.generic.UV_COORDS_1, shaderDefine: "DEBUG_UV_1" },
            {
                debugOutput: GltfState.DebugOutput.generic.OCCLUSION,
                shaderDefine: "DEBUG_OCCLUSION"
            },
            { debugOutput: GltfState.DebugOutput.generic.EMISSIVE, shaderDefine: "DEBUG_EMISSIVE" },

            { debugOutput: GltfState.DebugOutput.mr.BASECOLOR, shaderDefine: "DEBUG_BASE_COLOR" },
            { debugOutput: GltfState.DebugOutput.mr.ROUGHNESS, shaderDefine: "DEBUG_ROUGHNESS" },
            { debugOutput: GltfState.DebugOutput.mr.METALLIC, shaderDefine: "DEBUG_METALLIC" },

            {
                debugOutput: GltfState.DebugOutput.clearcoat.CLEARCOAT_FACTOR,
                shaderDefine: "DEBUG_CLEARCOAT_FACTOR"
            },
            {
                debugOutput: GltfState.DebugOutput.clearcoat.CLEARCOAT_ROUGHNESS,
                shaderDefine: "DEBUG_CLEARCOAT_ROUGHNESS"
            },
            {
                debugOutput: GltfState.DebugOutput.clearcoat.CLEARCOAT_NORMAL,
                shaderDefine: "DEBUG_CLEARCOAT_NORMAL"
            },

            {
                debugOutput: GltfState.DebugOutput.sheen.SHEEN_COLOR,
                shaderDefine: "DEBUG_SHEEN_COLOR"
            },
            {
                debugOutput: GltfState.DebugOutput.sheen.SHEEN_ROUGHNESS,
                shaderDefine: "DEBUG_SHEEN_ROUGHNESS"
            },

            {
                debugOutput: GltfState.DebugOutput.specular.SPECULAR_FACTOR,
                shaderDefine: "DEBUG_SPECULAR_FACTOR"
            },
            {
                debugOutput: GltfState.DebugOutput.specular.SPECULAR_COLOR,
                shaderDefine: "DEBUG_SPECULAR_COLOR"
            },

            {
                debugOutput: GltfState.DebugOutput.transmission.TRANSMISSION_FACTOR,
                shaderDefine: "DEBUG_TRANSMISSION_FACTOR"
            },
            {
                debugOutput: GltfState.DebugOutput.transmission.VOLUME_THICKNESS,
                shaderDefine: "DEBUG_VOLUME_THICKNESS"
            },

            {
                debugOutput: GltfState.DebugOutput.diffuseTransmission.DIFFUSE_TRANSMISSION_FACTOR,
                shaderDefine: "DEBUG_DIFFUSE_TRANSMISSION_FACTOR"
            },
            {
                debugOutput:
                    GltfState.DebugOutput.diffuseTransmission.DIFFUSE_TRANSMISSION_COLOR_FACTOR,
                shaderDefine: "DEBUG_DIFFUSE_TRANSMISSION_COLOR_FACTOR"
            },

            {
                debugOutput: GltfState.DebugOutput.iridescence.IRIDESCENCE_FACTOR,
                shaderDefine: "DEBUG_IRIDESCENCE_FACTOR"
            },
            {
                debugOutput: GltfState.DebugOutput.iridescence.IRIDESCENCE_THICKNESS,
                shaderDefine: "DEBUG_IRIDESCENCE_THICKNESS"
            },

            {
                debugOutput: GltfState.DebugOutput.retroreflection.RETROREFLECTION_FACTOR,
                shaderDefine: "DEBUG_RETROREFLECTION_FACTOR"
            },

            {
                debugOutput: GltfState.DebugOutput.anisotropy.ANISOTROPIC_STRENGTH,
                shaderDefine: "DEBUG_ANISOTROPIC_STRENGTH"
            },
            {
                debugOutput: GltfState.DebugOutput.anisotropy.ANISOTROPIC_DIRECTION,
                shaderDefine: "DEBUG_ANISOTROPIC_DIRECTION"
            },

            {
                debugOutput: GltfState.DebugOutput.volumeScatter.MULTI_SCATTER_COLOR,
                shaderDefine: "DEBUG_VOLUME_SCATTER_MULTI_SCATTER_COLOR"
            },
            {
                debugOutput: GltfState.DebugOutput.volumeScatter.SINGLE_SCATTER_COLOR,
                shaderDefine: "DEBUG_VOLUME_SCATTER_SINGLE_SCATTER_COLOR"
            },
            {
                debugOutput: GltfState.DebugOutput.msfs.MSFS_UV0_TILED,
                shaderDefine: "DEBUG_MSFS_UV0_TILED"
            },
            {
                debugOutput: GltfState.DebugOutput.msfs.MSFS_VERTEX_COLOR,
                shaderDefine: "DEBUG_MSFS_VERTEX_COLOR"
            },
            {
                debugOutput: GltfState.DebugOutput.msfs.MSFS_VERTEX_ALPHA,
                shaderDefine: "DEBUG_MSFS_VERTEX_ALPHA"
            },
            {
                debugOutput: GltfState.DebugOutput.msfs.MSFS_DETAIL_COLOR,
                shaderDefine: "DEBUG_MSFS_DETAIL_COLOR"
            },
            {
                debugOutput: GltfState.DebugOutput.msfs.MSFS_DETAIL_MASK,
                shaderDefine: "DEBUG_MSFS_DETAIL_MASK"
            },
            {
                debugOutput: GltfState.DebugOutput.msfs.MSFS_DETAIL_NORMAL,
                shaderDefine: "DEBUG_MSFS_DETAIL_NORMAL"
            },
            {
                debugOutput: GltfState.DebugOutput.msfs.MSFS_DETAIL_OCCLUSION,
                shaderDefine: "DEBUG_MSFS_DETAIL_OCCLUSION"
            },
            {
                debugOutput: GltfState.DebugOutput.msfs.MSFS_DETAIL_ROUGHNESS,
                shaderDefine: "DEBUG_MSFS_DETAIL_ROUGHNESS"
            },
            {
                debugOutput: GltfState.DebugOutput.msfs.MSFS_DETAIL_METALLIC,
                shaderDefine: "DEBUG_MSFS_DETAIL_METALLIC"
            },
            {
                debugOutput: GltfState.DebugOutput.msfs.MSFS_EXTRA_OCCLUSION,
                shaderDefine: "DEBUG_MSFS_EXTRA_OCCLUSION"
            },
            {
                debugOutput: GltfState.DebugOutput.msfs.MSFS_DIRT,
                shaderDefine: "DEBUG_MSFS_DIRT"
            },
            {
                debugOutput: GltfState.DebugOutput.msfs.MSFS_TIRE,
                shaderDefine: "DEBUG_MSFS_TIRE"
            },
            {
                debugOutput: GltfState.DebugOutput.msfs.MSFS_PEARL,
                shaderDefine: "DEBUG_MSFS_PEARL"
            },
            {
                debugOutput: GltfState.DebugOutput.msfs.MSFS_DECAL_WEIGHTS,
                shaderDefine: "DEBUG_MSFS_DECAL_WEIGHTS"
            },
            {
                debugOutput: GltfState.DebugOutput.msfs.MSFS_DECAL_RELIGHT,
                shaderDefine: "DEBUG_MSFS_DECAL_RELIGHT"
            },
            {
                debugOutput: GltfState.DebugOutput.msfs.MSFS_PARALLAX,
                shaderDefine: "DEBUG_MSFS_PARALLAX"
            },
            {
                debugOutput: GltfState.DebugOutput.msfs.MSFS_DRAW_ORDER,
                shaderDefine: "DEBUG_MSFS_DRAW_ORDER"
            }
        ];

        let mappingCount = 0;
        let mappingFound = false;
        for (let mapping of debugOutputMapping) {
            fragDefines.push(mapping.shaderDefine + " " + mappingCount++);
            if (state.renderingParameters.debugOutput == mapping.debugOutput) {
                fragDefines.push("DEBUG " + mapping.shaderDefine);
                mappingFound = true;
            }
        }

        if (mappingFound == false) {
            // fallback
            fragDefines.push("DEBUG DEBUG_NONE");
        }
    }

    applyLights() {
        const uniforms = [];
        for (const [node, light] of this.visibleLights) {
            uniforms.push(light.toUniform(node));
        }
        if (uniforms.length > 0) {
            this.shader.updateUniform("u_Lights", uniforms);
        }
    }

    applyEnvironmentMap(state, texSlotOffset) {
        const environment = state.environment;
        if (environment === undefined) {
            return texSlotOffset;
        }
        this.webGl.setTexture(
            this.shader.getUniformLocation("u_LambertianEnvSampler"),
            environment,
            environment.diffuseEnvMap,
            texSlotOffset++
        );

        this.webGl.setTexture(
            this.shader.getUniformLocation("u_GGXEnvSampler"),
            environment,
            environment.specularEnvMap,
            texSlotOffset++
        );
        this.webGl.setTexture(
            this.shader.getUniformLocation("u_GGXLUT"),
            environment,
            environment.lut,
            texSlotOffset++
        );

        this.webGl.setTexture(
            this.shader.getUniformLocation("u_CharlieEnvSampler"),
            environment,
            environment.sheenEnvMap,
            texSlotOffset++
        );
        this.webGl.setTexture(
            this.shader.getUniformLocation("u_CharlieLUT"),
            environment,
            environment.sheenLUT,
            texSlotOffset++
        );

        this.shader.updateUniform("u_MipCount", environment.mipCount);

        let rotMatrix4 = mat4.create();
        mat4.rotateY(
            rotMatrix4,
            rotMatrix4,
            (state.renderingParameters.environmentRotation / 180.0) * Math.PI
        );
        let rotMatrix3 = mat3.create();
        mat3.fromMat4(rotMatrix3, rotMatrix4);
        this.shader.updateUniform("u_EnvRotation", rotMatrix3);

        let envIntensity =
            state.renderingParameters.iblIntensity * state.environment.iblIntensityScale;

        if (state.renderingParameters.useIBL === false) {
            envIntensity = 0.0;
        }

        this.shader.updateUniform("u_EnvIntensity", envIntensity);

        return texSlotOffset;
    }

    destroy() {
        this.shaderCache.destroy();
    }
}

export { gltfRenderer };
