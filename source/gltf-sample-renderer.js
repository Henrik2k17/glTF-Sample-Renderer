import { GltfView } from "./GltfView/gltf_view.js";

import { GltfState } from "./GltfState/gltf_state.js";

import { ResourceLoader } from "./ResourceLoader/resource_loader.js";

import { ResourceLoaderUtils } from "./ResourceLoader/loader_utils.js";

import { isMsfsAsset, detectMsfsFrameRate } from "./gltf/msfs.js";

const Msfs = { isMsfsAsset, detectMsfsFrameRate };

export { GltfView, GltfState, ResourceLoader, ResourceLoaderUtils, Msfs };
