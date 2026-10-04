import { mat4, quat, vec3 } from "gl-matrix";

// Wireframe helpers for MSFS objects that have no geometry of their own: collision gizmos and
// fade volumes (ASOBO_gizmo_object, ASOBO_fade_object on meshes) and lights (ASOBO_street_light,
// ASOBO_advanced_light, ASOBO_sky_portal, the 2020 ASOBO_macro_light, KHR_lights_punctual).
// Conventions follow the 3ds Max exporter: gizmo translation/rotation are in the mesh node's
// space (Y up, box width/height/length along X/Y/Z) and MSFS lights shine along local +Z.

const TagColors = {
    Collision: [1.0, 0.3, 0.25],
    Road: [1.0, 0.65, 0.2],
    Ground: [0.4, 0.9, 0.3],
    SkinBoundingVolume: [0.7, 0.5, 1.0]
};
const UntaggedGizmoColor = [0.8, 0.8, 0.8];
const FadeColor = [0.3, 0.85, 1.0];
const PortalColor = [0.55, 0.8, 1.0];
const CircleSegments = 32;

class LineSet {
    constructor() {
        this.groups = new Map(); // color key -> {color, positions: number[]}
    }

    add(color, matrix, a, b) {
        const key = color.join(",");
        if (!this.groups.has(key)) {
            this.groups.set(key, { color, positions: [] });
        }
        const pa = vec3.transformMat4(vec3.create(), a, matrix);
        const pb = vec3.transformMat4(vec3.create(), b, matrix);
        this.groups.get(key).positions.push(...pa, ...pb);
    }

    // Circle in the plane spanned by unit vectors u and v around center.
    circle(color, matrix, center, u, v, radius) {
        let previous;
        for (let i = 0; i <= CircleSegments; i++) {
            const angle = (i / CircleSegments) * Math.PI * 2;
            const point = vec3.clone(center);
            vec3.scaleAndAdd(point, point, u, Math.cos(angle) * radius);
            vec3.scaleAndAdd(point, point, v, Math.sin(angle) * radius);
            if (previous !== undefined) {
                this.add(color, matrix, previous, point);
            }
            previous = point;
        }
    }

    // Arc of the given radius in the plane of +Z and side, from -angle to +angle off +Z.
    arc(color, matrix, side, angle, radius) {
        const steps = Math.max(2, Math.ceil((angle / Math.PI) * CircleSegments));
        let previous;
        for (let i = 0; i <= steps; i++) {
            const t = -angle + (2 * angle * i) / steps;
            const point = [side[0] * Math.sin(t) * radius, side[1] * Math.sin(t) * radius, Math.cos(t) * radius];
            if (previous !== undefined) {
                this.add(color, matrix, previous, point);
            }
            previous = point;
        }
    }

    box(color, matrix, width, height, length) {
        const [x, y, z] = [width / 2, height / 2, length / 2];
        const corners = [];
        for (const sx of [-1, 1]) {
            for (const sy of [-1, 1]) {
                for (const sz of [-1, 1]) {
                    corners.push([sx * x, sy * y, sz * z]);
                }
            }
        }
        // corners differ in exactly one axis along an edge
        for (let i = 0; i < 8; i++) {
            for (const bit of [1, 2, 4]) {
                if ((i & bit) === 0) {
                    this.add(color, matrix, corners[i], corners[i | bit]);
                }
            }
        }
    }

    sphere(color, matrix, radius) {
        const o = [0, 0, 0];
        this.circle(color, matrix, o, [1, 0, 0], [0, 1, 0], radius);
        this.circle(color, matrix, o, [0, 1, 0], [0, 0, 1], radius);
        this.circle(color, matrix, o, [1, 0, 0], [0, 0, 1], radius);
    }

    cylinder(color, matrix, radius, height) {
        const h = height / 2;
        this.circle(color, matrix, [0, h, 0], [1, 0, 0], [0, 0, 1], radius);
        this.circle(color, matrix, [0, -h, 0], [1, 0, 0], [0, 0, 1], radius);
        for (const [x, z] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
            this.add(color, matrix, [x * radius, h, z * radius], [x * radius, -h, z * radius]);
        }
    }

    // Light cone along +Z: rays at the half angle and the circle joining their ends.
    // Half angles of 90° and more open into a hemisphere or (at 180°) a sphere.
    cone(color, matrix, halfAngle, length) {
        const a = Math.min(halfAngle, Math.PI);
        if (a >= Math.PI - 0.01) {
            this.sphere(color, matrix, length * 0.5);
        } else {
            const radius = Math.sin(a) * length;
            const center = [0, 0, Math.cos(a) * length];
            for (let i = 0; i < 8; i++) {
                const phi = (i / 8) * Math.PI * 2;
                this.add(color, matrix, [0, 0, 0], [Math.cos(phi) * radius, Math.sin(phi) * radius, center[2]]);
            }
            this.circle(color, matrix, center, [1, 0, 0], [0, 1, 0], radius);
            // meridians show the spread, also for hemispheres
            this.arc(color, matrix, [1, 0, 0], a, length);
            this.arc(color, matrix, [0, 1, 0], a, length);
        }
        // direction
        this.add(color, matrix, [0, 0, 0], [0, 0, length * 1.3]);
    }

    result() {
        return [...this.groups.values()].map(({ color, positions }) => ({
            color,
            positions: new Float32Array(positions)
        }));
    }
}

function gizmoMatrix(nodeWorld, object) {
    const local = mat4.fromRotationTranslation(
        mat4.create(),
        object.rotation ?? quat.create(),
        object.translation ?? [0, 0, 0]
    );
    return mat4.multiply(local, nodeWorld, local);
}

function addMeshObjects(lines, node, mesh) {
    const world = node.getRenderedWorldTransform();
    for (const gizmo of mesh.extensions?.ASOBO_gizmo_object?.gizmo_objects ?? []) {
        const tag = gizmo.extensions?.ASOBO_tags?.tags?.[0];
        const color = TagColors[tag] ?? UntaggedGizmoColor;
        const matrix = gizmoMatrix(world, gizmo);
        const p = gizmo.params ?? {};
        if (gizmo.type === "box") {
            lines.box(color, matrix, p.width ?? 1, p.height ?? 1, p.length ?? 1);
        } else if (gizmo.type === "sphere") {
            lines.sphere(color, matrix, p.radius ?? 0.5);
        } else if (gizmo.type === "cylinder") {
            lines.cylinder(color, matrix, p.radius ?? 0.5, p.height ?? 1);
        }
    }
    for (const fade of mesh.extensions?.ASOBO_fade_object?.fade_objects ?? []) {
        if (fade.type === "sphere") {
            lines.sphere(FadeColor, gizmoMatrix(world, fade), fade.params?.radius ?? 0.5);
        }
    }
}

// Light color at full brightness, so dim colors stay visible.
function lightColor(color) {
    const c = color ?? [1, 1, 1];
    const max = Math.max(c[0], c[1], c[2], 1e-3);
    return [c[0] / max, c[1] / max, c[2] / max];
}

// The node's rotation and position without scale, so markers keep their size.
function unscaledWorld(node) {
    const world = node.getRenderedWorldTransform();
    const rotation = mat4.getRotation(quat.create(), world);
    quat.normalize(rotation, rotation);
    return mat4.fromRotationTranslation(mat4.create(), rotation, mat4.getTranslation(vec3.create(), world));
}

function addLights(lines, gltf, node, size) {
    const ext = node.extensions ?? {};
    const toRadians = Math.PI / 180;
    for (const name of ["ASOBO_street_light", "ASOBO_macro_light"]) {
        const light = ext[name];
        if (light !== undefined) {
            lines.cone(lightColor(light.color), unscaledWorld(node), ((light.cone_angle ?? 360) / 2) * toRadians, size);
        }
    }
    const advanced = ext.ASOBO_advanced_light;
    if (advanced !== undefined) {
        lines.cone(lightColor(advanced.color), unscaledWorld(node), ((advanced.outer_cone_angle ?? 360) / 2) * toRadians, size);
    }
    const portal = ext.ASOBO_sky_portal;
    if (portal !== undefined) {
        const matrix = unscaledWorld(node);
        // source radius in centimeters, as 3ds Max scene units
        lines.circle(PortalColor, matrix, [0, 0, 0], [1, 0, 0], [0, 1, 0], (portal.source_radius ?? 10) / 100);
        lines.cone(PortalColor, matrix, ((portal.outer_cone_angle ?? 180) / 2) * toRadians, size);
    }
    const lightIndex = ext.KHR_lights_punctual?.light;
    if (lightIndex !== undefined) {
        const light = gltf.extensions?.KHR_lights_punctual?.lights?.[lightIndex];
        if (light !== undefined) {
            // glTF lights shine along local -Z
            const matrix = mat4.rotateY(mat4.create(), unscaledWorld(node), Math.PI);
            const halfAngle = light.type === "spot" ? (light.spot?.outerConeAngle ?? Math.PI / 4) : light.type === "directional" ? 0 : Math.PI;
            lines.cone(lightColor(light.color), matrix, halfAngle, size);
        }
    }
}

/**
 * Builds the helper wireframes of the scene in world space, grouped by color.
 * @param {object} gltf
 * @param {number} sceneIndex
 * @param {{colliders: boolean, lights: boolean, size: number}} options - size: light marker
 *   length in meters
 * @returns {{color: number[], positions: Float32Array}[]} line list positions per color
 */
function buildMsfsHelperLines(gltf, sceneIndex, options) {
    const scene = gltf?.scenes?.[sceneIndex];
    if (scene === undefined || (!options.colliders && !options.lights)) {
        return [];
    }
    const lines = new LineSet();
    const stack = [...scene.nodes];
    while (stack.length > 0) {
        const node = gltf.nodes[stack.pop()];
        stack.push(...node.children);
        const mesh = gltf.meshes[node.mesh];
        if (options.colliders && mesh !== undefined) {
            addMeshObjects(lines, node, mesh);
        }
        if (options.lights) {
            addLights(lines, gltf, node, options.size);
        }
    }
    return lines.result();
}

export { buildMsfsHelperLines };
