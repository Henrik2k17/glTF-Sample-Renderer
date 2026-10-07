/**
 * Per-frame statistics for a performance overlay: CPU time of the steps of renderFrame and
 * drawScene, draw calls and triangles per render pass and, where the browser exposes
 * EXT_disjoint_timer_query_webgl2, the GPU time of a frame (reported a few frames later).
 * Does nothing unless enabled.
 */
class FrameProfiler {
    constructor(gl) {
        this.gl = gl;
        this.enabled = false;
        this.timer = undefined; // the timer query extension, fetched when first enabled
        this.pendingQueries = [];
        this.activeQuery = undefined;
        this.frame = undefined;
        /** The last finished frame: { cpu, sections, passes, draws, triangles } */
        this.lastFrame = undefined;
        /** GPU time of the most recent frame whose query finished, in ms (undefined if unsupported) */
        this.lastGpuMs = undefined;
        this.gpuSupported = undefined;
    }

    beginFrame() {
        this.frame = undefined;
        if (!this.enabled) {
            return;
        }
        this.frame = {
            start: performance.now(),
            sections: {},
            passes: {},
            pass: "main",
            draws: 0,
            triangles: 0
        };
        this.pollGpu();
        if (this.gpuSupported === undefined) {
            this.timer = this.gl.getExtension("EXT_disjoint_timer_query_webgl2");
            this.gpuSupported = this.timer !== null;
        }
        // one query at a time; frames while one is pending are not measured
        if (this.gpuSupported && this.activeQuery === undefined && this.pendingQueries.length < 4) {
            this.activeQuery = this.gl.createQuery();
            this.gl.beginQuery(this.timer.TIME_ELAPSED_EXT, this.activeQuery);
        }
    }

    endFrame() {
        if (this.frame === undefined) {
            return;
        }
        if (this.activeQuery !== undefined) {
            this.gl.endQuery(this.timer.TIME_ELAPSED_EXT);
            this.pendingQueries.push(this.activeQuery);
            this.activeQuery = undefined;
        }
        this.frame.cpu = performance.now() - this.frame.start;
        this.lastFrame = this.frame;
        this.frame = undefined;
    }

    /** @returns {number | undefined} a start time for time() */
    now() {
        return this.frame === undefined ? undefined : performance.now();
    }

    /** Adds the time since start (from now()) to a section. */
    time(section, start) {
        if (this.frame === undefined || start === undefined) {
            return;
        }
        this.frame.sections[section] = (this.frame.sections[section] ?? 0) + performance.now() - start;
    }

    /** Draw calls after this count for the given render pass. */
    setPass(pass) {
        if (this.frame !== undefined) {
            this.frame.pass = pass;
        }
    }

    countDraw(mode, count, instances = 1) {
        if (this.frame === undefined) {
            return;
        }
        // TRIANGLES 4, TRIANGLE_STRIP 5, TRIANGLE_FAN 6
        const triangles = mode === 4 ? count / 3 : mode === 5 || mode === 6 ? Math.max(0, count - 2) : 0;
        const pass = (this.frame.passes[this.frame.pass] ??= { draws: 0, triangles: 0 });
        pass.draws++;
        pass.triangles += triangles * instances;
        this.frame.draws++;
        this.frame.triangles += triangles * instances;
    }

    pollGpu() {
        const gl = this.gl;
        while (this.pendingQueries.length > 0) {
            const query = this.pendingQueries[0];
            if (!gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE)) {
                return;
            }
            const disjoint = gl.getParameter(this.timer.GPU_DISJOINT_EXT);
            if (!disjoint) {
                this.lastGpuMs = gl.getQueryParameter(query, gl.QUERY_RESULT) / 1e6;
            }
            gl.deleteQuery(query);
            this.pendingQueries.shift();
        }
    }
}

export { FrameProfiler };
