/**
 * Utility class providing static helper methods for resource loading operations,
 * such as extracting file extensions, resolving folder paths, normalizing relative
 * paths, and detecting absolute URLs.
 */
class ResourceLoaderUtils {
    /**
     * Extracts the file extension from a filename.
     * @param {string} filename - The filename or path to extract the extension from.
     * @returns {string|undefined} The lowercase file extension (without the leading dot),
     *   or `undefined` if the filename has no extension.
     */
    static getExtension(filename) {
        const split = filename.toLowerCase().split(".");
        if (split.length == 1) {
            return undefined;
        }
        return split[split.length - 1];
    }

    /**
     * Returns the directory portion of a file path, including the trailing slash.
     * @param {string} filePath - The full file path.
     * @returns {string} The path up to and including the last `/`, or an empty string
     *   if no `/` is present.
     */
    static getContainingFolder(filePath) {
        return filePath.substring(0, filePath.lastIndexOf("/") + 1);
    }

    /**
     * Normalizes a relative URL path by resolving `.` and `..` segments.
     * - Strips a leading `./` prefix.
     * - Collapses `/./` sequences to `/`.
     * - Resolves `/../` sequences by removing the preceding path segment.
     * @param {string} relativePath - The relative path to clean.
     * @returns {string} The normalized path with dot segments resolved.
     */
    static cleanRelativePath(relativePath) {
        if (relativePath.startsWith("./")) {
            relativePath = relativePath.substring(2);
        }
        while (relativePath.includes("/./")) {
            relativePath = relativePath.replace("/./", "/");
        }
        let searchIndex = relativePath.indexOf("/../");
        while (searchIndex !== -1) {
            let slashIndex = relativePath.lastIndexOf("/", searchIndex - 1);
            relativePath =
                relativePath.substring(0, slashIndex + 1) + relativePath.substring(searchIndex + 4);
            searchIndex = relativePath.indexOf("/../");
        }
        return relativePath;
    }

    /**
     * Determines whether a URL is absolute (i.e. contains a scheme such as `http:` or `data:`).
     * A URL is considered absolute when it contains a `:` that appears before any `/`.
     * @param {string} url - The URL string to test.
     * @returns {boolean} `true` if the URL is absolute, `false` otherwise.
     */
    static isAbsoluteUrl(url) {
        if (url.startsWith("data:")) {
            return false;
        }
        const colonIndex = url.indexOf(":");
        const slashIndex = url.indexOf("/");
        return colonIndex !== -1 && (slashIndex === -1 || colonIndex < slashIndex);
    }

    /**
     * Finds the dropped file a glTF URI refers to.
     * First tries the exact path resolved against the glTF's folder. If that fails, falls back
     * to a case-insensitive file name match, preferring the candidate that shares the most
     * trailing path segments with the URI. The fallback is needed for exporters such as the
     * MSFS one, which reference textures outside the model folder (e.g. "../../Assets/x.png").
     * @param {Array<[string, File]>} files - Dropped files as [path, file] pairs.
     * @param {string} uri - The URI from the glTF.
     * @param {string} gltfPath - Path of the glTF file, used to resolve relative URIs.
     * @returns {[string, File] | undefined} The matching [path, file] pair, if any.
     */
    static findFile(files, uri, gltfPath) {
        if (files === undefined || uri === undefined) {
            return undefined;
        }
        let actualPath = uri;
        if (!ResourceLoaderUtils.isAbsoluteUrl(uri)) {
            const parentPath = ResourceLoaderUtils.getContainingFolder(gltfPath ?? "");
            actualPath = ResourceLoaderUtils.cleanRelativePath(parentPath + uri);
        }
        const exactMatch = files.find((file) => file[0] == actualPath);
        if (
            exactMatch !== undefined ||
            uri.startsWith("data:") ||
            ResourceLoaderUtils.isAbsoluteUrl(uri)
        ) {
            return exactMatch;
        }

        let decodedUri = uri;
        try {
            decodedUri = decodeURI(uri);
        } catch {
            // keep the raw URI if it contains malformed escape sequences
        }
        const uriSegments = decodedUri
            .replace(/\\/g, "/")
            .toLowerCase()
            .split("/")
            .filter((segment) => segment !== "" && segment !== "." && segment !== "..");
        if (uriSegments.length === 0) {
            return undefined;
        }
        let bestMatch = undefined;
        let bestScore = 0;
        for (const file of files) {
            const fileSegments = file[0].replace(/\\/g, "/").toLowerCase().split("/");
            let score = 0;
            while (
                score < uriSegments.length &&
                score < fileSegments.length &&
                uriSegments[uriSegments.length - 1 - score] ===
                    fileSegments[fileSegments.length - 1 - score]
            ) {
                score++;
            }
            if (score > bestScore) {
                bestScore = score;
                bestMatch = file;
            }
        }
        return bestMatch;
    }
}

export { ResourceLoaderUtils };
