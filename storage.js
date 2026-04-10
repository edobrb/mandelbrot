/**
 * storage.js — localStorage persistence for settings and bookmarks.
 */

const SETTINGS_KEY = 'mandelbrot_settings';
const BOOKMARKS_KEY = 'mandelbrot_bookmarks';

/**
 * Persist the tunable subset of settings + state to localStorage.
 */
export function saveSettings(settings, state) {
    const data = {
        colors: settings.colors,
        weights: settings.weights,
        colorPeriod: settings.colorPeriod,
        zoomSpeed: settings.zoomSpeed,
        panSpeed: settings.panSpeed,
        keyZoomSpeed: settings.keyZoomSpeed,
        maxIterAdjustFactor: settings.maxIterAdjustFactor,
        maxiterMode: state.maxiterMode,
        baseMaxIter: state.baseMaxIter,
    };
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(data));
}

/**
 * Load previously saved settings. Returns null if nothing is stored.
 */
export function loadSavedSettings() {
    try {
        const raw = localStorage.getItem(SETTINGS_KEY);
        return raw ? JSON.parse(raw) : null;
    } catch {
        return null;
    }
}

// ---------- Bookmarks ----------

/**
 * @typedef {{ r: number, g: number, b: number, a: number }} ColorStop
 * @typedef {{ colors: ColorStop[], weights: number[], colorPeriod: number, maxiterMode: string, baseMaxIter: number }} BookmarkSettings
 * @typedef {{ name: string, x: string, y: string, viewport: number, settings?: BookmarkSettings }} Bookmark
 */

export function getBookmarks() {
    try {
        const raw = localStorage.getItem(BOOKMARKS_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch {
        return [];
    }
}

function setBookmarks(list) {
    localStorage.setItem(BOOKMARKS_KEY, JSON.stringify(list));
}

/** @param {Bookmark} bm */
export function addBookmark(bm) {
    const list = getBookmarks();
    list.push(bm);
    setBookmarks(list);
}

/** @param {number} idx */
export function removeBookmark(idx) {
    const list = getBookmarks();
    list.splice(idx, 1);
    setBookmarks(list);
}

// ---------- Share link ----------

/**
 * Encode current position + settings into a base64url string for use as a URL hash.
 * @param {object} settings
 * @param {object} state  — must have centerBF_X/Y (BigFloat) or centerX/Y
 */
export function encodeShareHash(settings, state) {
    const data = {
        x: state.centerBF_X ? state.centerBF_X.toDecimalString(40) : String(state.centerX),
        y: state.centerBF_Y ? state.centerBF_Y.toDecimalString(40) : String(state.centerY),
        v: state.viewportSizeY,
        mi: state.baseMaxIter,
        mm: state.maxiterMode,
        cp: settings.colorPeriod,
        c: settings.colors.map(({ r, g, b }) => [r, g, b]),
        w: settings.weights,
        zs: settings.zoomSpeed,
        ps: settings.panSpeed,
        kz: settings.keyZoomSpeed,
        mf: settings.maxIterAdjustFactor,
    };
    const json = JSON.stringify(data);
    // btoa needs a binary string; use TextEncoder to support arbitrary characters
    const bytes = new TextEncoder().encode(json);
    let binary = '';
    bytes.forEach(b => { binary += String.fromCharCode(b); });
    return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

/**
 * Decode a base64url share hash back into a plain object.
 * Returns null if the hash is missing or malformed.
 * @param {string} hash — raw hash string (with or without leading '#')
 */
export function decodeShareHash(hash) {
    try {
        const b64 = hash.replace(/^#/, '').replace(/-/g, '+').replace(/_/g, '/');
        const binary = atob(b64);
        const bytes = Uint8Array.from(binary, ch => ch.charCodeAt(0));
        const json = new TextDecoder().decode(bytes);
        return JSON.parse(json);
    } catch {
        return null;
    }
}
