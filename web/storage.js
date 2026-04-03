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

/** @typedef {{ name: string, x: string, y: string, viewport: number }} Bookmark */

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
