export const defaultSettings = {
    initialCenterX: -1.125,
    initialCenterY: 0.0,
    initialViewportSizeY: 3,
    initialMaxIter: 1000,

    maxiterMode: 'Dynamic',

    colors: [
        { r: 69,  g: 69,  b: 69,  a: 255 },
        { r: 169, g: 169, b: 169, a: 255 },
        { r: 0,   g: 0,   b: 0,   a: 255 },
        { r: 255, g: 0,   b: 0,   a: 255 },
        { r: 139, g: 0,   b: 0,   a: 255 },
        { r: 0,   g: 0,   b: 0,   a: 255 },
    ],

    weights: [1.0, 1.0, 1.0, 1.0, 1.0],

    gradientFunction: (x) => Math.log(x * 9 + 1) / Math.log(10),

    colorPeriod: 128,   // number of iterations before the color palette cycles

    zoomSpeed: 0.85,
    panSpeed: 0.05,
    keyZoomSpeed: 0.92,
    maxIterAdjustFactor: 1.5,
};
