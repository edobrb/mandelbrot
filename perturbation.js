/**
 * perturbation.js — Compute reference orbit at arbitrary precision on CPU.
 */

import { BigFloat } from './bigfloat.js';

/**
 * Compute the reference orbit for perturbation theory.
 *
 * @param {BigFloat} centerRe - Real part of center (arbitrary precision)
 * @param {BigFloat} centerIm - Imaginary part of center (arbitrary precision)
 * @param {number} maxIter - Maximum iterations
 * @param {number} bailoutSq - Bailout radius squared (default 4)
 * @returns {{ re: Float32Array, im: Float32Array, length: number }}
 */
export function computeReferenceOrbit(centerRe, centerIm, maxIter, bailoutSq = 4.0) {
    const re = new Float32Array(maxIter + 1);
    const im = new Float32Array(maxIter + 1);

    let zr = BigFloat.zero();
    let zi = BigFloat.zero();
    re[0] = 0;
    im[0] = 0;

    let len = maxIter;

    for (let i = 0; i < maxIter; i++) {
        const zr2 = zr.mul(zr);
        const zi2 = zi.mul(zi);
        const zri = zr.mul(zi);
        zr = zr2.sub(zi2).add(centerRe);
        zi = zri.add(zri).add(centerIm);

        const zrN = zr.toNumber();
        const ziN = zi.toNumber();
        re[i + 1] = Math.fround(zrN);
        im[i + 1] = Math.fround(ziN);

        if (zrN * zrN + ziN * ziN > bailoutSq) {
            len = i + 1;
            break;
        }
    }

    return { re, im, length: len };
}
