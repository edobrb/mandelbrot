/**
 * bigfloat.js — Arbitrary precision floating point using BigInt.
 * Represents value = mantissa * 2^exponent, where mantissa has PREC bits.
 */

const PREC = 256; // bits of working precision (~77 decimal digits)

function bigBitLen(n) {
    if (n <= 0n) return 0;
    let bits = 0;
    let v = n;
    // Fast path: skip 64 bits at a time
    while (v > 0xFFFFFFFFFFFFFFFFn) { v >>= 64n; bits += 64; }
    while (v > 0n) { v >>= 1n; bits++; }
    return bits;
}

export class BigFloat {
    constructor(m, e) {
        this.m = m;  // BigInt mantissa
        this.e = e;  // number exponent (base-2)
    }

    static zero() { return new BigFloat(0n, 0); }

    static fromNumber(n) {
        if (n === 0 || !isFinite(n)) return BigFloat.zero();
        const buf = new ArrayBuffer(8);
        new Float64Array(buf)[0] = n;
        const bits = new BigUint64Array(buf)[0];
        const sign = (bits >> 63n) !== 0n ? -1n : 1n;
        const rawExp = Number((bits >> 52n) & 0x7FFn);
        let frac = bits & ((1n << 52n) - 1n);
        if (rawExp === 0) {
            // subnormal
            return new BigFloat(sign * frac, -1074)._norm();
        }
        frac |= (1n << 52n);
        return new BigFloat(sign * frac, rawExp - 1023 - 52)._norm();
    }

    /**
     * Parse a decimal string like "-1.23456789012345678901234567890"
     */
    static fromString(s) {
        s = s.trim();
        if (s === '0') return BigFloat.zero();
        const neg = s.startsWith('-');
        if (neg) s = s.slice(1);
        if (s.startsWith('+')) s = s.slice(1);

        let dotIdx = s.indexOf('.');
        if (dotIdx < 0) dotIdx = s.length;

        const intPart = s.slice(0, dotIdx);
        const fracPart = s.slice(dotIdx + 1);
        const combined = intPart + fracPart;
        const fracDigits = fracPart.length;

        // value = combined * 10^(-fracDigits)
        let bigVal = BigInt(combined);
        if (neg) bigVal = -bigVal;

        if (fracDigits === 0) {
            return new BigFloat(bigVal, 0)._norm();
        }

        // Multiply by 2^(PREC + fracDigits*4) then divide by 10^fracDigits
        // to maintain precision
        const extraBits = PREC + fracDigits * 4;
        const shifted = bigVal << BigInt(extraBits);
        const pow10 = 10n ** BigInt(fracDigits);
        const result = shifted / pow10;
        return new BigFloat(result, -extraBits)._norm();
    }

    _norm() {
        if (this.m === 0n) { this.e = 0; return this; }
        const abs = this.m < 0n ? -this.m : this.m;
        const bits = bigBitLen(abs);
        const shift = PREC - bits;
        if (shift > 0) {
            this.m <<= BigInt(shift);
            this.e -= shift;
        } else if (shift < 0) {
            const s = BigInt(-shift);
            // Round to nearest
            this.m = (this.m >> s) + ((this.m >> (s - 1n)) & 1n);
            this.e += -shift;
        }
        return this;
    }

    toNumber() {
        if (this.m === 0n) return 0;
        const shift = PREC - 53;
        const m = Number(this.m >> BigInt(shift));
        return m * (2 ** (this.e + shift));
    }

    toF32() { return Math.fround(this.toNumber()); }

    add(b) {
        if (this.m === 0n) return b.clone();
        if (b.m === 0n) return this.clone();
        const diff = this.e - b.e;
        if (diff > PREC + 10) return this.clone();
        if (diff < -(PREC + 10)) return b.clone();
        let m1 = this.m, m2 = b.m, e;
        if (diff > 0) { m1 <<= BigInt(diff); e = b.e; }
        else if (diff < 0) { m2 <<= BigInt(-diff); e = this.e; }
        else { e = this.e; }
        return new BigFloat(m1 + m2, e)._norm();
    }

    sub(b) { return this.add(new BigFloat(-b.m, b.e)); }

    mul(b) { return new BigFloat(this.m * b.m, this.e + b.e)._norm(); }

    neg() { return new BigFloat(-this.m, this.e); }
    clone() { return new BigFloat(this.m, this.e); }

    isZero() { return this.m === 0n; }

    /**
     * Convert to decimal string with up to `digits` decimal places.
     */
    toDecimalString(digits = 50) {
        if (this.m === 0n) return '0';
        const neg = this.m < 0n;
        let absM = neg ? -this.m : this.m;
        const e = this.e;

        // value = absM * 2^e
        if (e >= 0) {
            const val = absM << BigInt(e);
            return (neg ? '-' : '') + val.toString();
        }

        // value = absM / 2^(-e)
        // Scale by 10^digits: absM * 10^digits / 2^(-e)
        const pow10 = 10n ** BigInt(digits);
        const negE = BigInt(-e);
        const scaled = (absM * pow10) >> negE;
        let str = scaled.toString();

        // Pad with leading zeros if needed
        if (str.length <= digits) {
            str = '0'.repeat(digits - str.length + 1) + str;
        }

        const intLen = str.length - digits;
        const intStr = str.slice(0, intLen) || '0';
        const fracStr = str.slice(intLen).replace(/0+$/, '') || '0';
        return (neg ? '-' : '') + intStr + '.' + fracStr;
    }

    toString() {
        return this.toDecimalString(60);
    }
}
