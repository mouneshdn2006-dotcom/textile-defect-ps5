package com.ps5.textileqc

import android.graphics.Bitmap
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Image pre-processing that reproduces what Ultralytics does on the Raspberry Pi / PC,
 * so the phone gives the same results as the Python program:
 *  - detector:   LetterBox to 512x512 (OpenCV INTER_LINEAR resize, grey 114 padding, centred)
 *  - classifier: Pillow BILINEAR resize of the shorter side to 224 + centre crop 224
 * Both produce RGB float tensors in [0,1], layout CHW.
 */
object ImageOps {

    class Letterbox(val chw: FloatArray, val gain: Float, val padLeft: Int, val padTop: Int)

    /** Pixels of a bitmap as ARGB ints. */
    fun pixels(bmp: Bitmap): IntArray {
        val px = IntArray(bmp.width * bmp.height)
        bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        return px
    }

    /** Same maths as Ultralytics LetterBox(new_shape=S, auto=False, center=True) + cv2.INTER_LINEAR. */
    fun letterbox(px: IntArray, w: Int, h: Int, s: Int = 512): Letterbox {
        val r = min(s.toDouble() / h, s.toDouble() / w)
        val nw = Math.rint(w * r).toInt()
        val nh = Math.rint(h * r).toInt()
        val dw = (s - nw) / 2.0
        val dh = (s - nh) / 2.0
        val top = Math.rint(dh - 0.1).toInt()
        val left = Math.rint(dw - 0.1).toInt()
        val plane = s * s
        val out = FloatArray(3 * plane) { 114f / 255f }

        // cv2 INTER_LINEAR: half-pixel centres, edge clamping
        val sx = w.toFloat() / nw
        val sy = h.toFloat() / nh
        val x0 = IntArray(nw); val x1 = IntArray(nw); val fxA = FloatArray(nw)
        for (x in 0 until nw) {
            var f = (x + 0.5f) * sx - 0.5f
            var i = floor(f).toInt()
            f -= i
            if (i < 0) { i = 0; f = 0f }
            if (i >= w - 1) { i = w - 1; f = 0f }
            x0[x] = i; x1[x] = min(i + 1, w - 1); fxA[x] = f
        }
        for (y in 0 until nh) {
            var fy = (y + 0.5f) * sy - 0.5f
            var iy = floor(fy).toInt()
            fy -= iy
            if (iy < 0) { iy = 0; fy = 0f }
            if (iy >= h - 1) { iy = h - 1; fy = 0f }
            val iy1 = min(iy + 1, h - 1)
            val row0 = iy * w
            val row1 = iy1 * w
            val oy = (y + top) * s
            for (x in 0 until nw) {
                val fx = fxA[x]
                val p00 = px[row0 + x0[x]]; val p01 = px[row0 + x1[x]]
                val p10 = px[row1 + x0[x]]; val p11 = px[row1 + x1[x]]
                val w00 = (1 - fx) * (1 - fy); val w01 = fx * (1 - fy)
                val w10 = (1 - fx) * fy; val w11 = fx * fy
                val o = oy + x + left
                for (c in 0 until 3) {
                    val shift = 16 - 8 * c // R, G, B
                    val v = ((p00 shr shift) and 255) * w00 + ((p01 shr shift) and 255) * w01 +
                            ((p10 shr shift) and 255) * w10 + ((p11 shr shift) and 255) * w11
                    out[c * plane + o] = min(255f, max(0f, v.roundToInt().toFloat())) / 255f
                }
            }
        }
        return Letterbox(out, r.toFloat(), left, top)
    }

    /** Pillow-style antialiased bilinear filter weights for one axis (only for the needed outputs). */
    private class Coeffs(val start: IntArray, val size: IntArray, val k: Array<FloatArray>)

    private fun coeffs(inSize: Int, outSize: Int, from: Int, count: Int): Coeffs {
        val scale = inSize.toDouble() / outSize
        val filterscale = max(scale, 1.0)
        val support = 1.0 * filterscale
        val ss = 1.0 / filterscale
        val start = IntArray(count); val size = IntArray(count)
        val k = Array(count) { FloatArray(0) }
        for (n in 0 until count) {
            val xx = from + n
            val center = (xx + 0.5) * scale
            var xmin = (center - support + 0.5).toInt(); if (xmin < 0) xmin = 0
            var xmax = (center + support + 0.5).toInt(); if (xmax > inSize) xmax = inSize
            xmax -= xmin
            val w = DoubleArray(xmax)
            var ww = 0.0
            for (x in 0 until xmax) {
                val t = kotlin.math.abs((x + xmin - center + 0.5) * ss)
                val v = if (t < 1.0) 1.0 - t else 0.0
                w[x] = v; ww += v
            }
            start[n] = xmin; size[n] = xmax
            k[n] = FloatArray(xmax) { if (ww != 0.0) (w[it] / ww).toFloat() else 0f }
        }
        return Coeffs(start, size, k)
    }

    /**
     * Classifier input from a region of the image: resize shorter side to [s] with Pillow BILINEAR
     * (antialiased), centre-crop s x s, RGB float CHW in [0,1].
     */
    fun classifierInput(px: IntArray, imgW: Int, cx0: Int, cy0: Int, cw: Int, ch: Int, s: Int = 224): FloatArray {
        val nw: Int; val nh: Int
        if (cw < ch) { nw = s; nh = (s.toLong() * ch / cw).toInt() } else { nh = s; nw = (s.toLong() * cw / ch).toInt() }
        val left = Math.rint((nw - s) / 2.0).toInt()   // Python round(): half to even
        val top = Math.rint((nh - s) / 2.0).toInt()
        val cx = coeffs(cw, nw, left, s)
        val cy = coeffs(ch, nh, top, s)
        // rows of the source needed for the vertical pass
        var rMin = Int.MAX_VALUE; var rMax = 0
        for (n in 0 until s) { rMin = min(rMin, cy.start[n]); rMax = max(rMax, cy.start[n] + cy.size[n]) }
        val rows = rMax - rMin
        // horizontal pass (rounded to 8 bit like Pillow)
        val tmp = Array(3) { FloatArray(rows * s) }
        for (r in 0 until rows) {
            val base = (cy0 + rMin + r) * imgW + cx0
            for (n in 0 until s) {
                val st = cx.start[n]; val kk = cx.k[n]
                var a = 0f; var b = 0f; var c = 0f
                for (i in kk.indices) {
                    val p = px[base + st + i]; val wv = kk[i]
                    a += ((p shr 16) and 255) * wv; b += ((p shr 8) and 255) * wv; c += (p and 255) * wv
                }
                val o = r * s + n
                tmp[0][o] = clip8(a); tmp[1][o] = clip8(b); tmp[2][o] = clip8(c)
            }
        }
        // vertical pass
        val plane = s * s
        val out = FloatArray(3 * plane)
        for (m in 0 until s) {
            val st = cy.start[m] - rMin; val kk = cy.k[m]
            for (n in 0 until s) {
                for (ch3 in 0 until 3) {
                    var v = 0f
                    val t = tmp[ch3]
                    for (i in kk.indices) v += t[(st + i) * s + n] * kk[i]
                    out[ch3 * plane + m * s + n] = clip8(v) / 255f
                }
            }
        }
        return out
    }

    private fun clip8(v: Float): Float = min(255f, max(0f, v.roundToInt().toFloat()))
}
