package com.ps5.textileqc

import kotlin.math.max
import kotlin.math.min

/** Detector post-processing (pure Kotlin, unit-tested against Ultralytics). */
object Post {
    /**
     * Class-agnostic NMS on the raw (1, 5, N) detector output [cx, cy, w, h, score]:
     * keep score > conf, sort by score, drop boxes with IoU > iou, at most maxDet.
     * Returns boxes [x0, y0, x1, y1, score] in 512x512 letterbox pixels.
     */
    fun nms(o: FloatArray, n: Int, conf: Float, iou: Float, maxDet: Int): List<FloatArray> {
        val cand = ArrayList<Int>()
        for (i in 0 until n) if (o[4 * n + i] > conf) cand.add(i)
        cand.sortByDescending { o[4 * n + it] }
        val keep = ArrayList<FloatArray>()
        for (i in cand) {
            val cx = o[i]; val cy = o[n + i]; val bw = o[2 * n + i]; val bh = o[3 * n + i]
            val b = floatArrayOf(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, o[4 * n + i])
            if (keep.all { iou(it, b) <= iou }) {
                keep.add(b)
                if (keep.size >= maxDet) break
            }
        }
        return keep
    }

    fun iou(a: FloatArray, b: FloatArray): Float {
        val ix = max(0f, min(a[2], b[2]) - max(a[0], b[0]))
        val iy = max(0f, min(a[3], b[3]) - max(a[1], b[1]))
        val inter = ix * iy
        val u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return if (u > 0) inter / u else 0f
    }

    /** Letterbox box -> original image pixels, clipped (Ultralytics scale_boxes). */
    fun toImage(b: FloatArray, gain: Float, padLeft: Int, padTop: Int, w: Int, h: Int): FloatArray = floatArrayOf(
        min(max((b[0] - padLeft) / gain, 0f), w.toFloat()),
        min(max((b[1] - padTop) / gain, 0f), h.toFloat()),
        min(max((b[2] - padLeft) / gain, 0f), w.toFloat()),
        min(max((b[3] - padTop) / gain, 0f), h.toFloat())
    )
}
