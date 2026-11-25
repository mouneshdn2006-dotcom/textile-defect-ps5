package com.ps5.textileqc

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import kotlin.math.max
import kotlin.math.min

data class Defect(
    val type: String, val severity: String, val detConf: Float, val typeConf: Float,
    val areaPct: Float, val box: RectF, val wholeImage: Boolean = false, val uncertain: Boolean = false
)

class InspectionResult(
    val annotated: Bitmap, val decision: String, val defects: List<Defect>,
    val ms: Long, val bits: String,
    val prepMs: Float = 0f, val detMs: Float = 0f, val clsMs: Float = 0f, val postMs: Float = 0f
)

/**
 * Two-stage inspection with the trained ExecuTorch models (same logic as final.py on the Pi):
 * YOLO26n detector (1 class "defect") -> YOLO26n-cls classifier (6 classes) on each box ->
 * severity from box size -> PASS / REWORK / REJECT.
 */
class Inspector private constructor(val bits: String, private val det: Module, private val cls: Module, private val names: List<String>) {

    companion object {
        const val DET_SIZE = 512
        const val CLS_SIZE = 224
        const val CONF = 0.10f      // tuned on the validation split (was 0.20)
        const val IOU = 0.5f          // Ultralytics predict default
        const val MAX_DET = 10
        const val UNSURE = 0.60f       // whole-image check below this confidence = "check manually"
        var TTA = false                // classifier test-time augmentation (flip)
        // Whole-image classifier check: minimum confidence per defect class, tuned on the validation split
        // separately for each precision (INT8 probabilities are less peaked than FP32).
        val WHOLE_CONF = mapOf(
            "int8" to mapOf("broken_yarn_needle" to 0.30f, "hole_burst" to 0.30f, "jute" to 0.70f, "lycra_run" to 0.30f, "stain_oil" to 0.30f),
            "fp32" to mapOf("broken_yarn_needle" to 0.70f, "hole_burst" to 0.25f, "jute" to 0.70f, "lycra_run" to 0.60f, "stain_oil" to 0.25f))
        val CRITICAL = setOf("hole_burst", "broken_yarn_needle")
        val SEV = listOf("Low", "Medium", "High")
        val SEV_COLOR = mapOf("Low" to Color.rgb(0, 200, 0), "Medium" to Color.rgb(255, 165, 0), "High" to Color.rgb(255, 40, 40))
        val DEC_COLOR = mapOf("PASS" to Color.rgb(46, 173, 91), "REWORK" to Color.rgb(240, 160, 32), "REJECT" to Color.rgb(229, 72, 77))

        /** Copies the bundled models out of the APK once and loads them. */
        fun load(ctx: Context, bits: String): Inspector {
            val dir = File(ctx.filesDir, "models").apply { mkdirs() }
            fun asset(name: String): String {
                val f = File(dir, name)
                val ver = ctx.packageManager.getPackageInfo(ctx.packageName, 0).lastUpdateTime
                if (!f.exists() || f.lastModified() < ver) {
                    ctx.assets.open("models/$name").use { inp -> f.outputStream().use { inp.copyTo(it) } }
                }
                return f.absolutePath
            }
            val det = Module.load(asset("detector_$bits.pte"))
            val cls = Module.load(asset("classifier_$bits.pte"))
            val names = ctx.assets.open("models/classes.txt").bufferedReader().readLines().map { it.trim() }.filter { it.isNotEmpty() }
            // warm-up run (first run of each model is slow while XNNPACK prepares its kernels)
            det.forward(EValue.from(Tensor.fromBlob(FloatArray(3 * DET_SIZE * DET_SIZE), longArrayOf(1, 3, DET_SIZE.toLong(), DET_SIZE.toLong()))))
            cls.forward(EValue.from(Tensor.fromBlob(FloatArray(3 * CLS_SIZE * CLS_SIZE), longArrayOf(1, 3, CLS_SIZE.toLong(), CLS_SIZE.toLong()))))
            return Inspector(bits, det, cls, names)
        }

        fun severity(type: String, areaPct: Float): String {
            var lvl = if (areaPct < 0.5f) 0 else if (areaPct < 2.0f) 1 else 2
            if (type in CRITICAL) lvl = min(lvl + 1, 2)
            return SEV[lvl]
        }

        fun decide(sevs: List<String>) = if ("High" in sevs) "REJECT" else if ("Medium" in sevs) "REWORK" else "PASS"
    }

    /** Classifier probabilities; with TTA the horizontally flipped crop is averaged in. */
    private fun classify(chw: FloatArray): FloatArray {
        val shape = longArrayOf(1, 3, CLS_SIZE.toLong(), CLS_SIZE.toLong())
        val p = cls.forward(EValue.from(Tensor.fromBlob(chw, shape)))[0].toTensor().dataAsFloatArray
        if (!TTA) return p
        val f = FloatArray(chw.size)
        val s = CLS_SIZE
        for (c in 0 until 3) for (y in 0 until s) { val row = c * s * s + y * s; for (x in 0 until s) f[row + x] = chw[row + s - 1 - x] }
        val q = cls.forward(EValue.from(Tensor.fromBlob(f, shape)))[0].toTensor().dataAsFloatArray
        return FloatArray(p.size) { (p[it] + q[it]) / 2f }
    }

    @Synchronized
    fun inspect(src: Bitmap): InspectionResult {
        val t0 = System.nanoTime()
        val w = src.width; val h = src.height
        val px = ImageOps.pixels(src)

        // ---- detector
        val lb = ImageOps.letterbox(px, w, h, DET_SIZE)
        val t1 = System.nanoTime()
        val out = det.forward(EValue.from(Tensor.fromBlob(lb.chw, longArrayOf(1, 3, DET_SIZE.toLong(), DET_SIZE.toLong()))))[0].toTensor()
        val o = out.dataAsFloatArray
        val n = out.shape()[2].toInt()           // (1, 5, N): cx, cy, w, h, score
        val t2 = System.nanoTime()
        val boxes = Post.nms(o, n, CONF, IOU, MAX_DET)
        var clsNs = 0L

        // ---- classifier on each box (+15% margin), severity
        val defects = ArrayList<Defect>()
        for (b in boxes) {
            val score = b[4]
            val (x0, y0, x1, y1) = Post.toImage(b, lb.gain, lb.padLeft, lb.padTop, w, h)
            val bw = x1 - x0; val bh = y1 - y0
            val cx0 = max(0f, x0 - .15f * bw).toInt(); val cy0 = max(0f, y0 - .15f * bh).toInt()
            val cx1 = min(w.toFloat(), x1 + .15f * bw).toInt(); val cy1 = min(h.toFloat(), y1 + .15f * bh).toInt()
            if (cx1 - cx0 < 1 || cy1 - cy0 < 1) continue
            val tc = System.nanoTime()
            val probs = classify(ImageOps.classifierInput(px, w, cx0, cy0, cx1 - cx0, cy1 - cy0, CLS_SIZE))
            clsNs += System.nanoTime() - tc
            var top = 0
            for (i in probs.indices) if (probs[i] > probs[top]) top = i
            var type = names.getOrElse(top) { "class_$top" }
            if (type == "good") type = "unclassified_defect"
            val area = 100f * bw * bh / (w.toFloat() * h)
            defects.add(Defect(type, severity(type, area), score, probs[top], area, RectF(x0, y0, x1, y1)))
        }
        // Second check when the detector finds nothing: classify the whole image.
        // Catches close-up defects on fabrics unlike the detector's training data.
        if (defects.isEmpty()) {
            val tc = System.nanoTime()
            val probs = classify(ImageOps.classifierInput(px, w, 0, 0, w, h, CLS_SIZE))
            clsNs += System.nanoTime() - tc
            var top = 0
            for (i in probs.indices) if (probs[i] > probs[top]) top = i
            val type = names.getOrElse(top) { "class_$top" }
            if (type != "good" && probs[top] >= (WHOLE_CONF[bits]?.get(type) ?: 0.6f)) {
                // unsure (< 60 %): ask the operator to check -> REWORK instead of REJECT
                val unsure = probs[top] < UNSURE
                val sev = if (unsure) "Medium" else if (type in CRITICAL) "High" else "Medium"
                defects.add(Defect(type, sev, 0f, probs[top], 100f, RectF(0f, 0f, w.toFloat(), h.toFloat()), wholeImage = true, uncertain = unsure))
            }
        }
        val decision = decide(defects.map { it.severity })
        val annotated = draw(src, defects, decision)
        val ms = (System.nanoTime() - t0) / 1_000_000
        val prep = (t1 - t0) / 1e6f; val detMs = (t2 - t1) / 1e6f; val clsMs = clsNs / 1e6f
        return InspectionResult(annotated, decision, defects, ms, bits, prep, detMs, clsMs, ms - prep - detMs - clsMs)
    }

    private fun draw(src: Bitmap, defects: List<Defect>, decision: String): Bitmap {
        val bmp = src.copy(Bitmap.Config.ARGB_8888, true)
        val c = Canvas(bmp)
        val unit = max(2f, min(bmp.width, bmp.height) / 220f)
        val box = Paint().apply { style = Paint.Style.STROKE; strokeWidth = unit * 1.2f; isAntiAlias = true }
        val txt = Paint().apply { color = Color.WHITE; textSize = unit * 7f; typeface = Typeface.DEFAULT_BOLD; isAntiAlias = true }
        val bg = Paint().apply { style = Paint.Style.FILL }
        for (d in defects) {
            val col = SEV_COLOR[d.severity] ?: Color.RED
            box.color = col; bg.color = col
            c.drawRect(d.box, box)
            val label = if (d.uncertain) "${d.type}? check manually" else "${d.type} ${d.severity}"
            val tw = txt.measureText(label)
            val ty = if (d.wholeImage) bmp.height - unit else max(txt.textSize + unit, d.box.top)
            c.drawRect(d.box.left, ty - txt.textSize - unit, d.box.left + tw + 2 * unit, ty, bg)
            c.drawText(label, d.box.left + unit, ty - unit * 1.5f, txt)
        }
        val big = Paint(txt).apply { textSize = unit * 14f }
        bg.color = DEC_COLOR[decision] ?: Color.GRAY
        c.drawRect(0f, 0f, big.measureText(decision) + 6 * unit, big.textSize + 5 * unit, bg)
        c.drawText(decision, 3 * unit, big.textSize + unit, big)
        return bmp
    }
}
