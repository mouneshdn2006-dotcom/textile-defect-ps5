package com.ps5.textileqc

import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.DataInputStream
import java.io.File
import kotlin.math.abs

/** Compares the phone pre/post-processing with Ultralytics outputs saved by gen_test.py. */
class PipelineTest {
    private val dir = File("/home/claude/apptest_data")

    private fun floats(name: String): FloatArray {
        val f = File(dir, name); val n = (f.length() / 4).toInt()
        DataInputStream(f.inputStream().buffered()).use { d -> return FloatArray(n) { d.readFloat() } }
    }

    private fun image(k: Int): Triple<IntArray, Int, Int> {
        DataInputStream(File(dir, "img$k.bin").inputStream().buffered()).use { d ->
            val w = d.readInt(); val h = d.readInt()
            return Triple(IntArray(w * h) { d.readInt() }, w, h)
        }
    }

    @Test
    fun matchesUltralytics() {
        assumeTrue("test vectors not on this computer – skipped", dir.exists())
        for (k in 0 until 6) {
            val (px, w, h) = image(k)
            // letterbox
            val lb = ImageOps.letterbox(px, w, h, 512)
            val ref = floats("lb$k.bin")
            var maxd = 0f; var off = 0
            for (i in ref.indices) { val d = abs(ref[i] - lb.chw[i]); if (d > maxd) maxd = d; if (d > 0.5f / 255) off++ }
            println("img$k ${w}x$h letterbox: max diff ${maxd * 255} levels, ${100.0 * off / ref.size}% values differ")
            assertTrue("letterbox $k", maxd * 255 <= 1.01f)

            // NMS + scaling (conf 0.01 for the test)
            val o = floats("det$k.bin"); val n = o.size / 5
            val mine = Post.nms(o, n, 0.01f, 0.5f, 10).map { Post.toImage(it, lb.gain, lb.padLeft, lb.padTop, w, h) + it[4] }
            val boxes = floats("boxes$k.bin"); val nb = boxes.size / 5
            println("   boxes mine ${mine.size} vs ultralytics $nb")
            assertTrue("box count $k", mine.size == nb)
            for (j in 0 until nb) for (c in 0 until 5) {
                val d = abs(mine[j][c] - boxes[5 * j + c])
                assertTrue("box $k/$j/$c diff $d", d < 0.05f)
            }

            // classifier input
            val (x0, y0, cw, ch) = File(dir, "cls$k.txt").readText().trim().split(" ").map { it.toInt() }
            val ci = ImageOps.classifierInput(px, w, x0, y0, cw, ch, 224)
            val cref = floats("cls$k.bin")
            var cmax = 0f; var coff = 0
            for (i in cref.indices) { val d = abs(cref[i] - ci[i]); if (d > cmax) cmax = d; if (d > 0.5f / 255) coff++ }
            println("   classifier input: max diff ${cmax * 255} levels, ${100.0 * coff / cref.size}% values differ")
            assertTrue("cls $k", cmax * 255 <= 2.01f)
        }
    }
}
