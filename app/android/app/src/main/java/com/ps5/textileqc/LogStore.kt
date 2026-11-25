package com.ps5.textileqc

import android.content.Context
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Inspection log as CSV (same columns as inspections.csv on the Raspberry Pi). */
class LogStore(ctx: Context) {
    val file = File(File(ctx.filesDir, "logs").apply { mkdirs() }, "inspections.csv")
    private val header = "time,source,decision,defect_type,severity,det_conf,type_conf,area_pct,advice,model_bits,ms"

    @Synchronized
    fun add(source: String, r: InspectionResult, advice: String) {
        if (!file.exists()) file.writeText(header + "\n")
        val now = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US).format(Date())
        val sb = StringBuilder()
        if (r.defects.isEmpty()) {
            sb.append(row(now, source, r.decision, "none", "", "", "", "0", advice, r.bits, r.ms))
        } else for (d in r.defects) {
            sb.append(row(now, source, r.decision, d.type, d.severity, "%.3f".format(Locale.US, d.detConf),
                "%.3f".format(Locale.US, d.typeConf), "%.2f".format(Locale.US, d.areaPct), advice, r.bits, r.ms))
        }
        file.appendText(sb.toString())
    }

    private fun row(vararg v: Any): String = v.joinToString(",") { q(it.toString()) } + "\n"
    private fun q(s: String) = if (s.any { it == ',' || it == '"' || it == '\n' }) "\"" + s.replace("\"", "\"\"") + "\"" else s

    /** Last [n] data rows, newest first, as short readable lines. */
    fun recent(n: Int = 40): List<String> {
        if (!file.exists()) return emptyList()
        val rows = parse(file.readText()).drop(1)
        return rows.takeLast(n).reversed().map { c ->
            val t = c.getOrElse(0) { "" }.replace('T', ' ')
            val type = c.getOrElse(3) { "" }
            val sev = c.getOrElse(4) { "" }
            val area = c.getOrElse(7) { "" }
            "$t  •  ${c.getOrElse(2) { "" }}  •  ${if (type == "none") "no defect" else "$type $sev (${area}%)"}  •  ${c.getOrElse(10) { "" }} ms"
        }
    }

    fun count(): Int = if (file.exists()) parse(file.readText()).size - 1 else 0

    class Stats(val images: Int, val decisions: Map<String, Int>, val types: Map<String, Int>, val avgMs: Double)

    /** Per-image summary of the whole log (rows of the same image share time + source). */
    fun stats(): Stats {
        if (!file.exists()) return Stats(0, emptyMap(), emptyMap(), 0.0)
        val rows = parse(file.readText()).drop(1).filter { it.size >= 11 }
        val images = rows.groupBy { it[0] + "|" + it[1] }
        val dec = HashMap<String, Int>(); val types = HashMap<String, Int>(); var ms = 0.0
        for ((_, r) in images) {
            dec[r[0][2]] = (dec[r[0][2]] ?: 0) + 1
            ms += r[0][10].toDoubleOrNull() ?: 0.0
            for (x in r) if (x[3] != "none") types[x[3]] = (types[x[3]] ?: 0) + 1
        }
        return Stats(images.size, dec, types, if (images.isEmpty()) 0.0 else ms / images.size)
    }

    private fun parse(text: String): List<List<String>> {
        val rows = ArrayList<List<String>>(); var cur = ArrayList<String>(); val f = StringBuilder(); var inQ = false; var i = 0
        while (i < text.length) {
            val ch = text[i]
            if (inQ) {
                if (ch == '"') { if (i + 1 < text.length && text[i + 1] == '"') { f.append('"'); i++ } else inQ = false } else f.append(ch)
            } else when (ch) {
                '"' -> inQ = true
                ',' -> { cur.add(f.toString()); f.clear() }
                '\n' -> { cur.add(f.toString()); f.clear(); rows.add(cur); cur = ArrayList() }
                else -> f.append(ch)
            }
            i++
        }
        if (f.isNotEmpty() || cur.isNotEmpty()) { cur.add(f.toString()); rows.add(cur) }
        return rows
    }
}
