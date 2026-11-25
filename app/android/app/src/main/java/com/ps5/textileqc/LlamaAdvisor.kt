package com.ps5.textileqc

import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmGenerationConfig
import org.pytorch.executorch.extension.llm.LlmModule

/**
 * Llama 3.2 1B Instruct (ExecuTorch .pte, XNNPACK 8da4w) running on the phone, offline.
 * Same prompt and settings as llama_advice.py on the Raspberry Pi.
 */
class LlamaAdvisor(modelPath: String, tokenizerPath: String) {

    companion object {
        const val SYSTEM = "You are a quality-control assistant in a textile factory. " +
            "Answer in at most 50 words, each action under 10 words, plain text, using exactly this format:\n" +
            "Likely cause: <one sentence>\nActions: 1) <action> 2) <action> 3) <action>"
        val STOP = listOf("<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>")
        val NICE = mapOf(
            "hole_burst" to "hole / burst", "stain_oil" to "oil stain", "broken_yarn_needle" to "broken yarn / needle break",
            "lycra_run" to "lycra run", "jute" to "jute contamination", "unclassified_defect" to "unclassified defect"
        )
        const val FABRIC = "woven cotton"

        fun prompt(defect: String, severity: String, sizePct: Float, decision: String): String {
            val user = "Fabric: $FABRIC. Defect: ${NICE[defect] ?: defect}. Severity: $severity. " +
                "Defect size: ${"%.1f".format(sizePct)}% of the inspected area. System decision: $decision. " +
                "What is the likely cause and what should the operator do?"
            return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + SYSTEM + "<|eot_id|>" +
                "<|start_header_id|>user<|end_header_id|>\n\n" + user + "<|eot_id|>" +
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
        }
    }

    private val module = LlmModule(modelPath, tokenizerPath, 0.0f)
    private val cache = HashMap<String, String>()
    @Volatile private var running = false
    @Volatile private var cancelled = false
    val loadSeconds: Float

    init {
        val t0 = System.nanoTime()
        module.load()
        loadSeconds = (System.nanoTime() - t0) / 1e9f
    }

    /** Stops an answer in progress (e.g. a new photo was taken). Unfinished answers are never cached. */
    fun stop() { if (running) { cancelled = true; module.stop() } }

    /**
     * Generates advice; [onText] receives the growing answer (call from a background thread).
     * Returns the final text and the seconds taken (0 when served from the cache).
     */
    @Synchronized
    fun advise(defect: String, severity: String, sizePct: Float, decision: String, useCache: Boolean = true,
               onText: (String) -> Unit): Pair<String, Float> {
        val key = "$defect|$severity|$decision"
        if (useCache) cache[key]?.let { onText(it); return it to 0f }
        cancelled = false
        val sb = StringBuilder()
        var stopped = false
        val t0 = System.nanoTime()
        val cfg = LlmGenerationConfig.create().echo(false).maxNewTokens(130).seqLen(512).temperature(0.0f).build()
        running = true
        try {
            module.resetContext()
            module.generate(prompt(defect, severity, sizePct, decision), cfg, object : LlmCallback {
                override fun onResult(result: String) {
                    if (stopped) return
                    sb.append(result)
                    val s = sb.toString()
                    if (STOP.any { s.contains(it) }) {
                        stopped = true
                        module.stop()
                    }
                    onText(clean(s))
                }
            })
        } finally {
            running = false
        }
        val text = clean(sb.toString())
        if (cancelled) return "" to -1f                       // interrupted: not a real answer
        if (text.length >= 20) cache[key] = text
        return text to (System.nanoTime() - t0) / 1e9f
    }

    private fun clean(s: String): String {
        var t = s
        for (m in STOP) { val i = t.indexOf(m); if (i >= 0) t = t.substring(0, i) }
        return t.trim()
    }

    fun close() = module.close()
}
