package com.ps5.textileqc

import android.content.Context
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/** Models and worker threads shared by the screens (loaded once per app start). */
object AppState {
    val visionThread: ExecutorService = Executors.newSingleThreadExecutor()
    val llamaThread: ExecutorService = Executors.newSingleThreadExecutor()

    @Volatile var inspector: Inspector? = null
    @Volatile var llama: LlamaAdvisor? = null
    @Volatile var llamaStatus: String = "Llama 3.2: not set up"

    init { System.loadLibrary("executorch") }

    fun llamaDir(ctx: Context) = File(ctx.filesDir, "llama").apply { mkdirs() }
    fun llamaModel(ctx: Context) = File(llamaDir(ctx), "model.pte")
    fun llamaTokenizer(ctx: Context) = File(llamaDir(ctx), "tokenizer.json")
    fun llamaReady(ctx: Context) = llamaModel(ctx).length() > 100_000_000L && llamaTokenizer(ctx).length() > 1000L

    /** Loads (or switches) the vision models. Call on visionThread. */
    fun loadVision(ctx: Context, bits: String): Inspector {
        inspector?.let { if (it.bits == bits) return it }
        val ins = Inspector.load(ctx.applicationContext, bits)
        inspector = ins
        return ins
    }

    /** Loads Llama if its files are on the phone. Call on llamaThread. */
    fun loadLlama(ctx: Context): LlamaAdvisor? {
        llama?.let { return it }
        if (!llamaReady(ctx)) { llamaStatus = "Llama 3.2: not set up (tap SET UP LLAMA)"; return null }
        llamaStatus = "Llama 3.2: loading…"
        return try {
            val l = LlamaAdvisor(llamaModel(ctx).absolutePath, llamaTokenizer(ctx).absolutePath)
            llama = l
            llamaStatus = "Llama 3.2 1B: ready (loaded in ${"%.1f".format(l.loadSeconds)} s, offline)"
            l
        } catch (e: Throwable) {
            llamaStatus = "Llama 3.2: failed to load – ${e.message}"
            null
        }
    }
}
