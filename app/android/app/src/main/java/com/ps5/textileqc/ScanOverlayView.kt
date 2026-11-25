package com.ps5.textileqc

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Shader
import android.util.AttributeSet
import android.view.View
import android.view.animation.LinearInterpolator

/** HUD overlay on the viewer: corner brackets always, a moving scan line while inspecting. */
class ScanOverlayView @JvmOverloads constructor(ctx: Context, attrs: AttributeSet? = null) : View(ctx, attrs) {

    private val cyan = Color.parseColor("#00E5FF")
    private val bracket = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = cyan; strokeWidth = 5f; style = Paint.Style.STROKE; alpha = 200 }
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply { strokeWidth = 4f }
    private val glow = Paint(Paint.ANTI_ALIAS_FLAG)
    private var pos = 0f
    private var anim: ValueAnimator? = null

    var scanning = false
        set(v) {
            field = v
            if (v) {
                anim?.cancel()
                anim = ValueAnimator.ofFloat(0f, 1f).apply {
                    duration = 1100; repeatMode = ValueAnimator.REVERSE; repeatCount = ValueAnimator.INFINITE
                    interpolator = LinearInterpolator()
                    addUpdateListener { pos = it.animatedValue as Float; invalidate() }
                    start()
                }
            } else { anim?.cancel(); anim = null; invalidate() }
        }

    override fun onDraw(c: Canvas) {
        val w = width.toFloat(); val h = height.toFloat(); val k = minOf(w, h) * 0.08f; val p = 10f
        c.drawLine(p, p, p + k, p, bracket); c.drawLine(p, p, p, p + k, bracket)
        c.drawLine(w - p, p, w - p - k, p, bracket); c.drawLine(w - p, p, w - p, p + k, bracket)
        c.drawLine(p, h - p, p + k, h - p, bracket); c.drawLine(p, h - p, p, h - p - k, bracket)
        c.drawLine(w - p, h - p, w - p - k, h - p, bracket); c.drawLine(w - p, h - p, w - p, h - p - k, bracket)
        if (scanning) {
            val y = p + (h - 2 * p) * pos
            glow.shader = LinearGradient(0f, y - 60f, 0f, y, Color.TRANSPARENT, Color.argb(90, 0, 229, 255), Shader.TileMode.CLAMP)
            c.drawRect(p, y - 60f, w - p, y, glow)
            line.color = cyan
            c.drawLine(p, y, w - p, y, line)
        }
    }

    override fun onDetachedFromWindow() { anim?.cancel(); super.onDetachedFromWindow() }
}
