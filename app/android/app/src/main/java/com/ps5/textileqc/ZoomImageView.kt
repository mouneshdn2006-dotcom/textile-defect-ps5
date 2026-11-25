package com.ps5.textileqc

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Matrix
import android.graphics.RectF
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import androidx.appcompat.widget.AppCompatImageView
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Result viewer with gestures:
 *  pinch = zoom, drag = pan (when zoomed), double-tap = zoom in/out,
 *  swipe left/right (when not zoomed) = next/previous result, long-press = share.
 */
class ZoomImageView @JvmOverloads constructor(ctx: Context, attrs: AttributeSet? = null) : AppCompatImageView(ctx, attrs) {

    var onSwipe: ((dir: Int) -> Unit)? = null      // -1 = swipe left (next), +1 = swipe right (previous)
    var onLongPress: (() -> Unit)? = null

    private val m = Matrix()
    private val base = Matrix()
    private var scale = 1f
    private val maxScale = 6f

    init { scaleType = ScaleType.MATRIX }

    private val scaleDetector = ScaleGestureDetector(ctx, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(d: ScaleGestureDetector): Boolean {
            val f = d.scaleFactor
            val ns = min(maxScale, max(1f, scale * f))
            val real = ns / scale
            scale = ns
            m.postScale(real, real, d.focusX, d.focusY)
            clamp(); imageMatrix = m
            return true
        }
    })

    private val gestures = GestureDetector(ctx, object : GestureDetector.SimpleOnGestureListener() {
        override fun onDown(e: MotionEvent) = true
        override fun onScroll(e1: MotionEvent?, e2: MotionEvent, dx: Float, dy: Float): Boolean {
            if (scale > 1.01f) { m.postTranslate(-dx, -dy); clamp(); imageMatrix = m; return true }
            return false
        }
        override fun onDoubleTap(e: MotionEvent): Boolean {
            animateTo(if (scale > 1.5f) 1f else 2.5f, e.x, e.y); return true
        }
        override fun onFling(e1: MotionEvent?, e2: MotionEvent, vx: Float, vy: Float): Boolean {
            if (scale > 1.01f || e1 == null) return false
            val dx = e2.x - e1.x; val dy = e2.y - e1.y
            if (abs(dx) > 120 && abs(dx) > 2 * abs(dy) && abs(vx) > 400) { onSwipe?.invoke(if (dx < 0) -1 else 1); return true }
            return false
        }
        override fun onLongPress(e: MotionEvent) { onLongPress?.invoke() }
    })

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (drawable == null) return super.onTouchEvent(event)
        scaleDetector.onTouchEvent(event)
        gestures.onTouchEvent(event)
        // don't scroll the page while zooming / panning
        parent?.requestDisallowInterceptTouchEvent(scale > 1.01f || event.pointerCount > 1)
        return true
    }

    override fun setImageBitmap(bm: android.graphics.Bitmap?) {
        super.setImageBitmap(bm)
        post { fit() }
    }

    override fun onSizeChanged(w: Int, h: Int, ow: Int, oh: Int) { super.onSizeChanged(w, h, ow, oh); fit() }

    /** Fit the image in the view (centred), reset zoom. */
    fun fit() {
        val d = drawable ?: return
        if (width == 0 || height == 0) return
        val iw = d.intrinsicWidth.toFloat(); val ih = d.intrinsicHeight.toFloat()
        if (iw <= 0 || ih <= 0) return
        val s = min(width / iw, height / ih)
        base.reset(); base.postScale(s, s); base.postTranslate((width - iw * s) / 2f, (height - ih * s) / 2f)
        m.set(base); scale = 1f; imageMatrix = m
    }

    private fun clamp() {
        val d = drawable ?: return
        val r = RectF(0f, 0f, d.intrinsicWidth.toFloat(), d.intrinsicHeight.toFloat())
        m.mapRect(r)
        var dx = 0f; var dy = 0f
        if (r.width() <= width) dx = (width - r.width()) / 2f - r.left
        else { if (r.left > 0) dx = -r.left; if (r.right < width) dx = width - r.right }
        if (r.height() <= height) dy = (height - r.height()) / 2f - r.top
        else { if (r.top > 0) dy = -r.top; if (r.bottom < height) dy = height - r.bottom }
        m.postTranslate(dx, dy)
    }

    private fun animateTo(target: Float, fx: Float, fy: Float) {
        val start = scale
        ValueAnimator.ofFloat(start, target).apply {
            duration = 220
            addUpdateListener {
                val v = it.animatedValue as Float
                if (target == 1f && v <= 1.001f) { fit(); return@addUpdateListener }
                val real = v / scale; scale = v
                m.postScale(real, real, fx, fy); clamp(); imageMatrix = m
            }
            start()
        }
    }
}
