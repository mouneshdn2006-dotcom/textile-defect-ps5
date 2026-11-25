package com.ps5.textileqc

import android.annotation.SuppressLint
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Size
import android.view.GestureDetector
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import java.io.File
import kotlin.math.abs

/**
 * Real-time inspection: camera frames -> detector + classifier on the phone -> boxes and decision.
 * Gestures: pinch = camera zoom, double-tap = torch, swipe down = close, shutter = inspect + Llama.
 */
class LiveActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_FRAME = "frame"
        const val EXTRA_BITS = "bits"
    }

    private lateinit var info: TextView
    private lateinit var decision: TextView
    private lateinit var zoomText: TextView
    private lateinit var view: ImageView
    @Volatile private var lastFrame: Bitmap? = null
    private var provider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    private var fps = 0f
    private var lastDecision = ""
    private var torch = false

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_live)
        info = findViewById(R.id.liveInfo)
        decision = findViewById(R.id.liveDecision)
        zoomText = findViewById(R.id.liveZoom)
        view = findViewById(R.id.liveImage)
        findViewById<ScanOverlayView>(R.id.liveScan).scanning = true
        findViewById<View>(R.id.btnBack).setOnClickListener { finish() }
        findViewById<View>(R.id.btnCapture).setOnClickListener { it.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY); capture() }
        val bits = intent.getStringExtra(EXTRA_BITS) ?: "int8"

        val scale = ScaleGestureDetector(this, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(d: ScaleGestureDetector): Boolean {
                val cam = camera ?: return true
                val z = cam.cameraInfo.zoomState.value ?: return true
                val nz = (z.zoomRatio * d.scaleFactor).coerceIn(z.minZoomRatio, z.maxZoomRatio)
                cam.cameraControl.setZoomRatio(nz)
                zoomText.text = "${"%.1f".format(nz)}×\nZOOM"
                return true
            }
        })
        val gestures = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDown(e: MotionEvent) = true
            override fun onDoubleTap(e: MotionEvent): Boolean {
                val cam = camera ?: return true
                if (cam.cameraInfo.hasFlashUnit()) {
                    torch = !torch; cam.cameraControl.enableTorch(torch)
                    view.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY)
                }
                return true
            }
            override fun onFling(e1: MotionEvent?, e2: MotionEvent, vx: Float, vy: Float): Boolean {
                if (e1 != null && e2.y - e1.y > 250 && abs(vy) > 800 && abs(e2.x - e1.x) < 200) { finish(); return true }
                return false
            }
        })
        view.setOnTouchListener { _, ev -> scale.onTouchEvent(ev); gestures.onTouchEvent(ev); true }

        AppState.visionThread.execute {
            try {
                AppState.loadVision(this, bits)
                runOnUiThread { startCamera() }
            } catch (e: Throwable) {
                runOnUiThread { info.text = "Could not load models: ${e.message}" }
            }
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val p = future.get()
            provider = p
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setResolutionStrategy(ResolutionStrategy(Size(1280, 960), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
                        .build()
                )
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(AppState.visionThread) { proxy ->
                try {
                    val raw = proxy.toBitmap()
                    val rot = proxy.imageInfo.rotationDegrees
                    val frame = if (rot != 0) Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, Matrix().apply { postRotate(rot.toFloat()) }, true) else raw
                    lastFrame = frame
                    val ins = AppState.inspector ?: return@setAnalyzer
                    val t0 = System.nanoTime()
                    val r = ins.inspect(frame)
                    val dt = (System.nanoTime() - t0) / 1e9f
                    fps = if (fps == 0f) 1f / dt else 0.8f * fps + 0.2f / dt
                    val text = "${r.defects.size} defect(s) · ${r.ms} ms · ${"%.1f".format(fps)} FPS · ${r.bits.uppercase()} · offline"
                    runOnUiThread {
                        view.setImageBitmap(r.annotated); info.text = text
                        decision.text = r.decision
                        decision.setTextColor(ContextCompat.getColor(this, when (r.decision) { "REJECT" -> R.color.reject; "REWORK" -> R.color.rework; else -> R.color.pass }))
                        if (r.decision != lastDecision && r.decision == "REJECT") view.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
                        lastDecision = r.decision
                    }
                } catch (e: Throwable) {
                    runOnUiThread { info.text = "Error: ${e.message}" }
                } finally {
                    proxy.close()
                }
            }
            try {
                p.unbindAll()
                camera = p.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
                info.text = "Point the camera at the fabric…"
            } catch (e: Exception) {
                info.text = "Camera error: ${e.message}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun capture() {
        val f = lastFrame ?: return
        val out = File(File(cacheDir, "photos").apply { mkdirs() }, "live_${System.currentTimeMillis()}.jpg")
        out.outputStream().use { f.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        setResult(RESULT_OK, Intent().putExtra(EXTRA_FRAME, out.absolutePath))
        finish()
    }

    override fun onDestroy() {
        provider?.unbindAll()
        super.onDestroy()
    }
}
