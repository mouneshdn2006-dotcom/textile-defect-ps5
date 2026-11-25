package com.ps5.textileqc

import android.Manifest
import android.app.ActivityManager
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.Gravity
import android.view.HapticFeedbackConstants
import android.view.View
import android.view.animation.OvershootInterpolator
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import com.google.android.material.bottomsheet.BottomSheetDialog
import java.io.File
import kotlin.math.max
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {

    private lateinit var image: ZoomImageView
    private lateinit var scan: ScanOverlayView
    private lateinit var empty: TextView
    private lateinit var hint: TextView
    private lateinit var decisionView: TextView
    private lateinit var mLatency: TextView
    private lateinit var mDefects: TextView
    private lateinit var mConf: TextView
    private lateinit var defectList: LinearLayout
    private lateinit var pipeline: TextView
    private lateinit var hudVision: TextView
    private lateinit var hudLlm: TextView
    private lateinit var hudDevice: TextView
    private lateinit var llamaStatus: TextView
    private lateinit var adviceView: TextView
    private lateinit var chipBits: TextView
    private lateinit var log: LogStore

    /** One inspected image of this session (swipe through them). */
    private class Entry(val r: InspectionResult, val source: String, var advice: String = "", var logged: Boolean = false)
    private val session = ArrayList<Entry>()
    private var cur = -1

    private var photoFile: File? = null
    private var pickedModel: Uri? = null
    @Volatile private var adviceRun = 0
    private var int8 = true
    private val bits get() = if (int8) "int8" else "fp32"

    // ------------------------------------------------------------ activity results
    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        val f = photoFile
        if (ok && f != null && f.length() > 0) inspectUri(Uri.fromFile(f), "camera_" + f.name)
    }
    private val pickImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) inspectUri(uri, displayName(uri) ?: "gallery_image")
    }
    private val live = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { res ->
        val path = res.data?.getStringExtra(LiveActivity.EXTRA_FRAME)
        if (path != null) inspectUri(Uri.fromFile(File(path)), "live_frame")
    }
    private var pendingCameraAction: (() -> Unit)? = null
    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) pendingCameraAction?.invoke() else toast("Camera permission is needed for photos and live mode")
        pendingCameraAction = null
    }
    private val pickLlamaModel = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@registerForActivityResult
        pickedModel = uri
        AlertDialog.Builder(this).setTitle("Step 2 of 2")
            .setMessage("Now choose tokenizer.json (the small file, about 9 MB).")
            .setPositiveButton("Choose") { _, _ -> pickLlamaTokenizer.launch(arrayOf("*/*")) }
            .setNegativeButton("Cancel", null).show()
    }
    private val pickLlamaTokenizer = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        val m = pickedModel
        if (uri != null && m != null) installLlama(m, uri)
    }

    // ------------------------------------------------------------ lifecycle
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        image = findViewById(R.id.image); scan = findViewById(R.id.scan); empty = findViewById(R.id.empty); hint = findViewById(R.id.hint)
        decisionView = findViewById(R.id.decision)
        mLatency = findViewById(R.id.mLatency); mDefects = findViewById(R.id.mDefects); mConf = findViewById(R.id.mConf)
        defectList = findViewById(R.id.defectList); pipeline = findViewById(R.id.pipeline)
        hudVision = findViewById(R.id.hudVision); hudLlm = findViewById(R.id.hudLlm); hudDevice = findViewById(R.id.hudDevice)
        llamaStatus = findViewById(R.id.llamaStatus); adviceView = findViewById(R.id.advice); chipBits = findViewById(R.id.chipBits)
        log = LogStore(this)

        val prefs = getPreferences(MODE_PRIVATE)
        int8 = prefs.getBoolean("int8", true)
        chipBits.text = bits.uppercase()
        chipBits.setOnClickListener {
            int8 = !int8; prefs.edit().putBoolean("int8", int8).apply()
            chipBits.text = bits.uppercase(); it.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY)
            loadVision()
        }

        findViewById<View>(R.id.btnCamera).setOnClickListener { it.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY); withCamera { openCamera() } }
        findViewById<View>(R.id.btnGallery).setOnClickListener { pickImage.launch("image/*") }
        findViewById<View>(R.id.btnSample).setOnClickListener { nextSample() }
        findViewById<View>(R.id.btnLive).setOnClickListener {
            withCamera { live.launch(Intent(this, LiveActivity::class.java).putExtra(LiveActivity.EXTRA_BITS, bits)) }
        }
        findViewById<View>(R.id.btnHistory).setOnClickListener { showStats() }
        findViewById<Button>(R.id.btnAdvice).setOnClickListener {
            val e = session.getOrNull(cur)
            if (e == null || e.r.defects.isEmpty()) toast("Inspect an image with a defect first") else askLlama(e, force = true)
        }
        findViewById<Button>(R.id.btnShare).setOnClickListener { shareReport() }
        findViewById<Button>(R.id.btnSetupLlama).setOnClickListener { setupLlamaDialog() }

        image.onSwipe = { dir -> swipe(dir) }
        image.onLongPress = { image.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS); shareReport() }

        adviceView.text = "Advice appears here after a REWORK / REJECT inspection."
        hudDevice.text = deviceInfo()
        loadVision()
        loadLlama()
    }

    override fun onDestroy() {
        session.filter { !it.logged }.forEach { log.add(it.source, it.r, it.advice); it.logged = true }
        super.onDestroy()
    }

    private fun deviceInfo(): String {
        val am = getSystemService(ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val soc = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE
        return "DEVICE ▸ ${Build.MODEL} · $soc · ${Runtime.getRuntime().availableProcessors()} cores · " +
            "${"%.1f".format(mi.totalMem / 1e9)} GB · ${Build.SUPPORTED_ABIS.firstOrNull() ?: ""}"
    }

    // ------------------------------------------------------------ models
    private fun loadVision() {
        val b = bits
        hudVision.text = "VISION ▸ loading ${b.uppercase()} models…"
        AppState.visionThread.execute {
            val msg = try {
                val t0 = System.nanoTime()
                AppState.loadVision(this, b)
                "VISION ▸ YOLO26n + cls · ExecuTorch ${b.uppercase()} · ready (${(System.nanoTime() - t0) / 1_000_000} ms) ✓"
            } catch (e: Throwable) {
                "VISION ▸ ✗ could not load models: ${e.message}"
            }
            runOnUiThread { hudVision.text = msg }
        }
    }

    private fun loadLlama() {
        setLlamaStatus(AppState.llamaStatus)
        AppState.llamaThread.execute {
            AppState.loadLlama(this)
            runOnUiThread { setLlamaStatus(AppState.llamaStatus) }
        }
    }

    private fun setLlamaStatus(s: String) {
        llamaStatus.text = s
        hudLlm.text = "LLM    ▸ " + s.replace("Llama 3.2 1B: ", "Llama 3.2 1B · ").replace("Llama 3.2: ", "Llama 3.2 · ")
    }

    // ------------------------------------------------------------ camera / images
    private fun withCamera(action: () -> Unit) {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) action()
        else { pendingCameraAction = action; cameraPermission.launch(Manifest.permission.CAMERA) }
    }

    private fun openCamera() {
        val dir = File(cacheDir, "photos").apply { mkdirs() }
        dir.listFiles()?.filter { it.lastModified() < System.currentTimeMillis() - 3_600_000 }?.forEach { it.delete() }
        val f = File(dir, "fabric_${System.currentTimeMillis()}.jpg")
        photoFile = f
        takePicture.launch(FileProvider.getUriForFile(this, "$packageName.files", f))
    }

    private fun displayName(uri: Uri): String? = try {
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        }
    } catch (e: Exception) { null }

    private fun sizeOf(uri: Uri): Long = try {
        contentResolver.query(uri, arrayOf(OpenableColumns.SIZE), null, null, null)?.use { c ->
            if (c.moveToFirst() && !c.isNull(0)) c.getLong(0) else -1L
        } ?: -1L
    } catch (e: Exception) { -1L }

    /** Decodes a photo (max 4096 px side), upright according to EXIF. */
    private fun decode(uri: Uri): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / sample > 4096) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample; inPreferredConfig = Bitmap.Config.ARGB_8888 }
        var bmp = contentResolver.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it, null, opts) }
            ?: throw IllegalArgumentException("Not an image")
        val rot = try { contentResolver.openInputStream(uri)!!.use { ExifInterface(it).rotationDegrees } } catch (e: Exception) { 0 }
        if (rot != 0) bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, Matrix().apply { postRotate(rot.toFloat()) }, true)
        return bmp
    }

    /** Runs the next bundled demo image (held-out test images). */
    private fun nextSample() {
        val names = (assets.list("demo") ?: emptyArray()).sorted()
        if (names.isEmpty()) return
        val prefs = getPreferences(MODE_PRIVATE)
        val i = prefs.getInt("sample", 0) % names.size
        prefs.edit().putInt("sample", i + 1).apply()
        val f = File(File(cacheDir, "photos").apply { mkdirs() }, "sample_$i.jpg")
        assets.open("demo/" + names[i]).use { inp -> f.outputStream().use { inp.copyTo(it) } }
        inspectUri(Uri.fromFile(f), "sample_" + names[i])
    }

    /** Swipe on the viewer: right = previous result, left = next result (or next sample at the end). */
    private fun swipe(dir: Int) {
        if (dir > 0 && cur > 0) { cur--; showEntry(session[cur], animate = false) }
        else if (dir < 0 && cur < session.size - 1) { cur++; showEntry(session[cur], animate = false) }
        else if (dir < 0) nextSample()
        else toast("First result of this session")
    }

    // ------------------------------------------------------------ inspection
    private fun inspectUri(uri: Uri, source: String) {
        AppState.llama?.stop()
        adviceRun++
        scan.scanning = true
        empty.visibility = View.GONE
        decisionView.text = "SCANNING…"
        decisionView.background = badge(ContextCompat.getColor(this, R.color.card2))
        val b = bits
        AppState.visionThread.execute {
            try {
                val bmp = decode(uri)
                runOnUiThread { image.setImageBitmap(bmp) }
                val ins = AppState.loadVision(this, b)
                val r = ins.inspect(bmp)
                runOnUiThread {
                    scan.scanning = false
                    // log the previous image once we move on
                    session.filter { !it.logged && it.r.defects.isEmpty() }.forEach { log.add(it.source, it.r, ""); it.logged = true }
                    val e = Entry(r, source)
                    session.add(e); if (session.size > 30) { val old = session.removeAt(0); if (!old.logged) log.add(old.source, old.r, old.advice) }
                    cur = session.size - 1
                    showEntry(e, animate = true)
                    if (r.defects.isEmpty()) { log.add(source, r, ""); e.logged = true } else askLlama(e, force = false)
                }
            } catch (e: Throwable) {
                runOnUiThread {
                    scan.scanning = false
                    decisionView.text = "ERROR"
                    defectList.removeAllViews(); defectList.addView(row("Could not inspect this image: ${e.message}", Color.GRAY))
                }
            }
        }
    }

    private fun badge(color: Int) = GradientDrawable().apply { cornerRadius = 16 * resources.displayMetrics.density; setColor(color) }

    private fun decisionColor(d: String) = ContextCompat.getColor(this, when (d) { "REJECT" -> R.color.reject; "REWORK" -> R.color.rework; else -> R.color.pass })

    private fun showEntry(e: Entry, animate: Boolean) {
        val r = e.r
        image.setImageBitmap(r.annotated)
        hint.visibility = View.VISIBLE
        decisionView.text = r.decision
        decisionView.background = badge(decisionColor(r.decision))
        if (animate) {
            decisionView.scaleX = 0.6f; decisionView.scaleY = 0.6f
            decisionView.animate().scaleX(1f).scaleY(1f).setDuration(380).setInterpolator(OvershootInterpolator(2.5f)).start()
            val fx = when (r.decision) {
                "REJECT" -> if (Build.VERSION.SDK_INT >= 30) HapticFeedbackConstants.REJECT else HapticFeedbackConstants.LONG_PRESS
                "REWORK" -> HapticFeedbackConstants.LONG_PRESS
                else -> if (Build.VERSION.SDK_INT >= 30) HapticFeedbackConstants.CONFIRM else HapticFeedbackConstants.VIRTUAL_KEY
            }
            decisionView.performHapticFeedback(fx)
        }
        mLatency.text = "${r.ms} ms\nLATENCY"
        mDefects.text = "${r.defects.size}\nDEFECTS"
        val best = r.defects.maxOfOrNull { it.typeConf }
        mConf.text = (if (best == null) "—" else "${(best * 100).roundToInt()}%") + "\nCONFIDENCE"

        defectList.removeAllViews()
        if (r.defects.isEmpty()) defectList.addView(row("✓  No defect found — fabric OK", ContextCompat.getColor(this, R.color.pass)))
        r.defects.forEachIndexed { i, d ->
            val col = when (d.severity) { "High" -> R.color.reject; "Medium" -> R.color.rework; else -> R.color.pass }
            val where = if (d.uncertain) "unsure – CHECK MANUALLY" else if (d.wholeImage) "whole-image check" else "size ${"%.2f".format(d.areaPct)}% · found ${bar(d.detConf)}"
            defectList.addView(row("●  ${i + 1}. ${d.type.uppercase()}  ·  ${d.severity.uppercase()}\n    type ${bar(d.typeConf)}  $where",
                ContextCompat.getColor(this, col)))
        }
        val tot = max(1f, r.prepMs + r.detMs + r.clsMs + r.postMs)
        fun line(n: String, v: Float) = "%-9s %s %6.1f ms".format(n, blocks(v / tot), v)
        pipeline.text = listOf("PIPELINE", line("prep", r.prepMs), line("detector", r.detMs), line("classify", r.clsMs),
            line("render", r.postMs), "total     ${r.ms} ms · ${r.bits.uppercase()} · ${cur + 1}/${session.size}").joinToString("\n")

        if (e.advice.isNotEmpty()) adviceView.text = e.advice
        else if (r.defects.isEmpty()) adviceView.text = "No defect — no action needed."
        else if (!animate) adviceView.text = "Tap REGENERATE for Llama advice on this result."
    }

    private fun bar(v: Float) = "${blocks(v, 8)} ${(v * 100).roundToInt()}%"
    private fun blocks(f: Float, n: Int = 12): String { val k = (f.coerceIn(0f, 1f) * n).roundToInt(); return "█".repeat(k) + "░".repeat(n - k) }

    private fun row(text: String, color: Int) = TextView(this).apply {
        this.text = text; setTextColor(color); textSize = 13f; typeface = android.graphics.Typeface.MONOSPACE
        setPadding(0, 6, 0, 6); gravity = Gravity.START
    }

    // ------------------------------------------------------------ Llama advice
    private fun askLlama(e: Entry, force: Boolean) {
        val r = e.r
        val worst = r.defects.maxByOrNull { Inspector.SEV.indexOf(it.severity) } ?: return
        val runId = ++adviceRun
        if (AppState.llama == null && !AppState.llamaReady(this)) {
            adviceView.text = "Llama is not set up on this phone yet. Tap SETUP."
            if (!e.logged) { log.add(e.source, r, ""); e.logged = true }
            return
        }
        adviceView.text = "▌ analysing ${worst.type} (${worst.severity})…"
        AppState.llama?.stop()
        AppState.llamaThread.execute {
            val l = AppState.loadLlama(this)
            runOnUiThread { setLlamaStatus(AppState.llamaStatus) }
            if (l == null) { runOnUiThread { if (runId == adviceRun) adviceView.text = AppState.llamaStatus }; return@execute }
            if (runId != adviceRun) return@execute
            try {
                val (text, secs) = l.advise(worst.type, worst.severity, worst.areaPct, r.decision, useCache = !force) { partial ->
                    if (runId == adviceRun) runOnUiThread { if (runId == adviceRun) adviceView.text = "$partial ▌" }
                }
                if (secs < 0f) return@execute
                runOnUiThread {
                    val note = if (secs == 0f) "⟲ from memory (same defect seen before)" else "⚡ generated in ${"%.1f".format(secs)} s · offline"
                    e.advice = "$text\n\n$note"
                    if (!e.logged) { log.add(e.source, r, text); e.logged = true }
                    if (runId == adviceRun && session.getOrNull(cur) === e) adviceView.text = e.advice
                }
            } catch (ex: Throwable) {
                runOnUiThread { if (runId == adviceRun) adviceView.text = "Llama error: ${ex.message}" }
            }
        }
    }

    // ------------------------------------------------------------ share
    private fun shareReport() {
        val e = session.getOrNull(cur) ?: run { toast("Nothing to share yet"); return }
        val f = File(File(cacheDir, "photos").apply { mkdirs() }, "inspection_report.jpg")
        f.outputStream().use { e.r.annotated.compress(Bitmap.CompressFormat.JPEG, 92, it) }
        val txt = StringBuilder("Textile QC inspection: ${e.r.decision}\n")
        e.r.defects.forEach { txt.append("• ${it.type} – ${it.severity} (type ${(it.typeConf * 100).roundToInt()}%)\n") }
        txt.append("${e.r.ms} ms on-device · ExecuTorch ${e.r.bits.uppercase()} · offline\n")
        if (e.advice.isNotEmpty()) txt.append("\nLlama 3.2 advice:\n${e.advice}\n")
        val uri = FileProvider.getUriForFile(this, "$packageName.files", f)
        val send = Intent(Intent.ACTION_SEND).setType("image/jpeg").putExtra(Intent.EXTRA_STREAM, uri)
            .putExtra(Intent.EXTRA_TEXT, txt.toString()).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        startActivity(Intent.createChooser(send, "Share inspection"))
    }

    // ------------------------------------------------------------ Llama setup (order-independent, no file-name checks)
    private fun setupLlamaDialog() {
        if (AppState.llamaReady(this)) {
            AlertDialog.Builder(this).setTitle("Llama 3.2 is installed ✓")
                .setMessage("The model (${AppState.llamaModel(this).length() / 1_000_000} MB) and tokenizer are already inside the app. " +
                    "Nothing more to do – advice works offline.\n\nRe-install only if you exported a new model.")
                .setPositiveButton("Close", null)
                .setNeutralButton("Re-install") { _, _ -> chooseLlamaFiles() }
                .show()
        } else chooseLlamaFiles()
    }

    private fun chooseLlamaFiles() {
        AlertDialog.Builder(this).setTitle("Set up Llama 3.2 (one time)")
            .setMessage("1. Download model.pte (914 MB) and tokenizer.json from Google Drive → Textile fabric → llama to the phone.\n" +
                "2. Tap CHOOSE and pick model.pte, then tokenizer.json.\n\nThe app copies them inside itself (1–2 minutes).")
            .setPositiveButton("Choose") { _, _ -> pickLlamaModel.launch(arrayOf("*/*")) }
            .setNegativeButton("Cancel", null).show()
    }

    private fun installLlama(a: Uri, b: Uri) {
        // decide which file is which by size (model ~914 MB, tokenizer ~9 MB) – names are unreliable on some phones
        val sa = sizeOf(a); val sb = sizeOf(b)
        val (model, tok) = if (sb > sa) b to a else a to b
        setLlamaStatus("Copying Llama files into the app…")
        AppState.llama?.let { it.stop(); it.close() }
        AppState.llama = null
        AppState.llamaThread.execute {
            try {
                val mOut = AppState.llamaModel(this); val tOut = AppState.llamaTokenizer(this)
                val mTmp = File(mOut.path + ".part"); val tTmp = File(tOut.path + ".part")
                copyWithProgress(tok, tTmp, "tokenizer")
                copyWithProgress(model, mTmp, "model")
                val first = tTmp.inputStream().use { s -> var c: Int; do { c = s.read() } while (c == ' '.code || c == '\n'.code || c == '\r'.code || c == 0xEF || c == 0xBB || c == 0xBF); c }
                when {
                    mTmp.length() < 100_000_000L -> throw IllegalStateException("model.pte looks too small (${mTmp.length() / 1_000_000} MB) – is the download complete?")
                    tTmp.length() > 100_000_000L || first != '{'.code -> throw IllegalStateException("the second file is not tokenizer.json")
                }
                mOut.delete(); tOut.delete(); tTmp.renameTo(tOut); mTmp.renameTo(mOut)
                runOnUiThread { setLlamaStatus("Loading Llama…") }
                AppState.loadLlama(this)
            } catch (e: Throwable) {
                AppState.llamaStatus = "Llama setup failed: ${e.message}"
            }
            runOnUiThread { setLlamaStatus(AppState.llamaStatus) }
        }
    }

    private fun copyWithProgress(src: Uri, dst: File, label: String) {
        val total = sizeOf(src)
        contentResolver.openInputStream(src)!!.use { inp ->
            dst.outputStream().use { out ->
                val buf = ByteArray(1 shl 20)
                var done = 0L; var lastPct = -1
                while (true) {
                    val n = inp.read(buf); if (n < 0) break
                    out.write(buf, 0, n); done += n
                    if (total > 0) {
                        val pct = (100 * done / total).toInt()
                        if (pct != lastPct) { lastPct = pct; runOnUiThread { setLlamaStatus("Copying $label… $pct%") } }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------ stats dashboard
    private fun showStats() {
        session.filter { !it.logged }.forEach { log.add(it.source, it.r, it.advice); it.logged = true }
        val st = log.stats()
        val sheet = BottomSheetDialog(this)
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; setPadding(48, 40, 48, 60); setBackgroundColor(ContextCompat.getColor(context, R.color.card))
        }
        fun t(s: String, size: Float, color: Int, bold: Boolean = false) = TextView(this).apply {
            text = s; textSize = size; setTextColor(color); typeface = if (bold) android.graphics.Typeface.create(android.graphics.Typeface.MONOSPACE, android.graphics.Typeface.BOLD) else android.graphics.Typeface.MONOSPACE
            setPadding(0, 8, 0, 8)
        }
        val cyan = ContextCompat.getColor(this, R.color.cyan); val txt = ContextCompat.getColor(this, R.color.text); val muted = ContextCompat.getColor(this, R.color.muted)
        box.addView(t("INSPECTION DASHBOARD", 15f, cyan, true))
        box.addView(t("${st.images} images inspected · avg ${"%.0f".format(st.avgMs)} ms on-device", 12f, muted))
        for (d in listOf("PASS", "REWORK", "REJECT")) {
            val n = st.decisions[d] ?: 0; val f = if (st.images == 0) 0f else n.toFloat() / st.images
            box.addView(t("%-7s %s %3d  (%d%%)".format(d, blocks(f, 14), n, (f * 100).roundToInt()), 13f, decisionColor(d), true))
        }
        if (st.types.isNotEmpty()) {
            box.addView(t("\nDEFECT TYPES", 13f, cyan, true))
            val maxN = st.types.values.max().toFloat()
            st.types.entries.sortedByDescending { it.value }.forEach { (k, v) -> box.addView(t("%-20s %s %d".format(k, blocks(v / maxN, 8), v), 12f, txt)) }
        }
        box.addView(t("\nRECENT", 13f, cyan, true))
        log.recent(12).forEach { box.addView(t(it, 11f, muted)) }
        val share = com.google.android.material.button.MaterialButton(this).apply {
            text = "Share inspections.csv"; setOnClickListener { sheet.dismiss(); shareCsv() }
        }
        box.addView(share)
        val sv = android.widget.ScrollView(this).apply { addView(box) }
        sheet.setContentView(sv)
        sheet.show()
    }

    private fun shareCsv() {
        if (!log.file.exists()) { toast("No log yet"); return }
        val uri = FileProvider.getUriForFile(this, "$packageName.files", log.file)
        val send = Intent(Intent.ACTION_SEND).setType("text/csv").putExtra(Intent.EXTRA_STREAM, uri).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        startActivity(Intent.createChooser(send, "Share inspections.csv"))
    }

    private fun toast(s: String) = Toast.makeText(this, s, Toast.LENGTH_SHORT).show()
}
