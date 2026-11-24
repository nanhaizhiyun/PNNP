package com.example.darkframecollector

import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.Spinner
import android.widget.TextView
import org.json.JSONObject
import java.io.File
import kotlin.math.max
import kotlin.math.pow

/**
 * 平场采集：为 PTC（光子转移曲线）标定 shot noise 增益 K 采集数据。
 *
 * 对每个 ISO 按曝光网格逐档连拍（默认 10 帧/档），信号覆盖黑电平+2DN ~ 85% 满阱：
 *  - 手动网格：起始/结束曝光 + 步数，log 均匀分布
 *  - 自动倍增：从起始曝光起逐档 ×2，直到帧均值 >= 0.7*白电平（或到设备上限）
 *
 * 输出 flat/ISO{iso}/ET{exp}ms_{idx}.dng + session.json(gt_mode=flat)。
 * 质检：平场均匀性（空间 std/mean > 15% 警告，可能没对准均匀表面）
 *       与过曝警告。离线拟合 Var_DN = g*mu_DN + sigma_read^2 得 K = 1/g。
 */
class FlatActivity : BaseCaptureActivity() {

    private lateinit var spMode: Spinner
    private lateinit var etIsos: EditText
    private lateinit var etExpStart: EditText
    private lateinit var etExpEnd: EditText
    private lateinit var etSteps: EditText
    private lateinit var etFrames: EditText
    private lateinit var tvProgress: TextView
    private lateinit var tvLog: TextView
    private lateinit var btnStart: Button
    private lateinit var btnStop: Button

    override val tag: String get() = "FlatCapture"

    companion object {
        private const val MODE_MANUAL = 0
        private const val MODE_AUTO = 1
        private const val SATURATION_FRACTION = 0.7   // 自动模式停止阈值（均值/白电平）
        private const val UNIFORMITY_LIMIT = 0.15     // 空间 std/mean 均匀性阈值
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        keepScreenOn()
        setContentView(R.layout.activity_flat)

        spMode = findViewById(R.id.spMode)
        etIsos = findViewById(R.id.etIsos)
        etExpStart = findViewById(R.id.etExpStart)
        etExpEnd = findViewById(R.id.etExpEnd)
        etSteps = findViewById(R.id.etSteps)
        etFrames = findViewById(R.id.etFrames)
        tvProgress = findViewById(R.id.tvProgress)
        tvLog = findViewById(R.id.tvLog)
        btnStart = findViewById(R.id.btnStart)
        btnStop = findViewById(R.id.btnStop)

        spMode.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item,
            listOf("手动网格（起始/结束/步数）", "自动倍增（到 0.7 倍白电平停止）"))
        spMode.setSelection(1)

        findViewById<Button>(R.id.btnPreset).setOnClickListener {
            etIsos.setText("100,200,400,800,1550")
            etExpStart.setText("0.5"); etExpEnd.setText("200"); etSteps.setText("15")
            etFrames.setText("10")
        }
        btnStart.setOnClickListener { ensureCameraPermission { startCollection() } }
        btnStop.setOnClickListener { running = false }
    }

    override fun onLogLine(msg: String) { tvLog.append("$msg\n") }

    override fun sessionExtras(): JSONObject =
        JSONObject().put("gt_mode", "flat")

    private fun startCollection() {
        val isos = parseInts(etIsos.text.toString())
        val expStartMs = etExpStart.text.toString().toDoubleOrNull() ?: 0.0
        val expEndMs = etExpEnd.text.toString().toDoubleOrNull() ?: 0.0
        val steps = etSteps.text.toString().toIntOrNull() ?: 0
        val frames = etFrames.text.toString().toIntOrNull() ?: 0
        val auto = spMode.selectedItemPosition == MODE_AUTO
        if (isos.isEmpty() || expStartMs <= 0.0 || frames <= 0) { toast("参数无效"); return }
        if (!auto && (expEndMs <= expStartMs || steps < 2)) { toast("手动模式需要有效的结束曝光与步数"); return }

        setUiRunning(true)
        running = true
        runInBg(onDone = { setUiRunning(false) }) {
            logUi("ISO范围=${isoRangeStr()} 曝光范围=${expRangeStr()} 白电平=${whiteLevel().toInt()} 黑电平=${"%.1f".format(blackLevel())}")
            logUi("对准均匀表面（白墙/匀光板），无纹理无阴影；环境光照在采集全程保持恒定")
            val cc = characteristics ?: throw IllegalStateException("相机未就绪")
            val isoRange = cc.get(android.hardware.camera2.CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
                ?: android.util.Range(100, 1600)
            val expRange = cc.get(android.hardware.camera2.CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
                ?: android.util.Range(1000000L, 33333333L)

            val exposures = if (auto) null else makeGrid(expStartMs, expEndMs, steps)
            var settingIdx = 0
            val total = isos.size * (if (auto) 20 else steps)   // 自动模式上限估 20 档
            for (iso in isos) {
                val isoAdj = iso.coerceIn(isoRange.lower, isoRange.upper)
                if (isoAdj != iso) logUi("提示: ISO$iso 被裁剪到设备支持范围 -> $isoAdj")
                val isoDir = File(sessionDir, "flat/ISO$isoAdj")

                if (auto) {
                    // 自动倍增：从起始曝光逐档 x2，均值达 0.7*白电平或到设备上限停止
                    var expMs = expStartMs
                    var level = 0
                    while (running) {
                        val expNs = (expMs * 1e6).toLong().coerceIn(expRange.lower, expRange.upper)
                        settingIdx++
                        runOnUiThread { tvProgress.text = "[$settingIdx/~] ISO$isoAdj ${expNs/1000000.0}ms (自动)" }
                        val mean = captureLevel(isoAdj, expNs, expNs / 1000000.0, frames, isoDir)
                        if (mean < 0) break
                        logUi("ISO$isoAdj ${expNs/1000000.0}ms: mean=${"%.1f".format(mean)} (${"%.1f".format(100*mean/whiteLevel())}% wp)")
                        if (mean >= SATURATION_FRACTION * whiteLevel() || expNs >= expRange.upper) break
                        expMs *= 2.0
                        level++
                        if (level > 24) { logUi("警告: 已扫 24 档仍未达 0.7 wp，检查照明或起始曝光"); break }
                    }
                } else {
                    for (expMs in exposures!!) {
                        if (!running) break
                        val expNs = (expMs * 1e6).toLong().coerceIn(expRange.lower, expRange.upper)
                        settingIdx++
                        runOnUiThread { tvProgress.text = "[$settingIdx/${isos.size * steps}] ISO$isoAdj ${expNs/1000000.0}ms" }
                        val mean = captureLevel(isoAdj, expNs, expNs / 1000000.0, frames, isoDir)
                        if (mean < 0) break
                        logUi("ISO$isoAdj ${expNs/1000000.0}ms: mean=${"%.1f".format(mean)} (${"%.1f".format(100*mean/whiteLevel())}% wp)")
                    }
                }
                if (!running) { logUi("用户停止"); return@runInBg }
            }
            logUi("平场采集完成。离线拟合: Var_DN = g*mu_DN + sigma_read^2, K = 1/g")
        }
    }

    /** 采一档：burst 拍摄 + 平场质检，返回首帧均值 DN（-1 表示失败/停止） */
    private fun captureLevel(iso: Int, expNs: Long, expMs: Double, frames: Int, isoDir: File): Double {
        var firstMean = -1.0
        var warnedUniformity = false
        var warnedOver = false
        val saved = captureBurst(iso, expNs, expMs, frames, isoDir,
            nameFn = { "FLAT_ISO${iso}_ET${expMs}ms_${"%03d".format(it)}.dng" },
            onFrame = { img, _, idx, mean ->
                if (idx == 0) firstMean = mean
                // 均匀性检查（抽样像素的空间 std/mean）
                val sp = spatialStats(img)
                if (sp.second / max(sp.first, 1.0) > UNIFORMITY_LIMIT && !warnedUniformity) {
                    warnedUniformity = true
                    logUi("!! 警告: 画面不均匀(std/mean=${"%.2f".format(sp.second / max(sp.first, 1.0))})，确认对准无纹理的均匀表面")
                }
                if (mean > whiteLevel() * 0.9 && !warnedOver) {
                    warnedOver = true
                    logUi("!! 警告: 均值接近白电平，该档可能过曝（剔除非线性区拟合）")
                }
            })
        if (saved < frames) logUi("警告: 只存了 $saved/$frames 帧")
        return firstMean
    }

    /** 抽样像素的空间均值与 std（每 16 像素抽 1） */
    private fun spatialStats(img: android.media.Image): Pair<Double, Double> {
        val plane = img.planes[0]
        val buf = plane.buffer.duplicate()
        val rowStride = plane.rowStride
        val pixStride = plane.pixelStride
        val w = img.width; val h = img.height
        val vals = ArrayList<Int>()
        val row = ByteArray(rowStride)
        var y = 0
        while (y < h) {
            buf.position(y * rowStride)
            buf.get(row, 0, rowStride)
            var x = 0
            while (x < w) {
                val off = x * pixStride
                vals.add((row[off].toInt() and 0xFF) or ((row[off + 1].toInt() and 0xFF) shl 8))
                x += 16
            }
            y += 16
        }
        val a = vals.map { it.toDouble() }.toDoubleArray()
        val mean = a.average()
        val std = kotlin.math.sqrt(a.map { (it - mean) * (it - mean) }.average())
        return Pair(mean, std)
    }

    /** log 均匀曝光网格 */
    private fun makeGrid(startMs: Double, endMs: Double, steps: Int): List<Double> {
        val logS = kotlin.math.log10(startMs); val logE = kotlin.math.log10(endMs)
        return (0 until steps).map { 10.0.pow(logS + (logE - logS) * it / (steps - 1)) }
    }

    private fun setUiRunning(running: Boolean) {
        btnStart.isEnabled = !running
        btnStop.isEnabled = running
        spMode.isEnabled = !running
        etIsos.isEnabled = !running
        etExpStart.isEnabled = !running
        etExpEnd.isEnabled = !running
        etSteps.isEnabled = !running
        etFrames.isEnabled = !running
        if (!running) tvProgress.text = "就绪"
    }
}
