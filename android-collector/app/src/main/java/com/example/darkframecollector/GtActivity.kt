package com.example.darkframecollector

import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.Spinner
import android.widget.TextView
import java.io.File

/**
 * GT 采集：为 PNNP 下游去噪训练收集干净 GT。
 *
 * 两种模式：
 *  1. 暗室多帧平均 —— 暗环境下对同一场景以固定参数连拍 N 帧，
 *     后期对齐平均得到高 SNR GT（等效 SNR 提升 sqrt(N)）。
 *     输出 gt/burst_avg/<场景>/*.dng（整组 burst 供离线平均）。
 *  2. 亮环境低 ISO 直拍 —— 明亮环境低 ISO 短曝光直接拍，
 *     单帧即接近干净 GT，可连拍数帧择优/冗余。
 *     输出 gt/direct/<场景>/*.dng。
 *
 * 注意：GT 不需要是低光图像——低光条件由训练时 PNNP 代理在线合成。
 * 多帧平均的离线流程（暗帧校正 -> 对齐 -> 平均）在 PC 端进行。
 */
class GtActivity : BaseCaptureActivity() {

    private lateinit var spMode: Spinner
    private lateinit var etScene: EditText
    private lateinit var etIso: EditText
    private lateinit var etExposure: EditText
    private lateinit var etFrames: EditText
    private lateinit var tvProgress: TextView
    private lateinit var tvLog: TextView
    private lateinit var btnStart: Button
    private lateinit var btnStop: Button

    override val tag: String get() = "GtCapture"
    override val rootDirName: String get() = "DarkFrameCollector"

    companion object {
        private const val MODE_BURST_AVG = 0   // 暗室多帧平均
        private const val MODE_DIRECT = 1      // 亮环境低 ISO 直拍
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        keepScreenOn()
        setContentView(R.layout.activity_gt)

        spMode = findViewById(R.id.spMode)
        etScene = findViewById(R.id.etScene)
        etIso = findViewById(R.id.etIso)
        etExposure = findViewById(R.id.etExposure)
        etFrames = findViewById(R.id.etFrames)
        tvProgress = findViewById(R.id.tvProgress)
        tvLog = findViewById(R.id.tvLog)
        btnStart = findViewById(R.id.btnStart)
        btnStop = findViewById(R.id.btnStop)

        spMode.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item,
            listOf("暗室多帧平均（burst 后期对齐平均）", "亮环境低ISO直拍"))

        findViewById<Button>(R.id.btnPreset).setOnClickListener {
            if (spMode.selectedItemPosition == MODE_BURST_AVG) {
                etIso.setText("800"); etExposure.setText("25"); etFrames.setText("25")
            } else {
                etIso.setText("50"); etExposure.setText("10"); etFrames.setText("3")
            }
        }
        btnStart.setOnClickListener { ensureCameraPermission { startCapture() } }
        btnStop.setOnClickListener { running = false }
    }

    override fun onLogLine(msg: String) { tvLog.append("$msg\n") }

    private fun startCapture() {
        val mode = spMode.selectedItemPosition
        val scene = etScene.text.toString().trim().ifEmpty { "scene" }
        val iso = etIso.text.toString().toIntOrNull() ?: 0
        val expMs = etExposure.text.toString().toDoubleOrNull() ?: 0.0
        val frames = etFrames.text.toString().toIntOrNull() ?: 0
        if (iso <= 0 || expMs <= 0.0 || frames <= 0) { toast("参数无效"); return }

        val modeName = if (mode == MODE_BURST_AVG) "burst_avg" else "direct"
        setUiRunning(true)
        running = true
        runInBg(onDone = { setUiRunning(false) }) {
            logUi("ISO范围=$isoRangeStr 曝光范围=$expRangeStr 白电平=${whiteLevel().toInt()} 黑电平=${"%.1f".format(blackLevel())}")
            val cc = characteristics ?: throw IllegalStateException("相机未就绪")
            val isoRange = cc.get(android.hardware.camera2.CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
                ?: android.util.Range(100, 1600)
            val expRange = cc.get(android.hardware.camera2.CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
                ?: android.util.Range(1000000L, 33333333L)
            val isoAdj = iso.coerceIn(isoRange.lower, isoRange.upper)
            val expNs = (expMs * 1e6).toLong().coerceIn(expRange.lower, expRange.upper)
            if (isoAdj != iso) logUi("提示: ISO 被裁剪到设备支持范围 -> $isoAdj")
            if (expNs != (expMs * 1e6).toLong()) logUi("提示: 曝光被裁剪到设备支持范围 -> ${expNs / 1000000.0}ms")

            val outDir = File(sessionDir, "gt/$modeName/$scene")
            runOnUiThread { tvProgress.text = "场景 '$scene' (${frames}帧) 采集中..." }
            logUi("模式=${if (mode == MODE_BURST_AVG) "暗室多帧平均" else "亮环境直拍"} ISO$isoAdj ${expNs / 1000000.0}ms × $frames 帧")

            var warnedOver = false
            var warnedUnder = false
            val saved = captureBurst(isoAdj, expNs, expNs / 1000000.0, frames, outDir,
                nameFn = { "GT_ISO${isoAdj}_ET${expNs / 1000000.0}ms_${"%03d".format(it)}.dng" },
                onFrame = { _, _, _, mean ->
                    val wp = whiteLevel(); val bl = blackLevel()
                    // 过曝检查（两种模式都需要）
                    if (mean > wp * 0.85 && !warnedOver) {
                        warnedOver = true
                        logUi("!! 警告: 帧均值 $mean 接近白电平 $wp，可能过曝（降低曝光或 ISO）")
                    }
                    // 欠曝/全黑检查（亮环境直拍模式：GT 应明亮）
                    if (mode == MODE_DIRECT && mean < bl * 1.5 && !warnedUnder) {
                        warnedUnder = true
                        logUi("!! 警告: 帧均值 $mean 接近黑电平，亮环境直拍模式画面过暗")
                    }
                })
            logUi("场景 '$scene': 已存 $saved/$frames 帧 -> ${outDir.absolutePath}")
            if (mode == MODE_BURST_AVG) {
                logUi("提示: 离线平均流程 = 暗帧校正 -> 对齐 -> 平均（见 PC 端工具）")
            }
        }
    }

    override fun sessionExtras(): org.json.JSONObject {
        val modeName = if (spMode.selectedItemPosition == MODE_BURST_AVG) "burst_avg" else "direct"
        return org.json.JSONObject()
            .put("gt_mode", modeName)
            .put("gt_scene", etScene.text.toString().trim().ifEmpty { "scene" })
    }

    private fun setUiRunning(running: Boolean) {
        btnStart.isEnabled = !running
        btnStop.isEnabled = running
        spMode.isEnabled = !running
        etScene.isEnabled = !running
        etIso.isEnabled = !running
        etExposure.isEnabled = !running
        etFrames.isEnabled = !running
        if (!running) tvProgress.text = "就绪"
    }
}
