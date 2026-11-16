package com.example.darkframecollector

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import java.io.File
import kotlin.math.max

/**
 * 暗帧采集：ISO × 曝光网格逐档连拍，输出 bias/ISO{iso}/*.dng。
 * 对应 PNNP 训练代码 DarkFrameDataset 的目录约定。
 */
class DarkActivity : BaseCaptureActivity() {

    private lateinit var etIsos: EditText
    private lateinit var etExposures: EditText
    private lateinit var etFrames: EditText
    private lateinit var tvProgress: TextView
    private lateinit var tvLog: TextView
    private lateinit var btnStart: Button
    private lateinit var btnStop: Button

    /** 质检阈值：raw 均值超过该 DN 值判定为"镜头未盖严" */
    private val darkMeanLimit = 1024.0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        keepScreenOn()
        setContentView(R.layout.activity_dark)

        etIsos = findViewById(R.id.etIsos)
        etExposures = findViewById(R.id.etExposures)
        etFrames = findViewById(R.id.etFrames)
        tvProgress = findViewById(R.id.tvProgress)
        tvLog = findViewById(R.id.tvLog)
        btnStart = findViewById(R.id.btnStart)
        btnStop = findViewById(R.id.btnStop)

        findViewById<Button>(R.id.btnPreset).setOnClickListener {
            etIsos.setText("50,64,100,200,400,800,1600,3200,6400")
            etExposures.setText("10,25,40")
            etFrames.setText("15")
        }
        btnStart.setOnClickListener { ensureCameraPermission { startCollection() } }
        btnStop.setOnClickListener { running = false }
    }

    override fun onLogLine(msg: String) { tvLog.append("$msg\n") }

    private fun startCollection() {
        val isos = parseInts(etIsos.text.toString())
        val expsMs = parseDoubles(etExposures.text.toString())
        val frames = etFrames.text.toString().toIntOrNull() ?: 0
        if (isos.isEmpty() || expsMs.isEmpty() || frames <= 0) { toast("参数无效"); return }

        setUiRunning(true)
        running = true
        runInBg(onDone = { setUiRunning(false) }) {
            logUi("ISO范围=$isoRangeStr 曝光范围=$expRangeStr 白电平=${whiteLevel().toInt()} 黑电平=${"%.1f".format(blackLevel())}")
            runCollection(isos, expsMs, frames)
        }
    }

    private fun runCollection(isos: List<Int>, expsMs: List<Double>, framesPerSetting: Int) {
        val cc = characteristics ?: throw IllegalStateException("相机未就绪")
        val isoRange = cc.get(android.hardware.camera2.CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
            ?: android.util.Range(100, 1600)
        val expRange = cc.get(android.hardware.camera2.CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
            ?: android.util.Range(1000000L, 33333333L)

        val totalSettings = isos.size * expsMs.size
        var settingIdx = 0
        for (iso in isos) {
            for (expMs in expsMs) {
                if (!running) { logUi("用户停止"); return }
                settingIdx++
                val expNs = (expMs * 1e6).toLong().coerceIn(expRange.lower, expRange.upper)
                val isoAdj = iso.coerceIn(isoRange.lower, isoRange.upper)
                val progress = "[$settingIdx/$totalSettings] ISO$isoAdj ${expMs}ms"
                runOnUiThread { tvProgress.text = progress }
                logUi("$progress 采集中...")

                val isoDir = File(sessionDir, "bias/ISO$isoAdj")
                var warnedBright = false
                val saved = captureBurst(isoAdj, expNs, expMs, framesPerSetting, isoDir,
                    nameFn = { "ISO${isoAdj}_ET${expMs}ms_${"%03d".format(it)}.dng" },
                    onFrame = { _, _, _, mean ->
                        // 暗帧质检：均值过高 = 漏光
                        val limit = max(darkMeanLimit, blackLevel() * 4)
                        if (mean > limit && !warnedBright) {
                            warnedBright = true
                            logUi("!! 警告: ISO$isoAdj 帧均值 $mean DN 过高，镜头可能未盖严（阈值 $limit）")
                        }
                    })
                logUi("ISO$isoAdj ${expMs}ms: 已存 $saved/$framesPerSetting 帧")
            }
        }
    }

    private fun setUiRunning(running: Boolean) {
        btnStart.isEnabled = !running
        btnStop.isEnabled = running
        etIsos.isEnabled = !running
        etExposures.isEnabled = !running
        etFrames.isEnabled = !running
        if (!running) tvProgress.text = "就绪"
    }
}
