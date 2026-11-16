package com.example.darkframecollector

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.widget.Button

/** 入口：选择采集任务 */
class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        findViewById<Button>(R.id.btnGt).setOnClickListener {
            startActivity(Intent(this, GtActivity::class.java))
        }
        findViewById<Button>(R.id.btnDark).setOnClickListener {
            startActivity(Intent(this, DarkActivity::class.java))
        }
    }
}
