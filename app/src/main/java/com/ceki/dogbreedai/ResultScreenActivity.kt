package com.ceki.dogbreedai

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity



class ResultScreenActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result_screen)

        findViewById<TextView>(R.id.textView).apply {
                text = intent.getStringExtra(EXTRA_MESSAGE)
        }

    }

}
