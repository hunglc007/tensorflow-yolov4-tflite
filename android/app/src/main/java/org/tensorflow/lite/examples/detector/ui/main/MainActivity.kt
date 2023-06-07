package org.tensorflow.lite.examples.detector.ui.main

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.examples.detector.databinding.ActivityMainBinding
import org.tensorflow.lite.examples.detector.extensions.getViewModelFactory
import org.tensorflow.lite.examples.detector.ui.detector.DetectorActivity
import java.io.IOException

open class MainActivity : AppCompatActivity() {

    companion object {
        const val TAG: String = "MainActivity"
    }

    private val viewModel by viewModels<MainViewModel> { getViewModelFactory() }

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        viewModel.setUpBitmaps(assets)
        binding.imageView.setImageBitmap(viewModel.bitmap)

        try {
            viewModel.setUpDetector(assets)
        } catch (e: IOException) {
            Log.e(TAG, "Exception initializing detector!")
            Log.e(TAG, e.stackTraceToString())

            Toast.makeText(
                baseContext, "Detector could not be initialized", Toast.LENGTH_SHORT
            ).show()
            finish()
        }

        viewModel.setUpDetectionProcessor(
            binding.imageView,
            binding.trackingOverlay,
            resources.displayMetrics
        )

        setUpListeners()
    }

    private fun setUpListeners() {
        binding.cameraButton.setOnClickListener {
            val intent = Intent(applicationContext, DetectorActivity::class.java)
            startActivity(intent)
        }

        binding.detectButton.setOnClickListener {
            viewModel.processImage()
        }
    }
}