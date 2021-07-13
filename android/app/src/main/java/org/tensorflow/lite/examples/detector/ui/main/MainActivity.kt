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

    private val mViewModel by viewModels<MainViewModel> { getViewModelFactory() }

    private lateinit var mBinding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        mBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(mBinding.root)

        mViewModel.setUpBitmaps(assets)
        mBinding.imageView.setImageBitmap(mViewModel.bitmap)

        try {
            mViewModel.setUpDetector(assets)
        } catch (e: IOException) {
            Log.e(TAG, "Exception initializing detector!")
            Log.e(TAG, e.stackTraceToString())

            Toast.makeText(
                baseContext, "Detector could not be initialized", Toast.LENGTH_SHORT
            ).show()
            finish()
        }

        mViewModel.setUpDetectionProcessor(
            mBinding.imageView,
            mBinding.trackingOverlay,
            resources.displayMetrics
        )

        setUpListeners()
    }

    private fun setUpListeners() {
        mBinding.cameraButton.setOnClickListener {
            val intent = Intent(applicationContext, DetectorActivity::class.java)
            startActivity(intent)
        }

        mBinding.detectButton.setOnClickListener {
            mViewModel.processImage()
        }
    }
}