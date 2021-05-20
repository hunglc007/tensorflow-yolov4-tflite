package org.tensorflow.lite.examples.detector.activities

import android.content.Intent
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.view.Surface
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.tensorflow.lite.examples.detector.Detector
import org.tensorflow.lite.examples.detector.DetectorFactory
import org.tensorflow.lite.examples.detector.databinding.ActivityMainBinding
import org.tensorflow.lite.examples.detector.enums.DetectionModel
import org.tensorflow.lite.examples.detector.helpers.DetectionProcessor
import org.tensorflow.lite.examples.detector.utils.ImageUtils
import java.io.IOException

open class MainActivity : AppCompatActivity() {

    companion object {
        const val TAG: String = "MainActivity"

        const val MINIMUM_SCORE: Float = 0.5f

        val DETECTION_MODEL: DetectionModel = DetectionModel.YOLO_V4_416_FP32

        /*
        * For some reason you have to use Surface.ROTATION_270 here
        */
        const val DEVICE_ROTATION: Int = Surface.ROTATION_270
    }

    private lateinit var mBinding: ActivityMainBinding

    private lateinit var mDetector: Detector
    private var mDetectionProcessor: DetectionProcessor? = null

    private lateinit var mSourceBitmap: Bitmap
    private lateinit var mCropBitmap: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        mBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(mBinding.root)

        mSourceBitmap = assets.open("kite.jpg").use { inputStream ->
            BitmapFactory.decodeStream(inputStream)
        }

        mCropBitmap = processBitmap(mSourceBitmap, DETECTION_MODEL.inputSize)
        mBinding.imageView.setImageBitmap(mCropBitmap)

        setUpDetector()
        lifecycleScope.launch(Dispatchers.Main) {
            setUpDetectionProcessor()
        }

        setUpListeners()
    }


    private fun setUpDetector() {
        try {
            mDetector = DetectorFactory.createDetector(assets, DETECTION_MODEL, MINIMUM_SCORE)
        } catch (e: IOException) {
            Log.e(TAG, "Exception initializing classifier!")
            Log.e(TAG, e.stackTraceToString())
            val toast: Toast = Toast.makeText(
                baseContext, "Classifier could not be initialized", Toast.LENGTH_SHORT
            )
            toast.show()
            finish()
        }
    }

    private suspend fun setUpDetectionProcessor() {
        while (mBinding.imageView.width == 0) {
            delay(200)
        }

        mDetectionProcessor = DetectionProcessor(
            context = baseContext,
            detector = mDetector,
            trackingOverlay = mBinding.trackingOverlay,
        )

        mDetectionProcessor!!.initializeTrackingLayout(
            mBinding.imageView.width,
            mBinding.imageView.height,
            mDetector.getDetectionModel().inputSize,
            DEVICE_ROTATION
        )
    }

    private fun setUpListeners() {
        mBinding.cameraButton.setOnClickListener {
            val intent = Intent(applicationContext, DetectorActivity::class.java)
            startActivity(intent)
        }

        mBinding.detectButton.setOnClickListener {
            lifecycleScope.launch(Dispatchers.Default) {
                mDetectionProcessor?.processImage(mCropBitmap)
            }
        }
    }

    private fun processBitmap(source: Bitmap, size: Int): Bitmap {
        val croppedBitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val frameToCropTransformations =
            ImageUtils.getTransformationMatrix(source.width, source.height, size, size, 0)

        val cropToFrameTransformations = Matrix()
        frameToCropTransformations.invert(cropToFrameTransformations)

        val canvas = Canvas(croppedBitmap)
        canvas.drawBitmap(source, frameToCropTransformations, null)

        return croppedBitmap
    }
}