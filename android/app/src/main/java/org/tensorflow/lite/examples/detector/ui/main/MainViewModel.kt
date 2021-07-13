package org.tensorflow.lite.examples.detector.ui.main

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Matrix
import android.util.DisplayMetrics
import android.view.Surface
import android.widget.ImageView
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.tensorflow.lite.examples.detector.misc.Constants
import org.tensorflow.lite.examples.detector.Detector
import org.tensorflow.lite.examples.detector.DetectorFactory
import org.tensorflow.lite.examples.detector.misc.DetectionProcessor
import org.tensorflow.lite.examples.detector.utils.ImageUtils
import org.tensorflow.lite.examples.detector.visualization.TrackingOverlayView

class MainViewModel : ViewModel() {

    companion object {
        const val DEVICE_ROTATION: Int = Surface.ROTATION_0
    }

    lateinit var bitmap: Bitmap
        private set

    private lateinit var mDetector: Detector
    private var mDetectionProcessor: DetectionProcessor? = null

    private lateinit var mSourceBitmap: Bitmap

    fun setUpBitmaps(assetManager: AssetManager) {
        mSourceBitmap = assetManager.open("kite.jpg").use { inputStream ->
            BitmapFactory.decodeStream(inputStream)
        }

        bitmap = processBitmap(mSourceBitmap, Constants.DETECTION_MODEL.inputSize)
    }

    fun setUpDetector(assetManager: AssetManager) {
        mDetector = DetectorFactory.createDetector(
            assetManager,
            Constants.DETECTION_MODEL,
            Constants.MINIMUM_SCORE
        )
    }

    fun setUpDetectionProcessor(
        imageView: ImageView,
        trackingOverlayView: TrackingOverlayView,
        displayMetrics: DisplayMetrics
    ) = viewModelScope.launch {
        while (imageView.width == 0) {
            delay(200)
        }

        mDetectionProcessor = DetectionProcessor(
            displayMetrics = displayMetrics,
            detector = mDetector,
            trackingOverlay = trackingOverlayView,
        )

        mDetectionProcessor!!.initializeTrackingLayout(
            imageView.width,
            imageView.height,
            mDetector.getDetectionModel().inputSize,
            DEVICE_ROTATION
        )
    }


    fun processImage() = viewModelScope.launch(Dispatchers.Default) {
        mDetectionProcessor?.processImage(bitmap)
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