package org.tensorflow.lite.examples.detector.helpers

import android.content.Context
import android.graphics.*
import android.util.DisplayMetrics
import android.util.Log
import org.tensorflow.lite.examples.detector.Detector
import org.tensorflow.lite.examples.detector.utils.ImageUtils
import org.tensorflow.lite.examples.detector.visualization.MultiBoxTracker
import org.tensorflow.lite.examples.detector.visualization.TrackingOverlayView
import kotlin.system.measureTimeMillis

class DetectionProcessor(
    context: Context,
    private var detector: Detector,
    private var trackingOverlay: TrackingOverlayView
) {
    private companion object {
        const val TAG: String = "DetectionProcessor"

        const val SHOW_SCORE: Boolean = true
    }

    private val mDisplayMetrics: DisplayMetrics = context.resources.displayMetrics

    private lateinit var mTracker: MultiBoxTracker
    private var mCroppedBitmap: Bitmap? = null
    private var mCropToFrameTransform: Matrix? = null

    private val mPaint: Paint = Paint().also {
        it.color = Color.RED
        it.style = Paint.Style.STROKE
        it.strokeWidth = 2.0f
    }

    fun initializeTrackingLayout(
        previewWidth: Int,
        previewHeight: Int,
        cropSize: Int,
        rotation: Int
    ) {
        Log.i(TAG, "Camera orientation relative to screen canvas : $rotation")
        Log.i(TAG, "Initializing with size ${previewWidth}x${previewHeight}")

        mCroppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)

        mCropToFrameTransform = ImageUtils.getTransformationMatrix(
            srcWidth = cropSize,
            srcHeight = cropSize,
            dstWidth = previewWidth,
            dstHeight = previewHeight,
            rotation = ((rotation + 3) % 4) * 90
        )

        mTracker = MultiBoxTracker(
            mDisplayMetrics,
            previewWidth,
            previewHeight,
            ((rotation + 1) % 4) * 90,
            mShowScore = SHOW_SCORE
        )
        trackingOverlay.setTracker(mTracker)
    }

    fun processImage(bitmap: Bitmap): Long {
        Log.v(TAG, "Running detection on image")
        val detections: List<Detector.Detection>
        val detectionTime: Long = measureTimeMillis {
            detections = detector.runDetection(bitmap)
        }

        Log.v(TAG, "Recognized objects : ${detections.size}")
        val cropCopyBitmap: Bitmap = Bitmap.createBitmap(mCroppedBitmap!!)
        val canvas = Canvas(cropCopyBitmap)

        for (detection in detections) {
            val boundingBox: RectF = detection.boundingBox
            canvas.drawRect(boundingBox, mPaint)
            mCropToFrameTransform!!.mapRect(boundingBox)
        }

        mTracker.trackResults(detections)
        trackingOverlay.postInvalidate()

        return detectionTime
    }

}