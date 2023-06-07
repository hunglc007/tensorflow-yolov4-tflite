package org.tensorflow.lite.examples.detector.visualization

import android.graphics.*
import android.graphics.Paint.Cap
import android.graphics.Paint.Join
import android.util.DisplayMetrics
import android.util.Log
import android.util.Pair
import android.util.TypedValue
import org.tensorflow.lite.examples.detector.Detector.Detection
import org.tensorflow.lite.examples.detector.utils.ImageUtils.getTransformationMatrix
import java.util.concurrent.locks.Lock
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.math.min

/**
 * A tracker that draws detection bounding boxes and scores (optional) on [Canvas].
 */
class MultiBoxTracker(
    displayMetrics: DisplayMetrics,
    private val frameWidth: Int,
    private val frameHeight: Int,
    private val orientation: Int,
    private val showScore: Boolean = true
) {

    private companion object {
        const val TAG: String = "MultiBoxTracker"
        const val TEXT_SIZE_DIP: Float = 18f
        const val MIN_SIZE: Float = 16.0f
        val COLORS: IntArray = intArrayOf(
            Color.BLUE,
            Color.RED,
            Color.GREEN,
            Color.YELLOW,
            Color.CYAN,
            Color.MAGENTA,
            Color.WHITE
        )
    }

    private val screenRectangles: MutableList<Pair<Float, RectF>> = mutableListOf()
    private val trackedDetections: MutableList<TrackedDetection> = mutableListOf()

    private var frameToCanvasMatrix: Matrix? = null

    private val borderedText: BorderedText = BorderedText(
        textSize = TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP,
            TEXT_SIZE_DIP,
            displayMetrics
        )
    )

    private val lock: Lock = ReentrantLock()

    fun trackResults(detections: List<Detection>) {
        screenRectangles.clear()

        val rgbFrameToScreen = Matrix(frameToCanvasMatrix)

        val detectionToTrack: List<Pair<Float, Detection>> = detections.mapNotNull { detection ->
            val detectionFrameRect: RectF = detection.boundingBox
            val detectionScreenRect = RectF()
            rgbFrameToScreen.mapRect(detectionScreenRect, detectionFrameRect)
            Log.d(
                TAG,
                "Result! Frame: ${detection.boundingBox} mapped to screen:$detectionScreenRect"
            )
            screenRectangles.add(Pair(detection.score, detectionScreenRect))
            if (detectionFrameRect.width() < MIN_SIZE || detectionFrameRect.height() < MIN_SIZE) {
                Log.w(TAG, "Degenerate rectangle : $detectionFrameRect")
                return@mapNotNull null
            }
            return@mapNotNull Pair(detection.score, detection)
        }

        if (detectionToTrack.isEmpty()) {
            Log.v(TAG, "Nothing to track.")
        }

        lock.withLock {
            trackedDetections.clear()
            for (rectangle in detectionToTrack) {
                val trackedRecognition = TrackedDetection()
                trackedRecognition.score = rectangle.first
                trackedRecognition.position = RectF(rectangle.second.boundingBox)
                trackedRecognition.title = rectangle.second.className
                trackedRecognition.boxPaint = Paint().also {
                    it.color = COLORS[trackedDetections.size % COLORS.size]
                    it.style = Paint.Style.STROKE
                    it.strokeWidth = 10.0f
                    it.strokeCap = Cap.ROUND
                    it.strokeJoin = Join.ROUND
                    it.strokeMiter = 100f
                }

                trackedDetections.add(trackedRecognition)
            }
        }
    }

    fun draw(canvas: Canvas) {
        val rotated = orientation % 180 != 0

        val targetWidth = if (rotated) frameHeight else frameWidth
        val targetHeight = if (rotated) frameWidth else frameHeight
        frameToCanvasMatrix = getTransformationMatrix(
            frameWidth,
            frameHeight,
            targetWidth,
            targetHeight,
            180
        )

        lock.withLock {
            for (trackedDetection in trackedDetections) {
                frameToCanvasMatrix!!.mapRect(trackedDetection.position)

                val cornerSize =
                    min(
                        trackedDetection.position.width(),
                        trackedDetection.position.height()
                    ) / 8.0f

                canvas.drawRoundRect(
                    trackedDetection.position,
                    cornerSize,
                    cornerSize,
                    trackedDetection.boxPaint
                )

                val labelString = if (showScore && trackedDetection.title.isNotBlank()) {
                    "%s %.2f%%".format(
                        trackedDetection.title,
                        100 * trackedDetection.score
                    )
                } else if (trackedDetection.title.isNotBlank()) {
                    trackedDetection.title
                } else if (showScore) {
                    "%.2f%%".format(100 * trackedDetection.score)
                } else ""

                borderedText.drawText(
                    canvas,
                    trackedDetection.position.left + cornerSize,
                    trackedDetection.position.top,
                    labelString,
                    trackedDetection.boxPaint
                )
            }
        }
    }

    private class TrackedDetection(
        var position: RectF = RectF(),
        var score: Float = 0f,
        var boxPaint: Paint = Paint(),
        var title: String = ""
    )

}