package org.tensorflow.lite.examples.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import androidx.core.graphics.scale
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.detector.Detector.Detection
import org.tensorflow.lite.examples.detector.enums.DetectionModel
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.util.PriorityQueue
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt


internal class YoloV4Detector(
    context: Context,
    private val detectionModel: DetectionModel,
    private val minimumScore: Float,
) : Detector {

    private companion object {
        const val TAG = "YoloV4Detector"
        const val NUM_THREADS = 4
        const val IS_GPU: Boolean = false
        const val IS_NNAPI: Boolean = false
        const val IS_XNNPACK: Boolean = false
    }

    private val inputSize: Int = detectionModel.inputSize

    // Config values.
    private val labels: List<String> = context.assets.open(
        // labels filename
        detectionModel.labelFilePath
            .split("file:///android_asset/")
            .toTypedArray()[1]
    )
        .use { it.readBytes() }
        .decodeToString()
        .trim()
        .split("\n")
        .map { it.trim() }
    private val interpreter: Interpreter =
        initializeInterpreter(FileUtil.loadMappedFile(context, detectionModel.modelFilename))
    private val nmsThresh = 0.6f

    // Pre-allocated buffers.
    private val intValues = IntArray(inputSize * inputSize)

    private val numBytesPerChannel = if (detectionModel.isQuantized) {
        1 // Quantized (int8)
    } else {
        4 // Floating point (fp32)
    }

    // input size * input size * pixel count (RGB) * pixel size (int8/fp32)
    private val input: Array<ByteBuffer> = arrayOf(
        ByteBuffer.allocateDirect(inputSize * inputSize * 3 * numBytesPerChannel)
            .order(ByteOrder.nativeOrder())
    )

    private val output: Map<Int, Array<Array<FloatArray>>> = mutableMapOf(
        0 to arrayOf(Array(detectionModel.outputSize) { FloatArray(numBytesPerChannel) }),
        1 to arrayOf(Array(detectionModel.outputSize) { FloatArray(labels.size) })
    )

    override fun getDetectionModel(): DetectionModel {
        return detectionModel
    }

    override fun runDetection(bitmap: Bitmap): List<Detection> {
        convertBitmapToByteBuffer(bitmap)
        val results = getDetections(bitmap.width, bitmap.height)

        return nms(results)
    }

    private fun initializeInterpreter(model: MappedByteBuffer): Interpreter {
        val options = Interpreter.Options()
        options.numThreads = NUM_THREADS
        options.setUseXNNPACK(false)

        when {
            IS_GPU -> {
                options.addDelegate(GpuDelegate())
            }

            IS_NNAPI -> {
                options.addDelegate(NnApiDelegate())
                options.useNNAPI = true
            }

            IS_XNNPACK -> {
                options.setUseXNNPACK(true)
            }
        }

        return Interpreter(model, options)
    }

    /**
     * Writes Image data into a [ByteBuffer].
     */
    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        val startTime = SystemClock.uptimeMillis()

        val fittedBitmap = scaleToFit(bitmap, inputSize, inputSize)
        val leftOffset = (inputSize - fittedBitmap.width) / 2F
        val topOffset = (inputSize - fittedBitmap.height) / 2F

        val scaledBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(scaledBitmap)
        canvas.drawBitmap(fittedBitmap, leftOffset, topOffset, null)

        scaledBitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)
        scaledBitmap.recycle()

        input[0].clear()
        for (pixel in intValues) {
            input[0].putFloat(Color.red(pixel) / 255.0f)
            input[0].putFloat(Color.green(pixel) / 255.0f)
            input[0].putFloat(Color.blue(pixel) / 255.0f)
        }
        Log.v(TAG, "ByteBuffer conversion time : ${SystemClock.uptimeMillis() - startTime} ms")
    }

    private fun scaleToFit(bitmap: Bitmap, preferredWidth: Int, preferredHeight: Int): Bitmap {
        val ratio: Float = bitmap.width.toFloat() / bitmap.height.toFloat()

        return if (ratio > 1) {
            val newHeight = (preferredWidth * ratio).roundToInt()
            bitmap.scale(preferredWidth, newHeight, false)
        } else {
            val newWidth = (preferredHeight * ratio).roundToInt()
            bitmap.scale(newWidth, preferredHeight, false)
        }
    }

    private fun getDetections(imageWidth: Int, imageHeight: Int): List<Detection> {
        interpreter.runForMultipleInputsOutputs(input, output)

        val boundingBoxes = output[0]!![0]
        val outScore = output[1]!![0]

        return outScore.zip(boundingBoxes)
            .mapIndexedNotNull { index, (classScores, boundingBoxes) ->
                val bestClassIndex: Int = labels.indices.maxBy { classScores[it] }
                val bestScore = classScores[bestClassIndex]

                if (bestScore <= minimumScore) {
                    return@mapIndexedNotNull null
                }

                val xPos = boundingBoxes[0]
                val yPos = boundingBoxes[1]
                val width = boundingBoxes[2]
                val height = boundingBoxes[3]
                val rectF = RectF(
                    max(0f, xPos - width / 2),
                    max(0f, yPos - height / 2),
                    min(imageWidth - 1.toFloat(), xPos + width / 2),
                    min(imageHeight - 1.toFloat(), yPos + height / 2)
                )

                return@mapIndexedNotNull Detection(
                    id = index.toString(),
                    className = labels[bestClassIndex],
                    detectedClass = bestClassIndex,
                    score = bestScore,
                    boundingBox = rectF
                )
            }
    }

    private fun nms(detections: List<Detection>): List<Detection> {
        val nmsList: MutableList<Detection> = mutableListOf()

        for (labelIndex in labels.indices) {
            val priorityQueue = PriorityQueue<Detection>(50)
            priorityQueue.addAll(detections.filter { it.detectedClass == labelIndex })

            while (priorityQueue.size > 0) {
                val previousPriorityQueue: List<Detection> = priorityQueue.toList()
                val max = previousPriorityQueue[0]
                nmsList.add(max)
                priorityQueue.clear()
                priorityQueue.addAll(previousPriorityQueue.filter {
                    boxIoU(max.boundingBox, it.boundingBox) < nmsThresh
                })
            }
        }

        return nmsList
    }

    private fun boxIoU(a: RectF, b: RectF): Float {
        return boxIntersection(a, b) / boxUnion(a, b)
    }

    private fun boxIntersection(a: RectF, b: RectF): Float {
        val w = overlap(
            (a.left + a.right) / 2,
            a.right - a.left,
            (b.left + b.right) / 2,
            b.right - b.left
        )

        val h = overlap(
            (a.top + a.bottom) / 2,
            a.bottom - a.top,
            (b.top + b.bottom) / 2,
            b.bottom - b.top
        )

        return if (w < 0F || h < 0F) 0F else w * h
    }

    private fun boxUnion(a: RectF, b: RectF): Float {
        val i = boxIntersection(a, b)
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i
    }

    private fun overlap(x1: Float, width1: Float, x2: Float, width2: Float): Float {
        val left1 = x1 - width1 / 2
        val left2 = x2 - width2 / 2
        val left = max(left1, left2)

        val right1 = x1 + width1 / 2
        val right2 = x2 + width2 / 2
        val right = min(right1, right2)

        return right - left
    }

}