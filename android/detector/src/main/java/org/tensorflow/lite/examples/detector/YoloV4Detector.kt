package org.tensorflow.lite.examples.detector

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.detector.Detector.Detection
import org.tensorflow.lite.examples.detector.enums.DetectionModel
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.*
import kotlin.math.max
import kotlin.math.min


@Suppress("UNNECESSARY_NOT_NULL_ASSERTION")
internal class YoloV4Detector(
    assetManager: AssetManager,
    private val detectionModel: DetectionModel,
    private val minimumScore: Float,
) : Detector {

    private companion object {
        const val TAG = "YoloV4Detector"
        const val NUM_THREADS = 4
        const val IS_GPU: Boolean = false
        const val IS_NNAPI: Boolean = false

    }

    private val inputSize: Int = detectionModel.inputSize

    // Config values.
    private val labels: List<String>
    private val interpreter: Interpreter
    private val nmsThresh = 0.6f

    // Pre-allocated buffers.
    private val intValues = IntArray(inputSize * inputSize)
    private val byteBuffer: Array<ByteBuffer>
    private val outputMap: MutableMap<Int, Array<Array<FloatArray>>> = HashMap()

    init {
        val labelsFilename = detectionModel.labelFilePath
            .split("file:///android_asset/")
            .toTypedArray()[1]

        labels = assetManager.open(labelsFilename)
            .use { it.readBytes() }
            .decodeToString()
            .trim()
            .split("\n")
            .map { it.trim() }

        interpreter = initializeInterpreter(assetManager)

        val numBytesPerChannel = if (detectionModel.isQuantized) {
            1 // Quantized (int8)
        } else {
            4 // Floating point (fp32)
        }

        // input size * input size * pixel count (RGB) * pixel size (int8/fp32)
        byteBuffer = arrayOf(
            ByteBuffer.allocateDirect(inputSize * inputSize * 3 * numBytesPerChannel)
        )
        byteBuffer[0].order(ByteOrder.nativeOrder())

        outputMap[0] = arrayOf(Array(detectionModel.outputSize) { FloatArray(numBytesPerChannel) })
        outputMap[1] = arrayOf(Array(detectionModel.outputSize) { FloatArray(labels.size) })
    }

    override fun getDetectionModel(): DetectionModel {
        return detectionModel
    }

    override fun runDetection(bitmap: Bitmap): List<Detection> {
        convertBitmapToByteBuffer(bitmap)
        val results = getDetections(bitmap.width, bitmap.height)

        return nms(results)
    }

    private fun initializeInterpreter(assetManager: AssetManager): Interpreter {
        val options = Interpreter.Options()
        options.setNumThreads(NUM_THREADS)

        when {
            IS_GPU -> {
                options.addDelegate(GpuDelegate())
            }
            IS_NNAPI -> {
                options.setUseNNAPI(true)
            }
            else -> {
                options.setUseXNNPACK(true)
            }
        }

        return assetManager.openFd(detectionModel.modelFilename).use { fileDescriptor ->
            val fileInputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileByteBuffer = fileInputStream.channel.map(
                FileChannel.MapMode.READ_ONLY,
                fileDescriptor.startOffset,
                fileDescriptor.declaredLength
            )

            return@use Interpreter(fileByteBuffer, options)
        }
    }

    /**
     * Writes Image data into a [ByteBuffer].
     */
    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        val startTime = SystemClock.uptimeMillis()
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        scaledBitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)
        scaledBitmap.recycle()

        byteBuffer[0].clear()
        for (pixel in intValues) {
            val r = (pixel and 0xFF) / 255.0f
            val g = (pixel shr 8 and 0xFF) / 255.0f
            val b = (pixel shr 16 and 0xFF) / 255.0f

            byteBuffer[0].putFloat(r)
            byteBuffer[0].putFloat(g)
            byteBuffer[0].putFloat(b)
        }
        Log.v(TAG, "ByteBuffer conversion time : ${SystemClock.uptimeMillis() - startTime} ms")
    }

    private fun getDetections(imageWidth: Int, imageHeight: Int): List<Detection> {
        interpreter.runForMultipleInputsOutputs(byteBuffer, outputMap as Map<Int, Any>)

        val boundingBoxes = outputMap[0]!![0]
        val outScore = outputMap[1]!![0]

        return outScore.zip(boundingBoxes)
            .mapIndexedNotNull { index, (classScores, boundingBoxes) ->
                val bestClassIndex: Int = labels.indices.maxByOrNull { classScores[it] }!!
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