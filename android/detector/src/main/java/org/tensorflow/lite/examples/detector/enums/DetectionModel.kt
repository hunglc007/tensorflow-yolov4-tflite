package org.tensorflow.lite.examples.detector.enums

/**
 * Enum which describes tflite models used by Detector.
 */
enum class DetectionModel(
    val modelFilename: String,
    val labelFilePath: String,
    val inputSize: Int,
    val outputSize: Int,
    val isQuantized: Boolean
) {
    YOLO_V4_416_FP32(
        "yolov4-416-fp32.tflite",
        "file:///android_asset/coco.txt",
        416,
        2535,
        false
    )
}