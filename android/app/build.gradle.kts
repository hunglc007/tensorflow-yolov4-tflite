plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {

    defaultConfig {
        applicationId = "org.tensorflow.lite.examples"
        minSdk = 23
        compileSdk = 35
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android.txt"), "proguard-rules.pro")
        }
    }

    buildFeatures {
        viewBinding = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_21
        targetCompatibility = JavaVersion.VERSION_21
    }
    lint {
        abortOnError = false
    }
    namespace = "org.tensorflow.lite.examples.detector"

}

dependencies {
    implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("*.jar"))))
    implementation(project(":detector"))

    implementation(libs.kotlinx.coroutines.android)

    implementation(libs.androidx.camera2)
    implementation(libs.androidx.cameraLifecycle)
    implementation(libs.androidx.cameraView)

    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.coordinatorLayout)
    implementation(libs.androidx.constraintLayout)
    implementation(libs.androidx.lifecycleRuntimeKtx)
    implementation(libs.androidx.lifecycleViewModelKtx)
    implementation(libs.androidx.activityKtx)

    implementation(libs.android.material)
    implementation(libs.google.gson)

    // Uncomment if you want to enable memory leak detection
//    implementation(libs.squareup.leakCanary)

    androidTestImplementation(libs.androidx.test.extJunit)
    androidTestImplementation(libs.androidx.test.extJunitKtx)
    androidTestImplementation(libs.google.truth)

}
