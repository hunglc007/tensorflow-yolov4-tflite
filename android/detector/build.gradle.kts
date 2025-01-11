plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
}

android {
    defaultConfig {
        minSdk = 23
        compileSdk = 35

        consumerProguardFiles("consumer-rules.pro")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_21
        targetCompatibility = JavaVersion.VERSION_21
    }

    kotlinOptions {
        jvmTarget = "21"
    }

    namespace = "org.tensorflow.lite.examples.detector"
}

dependencies {
    implementation(libs.androidx.coreKtx)

    implementation(libs.google.litert)
    implementation(libs.google.litertGpu)
    implementation(libs.google.litertSupport)
}