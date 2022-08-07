package org.tensorflow.lite.examples.detection.database;

public class Exit {

    private int id;

    private Integer num;

    private float latitude;

    private float longitude;

    public Exit(int id, Integer num, float latitude, float longitude) {
        this.id = id;
        this.num = num;
        this.latitude = latitude;
        this.longitude = longitude;
    }

    public int getId() {
        return id;
    }

    public Integer getNum() {
        return num;
    }

    public float getLatitude() {
        return latitude;
    }

    public float getLongitude() {
        return longitude;
    }
}
