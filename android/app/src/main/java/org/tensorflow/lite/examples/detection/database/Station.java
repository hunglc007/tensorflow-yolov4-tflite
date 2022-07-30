package org.tensorflow.lite.examples.detection.database;

public class Station {
    private int id;

    private String name;

    private float tracksLat;

    private float tracksLong;

    public Station(int id, String name, float tracksLat, float tracksLong) {
        this.id = id;
        this.name = name;
        this.tracksLat = tracksLat;
        this.tracksLong = tracksLong;
    }

    public int getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public float getTracksLat() {
        return tracksLat;
    }

    public float getTracksLong() {
        return tracksLong;
    }
}
