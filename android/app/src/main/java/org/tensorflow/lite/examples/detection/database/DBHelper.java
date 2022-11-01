package org.tensorflow.lite.examples.detection.database;

import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

public class DBHelper extends SQLiteOpenHelper {

    static final String DATABASE_NAME = "second_eyes.db";

    // DBHelper 생성자
    public DBHelper(Context context, int version) {
        super(context, DATABASE_NAME, null, version);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        db.execSQL("CREATE TABLE Station(ID INTEGER PRIMARY KEY AUTOINCREMENT, NAME TEXT, TRACKS_LAT REAL, TRACKS_LONG REAL)");
        db.execSQL("CREATE TABLE Exit(ID INTEGER PRIMARY KEY AUTOINCREMENT, NUM INTEGER, LATITUDE REAL, LONGITUDE REAL, STATION_ID INTEGER, FOREIGN KEY(STATION_ID) REFERENCES STATION(ID))");
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int i, int i1) {
        db.execSQL("DROP TABLE IF EXISTS Exit");
        db.execSQL("DROP TABLE IF EXISTS Station");
        onCreate(db);
    }


    public Station getStationByName(String name){
        SQLiteDatabase db = this.getReadableDatabase();
        Cursor cursor = db.rawQuery("select * from Station where name = ?", new String [] {name});
        if(cursor.getCount() != 0){
            System.out.println("지하철역을 잘못 읽어왔습니다.");
        }

        cursor.moveToFirst();
        int id = cursor.getInt(0);
        String stationName = cursor.getString(1);
        float trackLatitude = cursor.getFloat(2);
        float trackLongitude = cursor.getFloat(3);

        cursor.close();

        return new Station(id, stationName, trackLatitude, trackLongitude);
    }


    public Exit getExitByNum(int stationId, int number){
        SQLiteDatabase db = this.getReadableDatabase();
        Cursor cursor = db.rawQuery("select * from Exit where station_id = ? and num = ?", new String[]{String.valueOf(stationId), String.valueOf(number)});
        if(cursor.getCount() != 0){
            System.out.println("지하철역을 잘못 읽어왔습니다.");
        }

        cursor.moveToFirst();
        int id = cursor.getInt(0);
        int num = cursor.getInt(1);
        float latitude = cursor.getFloat(2);
        float longitude = cursor.getFloat(3);

        cursor.close();

        return new Exit(id, num, latitude, longitude);
    }


}
