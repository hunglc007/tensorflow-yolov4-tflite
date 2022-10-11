package org.tensorflow.lite.examples;
import com.odsay.odsayandroidsdk.API;
import com.odsay.odsayandroidsdk.ODsayData;
import com.odsay.odsayandroidsdk.ODsayService;
import com.odsay.odsayandroidsdk.OnResultCallbackListener;

@Override
protected void onCreate(Bundle savedInstanceState) {
        // 싱글톤 생성, Key 값을 활용하여 객체 생성
        ODsayService odsayService = ODsayService.init(Context, {발급받은 키값});
        // 서버 연결 제한 시간(단위(초), default : 5초)
        odsayService.setReadTimeout(5000);
        // 데이터 획득 제한 시간(단위(초), default : 5초)
        odsayService.setConnectionTimeout(5000);

        // 콜백 함수 구현
        OnResultCallbackListener onResultCallbackListener = new OnResultCallbackListener() {
// 호출 성공 시 실행
@Override
public void onSuccess(ODsayData odsayData, API api) {
        try {
        // API Value 는 API 호출 메소드 명을 따라갑니다.
        if (api == API.BUS_STATION_INFO) {
        String stationName = odsayData.getJson().getJSONObject("result").getString("stationName");
        Log.d(“Station name : %s”, stationName);
        }
        }catch (JSONException e) {
        e.printStackTrace();
        }
        }
// 호출 실패 시 실행
@Override
public void onError(int i, String s, API api) {
        if (api == API.BUS_STATION_INFO) {}
        }
        };
        // API 호출
        odsayService.requestBusStationInfo(“107475”, onResultCallbackListener());
        }
