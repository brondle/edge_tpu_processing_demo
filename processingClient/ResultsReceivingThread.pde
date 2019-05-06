import java.nio.charset.*;
import processing.net.*;

static final int MAX_DETECTED_OBJECTS = 20;

class ResultsReceivingThread extends Thread {
  // This is the port we are receiving detection results on
  int port = 9102;

  byte[] buffer = new byte[65536];
  Client client;

  boolean running = false;
  boolean available = false;

  // unlocked -> unlocking with approved face -> unlocked
  String lockState = "locked";

  float[][] boxes = new float[MAX_DETECTED_OBJECTS][4];
  String[] labels = new String[MAX_DETECTED_OBJECTS];
  int numDetections = 0;
  String faceClassification;

  ResultsReceivingThread(PApplet parent) {
    // create new tcp client
    client = new Client(parent, "127.0.0.1", port);
  }

  void start(){
    running = true;
    super.start();
  }

  void run() {
    while (running) {
      checkForNewAndUpdateResults();
    }
  }

  // call once to see if new result is available.
  // after it's called, it sets its value to false.
  boolean newResultsAvailable() {
    boolean currentAvailable = available;
    available = false;
    return currentAvailable;
  }

  byte[] receiveBuffer = new byte[65536];

  void parseResults(JSONObject resultsJson) {
    JSONArray detections = resultsJson.getJSONArray("detection");
    JSONArray classification = resultsJson.getJSONArray("classification");

    numDetections = 0;

    if (detections != null) {
      for (int i = 0; i < detections.size() && i < MAX_DETECTED_OBJECTS; i++) {
        JSONObject result = detections.getJSONObject(i);

        labels[i] = result.getString("label");

        JSONArray box = result.getJSONArray("box");

        boxes[i][0] = box.getFloat(0);
        boxes[i][1] = box.getFloat(1);
        boxes[i][2] = box.getFloat(2);
        boxes[i][3] = box.getFloat(3);

        numDetections++;
      }
    }

    if (classification != null) {
      faceClassification = classification.getString(0);
      println("got classification! face recognized as: " + faceClassification);
    }
  }

  void checkForNewAndUpdateResults(){
    if (client.available() > 0) {
      String base64String = client.readString();

      try {
        JSONObject obj = parseJSONObject(base64String);

        parseResults(obj);

        available = true;
      } catch (RuntimeException e) {
        println("bad json: ", base64String);
        e.printStackTrace();
      }
    }
  }

  int getNumDetections() {
    available = false;
    return numDetections;
  }

  float[][] getBoxes() {
    return boxes;
  }

  String[] getLabels() {
    return labels;
  }

  String getClassification() {
    return faceClassification;
  }
}
