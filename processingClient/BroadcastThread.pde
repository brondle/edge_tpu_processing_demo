import javax.imageio.*;
import java.awt.image.*;
import java.net.*;
import java.io.*;

class BroadcastThread extends Thread {
  // This are the ports we are sending images to
  int clientDetectionPort = 9100;
  int clientClassificationPort = 9101;

  String clientHost = "127.0.0.1";
  String broadcastHost = getRemoteBroadcastHost();
  int broadcastPort = getRemoteBroadcastPort();

  // This is our object that sends UDP out
  DatagramSocket ds;
  PImage lastImage;
  boolean newFrame = false;
  boolean running;

  float[] cropBox;

  BroadcastThread() {
    //println("Host and port:", host, port);
    // Setting up the DatagramSocket, requires try/catch
    try {
      ds = new DatagramSocket();
    } catch (SocketException e) {
      e.printStackTrace();
    }
  }

  void start() {
    running = true;
    super.start();
  }

  // we must implement run, this gets triggered by start()
  void run() {
    while (running) {
      if (newFrame) {
        broadcastFullImage(lastImage);
      }

      if (cropBox != null) {
        broadcastImageCrop(lastImage);
      }
    }
  }

  void update(PImage img) {
    lastImage = img;
    newFrame = true;
  }

  void disableCropToBroadcast() {
    cropBox = null;
  }

  void setCropToBroadcast(float[] box) {
    cropBox = box;
  }

  // Function to broadcast a PImage over UDP
  // Special thanks to: http://ubaa.net/shared/processing/udp/
  // (This example doesn't use the library, but you can!)
  void broadcastFullImage(PImage img) {
    // We need a buffered image to do the JPG encoding
    BufferedImage bimg = new BufferedImage(img.width,img.height, BufferedImage.TYPE_INT_RGB);

    // Transfer pixels from localFrame to the BufferedImage
    img.loadPixels();
    bimg.setRGB(0, 0, img.width, img.height, img.pixels, 0, img.width);

    // Need these output streams to get image as bytes for UDP communication
    ByteArrayOutputStream baStream = new ByteArrayOutputStream();
    BufferedOutputStream bos = new BufferedOutputStream(baStream);

    // Turn the BufferedImage into a JPG and put it in the BufferedOutputStream
    // Requires try/catch
    try {
      ImageIO.write(bimg, "jpg", bos);
    } catch (IOException e) {
      e.printStackTrace();
    }

    // Get the byte array, which we will send out via UDP!
    byte[] packet = baStream.toByteArray();

    // Send JPEG data as a datagram
    // println("Sending datagram with " + packet.length + " bytes");
    try {
      ds.send(new DatagramPacket(packet,packet.length, InetAddress.getByName(clientHost), clientDetectionPort));
      if (broadcastHost != "") {
        ds.send(new DatagramPacket(packet,packet.length, InetAddress.getByName(broadcastHost), broadcastPort));
      }
    } catch (Exception e) {
      e.printStackTrace();
    }

  }

  void broadcastImageCrop(PImage img) {
    // image classification model has a fixed size
    int IMG_SIZE = 224;

    BufferedImage bimg = new BufferedImage(IMG_SIZE, IMG_SIZE, BufferedImage.TYPE_INT_RGB);
    img.loadPixels();

    int x = Math.round(cropBox[0]);
    int y = Math.round(cropBox[1]);
    int w = Math.round(cropBox[2]) - x;
    int h = Math.round(cropBox[3]) - y;

    PImage croppedImg = img.get(x, y, w, h);
    croppedImg.loadPixels();
    croppedImg.resize(IMG_SIZE, IMG_SIZE);

    bimg.setRGB(0, 0, IMG_SIZE, IMG_SIZE, croppedImg.pixels, 0, IMG_SIZE);
    byte[] packet = createImageBytePacket(bimg);

    try {
      ds.send(new DatagramPacket(packet,packet.length, InetAddress.getByName(clientHost), clientClassificationPort));
      if (broadcastHost != "") {
        ds.send(new DatagramPacket(packet,packet.length, InetAddress.getByName(broadcastHost), broadcastPort));
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}

byte[] createImageBytePacket(BufferedImage bimg) {
    // Need these output streams to get image as bytes for UDP communication
    ByteArrayOutputStream baStream = new ByteArrayOutputStream();
    BufferedOutputStream bos = new BufferedOutputStream(baStream);

    // Turn the BufferedImage into a JPG and put it in the BufferedOutputStream
    // Requires try/catch
    try {
      ImageIO.write(bimg, "jpg", bos);
    } catch (IOException e) {
      e.printStackTrace();
    }

    // Get the byte array, which we will send out via UDP!
    byte[] packet = baStream.toByteArray();

    return packet;
}

String getRemoteBroadcastHost() {
  return getEnvValueOrDefault("BROADCAST_HOST", "");
}

int getRemoteBroadcastPort() {
  return getEnvValueOrDefault("BROADCAST_PORT", 0);
}
