// MicroDuck OpenRB-150 dual-host bridge: wired MacBook control with the Pi as the IMU relay.
//
// Hosts: USB (MacBook) and Serial2 (head board UART on J12 at PI_BAUD; pad 3 RX <- board TXD, pad 4 TX -> board RXD;
// A7Z UART2 pins 7 TX / 11 RX, Pi Zero pins 8 / 10).
// The DYNAMIXEL bus (Serial1) is driven by whichever host sent the last instruction packet; status packets go
// back to that host. Either host can therefore run the servos alone (the Pi bridge keeps working when no Mac is
// plugged in).
// IMU frames from the Pi (0xA5 0x5A, length byte, payload, 8-bit sum) are forwarded whole to USB, appended to the
// status batch of the Mac's next read: one USB write per control cycle. Only the newest pending frame is kept.
// A USB write can stall the loop until the host takes the data, and an instruction arriving meanwhile waits: IMU
// frames written on their own (185/s from the Pi) stretched the Mac's 14-servo read from 4.3 ms to 13.3 ms median
// and pushed 8 % of reads past its 18 ms timeout (measured 2026-10-15). Written immediately they also overflowed the
// Serial1 receive buffer (about half of all replies lost). Without a USB read for IMU_ALONE_US the frame goes out
// alone, so a host that only listens still gets the IMU.
// Every stream is parsed into whole packets before forwarding, so DYNAMIXEL status packets and IMU frames never
// interleave on the USB link. A partial packet is dropped after 3 ms of silence (resync).
// Rescue path for a servo that fell back to factory settings (57600 baud, 2026-11-13): a packet addressed to ID 253
// (no servo can have it) is not forwarded. Its parameters are a 4-byte little-endian baud rate followed by raw bytes;
// the bus is switched to that baud, the raw bytes are written, and the bus returns to BAUD. Replies at the other baud
// are not relayed; the host checks the result at BAUD afterwards (pi/rescue_factory_servo.py).
// Based on ROBOTIS usb_to_dynamixel.ino (Apache-2.0).

#include <Dynamixel2Arduino.h>

#define DXL_BUS Serial1
#define PI_HOST Serial2
#define USB_HOST Serial

const uint32_t BAUD = 1000000;
// The A7Z UART runs from a 24 MHz clock (base 1.5 Mbaud), so 1 Mbaud is out of its range and arrives garbled;
// 500 kbaud divides exactly on the A7Z, the Pi and the SAMD21, and carries the 53-byte IMU frames at 200/s easily.
const uint32_t PI_BAUD = 500000;
const int MAX_PKT = 512;

struct Parser {
  uint8_t buf[MAX_PKT];
  int len = 0;
  unsigned long lastByte = 0;
};

Dynamixel2Arduino dxl(DXL_BUS);
Parser fromUsb, fromPi, fromBus;
Stream *owner = &PI_HOST;  // host that receives DYNAMIXEL status packets
unsigned long ledOffAt = 0;
unsigned long lastInstr = 0, lastBusByte = 0, lastUsbFlush = 0;  // micros()
uint8_t imuPending[64];
int imuLen = 0;
const unsigned long QUIET_US = 800;
const unsigned long IMU_ALONE_US = 100000;  // no USB read for 100 ms: forward IMU frames on their own
// Status packets for the USB host are batched and written once the bus goes quiet: one USB write per sync read
// instead of one per servo (each CDC write can wait for a USB frame, which stretched a 14-servo read past 8 ms).
uint8_t usbOut[1024];
int usbOutLen = 0;
const unsigned long FLUSH_US = 150;

// Returns total length of the packet at p->buf if it is complete, 0 if more bytes are needed, -1 if the
// first byte cannot start a packet (caller drops one byte to resync).
int complete(Parser *p, bool allowImu) {
  if (p->len < 1) return 0;
  uint8_t b0 = p->buf[0];
  if (b0 == 0xFF) {
    if (p->len < 7) {
      const uint8_t hdr[4] = {0xFF, 0xFF, 0xFD, 0x00};
      for (int i = 1; i < p->len && i < 4; i++) if (p->buf[i] != hdr[i]) return -1;
      return 0;
    }
    if (p->buf[1] != 0xFF || p->buf[2] != 0xFD || p->buf[3] != 0x00) return -1;
    int total = 7 + (p->buf[5] | (p->buf[6] << 8));
    if (total > MAX_PKT) return -1;
    return p->len >= total ? total : 0;
  }
  if (allowImu && b0 == 0xA5) {
    if (p->len < 3) return (p->len < 2 || p->buf[1] == 0x5A) ? 0 : -1;
    if (p->buf[1] != 0x5A) return -1;
    int total = 3 + p->buf[2] + 1;
    return p->len >= total ? total : 0;
  }
  return -1;
}

void consume(Parser *p, int n) {
  memmove(p->buf, p->buf + n, p->len - n);
  p->len -= n;
}

// Pull bytes from src, emit whole packets via handler.
void pump(Stream &src, Parser *p, bool allowImu, void (*handler)(uint8_t *, int)) {
  int avail = src.available();
  unsigned long now = micros();
  if (avail <= 0) {
    if (p->len > 0 && now - p->lastByte > 3000) p->len = 0;
    return;
  }
  while (avail-- > 0 && p->len < MAX_PKT) p->buf[p->len++] = src.read();
  p->lastByte = now;
  if (p == &fromBus) lastBusByte = now;
  while (true) {
    int r = complete(p, allowImu);
    if (r == 0) break;
    if (r < 0) { consume(p, 1); continue; }
    handler(p->buf, r);
    consume(p, r);
  }
  if (p->len >= MAX_PKT) p->len = 0;
}

void blink() {
  digitalWrite(LED_BUILTIN, HIGH);
  ledOffAt = millis() + 20;
}

const uint8_t RESCUE_ID = 253;

// True when the packet was a rescue request and has been handled here.
bool rescue(uint8_t *b, int n) {
  if (b[4] != RESCUE_ID) return false;
  if (n < 15) return true;  // header 7 + instruction 1 + baud 4 + at least 1 raw byte + CRC 2
  uint32_t baud = (uint32_t)b[8] | ((uint32_t)b[9] << 8) | ((uint32_t)b[10] << 16) | ((uint32_t)b[11] << 24);
  // 4 Mbaud and over is out of the SAMD21's range at 16x oversampling: the write never completes and flush() hangs (2026-11-13)
  if (baud != 9600 && baud != 57600 && baud != 115200 && baud != 2000000 && baud != 3000000) return true;
  DXL_BUS.flush();
  dxl.begin(baud);
  DXL_BUS.write(b + 12, n - 14);
  DXL_BUS.flush();
  delay(20);  // the servo's EEPROM write and its reply at the other baud
  dxl.begin(BAUD);
  while (DXL_BUS.available()) DXL_BUS.read();
  fromBus.len = 0;
  blink();
  return true;
}

void usbPacket(uint8_t *b, int n) {
  if (b[0] == 0xFF && rescue(b, n)) return;
  if (b[0] == 0xFF) { owner = &USB_HOST; DXL_BUS.write(b, n); lastInstr = micros(); blink(); }
}

void piPacket(uint8_t *b, int n) {
  if (b[0] == 0xFF && rescue(b, n)) return;
  if (b[0] == 0xFF) { owner = &PI_HOST; DXL_BUS.write(b, n); lastInstr = micros(); blink(); }
  else if (n <= (int)sizeof(imuPending)) { memcpy(imuPending, b, n); imuLen = n; }  // sent when the bus is quiet
}

void busPacket(uint8_t *b, int n) {
  if (owner != &USB_HOST) { owner->write(b, n); return; }
  if (usbOutLen + n > (int)sizeof(usbOut)) { USB_HOST.write(usbOut, usbOutLen); usbOutLen = 0; }
  memcpy(usbOut + usbOutLen, b, n);
  usbOutLen += n;
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  USB_HOST.begin(BAUD);
  PI_HOST.begin(PI_BAUD);
  dxl.begin(BAUD);
}

void loop() {
  pump(USB_HOST, &fromUsb, false, usbPacket);
  pump(PI_HOST, &fromPi, true, piPacket);
  pump(DXL_BUS, &fromBus, false, busPacket);
  unsigned long now = micros();
  if (usbOutLen && now - lastBusByte > FLUSH_US && DXL_BUS.available() == 0) {
    if (imuLen && usbOutLen + imuLen <= (int)sizeof(usbOut)) {  // the newest IMU frame rides with the replies
      memcpy(usbOut + usbOutLen, imuPending, imuLen);
      usbOutLen += imuLen;
      imuLen = 0;
    }
    USB_HOST.write(usbOut, usbOutLen);
    usbOutLen = 0;
    lastUsbFlush = now;
  }
  if (imuLen && usbOutLen == 0 && now - lastUsbFlush > IMU_ALONE_US && now - lastBusByte > QUIET_US && fromBus.len == 0 &&
      DXL_BUS.available() == 0) {
    if (USB_HOST) USB_HOST.write(imuPending, imuLen);  // only meaningful with a Mac attached
    imuLen = 0;
  }
  if (ledOffAt && millis() > ledOffAt) {
    digitalWrite(LED_BUILTIN, LOW);
    ledOffAt = 0;
  }
}
