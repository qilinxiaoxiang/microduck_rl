# MicroDuck battery backpack: 2S LiPo plan (2026-10-15)

Status: plan; the DFR1202 sits in the battery backpack (Shawn, 2026-10-15; DECISIONS.md D-010). Parts are in the
Amazon cart, not paid; nothing is built or measured yet. Problems P5 (serviceable onboard
hardware) and P6 (first power and failure response). The bench backpack (layout F box, D-007) stays as it is.

## Summary

- **Battery:** Tattu 2S 650 mAh 75C with XT30, 57 x 31 x 12 mm, 43 g, two packs in rotation. It fits inside the
  layout F box outline. Expect about 20-25 min of walking per pack (15 min pessimistic) and about 30 min standing.
- **Servo branch:** battery -> switch -> OpenRB-150 J4, no converter. Every servo's Shutdown register goes from 53 to
  52 once, so the 8.4 V full charge does not cut torque.
- **A7Z branch:** the owned DFRobot DFR1202 (6-24 V in, 5.0 V out, 16 A) rides in the battery backpack after the
  switch and feeds the A7Z header pins 2 and 6. The bench backpack keeps its XL4015 #2 (5.0 V) for the A7Z.
- **Switch:** one XT30 inline switch between the battery and everything else, so one switch turns off the servos and
  the A7Z together.
- **Swap:** each backpack has two plugs: a female XT30 for the servo rail and a female MR30 for the A7Z's 5 V. The
  robot has a male XT30 inlet on J4 and a male MR30 on the A7Z lead. The two connector types cannot be crossed.

## 1. Power draw

Battery current is the servo bus current plus the 5 V branch divided by the buck efficiency (0.9). `budget.py`
computes it from `loads.json` and writes `budget.json`.

Servo bus at 7.4 V, simulation of the as-built V1.9 model running stand61/walk61
(`hardware-design-20260912/whole-robot-20260912/dynamics/power-2s-20261015/README.md`), mean / highest 1 s / highest
20 ms:

- stand61 steady: 0.43 / 0.45 / 0.46 A.
- walk61 vx 0.3: 0.71 / 0.87 / 2.14 A.
- Sit to stand: 0.50 / 1.33 / 6.59 A (1.75 A servo current limit).
- All 15 servos stalled at their current limit: about 25-29 A, briefly, in a fall or a jam.
- The model overstates holding current 3-10 times against the one hardware reading (0.165 A summed Present Current
  holding STAND at 5 V, 2026-10-13), so standing is probably nearer 0.3 A. Walking has no hardware reading yet; the
  pessimistic line below doubles it.

5 V branch (A7Z 2 GB, Radxa Camera 13M 214, BNO085, MAX98357A), from published measurements:

- A7Z idle 1.6-2.2 W (Jeff Geerling sbc-reviews issue 100, LinuxLinks), Geekbench multicore 4.3 W, all-core stress
  5.4-5.6 W; same SoC on the Cubie A7A reaches 9 W under stress with a heatsink.
- Camera module about 0.25-0.5 W plus ISP and encode on the SoC (estimate; no published figure for the IMX214).
- Wi-Fi module Quectel FCU760K: up to 353 mA at 3.3 V in transmit bursts.
- BNO085 about 14 mA at 3.3 V; amplifier 2.4 mA idle, about 0.1 A on a continuous tone at the -6 dB ceiling.
- Budget: about 4 W typical (0.8 A at 5 V), 7 W sustained heavy, 11 W short peak (2.2 A).

Battery current and continuous runtime (80 % of nameplate energy usable, 7.4 V):

- Idle, torque on, A7Z idle: 0.63 A. 2S 650: 50 min.
- Standing, camera on: 1.05 A. 2S 650: 30 min.
- Walking vx 0.3, typical compute: 1.31 A. 2S 650: 24 min; 2S 1000: 37 min; 2S 1300: 48 min.
- Walking with heavy compute: 1.76 A. 2S 650: 18 min.
- Walking, servo current doubled: 2.17 A. 2S 650: 14 min; 2S 1000: 22 min.
- Peak: about 8.3 A (sit to stand plus a 5 V spike), 13C for a 650 mAh pack; its 75C rating covers this and the stall
  case.

To replace the simulation with a measurement: log summed Present Current at 50 Hz while walking
(`guide/firmware/pi/current_watch.py`) and read the A7Z input with a USB or inline meter when it arrives (idle, policy
plus 1080p stream, `stress-ng --matrix 0`, an NPU model, and the lowest header voltage at boot).

## 2. Regulator for the A7Z

- **DFR1202 is the right part and is already owned.** 6-24 V input covers the whole 2S range (8.4 V full, 6.6 V at
  the software cutoff); 5.0 V +/-1 % output; 16 A is far above the 2.2 A peak; over 90 % efficient. Radxa asks for a
  5 V supply of 3 A or more.
- **Its cost is soldered pads.** The board has castellated edge pads only (`VIN`, `GND` shared by input and output,
  `VOUT`); D-007 dropped it for that reason. Section 5 lists the six wires it takes.
- **Placement: inside the battery backpack (Shawn, 2026-10-15).** It is used only on battery power; the bench
  backpack keeps XL4015 #2 at 5.0 V for the A7Z, so the robot carries no converter and the rear pack's converter
  channel stays empty. The cost is a second plug at every swap (section 5).
- **Not the 5991 or the kit XL4015.** The 5991 only negotiates USB-C PD and cannot take a battery; the XL4015 needs
  8 V or more in.
- **A7Z header power is supported** (Radxa docs: 5 V on pins 2 and 4). The pins tie straight to the board's 5 V net
  and back-feed the USB-C power port, so never plug a USB-C supply into the A7Z while it is header-powered, and keep
  the supply at or under 5.2 V (a 5.1 V zener sits on the USB side). The DFR1202's fixed 5.0 V and XL4015 #2 at
  5.00 V (4.95-5.10) meet both.

## 3. Battery

- **Pick: Tattu 2S 650 mAh 75C XT30**, Amazon B0727XZFDW, $9.16 each, sold by Gensace&Tattu Online and shipped by
  Amazon (same or next day to Pittsburgh). Page text: "57*31*12mm", "Approx Weight:43g", "Plug: XT30". Two packs, one in
  use and one charging.
- **Why 650 mAh.** It is the largest standard 2S XT30 pack that fits the layout F box outline (inner width about
  61.7 mm, inner depth about 18.9 mm): 57 mm along y, 12 mm thick. The switch and the DFR1202 also ride in the
  battery backpack, so its CAD decides the final outline. Larger 2S
  packs are 70 mm or longer or 18 mm or thicker and need a wider or deeper backpack, which moves the COM further back.
  20-25 min of walking per pack is enough for trial sessions; the bench backpack covers desk work.
- **Cheapest landed and fastest.** The same pack is $8.19 at ReadyMadeRC, but LiPo ships by ground at an estimated
  $9-13 below their free-shipping threshold, so Amazon at $9.16 delivered the next day is cheaper landed.
- **Larger pack if the measured runtime is short: Gaoneng GNB 930 mAh 2S 120C XT30** (Pyrodrone, $12.99, in stock;
  "61mm x 30mm x 16mm", "54g"). It is the largest standard-voltage 2S XT30 pack in stock at any US store checked on
  2026-10-15 (RDQ, Pyrodrone, CNHL US, genstattu, ReadyMadeRC; Amazon lists none between 850 and 1300 mAh). It gives
  about 43 % more runtime for 11 g more, but its 61 mm length nearly fills the 61.7 mm box width, so the backpack
  CAD must be drawn for it. Landed about $33-36 for two, arriving about Oct 19-22. Other candidates: Tattu 2S 800 45C
  XT30 (ReadyMadeRC $10.29, 52 x 30.5 x 15 mm), Lumenier 2S 850 35C XT30 (RDQ $16.49, 7-8 business-day ground).
- **Not 18650 or NP-F550.** A prebuilt STRIX Power Stix 2S 3200 mAh XT30 pack (ReadyMadeRC $21.99, 69 x 37 x 19.2 mm,
  101 g, 10 A continuous) gives about 50 min of walking but adds 58 g and does not fit the box width. Loose 18650
  cells need a 2S BMS and soldering. Pollen's NP-F550 (about 71 x 38 x 21 mm, 88-133 g) needs a printed holder with
  sprung contacts and has the same width problem.
- **Not 3S.** The 3S 650 two-pack chosen on 2026-10-14 fed the XL4015s; with no servo converter, 3S (12.6 V full)
  would overdrive the servos. It is removed from the cart.
- **Not HV packs** (7.6 V nominal, 8.7 V full), such as the OVONIC 2S 1100 or Tattu 450 HV: a full charge exceeds
  the 8.2 V top of the policies' training range.
- **Mass and balance.** The pack (43 g), the switch and the DFR1202 replace the two XL4015 and the 5991 in the same
  place behind the robot; the backpack CAD must rerun the COM check (`guide/print/back-plate/layout-F/mass_estimate.py`). The real robot already
  behaves as if its COM were about 10 mm forward of the model (`realgap-20261014`), so a small rearward shift should
  work in its favour (inference; the COM check decides).

## 4. Switch and connectors

- **Switch: REV Robotics Switch Cable and Bracket, REV-31-1387.** $13.00 at revrobotics.com, in stock. Page text:
  "Rated for >20A using a 12V battery", "Terminals are Male XT30(1) Female XT30(1)", "Panel mount in a rectangular hole
  27.5mm x 10.6mm". It is the standard FTC robot power switch, so RoboClub may have one; ask first. Nothing to solder.
  The backpack wall carries the 27.5 x 10.6 mm cutout.
- Amazon has no equivalent with XT30 on both ends from a known maker (the one listing found, 15 A, ships Oct 23-30
  from an unrated third-party seller, is $21.50).
- **Size.** REV publishes only the panel cutout; its STEP model gives about 32.4 x 38.1 x 13.6 mm overall and about
  29.6 mm behind the panel including the XT30 terminals, with 12 cm leads (REV docs: "XT30 Male - XT30 Female | 12
  cm"). Cut the 27.5 x 10.6 mm hole straight into the printed wall; the 45 x 31.5 x 3 mm steel bracket is not needed.
- **Genders** (named by the metal contacts, not the housing): the battery lead is female (Pyrodrone lists Tattu 2S
  packs as "XT30U-F"; REV: the Slim Battery is "wired with an XT30 female connector"), so the switch's male end takes
  the battery and its female end goes to the feed lead's male XT30 (section 5).
- **Not chosen:** Pololu Big Pushbutton Power Switch HP #2813 ($8.95; an `OFF` input would let the A7Z cut its own
  power after shutdown) and Big MOSFET Slide Switch HP #2815 ($8.49). Both need soldered terminals or wires plus two
  XT30 pigtails, and Pololu rates them at 6 A continuous with the MOSFET at 55 C. AndyMark am-4969 ($17, 20 A) has
  9.75 in leads that do not coil inside the box.
- **Landed price if bought:** $13.00 plus an estimated $6-9 ground shipping from Carrollton, TX, about Oct 20-22.
- **XT30 for battery voltage, MR30 for 5 V.** Every battery-voltage joint (battery, switch, feed lead, servo lead,
  robot servo inlet, bench 6.8 V output) is XT30, rated about 15 A continuous; the walking current is about 1.3 A and
  the sit-to-stand peak about 8 A. The A7Z's 5 V joint is an Amass MR30, a different shape, so 8.4 V cannot be
  plugged into the A7Z lead by mistake.
- **Owned:** one DFRobot FIT0586 XT30 male/female pair with solder cups: male half = robot
  servo inlet, female half = bench backpack 6.8 V output. The Amass XT30 and MR30 packs in the cart supply the rest.

## 5. Wiring

All power wire is 22 AWG stranded silicone, red `+` and black `-`. Every source side carries the female connector
(socket contacts, nothing live to touch); every robot side carries the male. No splices and no Y cables.

Robot side (fitted once, used by both backpacks):

1. **Servo inlet lead.** The owned DFRobot FIT0586 XT30 pair gives the male half. Solder a
   22 AWG pair into its two cups (heat shrink over each cup), about 150 mm long; the other ends go bare into the
   OpenRB-150 J4 screws (red `+`, black `-`). JP1 stays on `VIN(DXL)`.
2. **A7Z 5 V lead.** About 500 mm of 22 AWG pair along today's Pi lead route. The A7Z end is crimped into Dupont
   female terminals on header pin 2 (5 V) and pin 6 (GND); the rear-pack end is soldered into an Amass MR30 male,
   using the two outer contacts and leaving the middle one empty; mark the red side of every MR30 housing with a
   paint dot so all four MR30 halves are wired alike. A Dupont jumper
   (26-28 AWG) is too thin for this board: 1 m of loop at 2.2 A would drop about 0.3-0.45 V; the 22 AWG pair drops
   about 0.12 V. If the header reads under 4.85 V at boot or under stress, add a second pair on pins 4 and 9.

Battery backpack (Tattu pack, REV switch, DFR1202):

3. **Battery to switch.** The Tattu's female XT30 plugs into the switch's male end. Nothing to solder.
4. **Feed lead.** Amass XT30 male + about 80 mm of 22 AWG pair, plugged into the switch's female end; the far ends
   are soldered to the DFR1202 `VIN` and `GND` pads.
5. **Servo lead.** A second 22 AWG pair soldered to the same `VIN` and `GND` pads, about 100 mm, ending in an Amass
   XT30 female that plugs into the robot's servo inlet. The servo current passes through these two pad joints
   (about 1 A walking, 8 A for 20 ms at sit-to-stand), so tin both wires and the pad and keep the joint solid.
6. **5 V lead.** A third pair from `VOUT` and `GND`, about 100 mm, ending in an Amass MR30 female (same contacts as
   step 2) that plugs into the robot's A7Z lead.
   Pad count: `VIN` 2 wires, `GND` 3 wires, `VOUT` 1 wire, the same load the 2026-10-12 design put on these pads
   (input, OpenRB and Pi leads). `EN`, `AGND`, `PGOOD` stay empty. The board needs a printed hold-down in the
   backpack that leaves the pads reachable.

Bench backpack (V1.16 box, when used):

7. XL4015 #1 OUT (6.80 V) -> the owned FIT0586 female half on a 22 AWG pair (solder cups) -> robot servo inlet.
8. XL4015 #2 OUT (5.00 V) -> a short 22 AWG pair -> Amass MR30 female -> robot A7Z lead. This replaces the Pi's
   Dupont jumper lead.

Swapping backpacks: switch off (and shut the A7Z down first), unplug the XT30 and the MR30, plug in the other
backpack's two. The backpack CAD places both joints where they can be reached without opening the shells. Before the
first connection, meter each female: XT30 7-8.4 V (battery) or 6.80 V (bench), MR30 4.95-5.05 V with the correct
polarity.

## 6. Servo setting and software

- **Shutdown register, once per servo.** A full 2S pack reads 8.4 V, above the XL330 Max Voltage Limit (address 32,
  70 = 7.0 V, cannot be raised). With the factory Shutdown value 53 (address 63, bit 0 = Input Voltage Error) each servo
  drops torque and latches the error above 7.0 V. Write 52 to address 63 on IDs 1-15 with torque off, read it back,
  power-cycle. This is what Pollen runs (their maintainer reports shutdown = 52; the community build tutorial says to
  remove the input-voltage error trigger). It is a software setting and can be written back to 53 for the bench.
- `guide/firmware/pi/check_voltage_limit.py` needs a battery mode: accept Shutdown 52 and a rail of 6.4-8.4 V.
- **Battery guard in the control loop.** Every servo reports the bus voltage (Present Input Voltage, address 144).
  Read it about once a second: warn under 7.0 V under load (3.5 V per cell), and at 6.6 V (3.3 V per cell) stop
  walking, crouch, and shut the A7Z down. The walking policies were trained at 6.5-8.2 V, so this also keeps the
  policy inside its range.
- **Power-off order.** `sudo shutdown -h now` on the A7Z, wait 10-15 s, hold the robot, then switch off. The servos
  lose torque the moment the switch opens.

## 7. Charging and battery care

- Charge only with a balance charger on the JST-XH balance lead, in a LiPo bag, never on the robot and never
  unattended. 1C (0.65 A for a 650 mAh pack) is the conservative rate. The cart charger is fixed at 0.8 A, about 1.2C;
  Tattu publishes no charge rate for this pack (ReadyMadeRC and genstattu pages read 2026-10-15), and 1-2C is the usual
  allowance for FPV packs (inference). With an adjustable charger, set 0.6 A.
- New packs arrive at storage charge (about 3.8 V per cell). Charge fully before the first test; return unused packs
  to storage charge within a few days.
- Retire a pack that puffs, gets hot in use, or shows more than 0.05 V between cells after charging.
- The LiPo alarm on the balance lead is a backup only; unplug it between sessions because it drains the pack.

## 8. Parts

Owned, no purchase:

- DFRobot DFR1202 5 V buck converter.
- DFRobot FIT0586 XT30 male/female pair with solder cups.
- OpenRB-150, servos, A7Z (arriving 2026-10-16), the bench backpack parts.

Ask RoboClub (English names to use):

- **REV Robotics Switch Cable and Bracket (REV-31-1387)**, the FTC power switch with XT30 on both ends.
- **2S LiPo balance charger** with a JST-XH balance port (for example SkyRC iMAX B6, ISDT Q6 or ToolkitRC M6). If
  they lend one, delete the unselected cart charger; if not, tick it.
- **LiPo cell checker with low-voltage buzzer** (often sold as "1-8S LiPo battery checker / voltage alarm"). If they
  have none: RDQ 1-8S checker, $4.99, or any Amazon "1-8S lipo voltage checker".
- **22 AWG stranded silicone wire**, red and black, about 1.5 m of each (robot inlet 150 mm, A7Z lead 500 mm,
  backpack leads 280 mm, bench 5 V lead 100 mm, spare).
- **Female Dupont crimp terminals (2.54 mm) and 1-pin housings** plus a **Dupont crimping tool** (for example IWISS
  SN-28B or Engineer PA-09), to end the 22 AWG A7Z lead on the header pins.
- **Heat shrink tubing** about 3 mm and 6 mm, or Kapton tape.
- **Soldering station, rosin-core solder, flux**, and a **multimeter** if not at hand.

Buy (Amazon cart, read back 2026-10-15 01:04 EDT; selected subtotal $46.29, same-day delivery 5-10 PM except the
XT30 pack, which comes Oct 16):

- 2 x Tattu 2S 650 mAh 75C XT30, B0727XZFDW, $9.16 each, $18.32 (selected).
- 1 x Tenergy LiPo bag 7 x 9 in, B08QCMMGJT, $8.99 (selected; 4.8 stars from 2,415 ratings). Needed at home for
  charging and storage whatever RoboClub lends, and it lifts the order over the $25 same-day threshold.
- 1 x Amass XT30U 10 pairs with heat shrink, B07CTRH86M, $8.99 (selected; 4.6 stars from 217 ratings; arrives
  Oct 16). One male (feed lead) and one female (servo lead) go in the battery backpack; the rest are spares.
- 1 x Amass MR30 10 pairs, B0747MW9RX, $9.99 (selected; 4.8 stars from 168 ratings; rated 30 A continuous by the
  listing). One male on the robot's A7Z lead, one female each on the battery and bench backpack 5 V leads.
- 1 x LiPo Battery Charger 2S 3S Balance Charger, B0H4LC2J37, $14.99 (AC 100-240 V, 800 mA balance charge, 4.4 stars
  from 781 ratings), in the cart but unselected; tick it only if RoboClub lends no charger (arrives the next day).
- Not in the cart: the REV switch ($13.00 plus an estimated $6-9 shipping at revrobotics.com) if RoboClub has none.

Checked elsewhere and not cheaper landed: Tattu 650 at ReadyMadeRC ($8.19 plus ground shipping), ISDT UC2 USB
charger at RDQ ($8.99, 0.8 A, 5 V 2 A input, plus ground shipping of 7-8 business days), ToolkitRC M4AC ($35.99, an
upgrade only). Procurement record: `../../procurement/battery-backpack-20261015/`.

## 9. Order of work

1. RoboClub: switch, charger, checker, wire, crimp parts.
2. Batteries arrive: measure the real pack (length, width, thickness, lead exits); charge one fully on the balance
   charger in the bag; check both cells within 0.05 V.
3. Servo setting: write Shutdown 52 on IDs 1-15 and read back (script to write; runs on the bench supply first).
4. Backpack CAD with the measured pack: battery pocket and retention, REV switch cutout, DFR1202 hold-down with its
   leads, the XT30 and MR30 joint positions, balance lead access, same four fixing holes as the layout F box; COM and
   hip-sweep checks and the print gate. Explain the plug positions to Shawn before they shape the print.
5. Wiring per section 5; meter check of both plugs (step 5 values) before connecting servos or the A7Z.
6. First battery power: stand, then walk with `current_watch.py` logging, and the A7Z header voltage on a meter.

## 10. What changes the plan

- Measured walking draw above about 2 A from the battery (runtime under 15 min): consider a 2S 1000-1300 mAh pack
  and a deeper backpack.
- Any servo Input Voltage Error or overheating on the full pack: stop and revisit (D-007's 6.8 V rail stays the
  fallback through the bench backpack).
- A7Z header voltage under 4.85 V at boot or under stress: add the second 5 V/GND pair (pins 4 and 9).
- The DFR1202 `VIN`/`GND` joints warm up during the walking current test: move the servo branch off the pads (a
  commercial XT30 1-male-to-2-female Y lead; on 2026-10-15 Amazon only had unrated third-party ones arriving Oct 22 to
  Nov 9).
