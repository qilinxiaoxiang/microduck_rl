# Sim-vs-robot scorecard

Outcome matched in 5 of 12 replayed robot runs (sim majority over seeds [0, 1, 2]).

- **1018a-walk61-025** (bench-usb-20261018): robot up, sim {'up': 3} (max tilt robot 11.5, sim [5.7, 6.3, 5.7] deg); MATCH. Jitter robot 1.255 sim 0.213 (ratio 0.17), action jumps robot 0.349 sim [0.01, 0.011, 0.01].
- **1018b-ft3-020-first** (bench-usb-20261018): robot guard at 3.0 s, sim {'up': 3} (max tilt robot 41.3, sim [8.1, 5.1, 6.8] deg); MISS. Jitter robot 1.38 sim 0.186 (ratio 0.13), action jumps robot 0.433 sim [0.008, 0.004, 0.012].
- **1018b-ft3-020-second** (bench-usb-20261018): robot fell at 4.2 s, sim {'up': 3} (max tilt robot 92.0, sim [8.1, 5.1, 6.8] deg); MISS. Jitter robot 1.567 sim 0.186 (ratio 0.12), action jumps robot 0.619 sim [0.008, 0.004, 0.012].
- **1018d-walk61-025-first** (bench-usb-20261018): robot up, sim {'up': 3} (max tilt robot 7.7, sim [5.7, 6.3, 5.7] deg); MATCH. Jitter robot 0.896 sim 0.157 (ratio 0.18), action jumps robot 0.173 sim [0.007, 0.007, 0.007].
- **1018d-walk61-025-second** (bench-usb-20261018): robot up, sim {'up': 3} (max tilt robot 6.8, sim [5.7, 6.3, 5.7] deg); MATCH. Jitter robot 1.124 sim 0.157 (ratio 0.14), action jumps robot 0.299 sim [0.007, 0.007, 0.007].
- **1018g-ft3-020** (bench-usb-20261018): robot fell at 2.6 s, sim {'up': 3} (max tilt robot 57.2, sim [8.1, 5.1, 6.8] deg); MISS. Jitter robot 1.641 sim 0.161 (ratio 0.1), action jumps robot 0.61 sim [0.004, 0.004, 0.008].
- **1106-run1-table-scan** (battery-a7z-20261106): robot up, sim {'up': 3} (max tilt robot 10.9, sim [8.5, 8.1, 8.1] deg); MATCH. Jitter robot 0.187 sim 0.082 (ratio 0.44), action jumps robot 0.007 sim [0.001, 0.001, 0.0].
- **1106-run2-table-steer-flip** (battery-a7z-20261106): robot fell at 8.2 s, sim {'up': 3} (max tilt robot 129.6, sim [42.4, 37.6, 39.1] deg); MISS. Jitter robot 0.495 sim 0.109 (ratio 0.22), action jumps robot 0.103 sim [0.003, 0.0, 0.0].
- **1106-run3-table-lookdown** (battery-a7z-20261106): robot fell at 84.9 s, sim {'up': 3} (max tilt robot 60.3, sim [41.7, 43.3, 44.2] deg); MISS. Jitter robot 0.372 sim 0.099 (ratio 0.27), action jumps robot 0.049 sim [0.002, 0.002, 0.001].
- **1106-run4-table-aim-lookdown** (battery-a7z-20261106): robot fell at 84.5 s, sim {'up': 3} (max tilt robot 92.2, sim [9.6, 9.9, 9.6] deg); MISS. Jitter robot 0.369 sim 0.127 (ratio 0.34), action jumps robot 0.048 sim [0.005, 0.007, 0.008].
- **1106-run6-carpet-turns-05** (battery-a7z-20261106): robot fell at 27.4 s, sim {'up': 3} (max tilt robot 95.0, sim [37.9, 41.6, 44.0] deg); MISS. Jitter robot 0.499 sim 0.191 (ratio 0.38), action jumps robot 0.047 sim [0.011, 0.009, 0.004].
- **1106-run7-carpet-turns-03** (battery-a7z-20261106): robot up, sim {'up': 3} (max tilt robot 11.8, sim [8.9, 8.9, 8.5] deg); MATCH. Jitter robot 0.231 sim 0.087 (ratio 0.38), action jumps robot 0.008 sim [0.001, 0.001, 0.0].

Cited from earlier studies (not rerun):

- **1101-groundpick-bend** (charger-a7z-20261030): robot held bend: ankles stop at 1.25 rad, trunk 37.4 deg, camera 14.5 mm over the floor; sim as-built sim with the D-040 heel stop 1.36 rad: trunk 34.7 deg, camera 21.5 mm (limit 1.25 gives 37.6 deg, 15.6 mm); miss until the ankle cables move to the front socket; then the robot should stop near 1.36 rad. Source: microduck-sim/scan-20261101/README.md (closed_loop_port.json, closed_loop_heel.json); DECISIONS.md D-040.
- **1030-getup-side** (charger-a7z-20261030): robot stand61 from lying on the side: 0 of 2 rose; sim stand61 from the side: 10 of 10 rose (50 of 50 over all starts); miss. Source: microduck-sim/CURRENT-STATE.json facts.getup.
