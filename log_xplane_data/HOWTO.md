# HOW TO LOG X-PLANE DATA

## The maneuver enumaration

0:  Level Flight,
1:  Sustained Turn,
2:  Ascend,
3:  Descend,
4:  Reverse,
5:  Immelman,
6:  Aileron Roll,
7:  Split S,
8:  Chandelle,
9:  Lazy-8

## Steps

1. Start a flight in X-Plane.
2. From the Data Output menu chose below data groups and send them with UDP to local IP from 49005 port.
    * Mach, VVI, g-load
    * Joystics aileron/elevator/rudder
    * Flight controls aileron/elevator/rudder
    * Angular moments
    * Angular velocities
    * Pitch, roll, & headings
    * Angle of attack, sideslip, & paths
    * Throttle
    * Aerodynamic forces
3. Run the log.py program.
4. Press "space" to start logging.
5. Press the number corresponding to wanted maneuver to change current flight maneuver
6. Press "backsapce" to stop logging.
7. A file with date and time will appear with logs in the log.py directory.