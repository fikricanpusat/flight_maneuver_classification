import socket
import struct
import keyboard
import datetime

active_maneuver = -1

def determine_active_maneuver():
    global active_maneuver
    if keyboard.is_pressed('0'): active_maneuver =  0
    elif keyboard.is_pressed('1'): active_maneuver =  1
    elif keyboard.is_pressed('2'): active_maneuver =  2
    elif keyboard.is_pressed('3'): active_maneuver =  3
    elif keyboard.is_pressed('4'): active_maneuver =  4
    elif keyboard.is_pressed('5'): active_maneuver =  5
    elif keyboard.is_pressed('6'): active_maneuver =  6
    elif keyboard.is_pressed('7'): active_maneuver =  7
    elif keyboard.is_pressed('8'): active_maneuver =  8
    elif keyboard.is_pressed('9'): active_maneuver =  9
    elif keyboard.is_pressed('a'): active_maneuver =  10
    elif keyboard.is_pressed('b'): active_maneuver =  11
    elif keyboard.is_pressed('c'): active_maneuver =  12
    elif keyboard.is_pressed('d'): active_maneuver =  13
    elif keyboard.is_pressed('e'): active_maneuver =  14
    elif keyboard.is_pressed('f'): active_maneuver =  15

UDP_IP = "127.0.0.1"
UDP_PORT = 49005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

f = open(str(datetime.datetime.now()) + ".csv", "a+")
f.write("_Vind-_kias,\
    __VVI-__fpm,Gload-norml,Gload-axial,Gload-_side,_elev-stick,ailrn-stick,ruddr-stick,_elev-_surf,\
    ailrn-_surf,ruddr-_surf,____M-_ftlb,____L-_ftlb,____N-_ftlb,\
    ____Q-rad/s,____P-rad/s,____R-rad/s,pitch-__deg,_roll-__deg,hding-_true,hding-__mag,alpha-__deg,\
    _beta-__deg,hpath-__deg,vpath-__deg,_slip-__deg,thro1-_part,_lift-___lb,\
    _drag-___lb,_side-___lb,____L-lb-ft,____M-lb-ft,____N-lb-ft ,flight\n")

log_start = False
prev_maneuver = False
try:
    while True:
        data, addr = sock.recvfrom(437)
        floatdata = struct.unpack('=bbbbbiffffffffiffffffffiffffffffiffffffffiffffffffiffffffffiffffffffiffffffffiffffffffiffffffff', data)
        prev_maneuver = active_maneuver
        determine_active_maneuver()
        if keyboard.is_pressed('space') and log_start == False:
            log_start = True
            print("Logging started\nManeuver = " + str(active_maneuver))
        elif keyboard.is_pressed('backspace') and log_start == True:
            log_start = False
            print("Logging stopped")
        if log_start == True:
            if prev_maneuver != active_maneuver:
                print("Maneuver = " + str(active_maneuver))
            f.write(\
                str(floatdata[6]) + "," + \
                str(floatdata[17]) + "," + str(floatdata[19]) + "," + str(floatdata[20]) + "," + str(floatdata[21]) + "," + \
                str(floatdata[24]) + "," + str(floatdata[25]) + "," + str(floatdata[26]) + "," + \
                str(floatdata[33]) + "," + str(floatdata[34]) + "," + str(floatdata[35]) + "," + \
                str(floatdata[42]) + "," + str(floatdata[43]) + "," + str(floatdata[44]) + "," + \
                str(floatdata[51]) + "," + str(floatdata[52]) + "," + str(floatdata[53]) + "," + \
                str(floatdata[60]) + "," + str(floatdata[61]) + "," + str(floatdata[62]) + "," + str(floatdata[63]) + "," + \
                str(floatdata[69]) + "," + str(floatdata[70]) + "," + str(floatdata[71]) + "," + str(floatdata[72]) + "," + str(floatdata[76]) + "," + \
                str(floatdata[78]) + "," +\
                str(floatdata[87]) + "," + str(floatdata[88]) + "," + str(floatdata[89]) + "," + str(floatdata[90]) + "," + str(floatdata[91]) + "," + str(floatdata[92]) + "," + \
                str(active_maneuver) + "\n")

                
except KeyboardInterrupt:
    print("Ended!")
    f.close()
