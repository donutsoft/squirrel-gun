ssh pi@192.168.1.209 "rm -rf /home/pi/squirrel-daemon/*"
scp -r squirrel-daemon pi@192.168.1.209:/home/pi/
