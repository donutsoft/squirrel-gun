ssh pi@192.168.1.210 "killall python3"
ssh pi@192.168.1.210 "rm -rf /home/pi/squirrel-daemon/*"
scp -r squirrel-daemon pi@192.168.1.209:/home/pi/
ssh pi@192.168.1.210 "bash -lc 'cd /home/pi/squirrel-daemon && ./run.sh'"
