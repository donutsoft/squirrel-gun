ssh pi@192.168.1.155 "killall python3"
ssh pi@192.168.1.155 "rm -rf /home/pi/squirrel-daemon/*"
scp -r squirrel-daemon pi@192.168.1.155:/home/pi/
ssh pi@192.168.1.155 "bash -lc 'cd /home/pi/squirrel-daemon && ./run.sh'"
