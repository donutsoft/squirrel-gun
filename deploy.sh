ssh pi@squirrel.local "rm -rf /home/pi/squirrel-daemon"
scp -r squirrel-daemon pi@squirrel.local:/home/pi/
