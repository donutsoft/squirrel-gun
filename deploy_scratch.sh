mv scratch_py/.venv venv_bak
ssh pi@192.168.1.209 "killall python3"
ssh pi@192.168.1.209 "rm -rf /home/pi/scratch_py"
ssh pi@192.168.1.209 "mkdir /home/pi/scratch_py"
scp -r scratch_py pi@192.168.1.209:/home/pi/
ssh pi@192.168.1.209 "bash -lc 'cd /home/pi/scratch_py && uv run python3 sq_main.py'"
mv venv_bak scratch_py/.venv