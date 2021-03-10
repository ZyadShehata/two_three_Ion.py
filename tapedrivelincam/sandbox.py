import subprocess
import time
import signal
import sys
import os
print("startup")
#os.system("ping -t google.de")
#proc = subprocess.Popen([r"C:\Windows\System32\PING.EXE", r"-t", r"google.de"], stdin=subprocess.PIPE, stdout=sys.stdout.fileno(), stderr=sys.stdout.fileno(), shell=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
proc = subprocess.Popen([r"C:\Users\Photonscore\Desktop\erlangen-readout\erlangen_readout.exe", r"--log=stdout", r"--measure-for=20",r"--master-tac-window=4", r"--slave-tac-window=4", r"--master-tac-bias=2000", r"--slave-tac-bias=2000", r"c:\Users\Photonscore\Desktop\erlangen-readout\PX29110_master.profile", r"c:\Users\Photonscore\Desktop\erlangen-readout\PX29109_slave.profile", r"C:\Users\Photonscore\Desktop\testdir\OneIon10uW.photons"], stdin=subprocess.PIPE, stdout=sys.stdout.fileno(), stderr=sys.stdout.fileno(), creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, shell=True)
if(proc.wait() == 0):
    print("everything good")
else:
    print("error")


#time.sleep(5)  # just so it runs for a while
#os.kill()
#os.kill(proc.pid, signal.CTRL_C_EVENT)
#proc.send_signal(signal.CTRL_C_EVENT)
#print("send sig")



def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    for stderr_line in iter(popen.stderr.readline, ""):
        yield stderr_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

# Example
#for path in execute([r"C:\Users\Photonscore\Desktop\erlangen-readout\erlangen_readout.exe", r"--log=stdout", r"--master-tac-window=4", r"--slave-tac-window=4", r"--master-tac-bias=2000", r"--slave-tac-bias=2000", r"c:\Users\Photonscore\Desktop\erlangen-readout\PX29110_master.profile", r"c:\Users\Photonscore\Desktop\erlangen-readout\PX29109_slave.profile", r"C:\Users\Photonscore\Desktop\testdir\OneIon10uW.photons"]):
#    print(path, end="")
