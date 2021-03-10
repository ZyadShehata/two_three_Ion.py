import matplotlib.pyplot as plt
import numpy as np
import time
from include.Correlations import Correlator

#length and step in hardware bins
def build_boundaries(length, step):
    return np.linspace(0.5, length, num=(int(length/step)+1))

#numpy correlation function, quite fast
def correlate(times, channels, length, steps):
    channeldiffs = np.diff(channels)
    idx_pos = np.where(channeldiffs == 1)[0]
    idx_pos_two = idx_pos + 1
    idx_neg = np.where(channeldiffs == -1)[0]
    idx_neg_two = idx_neg + 1
    val_pos = np.subtract(times[idx_pos_two], times[idx_pos])
    val_neg = np.subtract(times[idx_neg_two], times[idx_neg])
    #create bins
    bins = build_boundaries(length, steps)
    histPos = np.histogram(val_pos, bins)
    histNeg = np.histogram(-1*val_neg, np.flipud(-1*bins))
    return histPos, histNeg

#simulation of random timetags
def sim_timetags(num_samples, filename): #size in GB, expected countrate per detector
    with open(filename, 'ab') as f:
        header = 0 #some meaningless header to simulate the one the qutag qill write
        f.write(header.to_bytes(40, byteorder='little'))
        channels = np.random.randint(1, 3, size=num_samples, dtype=np.int16)
        time_samples = np.zeros(num_samples, dtype=np.int64)
        mult = int(num_samples/10000)
        for i in range(10000):
            time_samples[i*mult:(i+1)*mult] = i
        times = np.zeros(num_samples, dtype=np.int64)
        for i in range(num_samples):
            if (i%int(num_samples/100) == 0 and i != 0):
                print(i/num_samples)
            times[i] = times[i-1]+time_samples[i]
        #plt.figure()
        #plt.plot(np.arange(times.size), times)
        #plt.show()
        for i in range(num_samples):
            f.write(int(times[i]).to_bytes(8, byteorder='little'))
            f.write(int(channels[i]).to_bytes(2, byteorder='little'))
    f.close()
    return

def read_chunk(old_offset, filepath): #chunksize
    #consider 40 byte header: first pointer has to be chosen accordingly
    with open(filepath, 'rb') as f:
        f.seek(old_offset, 0)
        dt = np.dtype([('timestamp', np.int64), ('channel', np.int16)])
        read_file = np.fromfile(f, dtype=dt)
        channels = read_file['channel']
        #print(channels)
        times = read_file['timestamp']
        #print(times)
        #mm = mmap.mmap(f.fileno(), len(input), access=mmap.ACCESS_READ)
        #for i in range(int(len(input)/10)):
        #    record = mm.read(10)
        #    channels[i] = record[8]+(record[9] << 8)
        #    times[i] = record[0]+(record[1] << 8)+(record[2] << 16)+(record[3] << 24)+(record[4] << 32)+(record[5] << 40)+(record[6] << 48)+(record[7] << 56)
        #mm.close()
        #dt = np.dtype([np.int64, np.int16])
        #read_file = np.fromfile(f, dtype=dt)
    f.close()
    return channels, times

#times = np.array([2, 4, 6, 8, 12, 13, 16, 19, 26, 28, 30, 35], dtype=np.int64)
#channels = np.array([1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1], dtype=np.int16)

filename = r'D:\Data\Stefan\2018_AstroLabTest\DirectBinary\Etalon80kTestTwo.bin'
filename = r'data/timetags.bin'
#filename = "D:/Data/Stefan/2018_AstroLabTest/TapedriveDev/30s80kEtalon532/data1.bin"
#filename = r'data/Etalon80kTest.bin'

#print("direct implement")

#sim_timetags(10000000, filename)

#starttime = time.time()
#offset = 40
#channels, times = read_chunk(offset, filename)

#print(time.time()-starttime)

#histOne, histTwo = correlate(times, channels, 10000, 1)
#totaloValues = np.concatenate([histTwo[0][1:], histOne[0][:-1]])
#totaloAxis = np.concatenate([histTwo[1][1:-1], histOne[1][1:-1]])

#print(time.time()-starttime)

#plt.figure()
#plt.plot(totaloAxis, totaloValues)
#plt.grid()
#plt.show()

#plt.clf()

print("numpy")

starttime = time.time()
Correlatoro = Correlator()
#Correlator.setJit(True)
Correlatoro.setLengthsAndSteps(10000, 50)
Correlatoro.setFilePath(filename)
Correlatoro.read_chunk()
print(time.time()-starttime)
starttime = time.time()
Correlatoro.Correlate()
print(time.time()-starttime)
print("second run")
starttime = time.time()
Correlatoro.Correlate()
print(time.time()-starttime)

print("numpy jitted")

starttime = time.time()
Correlatoro = Correlator()
Correlatoro.setJit(True)
Correlatoro.setLengthsAndSteps(10000, 50)
Correlatoro.setFilePath(filename)
Correlatoro.read_chunk()
print(time.time()-starttime)
starttime = time.time()
Correlatoro.Correlate()
print(time.time()-starttime)
print("second run (jitted & compiled)")
starttime = time.time()
Correlatoro.Correlate()
print(time.time()-starttime)

#plt.figure()
#plt.plot(Correlator.getHistograms()[1][:-1], Correlator.getHistograms()[0])
#plt.grid()
#plt.show()

#plt.clf()

print("loop")

starttime = time.time()
Correlatoro.resetHistograms()
Correlatoro.setMode("Loop")
Correlatoro.setJit(False)
Correlatoro.setFilePath(filename)
Correlatoro.read_chunk()
print(time.time()-starttime)
Correlatoro.Correlate()
print(time.time()-starttime)


print("loop jit")

starttime = time.time()
Correlatoro.resetHistograms()
Correlatoro.setMode("Loop")
Correlatoro.setJit(True)
Correlatoro.setFilePath(filename)
Correlatoro.read_chunk()
print(time.time()-starttime)
Correlatoro.Correlate()
print(time.time()-starttime)
print("second run (jitted & compiled)")
starttime = time.time()
Correlatoro.Correlate()
print(time.time()-starttime)

#plt.figure()
#plt.plot(Correlator.getHistograms()[1], Correlator.getHistograms()[0])
#plt.grid()
#plt.show()
