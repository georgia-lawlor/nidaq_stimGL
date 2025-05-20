from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.io import savemat
import numpy as np
import struct
import sys
import os


path_raw = 'test_nidaq.bin'


def convert2mat(path, plot=True):
    headerSize = 15
    
    file = open(path,'rb')
    header = []
    for h in range(headerSize):
        header.append( struct.unpack('B',file.read(1))[0])
    version     = header[0]
    fs          = header[1]*65536 + header[2]*256 + header[3]
    numCh       = header[4]
    year        = header[5]*256 + header[6]
    month       = header[7]
    day         = header[8]
    hour        = header[9]
    minute      = header[10]
    second      = header[11]
    microsecond = header[12]*65536 + header[13]*256 + header[14]
    timestamp = datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second, microsecond=microsecond)
    print('Sample Rate: ', fs)
    print('Number Channels: ', numCh)
    print('Start Time: ', timestamp)
    filesize = os.path.getsize(path) - headerSize

    data_len = filesize//8//numCh
    data_raw = np.ndarray((data_len, numCh), dtype=np.float64, buffer=file.read(filesize)).transpose()
    file.close()

    time_sec  = np.linspace(0,data_len/fs,data_len)
    time_date = np.array([timestamp + timedelta(seconds=t) for t in time_sec]).astype(np.datetime64)

    print('Saving .mat file...')
    savemat(os.path.splitext(path)[0]+'.mat', {'data_raw': data_raw, 'time_sec': time_sec, 'time_date': time_date})

    if plot:
        print('Plotting...')
        plt.figure()
        for n in range(numCh):
            if n == 0:
                ax = plt.subplot(numCh,1,n+1)
            else:
                plt.subplot(numCh,1,n+1, sharex=ax)
            plt.plot(time_sec, data_raw[n,:])
            plt.ylabel('Voltage')
        plt.xlabel('Time (sec)')
        plt.show()


if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) > 1:
        convert2mat(sys.argv[1])
    else:
        convert2mat(path_raw)