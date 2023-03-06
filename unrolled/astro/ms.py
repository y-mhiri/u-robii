#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 10:24:14 2021

@author: monnier, y-mhiri
last modified: 2023-02-14 16:01:42
"""

import numpy as np
import casacore.tables as tb


class MS:
    def __init__(self, msName="", DoReadData=True):
        self.msName = msName
        self.stationNames = None
        self.antennaPos   = None
        self.na           = None
        self.nbl          = None
        self.ref_freq     = None
        self.orig_freq    = None
        self.chan_freq    = None
        self.d_freq       = None
        self.chan_width   = None
        self.freq_mean    = None
        self.wavelength_chan= None
        self.nb_chan      = None
        self.dt           = None
        
        if self.msName == "":
            raise RuntimeError("No MS file is specified, please check if the path exists")
        
        if not(self.msName.endswith('.ms') or self.msName.endswith('.MS')):
            raise RuntimeError("File %s is not an MS file."%self.msName)
        
        self.readMSInfo()
        
        
        
        if DoReadData is True:
            self.MS_DATA = self.readData()
            self.printMSInfo()
        
    def giveMainTable(self):
        t = tb.table(self.msName, readonly=True, ack=False)
        return t.sort("TIME")
       
        
    
    def readData(self):
        
        DATA = dict.fromkeys(["ANTENNA1", "ANTENNA2", "UVW", "times", "uniq_times", "dnu", "dt", "data"])
        
        print("Reading Data from", self.msName)
        
        mainTable = tb.table(self.msName, readonly=True, ack=False)
        
        # Informations relatifs à la fréquence d'acquisition
        t_freq = tb.table(mainTable.getkeyword('SPECTRAL_WINDOW'), readonly=True, ack=False)

        ref_freq   = t_freq.getcol('REF_FREQUENCY') # [0] should depend on the spectral window ID (ici on ne travail que sur une seule) 
        orig_freq = t_freq.getcol('CHAN_FREQ')
        chan_freq = orig_freq
        d_freq = t_freq.getcol('CHAN_WIDTH') # Generic formulation pour sélectionner seulement une seule BW (on présume qu'elles sont toutes égales)
        chan_width = np.abs(t_freq.getcol('CHAN_WIDTH'))
        freq_mean = np.mean(chan_freq)        
        chan_wavelength = 299792458./chan_freq 

        nb_chan = len(chan_freq[0]) # Number of frequency channel in the the first spectral window  
        
        nRows = mainTable.nrows()
        
        #ordered_table = tb.taql("SELECT UVW, DATA, WEIGHT, FLAG, DATA_DESC_ID, ANTENNA1, ANTENNA2, TIME FROM $mainTable ORDERBY ANTENNA1, ANTENNA2, TIME")
        
        
        uvw = mainTable.getcol('UVW', 0, nRows)
        data_desc_id = mainTable.getcol('DATA_DESC_ID', 0, nRows)
        data = mainTable.getcol('DATA', 0, nRows)
        flag = mainTable.getcol('FLAG', 0, nRows)
        weight = mainTable.getcol('WEIGHT', 0, nRows)
        
        #TODO Trié les visibilités en fonction de la basline, temps, etc.

        mainTable.close()
        # ordered_table.close()

        self.nrows = nRows
        self.uvw = uvw
        self.data_desc_id = data_desc_id
        self.vis_data = data
        self.flag = flag
        self.weight = weight
        self.ref_freq       = ref_freq
        self.orig_freq      = orig_freq
        self.chan_freq      = chan_freq
        self.d_freq         = d_freq
        self.chan_width     = chan_width
        self.freq_mean      = freq_mean
        self.chan_wavelength= chan_wavelength
        self.nb_chan        = nb_chan
        

    def readMSInfo(self):
        
        print("Informations about", self.msName)
        mainTable = self.giveMainTable() 
        nRows = mainTable.nrows()
        print("Number of rows :", nRows)

        # Informations relatifs aux stations
        t_ant = tb.table(mainTable.getkeyword('ANTENNA'), readonly=True, ack=False)
        stationNames = t_ant.getcol('NAME')
        
        antennaPos  = t_ant.getcol('POSITION')
        na          = t_ant.getcol('POSITION').shape[0]
        nbl         = (na*(na-1)/2) + na


        nb_antenna = t_ant.nrows()
        self.max_baseline_length = - 1
        for p in range(0, nb_antenna):
            for q in range(p, nb_antenna):
                b = antennaPos[p, :] - antennaPos[q,:]
                self.max_baseline_length = max(np.linalg.norm(b), self.max_baseline_length)

        dt = mainTable.getcol("INTERVAL", 0, 1)


        self.stationNames   = stationNames        
        self.antennaPos     = antennaPos
        self.na             = na
        self.nbl            = nbl
        self.dt             = dt
        
        mainTable.close()
        
        
    def printMSInfo(self):
        print("Information about %s ." %(self.msName))
        print("%d Antennas from %s Stations." %(self.na, self.stationNames[0]))
        print("A total of %d baselines." %(self.nbl))
        
        print("Number of Channels : %d" %(self.nb_chan))
        print("Visibility polarization is ", self.vis_data.shape)
        print("")
        
def testMS():
    
    msFile =MS(msName="/home/monnier/These/MS_files/SNR_G55_10s.calib.ms")
    msFile.printMSInfo()
    

