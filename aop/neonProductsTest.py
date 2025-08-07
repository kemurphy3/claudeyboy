# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:26:24 2024

@author: tgoulden
"""


import sys
import os

currentFile = os.path.abspath(os.path.join(__file__))
nisLibDir = os.path.join(currentFile.lower().split('gold_pipeline')[0], 'Gold_Pipeline','ProcessingPipelines', 'lib', 'Python')


if nisLibDir not in sys.path:
    sys.path.insert(0, nisLibDir)

from aop_processing_utils_k8m_v1 import *

if __name__ == '__main__':

    YearSiteVisits = ['2022_ABBY_5'] # NIS
    #YearSiteVisits = ['2023_WLOU_4'] # Camera


    # For a complete run, be sure to have white balanced photos downloaded:
    # D:\2022\FullSite\D05\2022_LIRO_3\Processing\Camera\Preprocessing\Output
    # All other directories under D:\2022\FullSite\D05\2022_LIRO_3\ should be deleted.
    #
    # For a Camera run, be sure to also have AIGDSM and FlightlineBoundary downloaded:
    # D:\2022\FullSite\D05\2022_LIRO_3\Internal\DiscreteLidar\AIGDSM
    # D:\2022\FullSite\D05\2022_LIRO_3\Metadata\DiscreteLidar\FlightlineBoundary
    # All other directories under D:\2022\FullSite\D05\2022_LIRO_3\ should be deleted.

    for YearSiteVisit in YearSiteVisits:
        YSV = YearSiteVisitClass(YearSiteVisit)
        # YSV.downloadInternalLidarAigDsm()
        # aigdsmSapExists=False
        # for file in YSV.LidarInternalAigDsmFiles:
        #     if file.endswith('sap_vf_bf.hdr'):
        #         aigdsmSapExists = True
        
        # if aigdsmSapExists:
        #     pass
        # else:
        #     YSV.calcAigDsmSlopeAspect()
        #     YSV.getProductFiles()
        for mission in YSV.missions:
            if mission.flightday == '2022071115':
                print("Working on 2022071115")
                print('Downloading Spectrometer Raw data for '+ mission.missionId)
                #mission.downloadDataRawSpectrometerProcessing()
                print('Running Spectrometer flightline boundaries for '+ mission.missionId)
                mission.generateSpectrometerKmls()
                print('Running L1 Radiance Spectrometer Products for '+ mission.missionId)
                print('Running L1 DN to Radiance for '+mission.missionId)
                mission.processRadiance()
                print('Running L1 Radiance QA for '+mission.missionId)
                #mission.processRadianceQa(domain,visit)
                print('Running L1 Radiance Ortho for '+mission.missionId)
                mission.processOrthoRadiance(os.path.join(YSV.geoidFileDir,YSV.geoidFile),YSV.internalAigDsmFile,YSV.internalAigSapFile)
                print('Running L1 Radiance Ortho QA for '+mission.missionId)
                mission.generateOrthoQa()
                print('Running L1 Radiance H5 for '+mission.missionId)
                mission.processH5RadianceWriter()
                print('Generating L1 Radiance H5 RGB tifs for '+mission.missionId)
                mission.generateRgbRadianceTifs()                
                print('Running L1 Reflectance Spectrometer Products for '+ mission.missionId)
                print('Running Atcor Topo clips for '+mission.missionId)
                mission.clipTopoForAtcor(YSV.internalAigDsmFile)
                print('Running Smooth Elevation for '+mission.missionId)
                mission.processSmoothElevation()
                print('Running Slope Aspect for '+mission.missionId)
                mission.processSlopeAspect()
                print('Preparing ATCOR inputs for '+mission.missionId)
                mission.prepareAtcorInputs()
                print('Running ATCOR reflectance retrieval for '+mission.missionId)
                mission.processReflectance()
                print('Running post ATCOR updates for '+mission.missionId)
                mission.postAtcorUpdates()
                print('Running L1 reflectance H5 for '+mission.missionId)
                mission.postAtcorFileUpdates()
                mission.processH5ReflectanceWriter()
                print('Cleaning up atcor files for '+mission.missionId)
                #mission.cleanupAtcor()
                #mission.processL1ReflectanceSpectrometer(YSV.internalAigDsmFile)
                #mission.spectrometerProcessWorkflow(os.path.join(YSV.geoidFileDir,YSV.geoidFile),YSV.internalAigDsmFile,YSV.internalAigSapFile,YSV.spectrometerProductQaList,YSV.Domain,YSV.visit)

        # print('Running L3 reflectance H5 for '+YSV.YearSiteVisit)
        # YSV.processMosaicH5ReflectanceWriter()
        # YSV.getProductFiles()
        # YSV.cleanupL3H5()
        # YSV.processMosaicSpectrometerProducts()
        # YSV.processMosaicQa()
        # YSV.cleanMosaicProducts()
        #print('Finished processSpectrometerMosaicProducts '+YSV.YearSiteVisit)

        # Remove Lidar from Complete Run by commenting out self.processWaveformLidarWorkflow() in C: or D:\Gold_Pipeline\ProcessingPipelines\NIS\lib\Python\aop_processing_utils.py
        #YSV.processNeonProducts() # Complete Run
        #YSV.processSpectrometerProducts() # Spectrometer Run Only
        # YSV.previousYearSiteVisit  = '2021_STER_3'
        # YSV.obtainedPreviousSiteVisit = True 
        # print('Getting report summariues for '+YSV.YearSiteVisit)
        # YSV.getSpectrometerQaReports()
        
        # YSV.generateMosaicQaPdf()
        
        # YSV.cleanMosaicProducts()
        
        #YSV.cleanMosaicProducts()
        #YSV.generateMosaicReflectanceQa()
        #YSV.generateMosaicQaPdf()
        #YSV.downloadInternalLidarAigDsm()
        # YSV.calcAigDsmSlopeAspect()
        # YSV.getProductFiles()
        # YSV.downloadInternalLidarAigDsm()
        # aigdsmSapExists=False
        # for file in YSV.LidarInternalAigDsmFiles:
        #     if file.endswith('sap_vf_bf.hdr'):
        #         aigdsmSapExists = True
        
        # if aigdsmSapExists:
        #     pass
        # else:
        #     YSV.calcAigDsmSlopeAspect()
        #     YSV.getProductFiles()
        # YSV.downloadInternalLidarAigDsm()
        # aigdsmSapExists=False
        # for file in YSV.LidarInternalAigDsmFiles:
        #     if file.endswith('sap_vf_bf.hdr'):
        #         aigdsmSapExists = True
        
        # if aigdsmSapExists:
        #     pass
        # else:
        #     YSV.calcAigDsmSlopeAspect()
        #     YSV.getProductFiles()
       
        #YSV.processSpectrometerMosaic()
    #mission.payload.nisNominalTimingOffset = float(mission.payload.nisNominalTimingOffset) -1
                # print('Running Spectrometer flightline boundaries for '+ mission.missionId)
                # #mission.generateSpectrometerKmls()
                # print('Running L1 Radiance Spectrometer Products for '+ mission.missionId)
                # mission.processL1RadianceSpectrometer(os.path.join(YSV.geoidFileDir,YSV.geoidFile),YSV.internalAigDsmFile,YSV.internalAigSapFile,YSV.Domain,YSV.visit)
                # print('Running L1 Reflectance Spectrometer Products for '+ mission.missionId)
                # # print('Running Atcor Topo clips for '+mission.missionId)
                # mission.clipTopoForAtcor(YSV.internalAigDsmFile)
                # print('Running Smooth Elevation for '+mission.missionId)
                # mission.processSmoothElevation()
                # print('Running Slope Aspect for '+mission.missionId)
                # mission.processSlopeAspect()
                # print('Preparing ATCOR inputs for '+mission.missionId)
                # mission.prepareAtcorInputs()
                # print('Running ATCOR reflectance retrieval for '+mission.missionId)
                # mission.processReflectance()
                # print('Running post ATCOR updates for '+mission.missionId)
                # mission.postAtcorUpdates()
                # print('Running L1 reflectance H5 for '+mission.missionId)
                # mission.postAtcorFileUpdates()
                # mission.processH5ReflectanceWriter()
        #         print('Running L1 Radiance Ortho for '+mission.missionId)
        #         mission.processOrthoRadiance(os.path.join(YSV.geoidFileDir,YSV.geoidFile),YSV.internalAigDsmFile,YSV.internalAigSapFile)
        #         print('Running L1 Radiance Ortho QA for '+mission.missionId)
        #         mission.generateOrthoQa()
        #         print('Running L1 Radiance H5 for '+mission.missionId)
        #         mission.processH5RadianceWriter()
        #         print('Generating L1 Radiance H5 RGB tifs for '+mission.missionId)
        #         mission.generateRgbRadianceTifs()
        #         print('Cleaning up radiance files for '+mission.missionId)
        #         mission.cleanupRadiance()
        #         print('Cleaning up slope / aspect files for '+mission.missionId)
        #         mission.cleanupSlopeAspect()
        #         print('Running L1 Reflectance Spectrometer Products for '+ mission.missionId)
        #         mission.processL1ReflectanceSpectrometer(YSV.internalAigDsmFile)
        #         print('Running L2 Spectrometer Products for '+ mission.missionId)
        #         mission.processL2SpectrometerProducts()
        #         mission.organizeReflectanceFiles()
        #         print('Running L2 Spectrometer Product QA for '+ mission.missionId)
        #         mission.processFlightlineQa()
        #         print('Cleaning L2 Spectrometer data for '+ mission.missionId)
        #         mission.cleanL2SpectrometerProducts()                
        #         print("Finished 2022072515")
        #     else:
        #         mission.spectrometerProcessWorkflow(os.path.join(YSV.geoidFileDir,YSV.geoidFile),YSV.internalAigDsmFile,YSV.internalAigSapFile,YSV.spectrometerProductQaList,YSV.Domain,YSV.visit)
        # YSV.processSpectrometerMosaicProducts()
        # YSV.downloadInternalLidarAigDsm()
        # YSV.calcAigDsmSlopeAspect()
        # YSV.getProductFiles()
        
        # for mission in YSV.missions:
        #     if mission.flightday == '2023041315':
        #         print("Working on 2023041315")
        #         print('Downloading Spectrometer Raw data for '+ mission.missionId)
        #         #mission.downloadDataRawSpectrometerProcessing()
        #         print('Running Spectrometer flightline boundaries for '+ mission.missionId)
        #         #mission.generateSpectrometerKmls()
        #         print('Running L1 Radiance Spectrometer Products for '+ mission.missionId)
        #         print('Running L1 DN to Radiance for '+mission.missionId)
        #         #mission.processRadiance()
        #         print('Running L1 Radiance QA for '+mission.missionId)
        #         #mission.processRadianceQa(YSV.Domain,YSV.visit)
        #         print('Running L1 Radiance Ortho for '+mission.missionId)
        #         mission.processOrthoRadiance(os.path.join(YSV.geoidFileDir,YSV.geoidFile),YSV.internalAigDsmFile,YSV.internalAigSapFile)
        #         print('Running L1 Radiance Ortho QA for '+mission.missionId)
        #         #mission.generateOrthoQa()            
        #     else:
        #         continue
        #YSV.processCamera() # Camera Run Only
        #YSV.processWaveformLidarWorkflow() # Lidar Run Only