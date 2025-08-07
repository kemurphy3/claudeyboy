# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:39:40 2020

@author: tgoulden


classes and functions used for AOP processing pipelines

"""


import inspect
from pykml import parser
import simplekml
from shapely.geometry import box, mapping, Polygon, Point
import shutil
import lxml.etree as et
import os, sys, sqlite3, fileinput
from osgeo import gdal
from datetime import datetime
from subprocess import Popen, PIPE, check_call
import geopandas as gpd
import pandas as pd
import rasterio as rio
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from rasterio.transform import from_origin
from rasterio.coords import BoundingBox
from rasterio.mask import mask
import h5py
import csv
import zipfile
#from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import numpy as np
from skimage import exposure
from scipy.ndimage import convolve
import envi_file as evFile
import warnings
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import psutil
import matlab.engine
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
import re
import PyPDF2
import pyodbc
from scipy.stats import gaussian_kde
from PIL import Image
from idlpy import IDL
from pyproj import Transformer
import math
import laspy
from contextlib import contextmanager
from scipy.special import erf
import traceback
from pathlib import Path




"""
Created on Tue Jan  7 11:39:40 2020

@author: tgoulden

classes and functions used for AOP processing pipelines

"""
import inspect
from pykml import parser
import simplekml
from shapely.geometry import box, mapping, Polygon, Point
import shutil
import lxml.etree as et
import os, sys, sqlite3, fileinput
from osgeo import gdal
from datetime import datetime
from subprocess import Popen, PIPE, check_call
import geopandas as gpd
import pandas as pd
import rasterio as rio
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from rasterio.transform import from_origin
from rasterio.coords import BoundingBox
from rasterio.mask import mask
import h5py
import csv
import zipfile
#from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import numpy as np
from skimage import exposure
from scipy.ndimage import convolve
import envi_file as evFile
import warnings
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import psutil
import matlab.engine
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
import re
import PyPDF2
import pyodbc
from scipy.stats import gaussian_kde
from PIL import Image
from idlpy import IDL
from pyproj import Transformer
import math
import laspy
from contextlib import contextmanager
from scipy.special import erf



warnings.filterwarnings("ignore")

num_cpus = (psutil.cpu_count(logical=False)-1)*2

#pipelineHomeDir = os.path.join('D:',os.sep,'Gold_Pipeline','ProcessingPipelines')

currentPath = __file__

pipelineHomeDir = os.path.join(currentPath.lower().split('processingpipelines')[0],'processingpipelines')

pythonLibHomeDir = os.path.join(pipelineHomeDir,'lib','Python')

nisPipelineBase = os.path.join(pipelineHomeDir,'NIS')
lidarPipelineBase = os.path.join(pipelineHomeDir,'Lidar')
sbetPipelineBase = os.path.join(pipelineHomeDir,'SBET')

nisLib = os.path.join(nisPipelineBase,'lib')
nisLibPython = os.path.join(nisLib,'Python')

atcorHomeDir = os.path.join('C:',os.path.sep,'atcor_4_v73')

lidarSrcDir = os.path.join(lidarPipelineBase,'src')
lidarSrcL1Dir = os.path.join(lidarSrcDir,'L1')
lidarSrcL1RieglDir = os.path.join(lidarSrcL1Dir,'Riegl')
lidarSrcL1OptechDir = os.path.join(lidarSrcL1Dir,'Optech')
lidarSrcL1WaveformDir = os.path.join(lidarSrcL1Dir,'WaveformQA')
lidarSrcL3Dir = os.path.join(lidarSrcDir,'L3')
lidarResDir = os.path.join(lidarPipelineBase,'res')
lidarRieglResDir = os.path.join(lidarResDir,'Riegl')
lidarOptechResDir = os.path.join(lidarResDir,'Optech')
lidarOptechSrcDir = os.path.join(lidarSrcL1OptechDir ,'src')

lidarLibDir = os.path.join(lidarPipelineBase,'lib')
lidarLibPythonDir = os.path.join(lidarLibDir,'Python')
lidarLibMatlabDir = os.path.join(lidarLibDir,'Matlab')

nisSrcDir = os.path.join(nisPipelineBase,'src')
nisSrcL1Dir = os.path.join(nisSrcDir,'L1')
dn2RadProcessingDir = os.path.join(nisSrcL1Dir,'DN2Radiance')
orthoProcessingDir =  os.path.join(nisSrcL1Dir,'Ortho')
atcorProcessingDir =  os.path.join(nisSrcL1Dir,'ATCOR')
H5ProcessingDir = os.path.join(nisSrcL1Dir,'H5Writers')
brdfProcessingDir = os.path.join(nisSrcL1Dir,'BRDF')

nisSpectrometerMiscCodeDir = os.path.join(nisSrcDir,'misc')
nisSpectrometerQaCodeDir =  os.path.join(nisSpectrometerMiscCodeDir,'QA')

nisSrcL3Dir = os.path.join(nisSrcDir,'L3')
nisSpectrometerMosaicDir = os.path.join(nisSrcL3Dir,'ReflectanceAlbedo')

if dn2RadProcessingDir not in sys.path:
    sys.path.insert(0, dn2RadProcessingDir)

if nisSpectrometerQaCodeDir not in sys.path:
    sys.path.insert(0, nisSpectrometerQaCodeDir)

#from Dn2RadProcessing import *

if pythonLibHomeDir not in sys.path:
    sys.path.insert(0, pythonLibHomeDir)

if nisLibPython not in sys.path:
    sys.path.insert(0, nisLibPython)

if nisSrcDir not in sys.path:
    sys.path.insert(0, nisSrcDir)

if H5ProcessingDir not in sys.path:
    sys.path.insert(0, H5ProcessingDir)

if brdfProcessingDir not in sys.path:
    sys.path.insert(0, brdfProcessingDir)

if lidarSrcL3Dir not in sys.path:
    sys.path.insert(0, lidarSrcL3Dir)

if lidarSrcL1RieglDir not in sys.path:
    sys.path.insert(0, lidarSrcL1RieglDir)

if lidarSrcL1OptechDir not in sys.path:
    sys.path.insert(0, lidarSrcL1OptechDir)

if lidarSrcL1WaveformDir not in sys.path:
    sys.path.insert(0, lidarSrcL1WaveformDir)

from H5WriterFunctionRadiance import H5WriterFunctionRadiance
from H5WriterFunction import H5WriterFunction
from apply_brdf_correct import apply_brdf_correct
from extract_height_chm import extract_height_chm
from WaveformQA_update import WaveformQa

# from aop_download_utils import *
from aop_gcs_download import *

from nisL1qaQcPipeline import nisL1qaQcPipeline

from shapefile_merge import shapefile_merge

from make_riegl_atm import make_atm_files

from make_optech_met_file import make_met_file

from Dn2RadProcessing import * 

@contextmanager

def qa_check_and_retry(write_func, qa_func, *args, max_retries=3, **kwargs):
    """
    Attempts to write a file using write_func, then checks it with qa_func.
    Retries up to max_retries if QA fails.
    """
    for attempt in range(1, max_retries+1):
        try:
            write_func(*args, **kwargs)
            if qa_func(*args, **kwargs):
                return True
            else:
                print(f"[QA] Attempt {attempt}: QA check failed, retrying...")
        except Exception as e:
            print(f"[QA] Attempt {attempt}: Exception during write or QA: {e}\n{traceback.format_exc()}")
        time.sleep(2)  # brief pause before retry
    print(f"[QA] All {max_retries} attempts failed for {args}.")
    return False

def qa_check_h5_file(reflectanceEnviFile, elevationEnviFile, shadowEnviFile, *args, **kwargs):
    # Check that the H5 file exists and can be opened/read
    h5_path = kwargs.get('outputDir', None)
    if h5_path is None:
        # Try to infer from args
        h5_path = args[4] if len(args) > 4 else None
    if not h5_path or not os.path.exists(h5_path):
        return False
    try:
        # Try to open and read a key dataset
        with h5py.File(h5_path, 'r') as f:
            keys = list(f.keys())
            if not keys:
                return False
    except Exception:
        return False
    return True

def qa_check_envi_file(file_path, *args, **kwargs):
    # Check that the ENVI/ATCOR file exists and is readable
    if not os.path.exists(file_path):
        return False
    try:
        with rio.open(file_path) as src:
            arr = src.read(1)
            if arr is None or arr.size == 0:
                return False
    except Exception:
        return False
    return True


def suppress_output():
    # Save the current stdout and stderr
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    try:
        # Redirect stdout and stderr to a null device (suppress output)
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        # Close the null device and restore stdout and stderr
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr

def gaussian_response_matrix(original_wavelengths, original_fwhms, target_wavelengths):
    """Compute the spectral response matrix for convolving the original spectrum."""
    
    sigma = original_fwhms / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
    response_matrix = np.zeros((len(target_wavelengths), len(original_wavelengths)))

    for i, target_wavelength in enumerate(target_wavelengths):
        response_matrix[i, :] = np.exp(-0.5 * ((original_wavelengths - target_wavelength) / sigma) ** 2)

    # Normalize rows so each target band sums to 1 (energy conservation)
    response_matrix /= response_matrix.sum(axis=1, keepdims=True)
    
    return response_matrix

def resampleSpectrum(original_wavelengths, original_response, original_fwhms, target_wavelengths):
    """Simulates how the spectrum would look at different wavelengths using spectral convolution."""

    # Generate the spectral response matrix
    response_matrix = gaussian_response_matrix(original_wavelengths, original_fwhms, target_wavelengths)
    
    # Reshape original response for matrix multiplication: (N_pixels, M_bands)
    original_shape = original_response.shape
    reshaped_response = original_response.reshape(-1, original_shape[2])  # (pixels, bands)

    # Apply spectral convolution using matrix multiplication
    resampled_response = np.dot(reshaped_response, response_matrix.T)

 # Reshape back to (rows, cols, bands)
    return resampled_response.reshape(original_shape[0], original_shape[1], len(target_wavelengths))
def generateResampleH5Spectrum(h5File,targetWavelengths,outFolder,outputType,filePathToEnviProjCs):
    
    reflectanceArray, metadata, wavelengths = h5refl2array(h5File,'Reflectance')
    
    resampledResponse = resampleSpectrum(wavelengths,reflectanceArray,metadata['fwhm'],targetWavelengths)

    if outputType == 'ENVI':
        
        outFile = os.path.join(outFolder,os.path.splitext(os.path.basename(h5File))[0])
        metadata["wavelengths"] = targetWavelengths
        writeRasterToEnvi(resampledResponse,'Reflectance', metadata, targetWavelengths,outFile,filePathToEnviProjCs)
    
    elif outputType == 'GTIF':
        
        outFile = os.path.join(outFolder,os.path.basename(h5File).replace('.h5','.tif'))
        
        writeRasterToGtif(resampledResponse, metadata, targetWavelengths,outFile)

def getTimeFromObs(inObsFile):
    
    inObsFile = os.path.splitext(inObsFile)[0]
    obsData,metadata = readEnviRaster(inObsFile)
    obsData[obsData == float(metadata["data_ignore_value"])] = np.nan
    
    return obsData[9,:,:]

def getSunAzimuthFromObs(inObsFile):
    
    inObsFile = os.path.splitext(inObsFile)[0]
    obsData,metadata = readEnviRaster(inObsFile)
    obsData[obsData == float(metadata["data_ignore_value"])] = np.nan
    
    return obsData[3,:,:]

def getSunZenithromObs(inObsFile):
    
    inObsFile = os.path.splitext(inObsFile)[0]
    obsData,metadata = readEnviRaster(inObsFile)
    obsData[obsData == float(metadata["data_ignore_value"])] = np.nan
    
    return obsData[4,:,:]

def convertObsToSca(inObsFile,outScaFile):
    
    inObsFile = os.path.splitext(inObsFile)[0]
    
    obsData,metadata = readEnviRaster(inObsFile)
    
    noData = float(metadata["data_ignore_value"])
    
    scaData = obsData[[10,1,2,0],:,:]
    
    scaData[2,:,:] = np.cos(np.radians(scaData[2,:,:]))*scaData[3,:,:]
    scaData[0,:,:] = scaData[0,:,:]*100
    scaData[1,:,:] = scaData[1,:,:]*10
    
    scaData[0,obsData[0,:,:] == noData] = 9100
    scaData[1,obsData[0,:,:] == noData] = noData
    scaData[2,obsData[0,:,:] == noData] = 0
    scaData[3,obsData[0,:,:] == noData] = noData
    
    metadata["data_ignore_value"] = 9100
    metadata["num_bands"] = 4
    scaData = scaData.astype(np.int16)
    
    metadata["data_type"] = getRasterGdalDtype(scaData.dtype)
    writeEnviRaster(outScaFile+'.bsq',np.moveaxis(scaData,[0,1,2],[2,0,1]),metadata,needsBandMetadata=False)
    
    return obsData,scaData,metadata
    
    
def updateEnviHeaderFromBiltoBsq(inBsqEnviHdrFile,outBsqEnviHdrFile=None):
    
    with open(inBsqEnviHdrFile, 'r') as file:
        lines = file.readlines()

        # Update the interleave field if found
        updated_lines = []
        for line in lines:
            if line.lower().startswith('interleave'):
                parts = line.split('=')
                current_interleave = parts[1].strip().lower()
                
                if current_interleave == 'bil':
                    updated_line = f'interleave = bsq\n'
                    updated_lines.append(updated_line)
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
    
        # Determine the output file (overwrite or new file)
        outBsqEnviHdrFile = outBsqEnviHdrFile if outBsqEnviHdrFile else inBsqEnviHdrFile
        
        # Write the updated header to the file
        with open(outBsqEnviHdrFile, 'w') as file:
            file.writelines(updated_lines)
            
def changeEnviDataType(inEnviFile,outDataType,dataTypeUpdate,outEnviHdrFile=None):

    inEnviFile = os.path.splitext(inEnviFile)[0]

    inEnviHdrFile = inEnviFile+'.hdr'    

    rasterData,metadata = readEnviRaster(inEnviFile)
    
    rasterData = rasterData.astype(outDataType)
    
    rasterData.tofile(os.path.splitext(inEnviHdrFile)[0])
    
    with open(inEnviHdrFile, 'r') as file:
        lines = file.readlines()

        # Update the interleave field if found
        updated_lines = []
        for line in lines:
            if line.lower().startswith('data type'):
                updated_line = f'data type = '+str(dataTypeUpdate)+'\n'
                updated_lines.append(updated_line)
            else:
                updated_lines.append(line)
    
        # Determine the output file (overwrite or new file)
        outEnviHdrFile = outEnviHdrFile if outEnviHdrFile else inEnviHdrFile
        
        # Write the updated header to the file
        with open(outEnviHdrFile, 'w') as file:
            file.writelines(updated_lines)

def updateNoDataEnviHeader(inEnviHdrFile,outEnviHdrFile=None,noDataUpdate=-9999):
    
    with open(inEnviHdrFile, 'r') as file:
        lines = file.readlines()

        # Update the interleave field if found
        updated_lines = []
        for line in lines:
            if line.lower().startswith('data ignore value'):
                updated_line = f'data ignore value = '+str(noDataUpdate)+'\n'
                updated_lines.append(updated_line)
            else:
                updated_lines.append(line)
    
        # Determine the output file (overwrite or new file)
        outEnviHdrFile = outEnviHdrFile if outEnviHdrFile else inEnviHdrFile
        
        # Write the updated header to the file
        with open(outEnviHdrFile, 'w') as file:
            file.writelines(updated_lines)

def quickBiltoBsq(inBsqEnviFile,outBsqEnviFile=None,updateNoData = None):
    
    inBsqEnviFile = os.path.splitext(inBsqEnviFile)[0]
    
    print(inBsqEnviFile)
        
    outBsqEnviFile = os.path.splitext(outBsqEnviFile)[0]
    
    t0 = time.time()
    
    rasterData,metadata = readEnviRaster(inBsqEnviFile)
    
    updateEnviHeaderFromBiltoBsq(inBsqEnviFile+'.hdr',outBsqEnviHdrFile=outBsqEnviFile+'.hdr')
    
    nband,nrow,ncolumn = rasterData.shape
    
    if updateNoData is not None:
            
        rasterData[rasterData == float(metadata["data_ignore_value"])] = updateNoData
    
    outBsqEnviFile = outBsqEnviFile if outBsqEnviFile else inBsqEnviFile

    with open(outBsqEnviFile+'.bsq', 'wb') as file:
        for band in range(rasterData.shape[0]):
            rasterData[band,:,:].tofile(file)

    
    print('Finished '+outBsqEnviFile+' in '+str(round(time.time()-t0,1))+' seconds')


def makeBarChart(data,outputFile = None,title=None,xlabel=None,ylabel=None):
    
    xvalues = np.arange(len(data),dtype=np.int8)
    
    plt.bar(xvalues, data, color='skyblue', edgecolor='darkblue')
    
    plt.xlabel(xlabel)
    
    plt.ylabel(ylabel)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.title(title)
    
    if outputFile is not None:
        plt.savefig(outputFile, dpi=600)
        
    plt.close()


def prepDataForPca(dataCube,findingPcaComponents = True):
    
    height, width, bands = dataCube.shape
    
    dataCube = dataCube.reshape(-1,bands)

    dataCube = pd.DataFrame(dataCube)


    if findingPcaComponents:
        # Remove rows where all values are NaN
        dataCube = dataCube.dropna(how='all')

    nanColumns = dataCube.loc[:, dataCube.isna().all(axis=0)]

# Step 4: Remove columns with NaNs
    dataCube = dataCube.drop(columns=nanColumns)
    
    if not findingPcaComponents:
        # Remove rows where all values are NaN
        dataCube = dataCube.to_numpy()
        dataCube = dataCube.reshape(height, width, len(dataCube[0,:]))
    else:
        dataCube = dataCube.to_numpy()

    return dataCube


def getLaserRange(sbet,lasData):
    
    eastingAircraft = np.interp(lasData.gps_time,sbet.time, sbet.easting)
    northingAircraft = np.interp(lasData.gps_time,sbet.time, sbet.northing)
    elevationAircraft = np.interp(lasData.gps_time,sbet.time, sbet.elevation)
    
    laserRange = ((eastingAircraft-lasData.x)**2+(northingAircraft-lasData.y)**2+(elevationAircraft-lasData.z)**2)**0.5
    
    return laserRange


def renameLidarFiles(filelist,outputDir,basename,fileSuffix=None,flightline=False,tiles=False):

    for file in filelist:
        splitFileName = os.path.basename(file).split('_')
        if flightline:
            fileId = (splitFileName[0]+'_'+splitFileName[1])
        elif tiles:
            fileId = splitFileName[3]+'_'+splitFileName[4]
        if '.' in fileId:
            fileId = os.path.splitext(fileId)[0]
        ext = os.path.splitext(os.path.basename(file))[1]
        if fileSuffix is None:
            os.rename(file,os.path.join(outputDir,basename+fileId+ext))
        else:
            os.rename(file,os.path.join(outputDir,basename+fileId+'_'+fileSuffix+ext))


def interpolateSbetDataToLidar(lasData,sbet):
    
    roll = np.interp(lasData.gps_time,sbet.time, sbet.roll)
    pitch = np.interp(lasData.gps_time,sbet.time, sbet.pitch)
    yaw = np.interp(lasData.gps_time,sbet.time, sbet.heading)
    xError = np.interp(lasData.gps_time,sbet.errorTime, sbet.xError)
    yError = np.interp(lasData.gps_time,sbet.errorTime, sbet.yError)
    zError = np.interp(lasData.gps_time,sbet.errorTime, sbet.zError)
    rollError = np.interp(lasData.gps_time,sbet.errorTime, sbet.rollError)/60*np.pi/180 #given in arc-min
    pitchError = np.interp(lasData.gps_time,sbet.errorTime, sbet.pitchError)/60*np.pi/180 #given in arc-min
    yawError = np.interp(lasData.gps_time,sbet.errorTime, sbet.headingError)/60*np.pi/180 #given in arc-min

    return roll,pitch,yaw,xError,yError,zError,rollError,pitchError,yawError

def calcSimulatedUncertainty(lasData,sbet,payload):
    
    laserRange = getLaserRange(sbet,lasData)
    
    roll,pitch,yaw,xError,yError,zError,rollError,pitchError,yawError = interpolateSbetDataToLidar(lasData,sbet)
    try:
        SAngle = (lasData.scan_angle*0.006)*np.pi/180
    except:
        SAngle = (lasData.scan_angle_rank*0.006)*np.pi/180
        
    varcov = np.zeros((len(laserRange),6))
   
    varcov[:,0] = xError**2+((np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll))*np.sin(SAngle)*laserRange+(-np.cos(yaw)*np.sin(pitch)*np.sin(roll)+np.sin(yaw)*np.cos(roll))*np.cos(SAngle)*laserRange)*rollError**2*np.conj((np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll))*np.sin(SAngle)*laserRange+(-np.cos(yaw)*np.sin(pitch)*np.sin(roll)+np.sin(yaw)*np.cos(roll))*np.cos(SAngle)*laserRange)+(np.cos(yaw)*np.cos(pitch)*np.sin(roll)*np.sin(SAngle)*laserRange+np.cos(yaw)*np.cos(pitch)*np.cos(roll)*np.cos(SAngle)*laserRange)*pitchError**2*np.conj(np.cos(yaw)*np.cos(pitch)*np.sin(roll)*np.sin(SAngle)*laserRange+np.cos(yaw)*np.cos(pitch)*np.cos(roll)*np.cos(SAngle)*laserRange)+((-np.sin(yaw)*np.sin(pitch)*np.sin(roll)-np.cos(yaw)*np.cos(roll))*np.sin(SAngle)*laserRange+(np.cos(yaw)*np.sin(roll)-np.sin(yaw)*np.sin(pitch)*np.cos(roll))*np.cos(SAngle)*laserRange)*yawError**2*np.conj((-np.sin(yaw)*np.sin(pitch)*np.sin(roll)-np.cos(yaw)*np.cos(roll))*np.sin(SAngle)*laserRange+(np.cos(yaw)*np.sin(roll)-np.sin(yaw)*np.sin(pitch)*np.cos(roll))*np.cos(SAngle)*laserRange)+((np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll))*np.cos(SAngle)*laserRange-(np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll))*np.sin(SAngle)*laserRange)*payload.lidarScanAngleError**2*np.conj((np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll))*np.cos(SAngle)*laserRange-(np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll))*np.sin(SAngle)*laserRange)+((np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll))*np.sin(SAngle)+(np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll))*np.cos(SAngle))*payload.laserRangeError**2*np.conj((np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll))*np.sin(SAngle)+(np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll))*np.cos(SAngle))
    varcov[:,1] = ((np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll))*np.sin(SAngle)*laserRange+(-np.cos(yaw)*np.sin(pitch)*np.sin(roll)+np.sin(yaw)*np.cos(roll))*np.cos(SAngle)*laserRange)*rollError**2*np.conj((np.cos(yaw)*np.sin(roll)-np.sin(yaw)*np.sin(pitch)*np.cos(roll))*np.sin(SAngle)*laserRange+(np.cos(yaw)*np.cos(roll)+np.sin(yaw)*np.sin(pitch)*np.sin(roll))*np.cos(SAngle)*laserRange)+(np.cos(yaw)*np.cos(pitch)*np.sin(roll)*np.sin(SAngle)*laserRange+np.cos(yaw)*np.cos(pitch)*np.cos(roll)*np.cos(SAngle)*laserRange)*pitchError**2*np.conj(-np.sin(yaw)*np.cos(pitch)*np.sin(roll)*np.sin(SAngle)*laserRange-np.sin(yaw)*np.cos(pitch)*np.cos(roll)*np.cos(SAngle)*laserRange)+((-np.sin(yaw)*np.sin(pitch)*np.sin(roll)-np.cos(yaw)*np.cos(roll))*np.sin(SAngle)*laserRange+(np.cos(yaw)*np.sin(roll)-np.sin(yaw)*np.sin(pitch)*np.cos(roll))*np.cos(SAngle)*laserRange)*yawError**2*np.conj((-np.cos(yaw)*np.sin(pitch)*np.sin(roll)+np.sin(yaw)*np.cos(roll))*np.sin(SAngle)*laserRange+(-np.sin(yaw)*np.sin(roll)-np.cos(yaw)*np.sin(pitch)*np.cos(roll))*np.cos(SAngle)*laserRange)+((np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll))*np.cos(SAngle)*laserRange-(np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll))*np.sin(SAngle)*laserRange)*payload.lidarScanAngleError**2*np.conj((-np.sin(yaw)*np.sin(pitch)*np.sin(roll)-np.cos(yaw)*np.cos(roll))*np.cos(SAngle)*laserRange-(np.cos(yaw)*np.sin(roll)-np.sin(yaw)*np.sin(pitch)*np.cos(roll))*np.sin(SAngle)*laserRange)+((np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll))*np.sin(SAngle)+(np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll))*np.cos(SAngle))*payload.laserRangeError**2*np.conj((-np.sin(yaw)*np.sin(pitch)*np.sin(roll)-np.cos(yaw)*np.cos(roll))*np.sin(SAngle)+(np.cos(yaw)*np.sin(roll)-np.sin(yaw)*np.sin(pitch)*np.cos(roll))*np.cos(SAngle))
    varcov[:,2] = ((np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll))*np.sin(SAngle)*laserRange+(-np.cos(yaw)*np.sin(pitch)*np.sin(roll)+np.sin(yaw)*np.cos(roll))*np.cos(SAngle)*laserRange)*rollError**2*np.conj(-np.cos(pitch)*np.cos(roll)*np.sin(SAngle)*laserRange+np.cos(pitch)*np.sin(roll)*np.cos(SAngle)*laserRange)+(np.cos(yaw)*np.cos(pitch)*np.sin(roll)*np.sin(SAngle)*laserRange+np.cos(yaw)*np.cos(pitch)*np.cos(roll)*np.cos(SAngle)*laserRange)*pitchError**2*np.conj(np.sin(pitch)*np.sin(roll)*np.sin(SAngle)*laserRange+np.sin(pitch)*np.cos(roll)*np.cos(SAngle)*laserRange)+((np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll))*np.cos(SAngle)*laserRange-(np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll))*np.sin(SAngle)*laserRange)*payload.lidarScanAngleError**2*np.conj(-np.cos(pitch)*np.sin(roll)*np.cos(SAngle)*laserRange+np.cos(pitch)*np.cos(roll)*np.sin(SAngle)*laserRange)+((np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll))*np.sin(SAngle)+(np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll))*np.cos(SAngle))*payload.laserRangeError**2*np.conj(-np.cos(pitch)*np.sin(roll)*np.sin(SAngle)-np.cos(pitch)*np.cos(roll)*np.cos(SAngle))
    varcov[:,3] = yError**2+((np.cos(yaw)*np.sin(roll)-np.sin(yaw)*np.sin(pitch)*np.cos(roll))*np.sin(SAngle)*laserRange+(np.cos(yaw)*np.cos(roll)+np.sin(yaw)*np.sin(pitch)*np.sin(roll))*np.cos(SAngle)*laserRange)*rollError**2*np.conj((np.cos(yaw)*np.sin(roll)-np.sin(yaw)*np.sin(pitch)*np.cos(roll))*np.sin(SAngle)*laserRange+(np.cos(yaw)*np.cos(roll)+np.sin(yaw)*np.sin(pitch)*np.sin(roll))*np.cos(SAngle)*laserRange)+(-np.sin(yaw)*np.cos(pitch)*np.sin(roll)*np.sin(SAngle)*laserRange-np.sin(yaw)*np.cos(pitch)*np.cos(roll)*np.cos(SAngle)*laserRange)*pitchError**2*np.conj(-np.sin(yaw)*np.cos(pitch)*np.sin(roll)*np.sin(SAngle)*laserRange-np.sin(yaw)*np.cos(pitch)*np.cos(roll)*np.cos(SAngle)*laserRange)+((-np.cos(yaw)*np.sin(pitch)*np.sin(roll)+np.sin(yaw)*np.cos(roll))*np.sin(SAngle)*laserRange+(-np.sin(yaw)*np.sin(roll)-np.cos(yaw)*np.sin(pitch)*np.cos(roll))*np.cos(SAngle)*laserRange)*yawError**2*np.conj((-np.cos(yaw)*np.sin(pitch)*np.sin(roll)+np.sin(yaw)*np.cos(roll))*np.sin(SAngle)*laserRange+(-np.sin(yaw)*np.sin(roll)-np.cos(yaw)*np.sin(pitch)*np.cos(roll))*np.cos(SAngle)*laserRange)+((-np.sin(yaw)*np.sin(pitch)*np.sin(roll)-np.cos(yaw)*np.cos(roll))*np.cos(SAngle)*laserRange-(np.cos(yaw)*np.sin(roll)-np.sin(yaw)*np.sin(pitch)*np.cos(roll))*np.sin(SAngle)*laserRange)*payload.lidarScanAngleError**2*np.conj((-np.sin(yaw)*np.sin(pitch)*np.sin(roll)-np.cos(yaw)*np.cos(roll))*np.cos(SAngle)*laserRange-(np.cos(yaw)*np.sin(roll)-np.sin(yaw)*np.sin(pitch)*np.cos(roll))*np.sin(SAngle)*laserRange)+((-np.sin(yaw)*np.sin(pitch)*np.sin(roll)-np.cos(yaw)*np.cos(roll))*np.sin(SAngle)+(np.cos(yaw)*np.sin(roll)-np.sin(yaw)*np.sin(pitch)*np.cos(roll))*np.cos(SAngle))*payload.laserRangeError**2*np.conj((-np.sin(yaw)*np.sin(pitch)*np.sin(roll)-np.cos(yaw)*np.cos(roll))*np.sin(SAngle)+(np.cos(yaw)*np.sin(roll)-np.sin(yaw)*np.sin(pitch)*np.cos(roll))*np.cos(SAngle))
    varcov[:,4] = ((np.cos(yaw)*np.sin(roll)-np.sin(yaw)*np.sin(pitch)*np.cos(roll))*np.sin(SAngle)*laserRange+(np.cos(yaw)*np.cos(roll)+np.sin(yaw)*np.sin(pitch)*np.sin(roll))*np.cos(SAngle)*laserRange)*rollError**2*np.conj(-np.cos(pitch)*np.cos(roll)*np.sin(SAngle)*laserRange+np.cos(pitch)*np.sin(roll)*np.cos(SAngle)*laserRange)+(-np.sin(yaw)*np.cos(pitch)*np.sin(roll)*np.sin(SAngle)*laserRange-np.sin(yaw)*np.cos(pitch)*np.cos(roll)*np.cos(SAngle)*laserRange)*pitchError**2*np.conj(np.sin(pitch)*np.sin(roll)*np.sin(SAngle)*laserRange+np.sin(pitch)*np.cos(roll)*np.cos(SAngle)*laserRange)+((-np.sin(yaw)*np.sin(pitch)*np.sin(roll)-np.cos(yaw)*np.cos(roll))*np.cos(SAngle)*laserRange-(np.cos(yaw)*np.sin(roll)-np.sin(yaw)*np.sin(pitch)*np.cos(roll))*np.sin(SAngle)*laserRange)*payload.lidarScanAngleError**2*np.conj(-np.cos(pitch)*np.sin(roll)*np.cos(SAngle)*laserRange+np.cos(pitch)*np.cos(roll)*np.sin(SAngle)*laserRange)+((-np.sin(yaw)*np.sin(pitch)*np.sin(roll)-np.cos(yaw)*np.cos(roll))*np.sin(SAngle)+(np.cos(yaw)*np.sin(roll)-np.sin(yaw)*np.sin(pitch)*np.cos(roll))*np.cos(SAngle))*payload.laserRangeError**2*np.conj(-np.cos(pitch)*np.sin(roll)*np.sin(SAngle)-np.cos(pitch)*np.cos(roll)*np.cos(SAngle))
    varcov[:,5] = zError**2+(-np.cos(pitch)*np.cos(roll)*np.sin(SAngle)*laserRange+np.cos(pitch)*np.sin(roll)*np.cos(SAngle)*laserRange)*rollError**2*np.conj(-np.cos(pitch)*np.cos(roll)*np.sin(SAngle)*laserRange+np.cos(pitch)*np.sin(roll)*np.cos(SAngle)*laserRange)+(np.sin(pitch)*np.sin(roll)*np.sin(SAngle)*laserRange+np.sin(pitch)*np.cos(roll)*np.cos(SAngle)*laserRange)*pitchError**2*np.conj(np.sin(pitch)*np.sin(roll)*np.sin(SAngle)*laserRange+np.sin(pitch)*np.cos(roll)*np.cos(SAngle)*laserRange)+(-np.cos(pitch)*np.sin(roll)*np.cos(SAngle)*laserRange+np.cos(pitch)*np.cos(roll)*np.sin(SAngle)*laserRange)*payload.lidarScanAngleError**2*np.conj(-np.cos(pitch)*np.sin(roll)*np.cos(SAngle)*laserRange+np.cos(pitch)*np.cos(roll)*np.sin(SAngle)*laserRange)+(-np.cos(pitch)*np.sin(roll)*np.sin(SAngle)-np.cos(pitch)*np.cos(roll)*np.cos(SAngle))*payload.laserRangeError**2*np.conj(-np.cos(pitch)*np.sin(roll)*np.sin(SAngle)-np.cos(pitch)*np.cos(roll)*np.cos(SAngle))
    
    maxHorzError = np.zeros_like(varcov[:,0])
    
    counter = 0
    
    trace = varcov[:,0]+varcov[:,3]
    determinant = varcov[:,0]*varcov[:,3]-varcov[:,1]**2
    discriminant = trace**2 - 4 * determinant
    
    eigenvalue1 = (trace + discriminant**0.5) / 2
    eigenvalue2 = (trace - discriminant**0.5) / 2
    
    maxHorzError = np.max(np.vstack((eigenvalue1,eigenvalue2)).T,axis=1)
    
    return maxHorzError**0.5,varcov[:,5]**0.5

def generateSimulatedLidarUncertainty(lasFile,sbet,payload,outputDir):
    
    lasData = laspy.read(lasFile)

    horzError, vertError = calcSimulatedUncertainty(lasData,sbet,payload)
    
    lasData.z = horzError

    lasData.write(os.path.join(outputDir,os.path.basename(lasFile).replace('.las','_horizontal_uncertainty.las')))
    
    lasData.z = vertError
    
    lasData.write(os.path.join(outputDir,os.path.basename(lasFile).replace('.las','_vertical_uncertainty.las')))

def latlonToUtm(latitudes, longitudes, latLongEpsg,utmZoneEpsg):
    # Initialize arrays to store results
    utm_easting = np.zeros(latitudes.shape)
    utm_northing = np.zeros(latitudes.shape)

    transformer = Transformer.from_crs(latLongEpsg,utmZoneEpsg)
    easting, northing = transformer.transform(latitudes, longitudes)

    return easting, northing


def readFlatBinaryFile(file,dtype,numCols=1):
    
    # Open the binary file in read mode
    with open(file, 'rb') as file:
    # Read the entire file into a numpy array of doubles
        data = np.fromfile(file, dtype=dtype)
    # Determine the number of rows
    numRows = data.size // numCols
    # Reshape the array to have the correct number of columns
    data = data.reshape((numRows, numCols))
    
    return data

def getCleanSpectra(reflectanceArray, metadata, wavelengths):
    
    badBandMask = findBadBands(wavelengths,(metadata['bad_band_window1'],metadata['bad_band_window2'],np.array((380,435),dtype=np.float32),np.array((2440,2520),dtype=np.float32))).astype(np.int16)
    
    badBandMaskNoData = np.zeros_like(badBandMask)
    
    badBandMaskNoData[badBandMask==0] = metadata['noDataVal']
    
    reflectanceArray = reflectanceArray.astype(np.int16)
    
    return np.multiply(reflectanceArray,badBandMask[None,None,:])+badBandMaskNoData

def getCleanNormalizedDataCube(reflectanceArray,metadata,wavelengths):

    cleanDataCube = getCleanSpectra(reflectanceArray, metadata, wavelengths)

    #cleanDataCube[cleanDataCube==metadata['noDataVal']] = np.nan

    normalizedDataCube = normalizeSpectraSumSquares(cleanDataCube,metadata['noDataVal'])

    return normalizedDataCube

def getH5ReflectanceLogs(h5FileIn):
    
    #Read in reflectance hdf5 file 
    h5File = h5py.File(h5FileIn,'r')

    #Get the site name
    fileAttrsString = str(list(h5File.items()))
    fileAttrsStringSplit = fileAttrsString.split("'")
    sitename = fileAttrsStringSplit[1]
    
    ATCOR_log = h5File[sitename]['Reflectance']['Metadata']['Logs']['ATCOR_Processing_Log']
    ATCOR_inn = h5File[sitename]['Reflectance']['Metadata']['Logs']['ATCOR_input_file']
    ATCOR_shadow_processing_log = h5File[sitename]['Reflectance']['Metadata']['Logs']['Shadow_Processing_Log']
    ATCOR_skyview_processing_log = h5File[sitename]['Reflectance']['Metadata']['Logs']['Skyview_Processing_Log']
    
    rasterArray = h5File[sitename]['Reflectance']['Reflectance_Data']
    brdfCorrCheck = str(rasterArray.attrs['BRDF Correction'])
    
    if brdfCorrCheck == 'True': 
        BRDF_coeffs = h5File[sitename]['Reflectance']['Metadata']['Logs']['BRDF_COEFFS_JSON_for_HyTools']
        BRDF_config = h5File[sitename]['Reflectance']['Metadata']['Logs']['BRDF_Config_JSON_for_HyTools']
    else:
        BRDF_coeffs = None
        BRDF_config = None
    
    avSolarAzimuthAngles = h5File[sitename]['Reflectance']['Metadata']['Logs']['Solar_Azimuth_Angle']
    avSolarZenithAngles = h5File[sitename]['Reflectance']['Metadata']['Logs']['Solar_Zenith_Angle']
    
    return ATCOR_log,ATCOR_inn,ATCOR_shadow_processing_log,ATCOR_skyview_processing_log,BRDF_coeffs,BRDF_config,avSolarAzimuthAngles,avSolarZenithAngles

def plot_three_lines(x_values, y_values_list, prefix, output_filename):
    """
    Plots three lines with specified colors and labels, and saves the plot to a file.

    Parameters:
    x_values (numpy array): 1D array for the x-axis values.
    y_values_list (list of numpy arrays): List of three 1D arrays for the y-axis values.
    prefix (str): Prefix for the legend labels.
    output_filename (str): Filename to save the plot.b
    """
    # Check if the input list has exactly three elements
    if len(y_values_list) != 3:
        raise ValueError("y_values_list must contain exactly three numpy arrays.")

    # Define colors for the lines
    colors = ['blue', 'green', 'red']

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each line
    for i, y_values in enumerate(y_values_list):
        plt.plot(x_values, y_values, color=colors[i], label=f'{prefix} sample {i+1}')

    # Add labels and title
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance (%)')
    # Add legend
    plt.ylim([0,np.nanmax(y_values_list)+np.nanmax(y_values_list)*0.1])
    plt.legend()
    # Add grid
    plt.grid(True)
    # Save the plot to a file with the specified resolution
    plt.savefig(output_filename, dpi=600)

def getSpectraAtValue(h5File,raster,value):

    spectras = []    

    indexesAtValue = np.where(np.squeeze(raster) == value)    

    if len(indexesAtValue[0]) > 10:
        
        selectionMultiplier = np.floor(len(indexesAtValue[0])/5) 
        
        
        for counter in np.arange(3):    
            indexRow = indexesAtValue[0][int((counter+1)*selectionMultiplier)]
            indexCol = indexesAtValue[1][int((counter+1)*selectionMultiplier)]
            spatialIndexesToRead = [[None, None], [None, None]]
            spatialIndexesToRead[0][0] = int(indexRow) 
            spatialIndexesToRead[0][1] = int(indexRow+1)
            spatialIndexesToRead[1][0] = int(indexCol) 
            spatialIndexesToRead[1][1] = int(indexCol+1)
            
            spectra, metadata, wavelengths=h5refl2array(h5File,'Reflectance',spatialIndexesToRead = spatialIndexesToRead)
            
            badBandMask = findBadBands(wavelengths,(metadata['bad_band_window1'],metadata['bad_band_window2'],np.array((380,435),dtype=np.float32),np.array((2440,2520),dtype=np.float32))).astype(np.float32)
            
            spectra = np.squeeze(spectra.astype(np.float32))
            
            spectra[badBandMask==0] = np.nan
                    
            spectras.append(spectra)

    return spectras

def generateReflectancePlots(h5File,outPngPrefix,nirBand,swirBand,nirThreshold,swirThreshold):
    
    waterMask,metadata = getWaterMask(h5File,nirBand,swirBand,nirThreshold,swirThreshold)
        
    waterSpectras = getSpectraAtValue(h5File,waterMask,1)
    
    array, metadata, wavelengths=h5refl2array(h5File,'Reflectance',onlyMetadata=True)
    
    if waterSpectras:
    
        plot_three_lines(wavelengths, waterSpectras, 'Water', outPngPrefix+'_water_spectra.png')
    
    ddvArray, metadata, wavelengths=h5refl2array(h5File,'Dark_Dense_Vegetation_Classification')
    
    vegSpectras = getSpectraAtValue(h5File,ddvArray,2)
    
    if vegSpectras:
    
        plot_three_lines(wavelengths, vegSpectras, 'Vegetation', outPngPrefix+'_veg_spectra.png')
    

def generateWaterMask(h5File,outTif,nirBand,swirBand,nirThreshold,swirThreshold):
    
    waterMask,metadata = getWaterMask(h5File,nirBand,swirBand,nirThreshold,swirThreshold)
    
    EPSGCode,MosaicExtents,RasterCellHeight,RasterCellWidth,NoData = getGtifMetadataFromH5Metadata(metadata)
    
    writeRasterToTif(outTif,waterMask,MosaicExtents,EPSGCode,NoData,RasterCellWidth,RasterCellHeight)

def getWaterMask(h5File,nirBand,swirBand,nirThreshold,swirThreshold):
    
    array,metadata,wavelengths = h5refl2array(h5File, 'Reflectance', onlyMetadata = True)
    
    indexOfNirBand = np.where(np.abs(wavelengths-nirBand) == np.min(np.abs(wavelengths-nirBand)))[0]
    nirArray, metadata, wavelengths=h5refl2array(h5File,'Reflectance',bandIndexesToRead = [int(indexOfNirBand),int(indexOfNirBand+1)])
        
    indexOfSwirBand = np.where(np.abs(wavelengths-swirBand) == np.min(np.abs(wavelengths-swirBand)))[0]
    swirArray, metadata, wavelengths=h5refl2array(h5File,'Reflectance',bandIndexesToRead = [int(indexOfSwirBand),int(indexOfSwirBand+1)])
    
    boolNis = nirArray<100
    boolSwir = swirArray<50
    
    waterMask = (boolNis*boolSwir).astype(np.int)
    
    waterMask[nirArray==metadata['noDataVal']] = metadata['noDataVal'] 
    
    return waterMask,metadata

def getSpectrometerRGBQaReportSummaryFiles(mosaicDir,redMosaicFile,greenMosaicFile,blueMosaicFile,nirMosaicFile):

    summaryFilename = os.path.join(mosaicDir,'summary_data_RGB.txt')
    rgbNirSummaryDicts = []
    rgbNirNames=['Red','Green','Blue','NIR']
    rgbNirSummaryDicts.append(getRasterStats(redMosaicFile))
    rgbNirSummaryDicts.append(getRasterStats(greenMosaicFile))
    rgbNirSummaryDicts.append(getRasterStats(blueMosaicFile))
    rgbNirSummaryDicts.append(getRasterStats(nirMosaicFile))
    
    with open(summaryFilename, 'w') as f:
        f.write('Product,Min,Max,Mean,Median,Below Limit (%),Above Limit (%),Equal to Zero (%)\n')
        for bandName,summaryDict in zip(rgbNirNames,rgbNirSummaryDicts):
            f.write(bandName+' Mosaic,')
            values = ",".join(map(str, summaryDict.values()))
            f.write(values+'\n')

def getSpectrometerReflectanceDifferenceSummaryFiles(mosaicDir,rmsMosaicFile,maxDiffMosaicFile,maxDiffWavelengthMosaicFile):

    summaryFilename = os.path.join(mosaicDir,'summary_data_ReflectanceDifference.txt')
    reflectanceDiffsSummaryDicts = []
    reflectanceDiffNames=['Reflectance RMS','Reflectance Max Difference','Reflectance Max Difference WL']
    reflectanceDiffsSummaryDicts.append(getRasterStats(rmsMosaicFile))
    reflectanceDiffsSummaryDicts.append(getRasterStats(maxDiffMosaicFile))
    reflectanceDiffsSummaryDicts.append(getRasterStats(maxDiffWavelengthMosaicFile))
    
    with open(summaryFilename, 'w') as f:
        for bandName,summaryDict in zip(reflectanceDiffNames,reflectanceDiffsSummaryDicts):
            f.write(bandName+' Mosaic,')
            values = ",".join(map(str, summaryDict.values()))
            f.write(values+'\n')

def getSpectrometerAncillaryQaReportSummaryFiles(mosaicDir,mosaicPrefix,ancillaryProductList):

    for spectrometerAncillaryProduct in ancillaryProductList:
        
        mosaicFile = os.path.join(mosaicDir,mosaicPrefix+'_'+spectrometerAncillaryProduct+'_mosaic.tif')
        mosaicSummaryStatsDict = getRasterStats(mosaicFile)
        summaryFilename = os.path.join(mosaicDir,'summary_data_'+spectrometerAncillaryProduct+'.txt')
        with open(summaryFilename, 'w') as f:
            f.write(spectrometerAncillaryProduct+' Mosaic,')
            values = ",".join(map(str, mosaicSummaryStatsDict.values()))
            f.write(values)

def getSpectrometerQaReportSummaryFiles(mosaicFolder,mosaicPrefix,productQaList,previousYear = False, firstYear = False):
    
    if previousYear:
        mosaicPrefixWithLastYear = mosaicPrefix
        mosaicPrefix = mosaicPrefix.split('_')[0]+'_'+mosaicPrefix.split('_')[1]+'_'+mosaicPrefix.split('_')[2]
    
    for spectrometerProduct in productQaList:
        
        mosaicFile = os.path.join(mosaicFolder,mosaicPrefix+'_'+spectrometerProduct+'_mosaic.tif')
        if spectrometerProduct != 'Albedo':
            errorFile = os.path.join(mosaicFolder,mosaicPrefix+'_'+spectrometerProduct+'_error_mosaic.tif')
        
        if not firstYear:
            if previousYear:
                differenceFile = os.path.join(mosaicFolder,mosaicPrefixWithLastYear+'_'+spectrometerProduct+'_difference_mosaic.tif')
            else:
                differenceFile = os.path.join(mosaicFolder,mosaicPrefix+'_'+spectrometerProduct+'_difference_mosaic.tif')
                
        mosaicSummaryStatsDict = getRasterStats(mosaicFile)
        errorMosaicSummaryStatsDict = getRasterStats(errorFile)
        
        if not firstYear:
        
            differenceSummaryStatsDict = getRasterStats(differenceFile)
    
        summaryFilename = os.path.join(mosaicFolder,'summary_data_'+spectrometerProduct+'.txt')
    
        with open(summaryFilename, 'w') as f:
            f.write(spectrometerProduct+' Mosaic,')
            values = ",".join(map(str, mosaicSummaryStatsDict.values()))
            f.write(values+'\n')
            if spectrometerProduct == 'Albedo':
                pass
            else:
                f.write(spectrometerProduct+' Error Mosaic,')
                values = ",".join(map(str, errorMosaicSummaryStatsDict.values()))
                f.write(values+'\n')
            
            if not firstYear:
                f.write(spectrometerProduct+' Difference Mosaic,')
                values = ",".join(map(str, differenceSummaryStatsDict.values()))
                f.write(values)  

def getSpectrometerQaReportMaps(inputFiles,colorMapList,defaultCmap,errorCmap,differenceCmap):

    colorMaps = []

    for mosaicFile in inputFiles:
        
        cmap=defaultCmap    
        for rasterName in colorMapList:
            if rasterName in os.path.basename(mosaicFile): 
                cmap=colorMapList[rasterName]
        
        if '_error' in os.path.basename(mosaicFile):
            cmap=errorCmap
        elif 'difference' in  os.path.basename(mosaicFile):
            cmap=differenceCmap
            
        colorMaps.append(cmap)
     
    #for inputFile,colorMap in zip(inputFiles,colorMaps):
        #gtifFileAndCmap = zip(inputFile,colorMap)
        #plotGtifHelper(inputFile,colorMap) 
    with Pool(processes=30) as pool:
        processFunction = partial(plotGtifHelper)
        pool.map(processFunction,zip(inputFiles,colorMaps))
                

def generateAncillaryRastersFromH5s(ancillaryRasterList,reflectanceFiles,tempFolder,outputFolder,outPrefix,mosaicType):

    for h5File in reflectanceFiles:
        
        for ancillaryRaster in ancillaryRasterList:
            outTif = os.path.join(tempFolder,os.path.basename(h5File).replace('reflectance.h5',ancillaryRaster+'.tif'))
            convertH5RasterToGtif(h5File,ancillaryRaster,outTif)
            
    for ancillaryRaster in ancillaryRasterList:
        
        ancillaryRasterFiles = collectFilesInPath(tempFolder,ancillaryRaster+'.tif')
        
        generateMosaic(ancillaryRasterFiles,os.path.join(outputFolder,outPrefix+'_'+ancillaryRaster+'_mosaic.tif'),mosaicType=mosaicType)

def generateRgbCumulativeDistribution(inputRgbFile,outputPngPlotFile):
    
    with rio.open(inputRgbFile, 'r') as src:
        metadata = src.meta.copy()
        rgbImage = src.read()
        nodata = src.nodata
    
    red = rgbImage[0,:,:].flatten()
    green = rgbImage[1,:,:].flatten()
    blue = rgbImage[2,:,:].flatten()
    
    red = red[red!=nodata]
    green = green[green!=nodata]
    blue = blue[blue!=nodata]

    red=getPercentileRangeVector(red,5,95)
    green=getPercentileRangeVector(green,5,95)
    blue=getPercentileRangeVector(blue,5,95)
    
    plot_color_histograms(red/100, green/100, blue/100, os.path.join(outputPngPlotFile+'_RGB_histogram.png'))

def generateRgbAndNirTifMosaicFromH5s(SpectrometerL3ReflectanceFiles,outputDir,outputPrefix,mosaicType):
    
    redMosaicFilename = os.path.join(outputDir,outputPrefix+'_red_mosaic.tif')
    greenMosaicFileName = os.path.join(outputDir,outputPrefix+'_green_mosaic.tif')
    blueMosaicFileName = os.path.join(outputDir,outputPrefix+'_blue_mosaic.tif')
    nirMosaicFileName = os.path.join(outputDir,outputPrefix+'_NIR_mosaic.tif')
    rgbMosaicFileName = os.path.join(outputDir,outputPrefix+'_RGB_mosaic.tif')
    nirgbMosaicFileName = os.path.join(outputDir,outputPrefix+'_NIRGB_mosaic.tif')
    
    mosaicH5Band(SpectrometerL3ReflectanceFiles,redMosaicFilename,[54,55],mosaicType = mosaicType)
    
    mosaicH5Band(SpectrometerL3ReflectanceFiles,greenMosaicFileName,[34,35],mosaicType = mosaicType)
    
    mosaicH5Band(SpectrometerL3ReflectanceFiles,blueMosaicFileName,[18,19],mosaicType = mosaicType)
    
    mosaicH5Band(SpectrometerL3ReflectanceFiles,nirMosaicFileName,[85,86],mosaicType = mosaicType)
    
    stackTifImages([redMosaicFilename,greenMosaicFileName,blueMosaicFileName],rgbMosaicFileName)
    
    stackTifImages([nirMosaicFileName,greenMosaicFileName,blueMosaicFileName],nirgbMosaicFileName)

def getPercentileRangeVector(vector,lowerPercentile,upperPercentile):
    
    # Calculate the 5th and 95th percentiles
    percentile_5th = np.percentile(vector, lowerPercentile)
    percentile_95th = np.percentile(vector, upperPercentile)
    
    # Filter the vector to be within the 5th and 95th percentiles
    return vector[(vector >= percentile_5th) & (vector <= percentile_95th)]

def plot_color_histograms(redValues, greenValues, blueValues, outputFile):
    # Create histograms for each color
    subsampleFactor = 50
    redvaluesSubsampled = redValues[::subsampleFactor]
    greenValuesSubsampled = greenValues[::subsampleFactor]
    blueValuesSubsampled = blueValues[::subsampleFactor]

    redKde = gaussian_kde(redvaluesSubsampled)
    greenKde = gaussian_kde(greenValuesSubsampled)
    blueKde = gaussian_kde(blueValuesSubsampled)
    
    # Generate values for smooth lines
    xValuesRed = np.linspace(min(redValues),max(redValues),1000)
    xValuesBlue = np.linspace(min(blueValues),max(blueValues),1000)
    xValuesGreen = np.linspace(min(greenValues),max(greenValues),1000)
    
    redKdeScaled = redKde(xValuesRed) / np.sum(redKde(xValuesRed))
    blueKdeScaled = blueKde(xValuesBlue) / np.sum(blueKde(xValuesBlue))
    greenKdeScaled = greenKde(xValuesGreen) / np.sum(greenKde(xValuesGreen))
    
    # Plot smooth lines
    plt.plot(xValuesRed*100, redKdeScaled*100, color='red', label='Red', linewidth=3)
    plt.plot(xValuesGreen*100, greenKdeScaled*100, color='green', label='Green', linewidth=3)
    plt.plot(xValuesBlue*100, blueKdeScaled*100, color='blue', label='Blue', linewidth=3)

    plt.ylim(0,1.1*max(max(redKdeScaled*100),max(blueKdeScaled*100),max(greenKdeScaled*100)))
    # Add labels and legend
    plt.xlabel('Reflectance (%)')
    plt.ylabel('Relative Frequency (%)')
    plt.legend()

    # Save plot as PNG with 300 dpi
    plt.savefig(outputFile, dpi=300)

def stackTifImages(imageFileList,stackFileNameOut):
    
    imagesToStack = []
    
    for imageFile in imageFileList:
        
        with rio.open(imageFile, 'r') as src:
            metadata = src.meta.copy()
            imagesToStack.append(src.read())
            nodata = src.nodata
            
    imageStack = np.vstack(imagesToStack)
    metadata['count'] = len(imageFileList)
    with rio.open(stackFileNameOut, 'w', **metadata) as dst:
        dst.write(imageStack)

def mosaicH5Band(h5Files,outMosaicTif,bands,mosaicType = 'tiles'):
    
    outTempTifs = []
    
    for h5File in h5Files:
        
        array,metadata,wavelengths = h5refl2array(h5File, 'Reflectance', onlyMetadata = False, spatialIndexesToRead = None, bandIndexesToRead = (bands[0],bands[1]))
        originalDtype = array.dtype
        array = array.astype(np.float32)
        array[array==metadata['noDataVal']] = np.nan
        array = np.nanmean(array,axis=2)
        array[np.isnan(array)] = metadata['noDataVal']
        array = array.astype(originalDtype)
        EPSGCode,MosaicExtents,RasterCellHeight,RasterCellWidth,NoData = getGtifMetadataFromH5Metadata(metadata)
        
        outDir = os.path.dirname(outMosaicTif)
        
        outTempTif = os.path.join(outDir,os.path.basename(h5File).replace('.h5','.tif'))
        
        outTempTifs.append(outTempTif)
        
        writeRasterToTif(outTempTif,array,MosaicExtents,EPSGCode,NoData,RasterCellWidth,RasterCellHeight)
        
    generateMosaic(outTempTifs,outMosaicTif,mosaicType = mosaicType)
    
    for outTempTif in outTempTifs:
        os.remove(outTempTif)

def getRgbFromH5(inputH5):
    
    red,metadata,wavelengths = h5refl2array(inputH5, 'Reflectance', onlyMetadata = False, spatialIndexesToRead = None, bandIndexesToRead = (54,55))
    green,metadata,wavelengths = h5refl2array(inputH5, 'Reflectance', onlyMetadata = False, spatialIndexesToRead = None, bandIndexesToRead = (34,35))
    blue,metadata,wavelengths = h5refl2array(inputH5, 'Reflectance', onlyMetadata = False, spatialIndexesToRead = None, bandIndexesToRead = (18,19))
    
    return red,blue,green,metadata,wavelengths 

def generateRgbImageFromH5(inputH5,outTif):
    
    red,blue,green,metadata,wavelengths = getRgbFromH5(inputH5)
    
    rgb = np.dstack([red,blue,green])
    
    EPSGCode,MosaicExtents,RasterCellHeight,RasterCellWidth,NoData = getGtifMetadataFromH5Metadata(metadata)
    
    writeRasterToTif(outTif,rgb,MosaicExtents,EPSGCode,NoData,RasterCellWidth,RasterCellHeight)

def getRasterStats(inputGtif):
    
    with rio.open(inputGtif, 'r') as src:
        metadata = src.meta.copy()
        rasterData = src.read()
        nodata = src.nodata
        
    rasterData[rasterData==src.nodata] = np.nan
    
    minValue = np.nanmin(rasterData)
    maxValue = np.nanmax(rasterData)
    meanValue = np.nanmean(rasterData)
    medianValue = np.nanmedian(rasterData)
    ninetyFifthPercentile = np.nanpercentile(rasterData, 0.95)
    fifthPercentile = np.nanpercentile(rasterData, 0.05)
    percentZero = (np.sum(rasterData==0)/np.sum(np.isnan(rasterData)))*100
    
    summaryStatsDict = {"min": minValue,"max":maxValue,"mean": meanValue,"median":medianValue,"FifthPercentile":fifthPercentile,"NinetyFifthPercentile":ninetyFifthPercentile,"PercentZero":percentZero}
    
    return summaryStatsDict

def getMatchingTileFiles(fileList1,filelList2):
    
    currentTileFiles = []
    previousTileFiles = []
    
    for currentFile in fileList1:
        
        currentFileSplit = os.path.basename(currentFile).split('_')
        
        currentFileTileId = currentFileSplit[4]+'_'+currentFileSplit[5]
        
        for previousFile in filelList2:
            
            if currentFileTileId in os.path.basename(previousFile):
                
                currentTileFiles.append(currentFile)
                previousTileFiles.append(previousFile)
                break
            
    return currentTileFiles,previousTileFiles

def getOverlapDifferenceTif(filelist1,filelist2,outputDir):

    with Pool(processes=15) as pool:
        processFunction = partial(writeDifferenceGtifsHelper,outdir =outputDir)
        pool.map(processFunction,zip(filelist1,filelist2))
    
def getReflectanceMeanDifference(inputFiles):

    with Pool(processes=5) as pool:
        processFunction = partial(getDifferenceSum)
        reflectanceDifferenceResults = pool.map(processFunction,inputFiles)
    
    wavelengthSampleCount,sumWavelengthDifference = zip(*reflectanceDifferenceResults)
    
    meanDifferenceSpectra = np.sum(sumWavelengthDifference,axis=0)[:,None]/np.sum(wavelengthSampleCount,axis=0)[:,None]

    return wavelengthSampleCount,meanDifferenceSpectra

def getFlightlineReflectanceDifferenceSummaryStats(inputFiles,meanDifferenceSpectra,wavelengths):

    with Pool(processes=5) as pool:
        processFunction = partial(getReflectanceDifferenceSummaryStats,wavelengthMeanDifference=meanDifferenceSpectra.T,wavelengths=wavelengths)
        reflectanceSummaryStepsResults = pool.map(processFunction,inputFiles)
    
    rmsRasters,maxIndicesRasters,maxDiffRaster,maxDifferenceWavelengths,sumWavelengthDifferenceSquared,sumWavelengthVariance = zip(*reflectanceSummaryStepsResults)

    return rmsRasters,maxIndicesRasters,maxDiffRaster,maxDifferenceWavelengths,sumWavelengthDifferenceSquared,sumWavelengthVariance

def getRmsAndMaxWavelgthErrorRasters(rmsArrays,maxIndicesArrays,maxDiffArrays,originalTifFiles,outputRmsDir,outputMaxIndicesDir,outputMaxDiffDir,filebasenames):
    
    for rmsArray,maxIndicesArray,maxDiffArray,originTifFile,filebasename in zip(rmsArrays,maxIndicesArrays,maxDiffArrays,originalTifFiles,filebasenames):
        fileId = os.path.basename(originTifFile).split('_')[1]+'_'+os.path.basename(originTifFile).split('_')[2]
        outFileRms = os.path.join(outputRmsDir,filebasename+'_rmsDifference.tif')
        outFileMaxIndices = os.path.join(outputMaxIndicesDir,filebasename+'_maxWavelength.tif')
        outFileMaxDiff = os.path.join(outputMaxDiffDir,filebasename+'_maxDifference.tif')
        writeRasterToTifTransferMetadata(rmsArray,outFileRms,originTifFile)
        writeRasterToTifTransferMetadata(maxIndicesArray,outFileMaxIndices,originTifFile)
        writeRasterToTifTransferMetadata(maxDiffArray,outFileMaxDiff,originTifFile)

def getOverlappingImageFiles(fileList):
    
    filesToCompare1 = []
    filesToCompare2 = []
    
    for i,file1 in enumerate(fileList):
        
        bbox1 = getBbox(file1)
        polygon1 = Polygon([(bbox1[0], bbox1[1]), (bbox1[2], bbox1[1]), (bbox1[2], bbox1[3]), (bbox1[0], bbox1[3])])
        for file2 in fileList[i+1:]:
            
            bbox2 = getBbox(file2)
            polygon2 = Polygon([(bbox2[0], bbox2[1]), (bbox2[2], bbox2[1]), (bbox2[2], bbox2[3]), (bbox2[0], bbox2[3])])

            if polygon1.intersects(polygon2):
                filesToCompare1.append(file1)
                filesToCompare2.append(file2)
                
    return filesToCompare1,filesToCompare2

def getBbox(file):

    if os.path.splitext(os.path.basename(file))[1] == '.tif':

        with rasterio.open(tiff_file) as src:
            bbox = src.bounds

    if os.path.splitext(os.path.basename(file))[1] == '.h5':
        
        reflectanceArray, metadata, wavelengths = h5refl2array(file, 'Reflectance',onlyMetadata = True)
        
        bbox = (metadata['ext_dict']['xMin'],metadata['ext_dict']['yMin'],metadata['ext_dict']['xMax'],metadata['ext_dict']['yMax'])

    return bbox


def findOverlapRegionWithBboxes(bbox1, bbox2):
    """
    Find the overlapping region between two images given their bounding boxes.

    Args:
    - bbox1: Tuple containing the bounding box coordinates of the first image (x1, y1, x2, y2).
    - bbox2: Tuple containing the bounding box coordinates of the second image (x1, y1, x2, y2).

    Returns:
    - overlap_bbox: Tuple containing the coordinates of the overlapping bounding box (x1, y1, x2, y2).
    """

    # Extract coordinates from bounding boxes
    xMin1, yMin1, xMax1, yMax1 = bbox1
    xMin2, yMin2, xMax2, yMax2 = bbox2

    # Compute the coordinates of the overlapping bounding box
    overlapMinX = max(xMin1, xMin2)
    overlapMinY= max(yMin1, yMin2)
    overlapMaxX = min(xMax1, xMax2)
    overlapMaxY = min(yMax1, yMax2)

    # Check if there is any overlap
    if overlapMinX < overlapMaxX and overlapMinY < overlapMaxY:
        # Return the coordinates of the overlapping bounding box
        overlapBbox = (overlapMinX, overlapMinY, overlapMaxX, overlapMaxY)
    else:
        # No overlap
        overlapBbox = None

    return overlapBbox

def getOverlapIndices(bbox,overlap_bbox):

    xMin, yMin, xMax, yMax  = bbox
    overlapMinX, overlapMinY, overlapMaxX, overlapMaxY = overlap_bbox
    
    overlapIndicesCol = (int(overlapMinX - xMin), int(overlapMaxX - xMin))
    overlapIndicesRow = (int(yMax-overlapMaxY), int(yMax-overlapMinY))
    
    return overlapIndicesRow,overlapIndicesCol

def differenceOverlappingSectionsofH5(h5file1,h5file2):

    bbox1 = getBbox(h5file1)
    bbox2 = getBbox(h5file2)

    overlapBbox = findOverlapRegionWithBboxes(bbox1, bbox2)  
    
    if overlapBbox is None:
        difference = None
        return difference, overlapBbox                       
    overlapIndicesRow1,overlapIndicesCol1 = getOverlapIndices(bbox1,overlapBbox)
    overlapIndicesRow2,overlapIndicesCol2 = getOverlapIndices(bbox2,overlapBbox)
    
    reflectanceArray1, metadata1, wavelengths1 = h5refl2array(h5file1, 'Reflectance',spatialIndexesToRead = [overlapIndicesRow1,overlapIndicesCol1])
    reflectanceArray2, metadata2, wavelengths2 = h5refl2array(h5file2, 'Reflectance',spatialIndexesToRead= [overlapIndicesRow2,overlapIndicesCol2])
    
    reflectanceArray1=getCleanSpectra(reflectanceArray1, metadata1, wavelengths1)
    reflectanceArray2=getCleanSpectra(reflectanceArray2, metadata2, wavelengths2)

    difference = reflectanceArray1 - reflectanceArray2
    
    reflectanceArray, metadata, wavelengths = h5refl2array(h5file1, 'Reflectance',onlyMetadata = True)
    
    difference[reflectanceArray1==metadata['noDataVal']] = metadata['noDataVal']
    
    difference[reflectanceArray2==metadata['noDataVal']] = metadata['noDataVal']                                                                            
    
    return difference, overlapBbox

def getH5Difference(h5file1,h5file2):
    
    difference, overlapBbox = differenceOverlappingSectionsofH5(h5file1,h5file2)
    
    if overlapBbox is None or difference is None:
        metadata = None
        return difference,metadata
                                                 
    reflectanceArray, metadata, wavelengths = h5refl2array(h5file1, 'Reflectance',onlyMetadata = True)
    
    metadata['extent'] = overlapBbox 
    
    return difference,metadata

def writeH5DifferneceToGtif(h5file1,h5file2,outFile):

    difference,metadata = getH5Difference(h5file1,h5file2)        

    if difference is None:
        return None
    elif (difference==metadata['noDataVal']).all():
        return None
    else:
        #difference[np.isnan(difference)] = metadata['noDataVal'] 
            
        writeTifHdf5Metadata(difference,metadata,outFile)

def getDifferenceSum(tifFile):

    print('Working on mean calcs for: '+tifFile)
    
    with rio.open(tifFile, 'r') as src:
        metadata = src.meta.copy()
        dataCubeDifference = src.read()
        nodata = src.nodata
    
    dataCubeDifference = dataCubeDifference.astype(np.float32)
    
    dataCubeDifference[dataCubeDifference==nodata] = np.nan
   
    wavelengthSampleCount = np.sum((~np.isnan(dataCubeDifference)).astype(np.bool)*1,axis=(1,2))
    
    sumWavelengthDifference = np.nansum(dataCubeDifference,axis=(1,2))
    
    return wavelengthSampleCount,sumWavelengthDifference

def getReflectanceDifferenceSummaryStats(tifFile,wavelengthMeanDifference,wavelengths):
   
    print('Working on summary stats calcs for: '+os.path.basename(tifFile))
    
    with rio.open(tifFile, 'r') as src:
        metadata = src.meta.copy()
        dataCubeDifference = src.read()
        nodata = src.nodata
    
    dataCubeDifference = dataCubeDifference.astype(np.float32)
    
    dataCubeDifference[dataCubeDifference==nodata] = np.nan
    
    RMS = np.nanmean(dataCubeDifference**2,axis=0)**0.5
    
    maxDiff = np.nanmax(dataCubeDifference,axis=0)
 
    maxDifferenceWavelengths = np.nanmax(np.abs(dataCubeDifference),axis=(1,2))
    maskedArray = np.ma.masked_invalid(dataCubeDifference)
    maxIndices = np.nanargmax(maskedArray , axis=0).astype(np.float32)
    maskedArray = []
    maxIndices = np.where(np.all(np.isnan(dataCubeDifference), axis=0), np.nan, maxIndices)

    sumWavelengthDifferenceSquared = np.nansum(dataCubeDifference**2,axis=(1,2))
   
    sumWavelengthVariance = np.nansum((dataCubeDifference-wavelengthMeanDifference.T[:,:,np.newaxis])**2,axis=(1,2))
    dataCubeDifference = []
    RowCounter = 0
    for Row in maxIndices:
        ColCounter=0
        for Cell in Row:
            if np.isnan(Cell):
                ColCounter=ColCounter+1
                continue

            maxIndices[RowCounter,ColCounter] = wavelengths[int(Cell)]
            ColCounter=ColCounter+1
        RowCounter = RowCounter+1
  
    return RMS,maxIndices,maxDiff,maxDifferenceWavelengths,sumWavelengthDifferenceSquared,sumWavelengthVariance

def writeDifferenceGtifsHelper(files,outdir):
    file1,file2 = files
    print('Working on '+os.path.basename(file1)+' and '+os.path.basename(file2))
    file1baseSplit = os.path.basename(file1).split('_')
    file2baseSplit = os.path.basename(file2).split('_')
    outname = file1baseSplit[2]+'_'+file1baseSplit[4]+'_'+file1baseSplit[5]+'_'+file2baseSplit[4]+'_'+file2baseSplit[5]+'_difference.tif' 
    writeH5DifferneceToGtif(file1,file2,os.path.join(outdir,outname))    

def generateReflectanceDifferenceLinePlots(wavelengths,meanDifferenceSpectra,wavelengthStandardDeviation,wavelengthTotalRms,maxDifferenceWavelengthsTotal,scaleReflectanceFactor,outputDir,outputFileBaseName):        
    
    plt.subplots(figsize=(10, 8))
    plt.plot(wavelengths,meanDifferenceSpectra/scaleReflectanceFactor,'b-')
    plt.xlabel('Wavelength',fontsize=12)
    plt.ylabel('Mean of Difference (%)',fontsize=12)
    plt.savefig(os.path.join(outputDir,outputFileBaseName+'_mean_reflectance_diff.png'),dpi=600)
    plt.close()
    
    plt.subplots(figsize=(10, 8))
    plt.plot(wavelengths,wavelengthStandardDeviation/scaleReflectanceFactor,'r-')
    plt.xlabel('Wavlength',fontsize=12)
    plt.ylabel('St Dev of Difference (%)',fontsize=12)
    plt.savefig(os.path.join(outputDir,outputFileBaseName+'_stddev_reflectance_diff.png'),dpi=600)
    plt.close()
    
    plt.subplots(figsize=(10, 8))
    plt.plot(wavelengths,wavelengthTotalRms/scaleReflectanceFactor,'g-')
    plt.xlabel('Wavlength',fontsize=12)
    plt.ylabel('RMS of Difference (%)',fontsize=12)
    plt.savefig(os.path.join(outputDir,outputFileBaseName+'_RMS_reflectance_diff.png'),dpi=600)
    plt.close()
    
    plt.subplots(figsize=(10, 8))
    plt.plot(wavelengths,maxDifferenceWavelengthsTotal/scaleReflectanceFactor,'k-')
    plt.xlabel('Wavlength',fontsize=12)
    plt.ylabel('Max Difference (%)',fontsize=12)
    plt.savefig(os.path.join(outputDir,outputFileBaseName+'_max_reflectance_diff.png'),dpi=600)
    plt.close()
    
    plt.subplots(figsize=(10, 8))
    plt.plot(wavelengths[:,None],meanDifferenceSpectra/scaleReflectanceFactor,'b-',label='Mean Difference')
    plt.fill_between(wavelengths, (meanDifferenceSpectra/scaleReflectanceFactor - wavelengthStandardDeviation/scaleReflectanceFactor).flatten(), (meanDifferenceSpectra/scaleReflectanceFactor + wavelengthStandardDeviation/scaleReflectanceFactor).flatten(), color='lightblue', label='Standard Deviation')
    plt.xlabel('Wavlength',fontsize=12)
    plt.ylabel('Reflectance Difference (%)',fontsize=12)
    plt.legend()
    plt.savefig(os.path.join(outputDir,outputFileBaseName+'_mean_standard_deviation_reflectance_diff.png'),dpi=600)
    plt.close()

def generateNisKml(inputKml,outputKml,color):
    # Read KML file and parse coordinates of the polygon
    
    label = os.path.splitext(os.path.basename(inputKml))[0]
    with open(inputKml) as f:
        doc = parser.parse(f)
        
    coordinates = []
    for pm in doc.findall('.//{http://www.opengis.net/kml/2.2}Placemark'):
        if hasattr(pm, 'Polygon'):
            coords = str(pm.Polygon.outerBoundaryIs.LinearRing.coordinates).split()
            coords = [(float(coord.split(',')[0]), float(coord.split(',')[1])) for coord in coords]
            coordinates.append(coords)

    # Create Shapely polygon from coordinates
    polygon = Polygon(coordinates[0])  # Assuming there's only one polygon in the KML file
    decimated_coords = []
    for i in range(0, len(coordinates[0]), 10):
        decimated_coords.append(coords[i])

    simplified_polygon = Polygon(decimated_coords)
    
       # Create output KML and assign placemark at the center
    kml = simplekml.Kml()
    center = Point(simplified_polygon.centroid.x, simplified_polygon.centroid.y)
    kml.newpoint(name=label, coords=[(center.x, center.y)])
    
    # Color the polygon perimeter and shade the polygon based on attributes
    style = simplekml.Style()
    style.linestyle.width = 5 
    if color == 'Red':
        style.linestyle.color = simplekml.Color.red
        style.polystyle.color = simplekml.Color.rgb(200, 50, 50,a=125)  # Red with transparency
    elif color == 'Green':  # You can replace this condition with your own logic for color assignment
        style.linestyle.color = simplekml.Color.green
        style.polystyle.color = simplekml.Color.rgb(50,200,50,a=125)  # Green with transparency
    elif color == 'Yellow':
        style.linestyle.color = simplekml.Color.yellow
        style.polystyle.color = simplekml.Color.rgb(155, 155, 20,a=125)  # Yellow with transparency
    
    # Add simplified polygon to the KML with assigned style
    outer = kml.newpolygon(name=label)
    outer.altitudemode= simplekml.AltitudeMode.clamptoground
    outer.outerboundaryis = simplified_polygon.exterior.coords[:]
    outer.style = style
    
    # Save KML file
    kml.save(outputKml)

def AddNoDataToTIFF(InTIFFfile):

    if os.path.isdir(os.path.dirname(InTIFFfile)):
        # Find each of the 7 TIFF files associated with the input example/root file.
        ThisPath = os.path.dirname(InTIFFfile)
        #ThisFileBasename = os.path.basename(InTIFFfile)
        TheseTIFFs = [file for file in os.listdir(ThisPath) if (file.endswith('.tif'))]
        for dataTIFF in TheseTIFFs:
            DataSet = gdal.Open(ThisPath + '/' + dataTIFF, gdal.GA_Update)
            DataSet.GetRasterBand(1).SetNoDataValue(float(-9999.0))
            DataSet = None
        return InTIFFfile
    else:
        return 'Failure'

def writeCameraSummaryFile(outFile, parameterDict):
    
    with open(outFile, 'w') as file:
        for item, value in parameterDict.items():
            if isinstance(value, list):
                value_str = '[' + ', $\n\t'.join(f"'{item}'" for item in value) + ']'
            else:
                value_str = f"'{value}'"
            file.write(f"{item} = {value_str}\n\n")

def previousSiteVisit(yearSiteVisit):

    yearSiteVisitSplit = yearSiteVisit.split('_')

    return yearSiteVisitSplit[1]+'_'+str(int(yearSiteVisitSplit[2])-1)

def getLeapSeconds(flightdayHour):

    if flightdayHour < 20120701:
        leapSeconds = 15
    elif flightdayHour < 20150630:
        leapSeconds = 16
    elif  flightdayHour < 20161231:
        leapSeconds=17
    elif flightdayHour < 20251029:
        leapSeconds = 18

    return leapSeconds

def getFltLogLineNumber(fltLogData, site,DateLine = None):

    # Input
        # csvFname = Flight Log CSV FileName,     eg: 'D:\Raw\2018\P3\C1\2018071415_P3C1\L0\Ancillary\FlightLogs\2018071415_P3C1_NIS.csv'
        # DateLine = Date and 6 digit LineNumber, eg: '2018071415_165617'

    # Output
        # if DateLine is None:
            # Return a Pandas DataFrame with all DateLine in the CSV including newLineNumber (L???-?)
        # else
            # Return Pandas DataFrame for the input DateLine

    #print('Retrieve Flight Plan Line Number: {}'.format(csvFname))
    # Filter Flight Log Data
    fltLogData['Filename'] = fltLogData['Filename'].astype(int)
    fltLogData = fltLogData[np.isfinite(fltLogData['Filename'])].reset_index(drop = True) # Drop Row where Filename is NaN
    fltLogData = fltLogData[fltLogData.LineNumber != 'GT'].reset_index(drop = True) # Drop Row where Line is GT
    fltLogData = fltLogData[fltLogData.LineNumber != 'AT'].reset_index(drop = True) # Drop Row where Line is AT
    fltLogData = fltLogData[fltLogData.LineNumber != 'Adhoc'].reset_index(drop = True) # Drop Row where Line is AT
    fltLogData = fltLogData[fltLogData.Site.str.contains(site)].reset_index(drop = True)
    fltLogData['LineNumber'] = fltLogData['LineNumber'].astype(int)

    # Sort by Six Digit LineNumber
    fltLogData = fltLogData.sort_values('Filename').reset_index(drop = True)

    # Create DateLine and newLineNumber Column
    fltLogData['DateLine']      = fltLogData['FlightID'].astype(int).astype(str).str.slice(stop=8) + '_' + fltLogData['Filename'].astype(int).astype(str) #'YYYYNNDDHH_HHMMSS'
    fltLogData['newLineNumber'] = 'L' + fltLogData['LineNumber'].apply(lambda x: '{0:0>3}'.format(x)) + '-1'

    duplicateRow = fltLogData[fltLogData.duplicated(['newLineNumber'], keep = False)]

    if duplicateRow.shape[0] != 0:
        opData  = pd.DataFrame()
        lineNum = list(set(duplicateRow['LineNumber'].values))
        for line in lineNum:
            tempData = duplicateRow[duplicateRow['LineNumber'] == line]
            tempData = tempData.sort_values('Filename').reset_index(drop = True)
            #print(tempData)
            for i in np.arange(tempData.shape[0]):
                tempStr = tempData.loc[i, 'newLineNumber'].split('-')[0]
                tempStr = '{}-{}'.format(tempStr, i+1)
                #print(i, tempStr)
                tempData.loc[i, 'newLineNumber'] = tempStr
                del tempStr
            #print(tempData)
            opData = opData.append(tempData)
            del tempData
        del duplicateRow
        #print(opData)
        opData = opData.reset_index(drop = True)
        #print(opData)
        for i in np.arange(opData.shape[0]):
            tempDateLine = opData.loc[i, 'DateLine']
            tempData     = opData.loc[i, 'newLineNumber']
            tempIdx = fltLogData.index[fltLogData['DateLine'] == tempDateLine]
            fltLogData.loc[tempIdx, 'newLineNumber'] = tempData
            del tempIdx, tempData, tempDateLine
        del opData

    if DateLine is not None:
        tempIdx = fltLogData.index[fltLogData['DateLine'] == DateLine]
        fltLogData = fltLogData.loc[tempIdx, :]['newLineNumber'].values[0]
        #print('getFltLogLineNumber          DateLine: {}     Flight Plan Line Number: {}'.format(DateLine, fltLogData))
        del tempIdx

    return fltLogData

def format_northing(value, _):
    return f"{int(value)}"

def parseEnviProjCs(filePathToEnviProjCs,epsg):
    with open(filePathToEnviProjCs, 'r') as file:
        for line in file:
            # Use regular expression to find a number and the following string
            match = re.match(r'^\s*(\d+)\s+(.*)$', line)
            if match:
                number = int(match.group(1))
                if number == epsg:
                    return match.group(2).strip()

def differenceGeotiffFiles(inputGtifFile1,inputGtifFile2):

    with rio.open(inputGtifFile1) as src1, rio.open(inputGtifFile2) as src2:
        # Get the bounding boxes of the two datasets
        bounds1 = src1.bounds
        bounds2 = src2.bounds

        # Calculate the intersection of the bounding boxes
        overlap_min_x = max(bounds1.left, bounds2.left)
        overlap_min_y = max(bounds1.bottom, bounds2.bottom)
        overlap_max_x = min(bounds1.right, bounds2.right)
        overlap_max_y = min(bounds1.top, bounds2.top)

        # Check if there is an actual overlap
        if overlap_min_x < overlap_max_x and overlap_min_y < overlap_max_y:
            overlap = BoundingBox(left=overlap_min_x, bottom=overlap_min_y, right=overlap_max_x, top=overlap_max_y)

            # Convert the overlapping extent to pixel coordinates for each dataset
            window1 = src1.window(overlap_min_x, overlap_min_y, overlap_max_x, overlap_max_y)
            window2 = src2.window(overlap_min_x, overlap_min_y, overlap_max_x, overlap_max_y)

            # Read the overlapping portions of the rasters
            data1 = src1.read(window=window1)
            data2 = src2.read(window=window2)

            # Replace no-data values with np.nan
            data1[data1==src1.nodata] = np.nan
            data2[data2==src2.nodata] = np.nan

            # Perform subtraction
            result = data1 - data2

            # Set no-data values in difference raster
            result[np.isnan(result)] = src1.nodata

            # Create a profile for the output raster
            profile = src1.profile
            profile.update(width=window1.width, height=window1.height, transform=src1.window_transform(window1),nodata = src1.nodata)

    return result, profile

def updateGeotiffFileData(inGeotiffPath, newData):

    gdaldataset = gdal.Open(inGeotiffPath,gdal.GA_Update)

    gdaldataset.GetRasterBand.WriteArray(newData)

    gdaldataset = None

def smoothGeotiffFile(inGeotiffPath, outGeotiffDir, windowSize):
    
    outGeotiffPath = os.path.join(outGeotiffDir,os.path.basename(inGeotiffPath).replace('.tif','_filtered.tif'))
    # Open the input GeoTIFF file
    with rio.open(inGeotiffPath, 'r') as src:
        # Get the metadata from the source dataset
        metadata = src.meta.copy()

        # Read the data from all bands
        data = src.read()
        nodata = src.nodata

    data[data==nodata] = np.nan
        # Apply the modification function to the data
    smoothData = smoothImageIgnoreNans(np.squeeze(data), windowSize)
    smoothData = smoothData[None,:,:]
    # Write the modified data to a new GeoTIFF file
    smoothData[np.isnan(smoothData)] = nodata
    with rio.open(outGeotiffPath, 'w', **metadata) as dst:
        dst.write(smoothData)

def plot_classified_geotiff(input_path,classLabels,classColors, title='Classified Image', save_path=None, variable=''):
    # Open the GeoTIFF file
    with rio.open(input_path) as src:
        # Read the raster data

        raster_data = src.read(masked=True)  # Use masked=True to handle nodata values
        common_mask = src.read_masks(1) # Assuming you want to use the mask from the first band
        nodata = src.nodata
        #bands = np.ma.masked_array(raster_data, mask=np.broadcast_to(common_mask == 0, raster_data.shape))
        
        # Get the bounding box of the raster
        left, bottom, right, top = src.bounds

        # Create a plot
        fig, ax = plt.subplots(figsize=(10, 8))
   
        cmap = ListedColormap([classColors[i] for i in range(0, len(classColors) - 1)]) 
   
        im = ax.imshow(np.moveaxis(raster_data,0,-1), cmap=cmap,extent=[left, right, bottom, top])
        
        legend_elements = [Patch(facecolor=color, label=f'{label}') for label, color in zip(classLabels,classColors)]

        # Add legend
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.6, 1), loc='upper right',fontsize=12)
        
        # Set axis labels
        ax.set_xlabel('UTM Easting (m)')
        ax.set_ylabel('UTM Northing (m)')

        # Set the title
        ax.set_title(title)
        ax.tick_params(axis='both', which='major', labelsize=12)
        # Format northing values as integers
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        ax.yaxis.set_major_formatter(FuncFormatter(format_northing))

        # Save the plot as a PNG file if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, transparent=False,dpi=600)
            #print(f"Plot saved as {save_path}")

        # Show the plot
        #plt.show()
        plt.close('all')

def plot_multiband_geotiff(input_path, title='Raster Plot', stretch=None,save_path=None, nodata_color='black',variable='',classLabels=None,classColors=None,bandsToKeep = [0,1,2]):
    # Open the GeoTIFF file
    with rio.open(input_path) as src:
        # Read the raster data
        bandsToKeep = bandsToKeep+[-1]
        raster_data = src.read(masked=True)  # Use masked=True to handle nodata values
        common_mask = src.read_masks(1) # Assuming you want to use the mask from the first band
        nodata = src.nodata
        bands = np.ma.masked_array(raster_data, mask=np.broadcast_to(common_mask == 0, raster_data.shape))
        
        # Get the bounding box of the raster
        left, bottom, right, top = src.bounds

        # Create a plot
        fig, ax = plt.subplots(figsize=(12, 8))

        if stretch == 'linear5':
            dataForStretch = bands.compressed().flatten()
            dataForStretch = dataForStretch[dataForStretch!=nodata]                               
            percentiles = np.percentile(dataForStretch, q=[5, 95])
            min_val = percentiles[0]
            max_val = percentiles[1]
            raster_data = [(band - min_val) / (max_val - min_val) for band in bands]
            raster_data= (np.clip(raster_data, 0, 1)).astype(np.float32)  # Clip values to [0, 1] range
        else:
            if raster_data.dtype == np.uint8:
                raster_data = (raster_data/255).astype(np.float32)
            else:
                raster_data = (raster_data/np.max(raster_data)).astype(np.float32)
             
            
        alpha_channel = np.where(common_mask, 1, 0)[None,:,:]
        raster_data = np.vstack((raster_data,alpha_channel))            
        
        im = ax.imshow(np.moveaxis(raster_data[bandsToKeep,:,:],0,-1), extent=[left, right, bottom, top])
               
        if classLabels is not None:
            legend_elements = [Patch(facecolor=color, label=f'{label}') for label, color in zip(classLabels,classColors)]

            # Add legend
            plt.legend(handles=legend_elements, bbox_to_anchor=(1.7, 1), loc='upper right',fontsize=12)
        plt.tight_layout()
        # Set axis labels
        ax.set_xlabel('UTM Easting (m)',fontsize=12)
        ax.set_ylabel('UTM Northing (m)',fontsize=12)

        # Set the title
        ax.set_title(title)
        
        # Set the font size of the tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Format northing values as integers
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        ax.yaxis.set_major_formatter(FuncFormatter(format_northing))

        # Save the plot as a PNG file if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, transparent=False,dpi=600)
            print(f"Plot saved as {save_path}")

        # Show the plot
        #plt.show()
        plt.close('all')

def plot_dates_geotiff(input_path, title='', save_path=None):
    # Open the GeoTIFF file
    with rio.open(input_path) as src:
        # Read the raster data
        raster_data = src.read(1, masked=True)  # Use masked=True to handle nodata values
        common_mask = src.read_masks(1)  # Assuming you want to use the mask from the first band
        nodata = src.nodata

        # Extract unique values from the raster data
        unique_values = np.unique(raster_data[raster_data != nodata])

        # Convert unique values to date format
        def convert_to_date(value):
            try:
                return datetime.strptime(str(int(value)), '%Y%m%d').date()
            except ValueError:
                return None

        unique_dates = [convert_to_date(value) for value in unique_values if value != -9999]

        # Create class labels and colors
        classLabels = [date.strftime('%Y-%m-%d') for date in unique_dates]
        classColors = plt.cm.viridis(np.linspace(0, 1, len(unique_dates)))

        # Get the bounding box of the raster
        left, bottom, right, top = src.bounds

        # Create a plot
        fig, ax = plt.subplots(figsize=(10, 8))

        cmap = ListedColormap(classColors)

        im = ax.imshow(raster_data, cmap=cmap, extent=[left, right, bottom, top])

        legend_elements = [Patch(facecolor=color, label=label) for label, color in zip(classLabels, classColors)]

        # Add legend
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.6, 1), loc='upper right', fontsize=12)

        # Set axis labels
        ax.set_xlabel('UTM Easting (m)')
        ax.set_ylabel('UTM Northing (m)')

        # Rotate easting axis tick labels and format them as integers
        #ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
        #ax.set_xticklabels(ax.get_xticks(), rotation=45, ha='right')

        # Format northing values as integers without commas
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))

        # Set the title
        ax.set_title(title)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Save the plot as a PNG file if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, transparent=False, dpi=600)
            print(f"Plot saved as {save_path}")

        # Show the plot
        # plt.show()
        plt.close('all')

def plot_geotiff(input_path, title='Raster Plot', cmap='viridis', save_path=None, nodata_color='black', variable='', vmin=None, vmax=None):
    # Open the GeoTIFF file
    with rio.open(input_path) as src:
        # Read the raster data
        raster_data = src.read(1,masked=True)  # Use masked=True to handle nodata values
        raster_data[raster_data==src.nodata] = np.median(raster_data[raster_data!=src.nodata])
        raster_data_smooth = smoothImage(raster_data, 5)
        raster_data_smooth = np.ma.masked_where(raster_data.mask, raster_data_smooth)
        # Calculate mean and standard deviation
        # mean_value = raster_data.mean()
        # std_value = raster_data.std()

        # Get the bounding box of the raster
        left, bottom, right, top = src.bounds

        # Create a plot
        fig, ax = plt.subplots(figsize=(10, 8))

        dataForLimits = raster_data[~raster_data.mask].flatten()

        quartile10 = np.percentile(dataForLimits, 10)
        quartile90 = np.percentile(dataForLimits, 90)

        if vmin and vmax:
            im = ax.imshow(raster_data_smooth, extent=[left, right, bottom, top], cmap=cmap, alpha=0.9,
                           vmin=vmin, vmax=vmax, interpolation='none', rasterized=True)
        else:
            im = ax.imshow(raster_data_smooth, extent=[left, right, bottom, top], cmap=cmap, alpha=0.9,
                           vmin=quartile10, vmax=quartile90, interpolation='none', rasterized=True)

        # Plot the raster data with nodata values as transparent
        #im = ax.imshow(raster_data_smooth, extent=[left, right, bottom, top], cmap=cmap, alpha=0.9,
        #               vmin=np.max((mean_value - std_value,np.min((raster_data)))), vmax=np.min((mean_value + std_value,np.max((raster_data)))), interpolation='none', rasterized=True)

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax, label=variable)

        # Set axis labels
        ax.set_xlabel('UTM Easting (m)')
        ax.set_ylabel('UTM Northing (m)')
        ax.set_aspect('equal', adjustable='box')

        x_ticks = (np.arange(round(left,-3), round(right,-3), 1000)).astype(np.int32)

        y_ticks = (np.arange(bottom, top, 1000)).astype(np.int32)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks,rotation=90)
        ax.set_yticks(y_ticks)
       
        # Set the title
        ax.set_title(title)

        # Format northing values as integers
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        ax.yaxis.set_major_formatter(FuncFormatter(format_northing))
        ax.tick_params(axis='both', which='major', labelsize=12)                      

        # Save the plot as a PNG file if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, transparent=False,dpi=600)
            print(f"Plot saved as {save_path}")

        # Show the plot
        #plt.show()
        plt.close('all')

def write_stats_to_file(Data1, Data2, descriptor1, descriptor2, file_path):
    """
    Write mean, standard deviation, and percent difference in mean to a text file.

    Parameters:
    - Data1 (numpy.ndarray): First set of data.
    - Data2 (numpy.ndarray): Second set of data.
    - descriptor1 (str): Descriptor for Data1.
    - descriptor2 (str): Descriptor for Data2.
    - file_path (str): File path to save the statistics.

    Returns:
    None
    """
    # Calculate mean and standard deviation for each vector
    mean1, std1 = np.mean(Data1), np.std(Data1)
    mean2, std2 = np.mean(Data2), np.std(Data2)

    # Calculate percent difference in mean
    percent_diff_mean = ((mean2 - mean1) / mean1) * 100

    # Write statistics to the text file
    with open(file_path, 'w') as file:
        file.write(f"1: Mean of {descriptor1}: {mean1}\n")
        file.write(f"2: Standard Deviation of {descriptor1}: {std1}\n")
        file.write(f"3: Mean of {descriptor2}: {mean2}\n")
        file.write(f"4: Standard Deviation of {descriptor2}: {std2}\n")
        file.write(f"5: Percent Difference in Mean: {percent_diff_mean:.2f}%\n")

def plot_histograms_matplotlib(data1, data2,data_label_one, data_label_two,title,save_path=None):
    """
    Plot histograms for two sets of data using Matplotlib.

    Parameters:
    - data1 (numpy.ndarray): First set of data.
    - data2 (numpy.ndarray): Second set of data.
    - save_path (str or None, optional): File path to save the plot. Default is None.

    Returns:
    None
    """
    # Calculate mean and standard deviation for each vector
    mean1, std1 = np.mean(data1), np.std(data1)
    mean2, std2 = np.mean(data2), np.std(data2)

    # Calculate the maximum value for the histogram bins
    max_value = max(mean1 + 3 * std1, mean2 + 3 * std2)

    # Create histograms with bins for each integer between zero and max_value
    bins = np.arange(int(int(max_value)/100), int(max_value) + 2, int(int(max_value)/100))  # Adjusting bin edges to align with integers

    hist, edges = np.histogram(data1, bins=bins)

    plt.hist(data1, bins=bins, alpha=1.0, color='blue', label=data_label_one)
    plt.hist(data2, bins=bins, alpha=1.0, color='red', label=data_label_two)

    plt.xlim(0, max_value)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path,bbox_inches='tight', pad_inches=0.1, dpi=600)

    #plt.show()
    plt.close('all')

def plotGtifHistogram(inGeotiffPath,outHistogramPlot,rasterVariable,xlims = None):

    with rio.open(inGeotiffPath, 'r') as src:
        # Get the metadata from the source dataset
        metadata = src.meta.copy()

        # Read the data from all bands
        data = src.read()
        nodata = src.nodata

    data = np.squeeze(data).flatten()

    data = data[data!=src.nodata]
    
    binType = 'fd'
    manualBins = None
    if 'Wavelength' in rasterVariable:
        binType = 'manual'
        manualBins = np.arange(380,2510,5)

    if xlims:
        plot_histogram(data, variable_name=rasterVariable, abs_value=False, filter_stddev=True, bins=binType, cumulative=False, title=' ', xlabel=rasterVariable, save_path=outHistogramPlot, xlims=xlims,manualBins=manualBins)
    else:
        plot_histogram(data, variable_name=rasterVariable, abs_value=False, filter_stddev=True, bins=binType, cumulative=False, title=' ', xlabel=rasterVariable, save_path=outHistogramPlot,manualBins = manualBins)

def plot_histogram(data, variable_name='Variable', abs_value=False, filter_stddev=True, bins='fd', cumulative=False, title=None, xlabel=None, save_path=None, xlims=None,manualBins = None):
    """
    Plot a histogram or cumulative distribution.

    Parameters:
    - data: NumPy array or list, the data to be plotted.
    - variable_name: str, the name of the variable being plotted.
    - abs_value: bool, if True, take the absolute value of the variable.
    - filter_stddev: bool, if True, filter the data within two standard deviations around the mean.
    - bins: int, array, 'auto', 'fd' (Freedman-Diaconis rule), 'sqrt' (square root of the number of samples), or 'manual' (to provide custom bin edges), the number of bins or bin calculation method. Default is 'auto'.
    - cumulative: bool, if True, plot the cumulative distribution instead of the histogram.
    - title: str, the title of the plot.
    - xlabel: str, the label for the x-axis.
    - save_path: str, if provided, save the plot to the specified file path.

    Returns:
    None
    """
    #print(variable_name)
    # Optionally take the absolute value of the variable
    data = data.astype(np.float32)
    
    # remove zero values for CHM rasters since the CHMs usually have too many zeros
    print(variable_name)
    if 'chm' in variable_name.lower():
        print('Removing zero values for CHM / CHM difference raster')
        data = data[data!=0]
        
    if abs_value:
        data = np.abs(data)

    if isinstance(data, np.ma.MaskedArray):
        data = data[~data.mask].flatten()
    # Optionally filter the data within two standard deviations around the mean
    if filter_stddev:
        mean_value = np.mean(data)
        std_value = np.std(data)
        #data = data[(data >= mean_value - 3 * std_value) & (data <= mean_value + 3 * std_value)]
        data0 = data
        quartile5 = np.percentile(data, 5)
        quartile95 = np.percentile(data, 95)
        data = data[data >= quartile5]
        data = data[data <= quartile95]

    # Determine the number of bins based on the specified method
    if bins == 'fd':
        bin_width = 2 * (np.percentile(data, 75) - np.percentile(data, 25)) / (data.size ** (3/10))
        if np.isnan(bin_width) or bin_width == 0:
            # CHM / CHM diff may have a lot of zero values. Try removing those first and re-calculate the bin width and bins to generate a histogram
            data = data0[data0 != 0]
            quartile2 = np.percentile(data, 2)
            quartile98 = np.percentile(data, 98)
            data = data[data >= quartile2]
            data = data[data <= quartile98]
            bin_width = 2 * (np.percentile(data, 98) - np.percentile(data, 2)) / (data.size ** (3/10))
            print('bin width: ',bin_width)
            if np.isnan(bin_width) or bin_width==0:
                print(variable_name+ ' had only 1 value, did not make histogram')
                uniqueValue = np.unique(data)[0]
                count = 100
                plt.bar(uniqueValue, count,edgecolor='darkblue', alpha=0.7,color='lightblue', width=1) 
                plt.xlabel(xlabel if xlabel else variable_name,fontsize=12)
                plt.ylabel('Normalized Frequency (%)',fontsize=12)
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=600)
                plt.close('all')
                return
                       
            bins = max(250, int((np.nanmax(data) - np.nanmin(data)) / bin_width))  # Ensure at least 10 bins
        
        # original code
        # if np.isnan(bin_width) or bin_width==0:
        #     print(variable_name+ ' had only 1 value, did not make histogram')
        #     return
        # bins = int((np.nanmax(data) - np.nanmin(data)) / bin_width)
        
        # another potential solution ...
        # if np.isnan(bin_width) or bin_width==0:
        #    print(variable_name+ ' had only 1 value, did not make histogram')
        #    uniqueValue = np.unique(data)[0]
        #    count = 100
        #    plt.bar(uniqueValue, count,edgecolor='darkblue', alpha=0.7,color='lightblue', width=1) 
        #    plt.xlabel(xlabel if xlabel else variable_name,fontsize=12)
        #    plt.ylabel('Normalized Frequency (%)',fontsize=12)
        #    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=600)
        #    return
        # bins = int((np.nanmax(data) - np.nanmin(data)) / bin_width)
    elif bins == 'sqrt':
        bins = int(np.nansqrt(data.size))
    elif bins == 'manual':
        # Provide your custom bin edges here
        bins = manualBins
    else:
        bins = int(bins)

    # Plot either a histogram or cumulative distribution with blue bars
    if cumulative:
        plt.hist(data, bins=bins, edgecolor='darkblue', alpha=0.7, cumulative=True, density=True, color='lightblue')
    else:
        plt.hist(data, bins=bins, edgecolor='darkblue', alpha=0.7, density=True, color='lightblue')

    # Set plot labels and title
    if cumulative:
        plt.title(title)
    else:
        plt.title(title)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    if xlims:
        plt.xlim(xlims[0], xlims[1])
    else:
        plt.xlim(np.nanmin(data), np.nanmax(data))
    # plt.xlim( quartile5, quartile95 )
    plt.xlabel(xlabel if xlabel else variable_name,fontsize=12)
    plt.ylabel('Normalized Frequency (%)' if not cumulative else 'Cumulative Probability',fontsize=12)

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=600)
        print(f"Plot saved as {save_path}")

    # Show the plot
    # plt.show()
    plt.close('all')


def batchRemoveBuffers(files,outputDir,tileBuffer):
    
    os.makedirs(outputDir,exist_ok=True)                                    

    for file in files:

        data, metadata  = removeBuffer(file,tileBuffer)

        if data is not None:

            with rio.open(os.path.join(outputDir,os.path.basename(file)), 'w', **metadata) as dst:
                dst.write(data)
             

def removeBuffer(inputGtif,buffer_size):
    """
    Remove a buffer from the outside of a raster image and adjust the raster size.

    Parameters:
    - input_path (str): Path to the input raster file.
    - output_path (str): Path to the output raster file.
    - buffer_size (int): Size of the buffer to be removed.

    Returns:
    None
    """
    with rio.open(inputGtif) as src:
        # Get metadata of the original raster
        metadata = src.meta.copy()

        # Update the metadata to adjust the size and transform
        metadata['width'] -= 2 * buffer_size
        metadata['height'] -= 2 * buffer_size
        metadata['transform'] = from_origin(
            src.bounds.left + buffer_size * src.res[0],
            src.bounds.top - buffer_size * src.res[1],
            src.res[0],
            src.res[1]
        )

        # Read the non-buffered region of the raster data

        if src.height <= 2*buffer_size or src.width <= 2*buffer_size:
            data = None
        else:
            data = src.read(window=((buffer_size, src.height - buffer_size), (buffer_size, src.width - buffer_size)))

    # Write the adjusted raster to a new file
    return data, metadata

def getRasterValuesAtCoordinates(geotiffPath, coordinates):
    """
    Get raster values from a GeoTIFF file at a list of coordinates.

    Parameters:
    geotiff_path (str): Path to the GeoTIFF file.
    coordinates (list of tuples): List of (easting, northing) coordinates.

    Returns:
    list: Raster values corresponding to the input coordinates.
    """
    raster_values = []
    with rio.open(geotiffPath) as src:
        for coord in coordinates:
            easting, northing = coord

            # Transform easting and northing to pixel coordinates
            row, col = src.index(easting, northing)
            # Read raster value at the specified pixel coordinates
            raster_value = src.read(1, window=((row, row + 1), (col, col + 1)),boundless=True)
            if raster_value.size == 0:
                raster_values.append(np.nan)
            else:
                raster_values.append(raster_value[0][0])
    return raster_values

def accessDbToDataframes(dbFilePath,tableNames):

    dfs = []    

    with pyodbc.connect(r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ='+dbFilePath) as conn:
        for tableName in tableNames:

            sql_query = f'SELECT * FROM {tableName}'
                                     

            # Use pandas to read data from the Access database table into a DataFrame
            df = pd.read_sql(sql_query, conn)
            dfs.append(df)
    time.sleep(5)
    return dfs

def calculateUncertainty(inputTxtFile,outputDir,campaign,sensor,year,utmZone):

    os.makedirs(outputDir,exist_ok=True)
    matlabEng = matlab.engine.start_matlab()
    matlabEng.cd(lidarSrcL3Dir, nargout=0)
    matlabEng.calc_simulated_uncertainty(inputTxtFile,outputDir+'/',campaign,sensor,year,utmZone)

    return 1

def write_file_paths_to_text(directory, file_wildcard, output_file):
    """
    Write full paths of files with a specific extension in a directory to a text file.

    Args:
        directory (str): Path to the directory.
        file_extension (str): Desired file extension (e.g., '.txt', '.jpg').
        output_file (str): Path to the output text file.
    """

    with open(output_file, 'w') as f:
        for root, _, files in os.walk(directory):
            for file in files:
                if file_wildcard in file:
                    file_path = os.path.join(root, file)
                    f.write(file_path + '\n')

def makeFileListForLastools(inFolder,outName,extension):

    files = [os.path.join(inFolder,'') +
        file for file in os.listdir(inFolder) if file.endswith(extension)]

    fileList = os.path.join(inFolder,outName)
    with open(fileList , 'w') as fp:
        fp.write('\n'.join(files))

    return fileList

def executeLastoolsCommand(lastoolExec,lastoolCommandDict,outputDir=None):
    
       
    lastoolsCommand = lastoolExec
    for key, value in lastoolCommandDict.items():
        lastoolsCommand += ' '
        if not value:
            lastoolsCommand = lastoolsCommand + '-'+str(key)
        else:
            lastoolsCommand = lastoolsCommand + '-'+str(key)+' '+str(value)
    
    if outputDir is not None:
        os.makedirs(outputDir,exist_ok=True)
        lastoolsCommand = lastoolsCommand+' > '+os.path.join(outputDir,'lastools_output.txt')+' 2>&1' 
    else:
        lastoolsCommand = lastoolsCommand 
    with suppress_output():
        os.system(lastoolsCommand)
    return lastoolsCommand

def normalizeSpectraSumSquares(dataCube,noDataVal):
    """
    Normalize each spectrum independently in a hyperspectral data cube.

    Args:
        data_cube (numpy.ndarray): Input data cube with dimensions (rows, columns, bands).

    Returns:
        normalized_cube (numpy.ndarray): Data cube with normalized spectra.
        #Thanks to GPT-4
    """
    rows, cols, bands = dataCube.shape

    dtypeData = dataCube.dtype

    # Reshape the data cube into a 2D array (pixels, bands)
    data2d = dataCube.reshape(-1, bands).astype(np.float32)
    
    data2d[data2d==noDataVal] = np.nan

    # Normalize each spectrum independently
    normalizedData = data2d / (np.nansum(data2d**2,axis=1)**0.5)[:,None]

    # Reshape normalized data back to the original shape
    normalizedData = (normalizedData.reshape(rows, cols, bands))*10000
            
    normalizedData[dataCube == noDataVal] = noDataVal

    return normalizedData.astype(dtypeData)

def performMnf(dataCube):

    """
    Perform Minimum Noise Fraction (MNF) transformation on a hyperspectral data cube.

    Args:
        data_cube (numpy.ndarray): Input data cube with dimensions (rows, columns, bands).

    Returns:
        mnf_cube (numpy.ndarray): MNF-transformed data cube.
    """
    rows, cols, bands = dataCube.shape

    # Reshape the data cube into a 2D array (pixels, bands)
    data2d = dataCube.reshape(-1, bands)

    # Center the data
    meanSpectrum = np.mean(data2d, axis=0)
    centeredData = data2d - meanSpectrum

    # Perform PCA
    pca = PCA()
    pca_components = pca.fit_transform(centeredData)

    # Reorder the PCA components by eigenvalues
    sortedIndices = np.argsort(pca.explained_variance_)[::-1]
    mnfComponents = pca_components[:, sortedIndices]

    # Reshape MNF components back to the original shape
    mnfCube = mnfComponents.reshape(rows, cols, bands)
    mnfCube = np.divide((mnfCube-np.min(mnfCube,axis=(0,1))),(np.max(mnfCube,axis=(0,1))-np.min(mnfCube,axis=(0,1)))[None,None,:])
    return mnfCube

def performPca(dataCube):
    """
    Perform Principal Component Analysis (PCA) on a three-dimensional data cube.

    Args:
        data_cube (numpy.ndarray): Input data cube with dimensions (rows, columns, bands).

    Returns:
        pca_components (numpy.ndarray): PCA components with dimensions (rows, columns, bands).
        pca_explained_variance (numpy.ndarray): Explained variance ratio for each component.
    #Thanks to GPT-4
    """
    #rows, cols, bands = dataCube.shape

    # Reshape the data cube into a 2D array (pixels, bands)
    #data2d = dataCube.reshape(-1, bands)

    # Perform PCA
    pca = PCA()
    pcaComponents = pca.fit(dataCube)

    # Get explained variance ratio for each component
    pcaExplainedVariance = pca.explained_variance_ratio_

    return pca, pcaComponents, pcaExplainedVariance

def applyPca(pca,dataCube):
    
    rows, cols, bands = dataCube.shape

    # Reshape the data cube into a 2D array (pixels, bands)
    data2d = dataCube.reshape(-1, bands)

    pcaComponents = pca.transform(data2d)

    # Reshape PCA components back to the original shape
    pcaComponents = pcaComponents.reshape(rows, cols, bands)

    return pcaComponents

def findBadBands(wavelengths,badBandWindows):

    """
Find values in the data array that are outside of the specified ranges.

Args:
    wavelentghs(numpy.ndarray): Array of wavelengths.
    badBandWindows (list of numpy.ndarray): Variable number of ni,py range tuples, each containing minimum and maximum values.

Returns:
    outside_values (numpy.ndarray): Array of values outside of the specified ranges.
"""
#Thanks GPT4 for getting us started...but not getting it quite right.

    outsideMask = np.ones_like(wavelengths, dtype=bool)

    for min_val, max_val in badBandWindows:
        outsideMask *= (wavelengths < min_val) | (wavelengths > max_val)

    return outsideMask

def normalizeBands(dataCube):

    rows, cols, bands = dataCube.shape

    for band in np.arange(bands):
            dataCube[:, :, band] = StandardScaler().fit_transform(dataCube[:, :, band])

    return dataCube

def getDateRaster(h5ReflectanceFile):

    #Read in reflectance hdf5 file
    h5File = h5py.File(h5ReflectanceFile,'r')

    #Get the site name
    fileAttrsString = str(list(h5File.items()))
    fileAttrsStringSplit = fileAttrsString.split("'")
    sitename = fileAttrsStringSplit[1]

    #Extract the data selection index raster
    DataSelectionIndex = h5File[sitename]['Reflectance']['Metadata']['Ancillary_Imagery']['Data_Selection_Index']
    DataSelectionIndexArray = h5File[sitename]['Reflectance']['Metadata']['Ancillary_Imagery']['Data_Selection_Index'][()]
    #Create dictionary containing relevant metadata information
    DataSelectionIndexLookup = str(DataSelectionIndex.attrs['Data_Files'])
    DataSelectionIndexLookupSplit = DataSelectionIndexLookup.split(',')

    DateRaster = np.copy(DataSelectionIndex)
    #DateRaster = DateRaster.astype('uint32')
    RowCounter = 0

    for Row in DateRaster:
        CellCounter=0
        for Cell in Row:
            if Cell != -9999:

                H5FileName = DataSelectionIndexLookupSplit[Cell]
                Date = H5FileName.split('_')[4]
                DateRaster[RowCounter,CellCounter] = int(Date)
            else:
                DateRaster[RowCounter,CellCounter] = -9999
            CellCounter=CellCounter+1
        RowCounter = RowCounter+1

    metadata = {}
    rasterShape = DateRaster.shape
    metadata['noDataVal'] = -9999
    metadata['map info'] =  h5File[sitename]['Reflectance']['Metadata']['Coordinate_System']['Map_Info'][()]
    metadata['proj4'] =  h5File[sitename]['Reflectance']['Metadata']['Coordinate_System']['Proj4'][()]
    metadata['epsg'] = int(h5File[sitename]['Reflectance']['Metadata']['Coordinate_System']['EPSG Code'][()])

    mapInfo = h5File[sitename]['Reflectance']['Metadata']['Coordinate_System']['Map_Info'][()]
    mapInfo_string = str(mapInfo); #print('Map Info:',mapInfo_string)\n",
    mapInfo_split = mapInfo_string.split(",")

    metadata['res'] = {}
    metadata['res']['pixelWidth'] = float(mapInfo_split[5])
    metadata['res']['pixelHeight'] = float(mapInfo_split[6])

    xMin = float(mapInfo_split[3]) #convert from string to floating point number\n",
    yMax = float(mapInfo_split[4])
    #Calculate the xMax and yMin values from the dimensions\n",
    xMax = xMin + (rasterShape[1]*float(metadata['res']['pixelWidth'])) #xMax = left edge + (# of columns * resolution)\n",
    yMin = yMax - (rasterShape[0]*float(metadata['res']['pixelHeight'])) #yMin = top edge - (# of rows * resolution)\n",
    metadata['extent'] = (xMin,xMax,yMin,yMax)
    #Extract map information: spatial extent & resolution (pixel size)

    h5File.close

    return DateRaster,metadata

# TODO: Make this MULTIPROCESS

def combineChms(chmProcessingFiles):

    basenameList = []
    metaDataFileList = []

    for file in chmProcessingFiles:
        fileSplit = os.path.basename(file).split('_')
        basenameList.append('_'.join(fileSplit[:5]))

    uniqueBasenames = list(set(basenameList))
    stackedChmArrays = []
    for uniqueBasename in uniqueBasenames:

        matchingFiles = [file for file in chmProcessingFiles if uniqueBasename in file]
        chmArrayList = []
        metaDataFileList.append(matchingFiles[0])
        for matchingFile in matchingFiles:
            
            if os.path.getsize(matchingFile) < 10:
                continue

            ul_x, lr_y, lr_x, ul_y, nodata,chmArray = getCurrentFileExtentsAndData(matchingFile,Band=1)
            chmArrayList.append(chmArray)

        stackedChmArrays.append(np.max(np.stack(chmArrayList, axis=0),axis=0))

    return stackedChmArrays,metaDataFileList

def getMaxTwoRasters(gtifFileOne,gtifFileTwo):

    with rio.open(gtifFileOne) as src1, rio.open(gtifFileTwo) as src2:
        # Read raster data
        data1 = src1.read(1)
        data2 = src2.read(1)

        # Choose the maximum value for each cell
        maxData = np.maximum(data1, data2)

        # Create the output geotiff file
        metadata = src1.profile.copy()

    return maxData, metadata

def retileTiles(files,tileSize):

    Mosaic,MosaicExtents,EPSGCode,Nodata,RasterCellWidth,RasterCellHeight = getMosaicAndExtents(files)

    mosaicRows = Mosaic.shape[0]
    mosaicCols = Mosaic.shape[1]

    startX = int(MosaicExtents[0] - np.mod(MosaicExtents[0],tileSize))
    startY = int(MosaicExtents[3] - np.mod(MosaicExtents[3],tileSize)+tileSize)

    endX = int(MosaicExtents[2] - np.mod(MosaicExtents[2],tileSize))+tileSize;
    endY = int(MosaicExtents[1] - np.mod(MosaicExtents[1],tileSize));

    startIndexFootprintX = int(np.ceil(np.abs(startY - MosaicExtents[3])));
    startIndexFootprintY = int(np.ceil(np.abs(startX - MosaicExtents[0])));

    mosaicToTile = np.zeros((int(np.ceil((startY-endY))),int(np.ceil((endX-startX))),1),dtype=Mosaic.dtype)+Nodata

    mosaicToTile[startIndexFootprintX:startIndexFootprintX+mosaicRows,startIndexFootprintY:startIndexFootprintY+mosaicCols] = Mosaic

    tiles = []
    tileExtents = []
    tileExtent = {}
    tileExtent['ext_dict'] = {}

    for tileStartY in range(0,mosaicToTile.shape[0],tileSize):
        for tileStartX in range(0,mosaicToTile.shape[1],tileSize):
            tileChunk = mosaicToTile[tileStartY:tileStartY+tileSize,tileStartX:tileStartX+tileSize,0]
            if  np.all(tileChunk == Nodata):
                continue
            tileExtent['ext_dict']['xMin'] = tileStartX+startX
            tileExtent['ext_dict']['xMax'] = tileStartX+startX+tileSize
            tileExtent['ext_dict']['yMin'] = startY-tileStartY-tileSize
            tileExtent['ext_dict']['yMax'] = startY-tileStartY
            tiles.append(tileChunk)
            tileChunk = None
            tileExtents.append(tileExtent)
            tileExtent = {}
            tileExtent['ext_dict'] = {}

    return tiles,tileExtents

def writeTifHdf5Metadata(raster,metadata,outFile):

    EPSGCode = metadata['EPSG']
    MosaicExtents = metadata['extent']
    RasterCellHeight = int(float(metadata['res']['pixelHeight']))
    RasterCellWidth = int(float(metadata['res']['pixelWidth']))
    NoData = metadata['noDataVal']

    writeRasterToTif(outFile,raster,MosaicExtents,EPSGCode,NoData,RasterCellWidth,RasterCellHeight)

def generateDateTifs(h5File,outFolder):

    dateRaster,metadata = getDateRaster(h5File)
    basename = os.path.basename(h5File)
    outFile = os.path.join(outFolder,basename.replace('reflectance.h5','date.tif'))

    writeTifHdf5Metadata(dateRaster,metadata,outFile)

def generateQaTif(h5File,makeCog=True):

    qaRasterNames = ['Aerosol_Optical_Thickness','Aspect',\
                   'Cast_Shadow','Dark_Dense_Vegetation_Classification',\
                   'Haze_Cloud_Water_Map','Illumination_Factor','Path_Length',\
                   'Sky_View_Factor','Slope','Smooth_Surface_Elevation',\
                   'Visibility_Index_Map','Water_Vapor_Column','to-sensor_Azimuth_Angle',\
                   'to-sensor_Zenith_Angle']

    qaRasterArrayList = []

    for qaRasterName in qaRasterNames:
        qaArray, qaMetadata, wavelengths = h5refl2array(h5File, qaRasterName)
        qaArray = qaArray.astype(np.float32)
        qaArray[qaArray==qaMetadata['noDataVal']]=-9999
        qaRasterArrayList.append(qaArray)

    if 'DP3' in h5File:

        qaArray, qaMetadata, wavelengths = h5refl2array(h5File, 'Weather_Quality_Indicator')
        qaArray = qaArray.astype(np.float32)
        weatherArray = np.zeros_like(qaArray[:,:,0])
        weatherArray[np.multiply(np.multiply(qaArray[:,:,0]==255,qaArray[:,:,1]==0),qaArray[:,:,2]==0)] = 3
        weatherArray[np.multiply(np.multiply(qaArray[:,:,0]==255,qaArray[:,:,1]==255),qaArray[:,:,2]==0)] = 2
        weatherArray[np.multiply(np.multiply(qaArray[:,:,0]==0,qaArray[:,:,1]==255),qaArray[:,:,2]==0)]  = 1
        weatherArray[weatherArray==0] = -9999
        qaRasterArrayList.append(weatherArray)
        dateRaster,dateRasterMetadata = getDateRaster(h5File)
        qaRasterArrayList.append(dateRaster.astype(np.float32))

    fullQaArray = np.stack(qaRasterArrayList,axis=0)

    EPSGCode,MosaicExtents,RasterCellHeight,RasterCellWidth,NoData = getGtifMetadataFromH5Metadata(qaMetadata)

    outFile = h5File.replace('reflectance.h5','ReflectanceQA.tif')
    writeRasterToTif(outFile,np.moveaxis(fullQaArray,[0,1,2],[2,0,1]),MosaicExtents,EPSGCode,NoData,RasterCellWidth,RasterCellHeight)

    if makeCog:
        writeRasterToCog(outFile)

def generateTifs(h5File,rasterName,outFile,makeCog=True,bands='all'):

    if outFile == 'same':
        outFile = h5File.replace('reflectance.h5',rasterName+'.tif')
    
    if bands != 'all':
        bandIndexesToread = [np.min(bands),np.max(bands)]
    
    array, Metadata, wavelengths = h5refl2array(h5File, rasterName,bandIndexesToRead = bandIndexesToread)
    
    EPSGCode,MosaicExtents,RasterCellHeight,RasterCellWidth,NoData = getGtifMetadataFromH5Metadata(Metadata)

    if bands != 'all':
        bands = bands - np.min(bands)
        bands[-1] = bands[-1]-1
        array = array[:,:,bands]

    writeRasterToTif(outFile,array,MosaicExtents,EPSGCode,NoData,RasterCellWidth,RasterCellHeight)
    if makeCog:
        writeRasterToCog(outFile)

    return outFile

def getCircularMask(buffer):

    size = buffer*2+1
    y,x = np.ogrid[-buffer:size-buffer, -buffer:size-buffer]
    mask = x*x + y*y <= buffer**2

    array = np.ones((size, size))
    array[~mask] = np.nan

    return array

def run_system_command(cmd):
    with Popen(cmd, stdout=PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='') # print standard output for each line
        if p.returncode != 0:
            print('returncode:',p.returncode)
    return cmd

def differenceH5Files(h5File1,h5File2,outputFormat,outDir = None,skipRasters = [],filePathToEnviProjCs=None,spatialIndexesToRead = None,bandIndexesToRead = None):

    if outDir is None:
        outDir = os.path.dirname(h5_filename)
        
    h5File1Name=h5File1.replace('\\','/')
    h5File1NameSplit = h5File1Name.split('/')

    h5File2Name=h5File2.replace('\\','/')
    h5File2NameSplit = h5File2Name.split('/')

    yearNum = int(h5File1NameSplit[1])

    print('In getAll Rasters')
    if 'Reflectance' in h5File1Name:
        rasterNames = ['Reflectance','Aerosol_Optical_Thickness','Aspect',\
                      'Cast_Shadow','Dark_Dense_Vegetation_Classification',\
                      'Haze_Cloud_Water_Map','Illumination_Factor','Path_Length',\
                      'Sky_View_Factor','Slope','Smooth_Surface_Elevation',\
                      'Visibility_Index_Map','Water_Vapor_Column','to-sensor_Azimuth_Angle',\
                      'to-sensor_Zenith_Angle']

        if 'DP3' in h5File1Name:
            rasterNames.append('Data_Selection_Index')
            rasterNames.append('Weather_Quality_Indicator')
            rasterNames.append('Aquisition_Date')
    elif 'Radiance' in h5File1Name:
        rasterNames = ['Radiance','IGM_Data','OBS_Data']

        if yearNum == 2013 or yearNum >= 2019:
            rasterNames.append('GLT_Data')

    if skipRasters:
        
        for skipRaster in skipRasters:
            rasterNames.remove(skipRaster)

    for rasterName in rasterNames:
        print(rasterName)
        t1 = time.time()

        if 'reflectance' in h5File1Name:
            outFile = os.path.join(outDir,os.path.basename(h5File1Name).replace('reflectance.h5',rasterName))
        elif 'radiance' in h5File1Name:
            outFile = os.path.join(outDir,os.path.basename(h5File1Name).replace('radiance.h5',rasterName))

        if os.path.exists(outFile):
            print('Already exists: ' + outFile)
            continue
        
        if outputFormat == 'gtif':
            outFile = outFile+'_difference.tif'
            convertH5RasterDifferenceToGtif(h5File1Name,h5File2Name,rasterName,outFile,spatialIndexesToRead = spatialIndexesToRead,bandIndexesToRead = bandIndexesToRead)
#        elif outputFormat == 'ENVI':
#            outFile = outFile+'_difference.bsq'
#            convertH5RasterToEnvi(h5_filename,rasterName,outFile,filePathToEnviProjCs,spatialIndexesToRead = spatialIndexesToRead,bandIndexesToRead = bandIndexesToRead)

def getAllRastersFromH5(h5_filename,outputFormat='gtif',outDir = None,skipRasters = [],keepRasters = [],filePathToEnviProjCs=None,spatialIndexesToRead = None,bandIndexesToRead = None):
        if outDir is None:
            outDir = os.path.dirname(h5_filename)
            
        h5_filename=h5_filename.replace('\\','/')
        h5_filename_split = h5_filename.split('/')

        yearNum = int(h5_filename_split[1])

        print('In getAll Rasters')

        if keepRasters:
            rasterNames = keepRasters
        else:

            if 'Reflectance' in h5_filename:
                rasterNames = ['Reflectance','Aerosol_Optical_Thickness','Aspect',\
                              'Cast_Shadow','Dark_Dense_Vegetation_Classification',\
                              'Haze_Cloud_Water_Map','Illumination_Factor','Path_Length',\
                              'Sky_View_Factor','Slope','Smooth_Surface_Elevation',\
                              'Visibility_Index_Map','Water_Vapor_Column','to-sensor_Azimuth_Angle',\
                              'to-sensor_Zenith_Angle']

                if 'DP3' in h5_filename:
                    rasterNames.append('Data_Selection_Index')
                    rasterNames.append('Weather_Quality_Indicator')
                    rasterNames.append('Aquisition_Date')
            elif 'Radiance' in h5_filename:
                rasterNames = ['Radiance','IGM_Data','OBS_Data']

                if yearNum == 2013 or yearNum >= 2019:
                    rasterNames.append('GLT_Data')

            if skipRasters:
                for skipRaster in skipRasters:
                    rasterNames.remove(skipRaster)

        for rasterName in rasterNames:
            print(rasterName)
            t1 = time.time()

            if 'reflectance' in h5_filename:
                outFile = os.path.join(outDir,os.path.basename(h5_filename).replace('reflectance.h5',rasterName))
            elif 'radiance' in h5_filename:
                outFile = os.path.join(outDir,os.path.basename(h5_filename).replace('radiance.h5',rasterName))

            if os.path.exists(outFile):
                print('Already exists: ' + outFile)
                continue
            
            if outputFormat == 'gtif':
                outFile = outFile+'.tif'
                convertH5RasterToGtif(h5_filename,rasterName,outFile,spatialIndexesToRead = spatialIndexesToRead,bandIndexesToRead = bandIndexesToRead)
            elif outputFormat == 'ENVI':
                outFile = outFile+'.bsq'
                convertH5RasterToEnvi(h5_filename,rasterName,outFile,filePathToEnviProjCs,spatialIndexesToRead = spatialIndexesToRead,bandIndexesToRead = bandIndexesToRead)

def getGtifMetadataFromH5Metadata(h5Metadata):
    
    EPSGCode = h5Metadata['EPSG']
    MosaicExtents = np.array(h5Metadata['extent'],dtype=np.float)
    RasterCellHeight = int(float(h5Metadata['res']['pixelHeight']))
    RasterCellWidth = int(float(h5Metadata['res']['pixelWidth']))
    NoData = h5Metadata['noDataVal']
    
    return EPSGCode,MosaicExtents,RasterCellHeight,RasterCellWidth,NoData

def convertH5RasterToEnvi(h5File,rasterName,outEnviFile,filePathToEnviProjCs,spatialIndexesToRead = None,bandIndexesToRead = None):

    raster, metadata, wavelengths = h5refl2array(h5File, rasterName,spatialIndexesToRead = spatialIndexesToRead, bandIndexesToRead = bandIndexesToRead )
    
    writeRasterToEnvi(raster, rasterName, metadata, wavelengths,outEnviFile,filePathToEnviProjCs)

def writeRasterToEnvi(raster, rasterName,metadata, wavelengths,outEnviFile,filePathToEnviProjCs):

    if raster.ndim == 2:
        raster = raster[:,:,None]
        metadata['shape'] = raster.shape
        
    if isinstance(metadata["bandNames"],np.ndarray):
        metadata["bandNames"] = metadata["bandNames"].astype('U')[0]
    
    enviProjCs = parseEnviProjCs(filePathToEnviProjCs,metadata['EPSG'])
    
    if 'Sky_View_Factor' == rasterName or 'Cast_Shadow' == rasterName:
        raster = raster.astype(np.uint8)
    if 'Slope' == rasterName or 'Smooth_Surface_Elevation' == rasterName or 'Aspect' == rasterName:
        raster = raster.astype(np.float64)

    gdalDtype = getRasterGdalDtype(raster.dtype)
    
    enviMetadata = {
    "geotransform": (metadata['ext_dict']['xMin'], metadata['res']['pixelWidth'],-0.0,metadata['ext_dict']['yMax'],0.0,-1.0*metadata['res']['pixelHeight']),
    "projection": enviProjCs,
    "data_type": gdalDtype,
    "data_ignore_value": float(metadata['noDataVal']),
    "num_bands": int(metadata['shape'][2]),
    "width": int(metadata['shape'][1]),
    "height": int(metadata['shape'][0]),
    "band_names": metadata["bandNames"].split(','),
    "wavelengths": metadata["wavelengths"],
    "fwhm": metadata["fwhm"]
    # Add more metadata fields as needed
    }
    
    
    if 'Classes' in metadata:
        enviMetadata['Classes'] = metadata['Classes']
    if 'Class_Lookup' in metadata:
        enviMetadata['Class_Lookup'] = metadata['Class_Lookup']
    if 'Class_Names' in metadata:
        enviMetadata['Class_Names'] = metadata['Class_Names']
        

    needsBandMetadata = False
    if rasterName == 'Reflectance' or rasterName == 'Radiance':
        needsBandMetadata = True

    writeEnviRaster(outEnviFile,np.squeeze(raster),enviMetadata,needsBandMetadata=needsBandMetadata)

def convertH5RasterDifferenceToGtif(h5File1,h5File2,rasterName,outTif,spatialIndexesToRead = None,bandIndexesToRead = None):

    array1, Metadata1, wavelengths1 = h5refl2array(h5File1, rasterName,spatialIndexesToRead = spatialIndexesToRead,bandIndexesToRead = bandIndexesToRead)

    array2, Metadata2, wavelengths2 = h5refl2array(h5File2, rasterName,spatialIndexesToRead = spatialIndexesToRead,bandIndexesToRead = bandIndexesToRead)

    difference = array1-array2
    
    difference[array1==Metadata1['noDataVal']] = Metadata1['noDataVal']

    difference[array2==Metadata2['noDataVal']] = Metadata1['noDataVal']

    EPSGCode,MosaicExtents,RasterCellHeight,RasterCellWidth,NoData = getGtifMetadataFromH5Metadata(Metadata1)
    
    if np.all(difference[difference!=Metadata1['noDataVal']] == 0):
        print(rasterName+' for file '+os.path.basename(h5File1)+' and '+os.path.basename(h5File2) + ' showed no difference')
    else:
        writeRasterToTif(outTif,difference,MosaicExtents,EPSGCode,NoData,RasterCellWidth,RasterCellHeight)

def convertH5RasterToGtif(h5File,rasterName,outTif,spatialIndexesToRead = None,bandIndexesToRead = None):

    array, metadata, wavelengths = h5refl2array(h5File, rasterName,spatialIndexesToRead = spatialIndexesToRead,bandIndexesToRead = bandIndexesToRead)

    writeRasterToGtif(array, metadata, wavelengths,outTif)
    
def writeRasterToGtif(array,metadata, wavelengths,outTif):
    
    EPSGCode,MosaicExtents,RasterCellHeight,RasterCellWidth,NoData = getGtifMetadataFromH5Metadata(metadata)

    writeRasterToTif(outTif,array,MosaicExtents,EPSGCode,NoData,RasterCellWidth,RasterCellHeight)

def h5refl2array(h5_filename, raster, onlyMetadata = False, spatialIndexesToRead = None, bandIndexesToRead = None):
    hdf5_file = h5py.File(h5_filename,'r')
    #print('h5refl2array')
    #Get the site name
    sitename = str(list(hdf5_file.items())).split("'")[1]
    productType = str(list(hdf5_file[sitename].items())).split("'")[1]


    if productType == 'Reflectance':
        productBaseLoc = hdf5_file[sitename]['Reflectance']
    elif productType == 'Radiance':
        productBaseLoc = hdf5_file[sitename]['Radiance']

    if raster == 'Reflectance':
        raster = 'Reflectance_Data'
        productLoc = productBaseLoc
    elif raster == 'Radiance':
        productLoc = productBaseLoc
    elif raster == 'to-sensor_Azimuth_Angle' or raster == 'to-sensor_Zenith_Angle':
        productLoc = productBaseLoc['Metadata']
    elif raster == 'GLT_Data' or raster == 'IGM_Data' or raster == 'OBS_Data':
        productLoc = productBaseLoc['Metadata']['Ancillary_Rasters']
    else:
        productLoc = productBaseLoc['Metadata']['Ancillary_Imagery']

    if 'DP3' in h5_filename and raster == 'to-sensor_Azimuth_Angle':
        raster = 'to-sensor_azimuth_angle'

    if 'DP3' in h5_filename and raster == 'to-sensor_Zenith_Angle':
        raster = 'to-sensor_zenith_angle'

    metadata = {}
    if raster == 'Radiance':
         rasterArray = productLoc['RadianceDecimalPart']
    else:
        rasterArray = productLoc[raster]
        
    if raster == 'Reflectance_Data':
        metadata['bandNames'] = 'Reflectance'
    elif raster == 'Radiance':
        metadata['bandNames'] = 'Radiance'
    elif raster == 'BDE':
        metadata['bandNames'] = 'BadDetectorElements'
    else:
        if 'Band_Names' in rasterArray.attrs:
            metadata['bandNames'] = rasterArray.attrs['Band_Names']
    
    
    if 'Scale_Factor' in rasterArray.attrs:
        metadata['scaleFactor'] = float(rasterArray.attrs['Scale_Factor'])
    elif 'Scale' in rasterArray.attrs:
        metadata['scaleFactor'] = float(rasterArray.attrs['Scale'])
    else:
        metadata['scaleFactor'] = 1.0

    rasterShape = rasterArray.shape
    wavelengths = productBaseLoc['Metadata']['Spectral_Data']['Wavelength'][:]
    metadata['fwhm'] = productBaseLoc['Metadata']['Spectral_Data']['FWHM'][:]
    metadata['wavelengths'] = wavelengths
    
    #Create dictionary containing relevant metadata information    #Create dictionary containing relevant metadata information
    
    metadata['shape'] = rasterShape
    metadata['mapInfo'] = productBaseLoc['Metadata']['Coordinate_System']['Map_Info'][()]
    #Extract no data value & set no data value to NaN\n",

    
    if raster == 'Reflectance_Data':

        metadata['bad_band_window1'] = (productLoc.attrs['Band_Window_1_Nanometers'])
        metadata['bad_band_window2'] = (productLoc.attrs['Band_Window_2_Nanometers'])

    metadata['projection'] = productBaseLoc['Metadata']['Coordinate_System']['Proj4'][()]
    metadata['EPSG'] = int(productBaseLoc['Metadata']['Coordinate_System']['EPSG Code'][()])
    mapInfo = productBaseLoc['Metadata']['Coordinate_System']['Map_Info'][()]
    mapInfo_string = str(mapInfo); #print('Map Info:',mapInfo_string)\n",
    mapInfo_split = mapInfo_string.split(",")
    #Extract the resolution & convert to floating decimal number
    metadata['res'] = {}
    metadata['res']['pixelWidth'] = float(mapInfo_split[5])
    metadata['res']['pixelHeight'] = float(mapInfo_split[6])
    #Extract the upper left-hand corner coordinates from mapInfo\n",
    xMin = float(mapInfo_split[3]) #convert from string to floating point number\n",
    yMax = float(mapInfo_split[4])
    #Calculate the xMax and yMin values from the dimensions\n",
    xMax = xMin + (rasterShape[1]*float(metadata['res']['pixelWidth'])) #xMax = left edge + (# of columns * resolution)\n",
    yMin = yMax - (rasterShape[0]*float(metadata['res']['pixelHeight'])) #yMin = top edge - (# of rows * resolution)\n",
    metadata['extent'] = (xMin,xMax,yMin,yMax)
    metadata['ext_dict'] = {}
    metadata['ext_dict']['xMin'] = xMin
    metadata['ext_dict']['xMax'] = xMax
    metadata['ext_dict']['yMin'] = yMin
    metadata['ext_dict']['yMax'] = yMax

    if 'Classes' in rasterArray.attrs:
        metadata['Classes'] = float(rasterArray.attrs['Classes'])
    if 'Class_Lookup' in rasterArray.attrs:
        metadata['Class_Lookup'] = rasterArray.attrs['Class_Lookup'].astype(np.float32)
    if 'Class_Names' in rasterArray.attrs:
        metadata['Class_Names'] = rasterArray.attrs['Class_Names']
    if spatialIndexesToRead is None or spatialIndexesToRead == 'all':
        
        indexesRow = (int(0),int(rasterShape[0]))
        indexesCol = (int(0),int(rasterShape[1]))
        spatialIndexesToRead = [indexesRow,indexesCol]
    
    if 'Data_Ignore_Value' in rasterArray.attrs:
        metadata['noDataVal'] = float(rasterArray.attrs['Data_Ignore_Value'])                                               
    if onlyMetadata:
        hdf5_file.close()
        rasterArray = []
    elif raster == 'Radiance' or raster == 'Reflectance_Data':  
        
        if bandIndexesToRead is None:
            bandIndexesToRead = (int(0),int(rasterShape[2]))
            
        if raster == 'Reflectance_Data':
            rasterArray = rasterArray[spatialIndexesToRead[0][0]:spatialIndexesToRead[0][1],spatialIndexesToRead[1][0]:spatialIndexesToRead[1][1],bandIndexesToRead[0]:bandIndexesToRead[1]]
        elif raster == 'Radiance': 
        
            rasterArray = productLoc['RadianceIntegerPart'][spatialIndexesToRead[0][0]:spatialIndexesToRead[0][1],spatialIndexesToRead[1][0]:spatialIndexesToRead[1][1],bandIndexesToRead[0]:bandIndexesToRead[1]] + productLoc['RadianceDecimalPart'][spatialIndexesToRead[0][0]:spatialIndexesToRead[0][1],spatialIndexesToRead[1][0]:spatialIndexesToRead[1][1],bandIndexesToRead[0]:bandIndexesToRead[1]]/metadata['scaleFactor']
            rasterArray[rasterArray==productLoc['RadianceIntegerPart'].attrs['Data_Ignore_Value']+productLoc['RadianceDecimalPart'].attrs['Data_Ignore_Value']/metadata['scaleFactor']]=-9999
            metadata['noDataVal'] = -9999
    else:
        
        rasterArray = rasterArray[spatialIndexesToRead[0][0]:spatialIndexesToRead[0][1],spatialIndexesToRead[1][0]:spatialIndexesToRead[1][1]] 
        if 'noDataVal' not in metadata.keys():

            if rasterArray.dtype == np.uint8:
                metadata['noDataVal'] = 0
            else:
                metadata['noDataVal'] = -9999
        
        hdf5_file.close()

    return rasterArray, metadata, wavelengths

def reflectanceResampling(h5File,bandMin,bandMax):

    reflArray, metadata, wavelengths = h5refl2array(h5File,'Reflectance',onlyMetadata =True)
    wavelengths = wavelengths[:]

    if (bandMax-bandMin < 6):

        bandMean = (bandMax+bandMin)/2
        indexOfBand = np.where(np.abs(wavelengths-bandMean) == np.min(np.abs(wavelengths-bandMean)))[0]
        reflArray, metadata, wavelengths=h5refl2array(h5File,'Reflectance',bandIndexesToRead = [int(indexOfBand),int(indexOfBand+1)])
        return reflArray

    #Get the shoulder and center band indices. "Left" is short wavelength, "Right" is longer, "Center" is everything in between.
    LeftShoulderLimit = bandMin - 2.9
    LeftShoulderMax = bandMin + 2.9
    LeftShoulderIndices = np.where((wavelengths >= LeftShoulderLimit) == (wavelengths < LeftShoulderMax))

    if len(LeftShoulderIndices) == 0:
        LeftShoulderIndices = []

    RightShoulderMax = bandMax - 2.9
    RightShoulderLimit = bandMax + 2.9
    RightShoulderIndices = np.where((wavelengths > RightShoulderMax) == (wavelengths < RightShoulderLimit))

    if len(RightShoulderIndices) == 0:
        RightShoulderIndices = []
    CenterPassIndices = np.where((wavelengths > LeftShoulderMax) == (wavelengths < RightShoulderMax))

    #Compute the weights for the input bands to make our composite band
    sigma = 2.5
    if len(LeftShoulderIndices) == 0:
      LeftShoulderWeights = []
    else:
      LeftShoulderWeights = np.exp(-(wavelengths[LeftShoulderIndices] - LeftShoulderMax)**2/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))* 6.25

    if len(RightShoulderIndices) == 0:
      RightShoulderWeights = []
    else:
      RightShoulderWeights = np.exp(-(wavelengths[RightShoulderIndices] - RightShoulderMax)**2/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))* 6.25

    CenterPassWeights = np.zeros(len(CenterPassIndices[0])) + 1.0
    ThisBandIndices = np.concatenate((LeftShoulderIndices[0],CenterPassIndices[0],RightShoulderIndices[0]))
    ThisBandWeights = np.concatenate((LeftShoulderWeights,CenterPassWeights,RightShoulderWeights))

    ThisBandIndices = np.unique(ThisBandIndices)
    theseInputBands, metadata, wavelengths = h5refl2array(h5File,'Reflectance',bandIndexesToRead = [np.min(ThisBandIndices),np.max(ThisBandIndices)+1])

    thisSpectrallyResampledBand = np.sum(theseInputBands*ThisBandWeights,axis=2)/np.sum(ThisBandWeights)

    thisSpectrallyResampledBand[theseInputBands[:,:,0]==metadata['noDataVal']]  = metadata['noDataVal']

    return thisSpectrallyResampledBand

def getGeneralIndexRatio(Band1,Band2):

    return (Band1-Band2)/(Band1+Band2)

def getGeneralIndexRatioError(Band1,Band2,varBand1,varBand2,covarBand1Band2):

    return (covarBand1Band2*(1/(Band2 + Band1) + (Band2 - Band1)/(Band2 + Band1)**2) - varBand2*(1/(Band2 + Band1) - (Band2 - Band1)/(Band2 + Band1)**2))*((np.conj(Band2) - np.conj(Band1))/(np.conj(Band2) + np.conj(Band1))**2 - 1/(np.conj(Band2) + np.conj(Band1))) - (covarBand1Band2*(1/(Band2 + Band1) - (Band2 - Band1)/(Band2 + Band1)**2) - varBand1*(1/(Band2 + Band1) + (Band2 - Band1)/(Band2 + Band1)**2))*((np.conj(Band2) - np.conj(Band1))/(np.conj(Band2) + np.conj(Band1))**2 + 1/(np.conj(Band2) + np.conj(Band1)))

def getSimpleRatio(Band1,Band2):

    return Band1/Band2

def getSimpleRatioError(Band1,Band2,varBand1,varBand2,covarBand1Band2):

    return (varBand1/Band2 - (covarBand1Band2*Band1)/Band2**2)/np.conj(Band2) - (np.conj(Band1)*(covarBand1Band2/Band2 - (Band1*varBand2)/Band2**2))/np.conj(Band2)**2

def getEvi(NIR,Red,Blue,G,C1,C2,L):

    return G*((NIR-Red)/(NIR+C1*Red-C2*Blue+L))

def getEviError(NIR,Red,Blue,varNirBand,varRedBand,varBlueBand,covarNirRedBand,covarNirBlueBand,covarRedBlueBand,G,C1,C2,L):

    return (np.conj(G)/(np.conj(L) + np.conj(NIR) + np.conj(C1)*np.conj(Red) - np.conj(C2)*np.conj(Blue)) - (np.conj(C1)*np.conj(G)*(np.conj(Red) - np.conj(NIR)))/(np.conj(L) + np.conj(NIR) + np.conj(C1)*np.conj(Red) - np.conj(C2)*np.conj(Blue))**2)*(varRedBand*(G/(L + NIR + C1*Red - C2*Blue) - (C1*G*(Red - NIR))/(L + NIR + C1*Red - C2*Blue)**2) - covarNirRedBand*(G/(L + NIR + C1*Red - C2*Blue) + (G*(Red - NIR))/(L + NIR + C1*Red - C2*Blue)**2) + (C2*G*covarRedBlueBand*(Red - NIR))/(L + NIR + C1*Red - C2*Blue)**2) - (np.conj(G)/(np.conj(L) + np.conj(NIR) + np.conj(C1)*np.conj(Red) - np.conj(C2)*np.conj(Blue)) + (np.conj(G)*(np.conj(Red) - np.conj(NIR)))/(np.conj(L) + np.conj(NIR) + np.conj(C1)*np.conj(Red) - np.conj(C2)*np.conj(Blue))**2)*(covarNirRedBand*(G/(L + NIR + C1*Red - C2*Blue) - (C1*G*(Red - NIR))/(L + NIR + C1*Red - C2*Blue)**2) - varNirBand*(G/(L + NIR + C1*Red - C2*Blue) + (G*(Red - NIR))/(L + NIR + C1*Red - C2*Blue)**2) + (C2*G*covarNirBlueBand*(Red - NIR))/(L + NIR + C1*Red - C2*Blue)**2) + (np.conj(C2)*np.conj(G)*(np.conj(Red) - np.conj(NIR))*(covarRedBlueBand*(G/(L + NIR + C1*Red - C2*Blue) - (C1*G*(Red - NIR))/(L + NIR + C1*Red - C2*Blue)**2) - covarNirBlueBand*(G/(L + NIR + C1*Red - C2*Blue) + (G*(Red - NIR))/(L + NIR + C1*Red - C2*Blue)**2) + (C2*G*varBlueBand*(Red - NIR))/(L + NIR + C1*Red - C2*Blue)**2))/(np.conj(L) + np.conj(NIR) + np.conj(C1)*np.conj(Red) - np.conj(C2)*np.conj(Blue))**2

def getArvi(NIR,Red,Blue,gamma):

    return (NIR -(Red-gamma*(Blue-Red)))/(NIR +(Red-gamma*(Blue-Red)))

def getArviError(NIR,Red,Blue,varNirBand,varRedBand,varBlueBand,covarNirRedBand,covarNirBlueBand,covarRedBlueBand,gamma):

    return (np.conj(gamma)/(np.conj(Red) + np.conj(NIR) + np.conj(gamma)*(np.conj(Red) - np.conj(Blue))) - (np.conj(gamma)*(np.conj(Red) - np.conj(NIR) + np.conj(gamma)*(np.conj(Red) - np.conj(Blue))))/(np.conj(Red) + np.conj(NIR) + np.conj(gamma)*(np.conj(Red) - np.conj(Blue)))**2)*(covarNirBlueBand*(1/(Red + NIR + gamma*(Red - Blue)) + (Red - NIR + gamma*(Red - Blue))/(Red + NIR + gamma*(Red - Blue))**2) - covarRedBlueBand*((gamma + 1)/(Red + NIR + gamma*(Red - Blue)) - ((gamma + 1)*(Red - NIR + gamma*(Red - Blue)))/(Red + NIR + gamma*(Red - Blue))**2) + varBlueBand*(gamma/(Red + NIR + gamma*(Red - Blue)) - (gamma*(Red - NIR + gamma*(Red - Blue)))/(Red + NIR + gamma*(Red - Blue))**2)) - ((np.conj(gamma) + 1)/(np.conj(Red) + np.conj(NIR) + np.conj(gamma)*(np.conj(Red) - np.conj(Blue))) - ((np.conj(gamma) + 1)*(np.conj(Red) - np.conj(NIR) + np.conj(gamma)*(np.conj(Red) - np.conj(Blue))))/(np.conj(Red) + np.conj(NIR) + np.conj(gamma)*(np.conj(Red) - np.conj(Blue)))**2)*(covarNirRedBand*(1/(Red + NIR + gamma*(Red - Blue)) + (Red - NIR + gamma*(Red - Blue))/(Red + NIR + gamma*(Red - Blue))**2) - varRedBand*((gamma + 1)/(Red + NIR + gamma*(Red - Blue)) - ((gamma + 1)*(Red - NIR + gamma*(Red - Blue)))/(Red + NIR + gamma*(Red - Blue))**2) + covarRedBlueBand*(gamma/(Red + NIR + gamma*(Red - Blue)) - (gamma*(Red - NIR + gamma*(Red - Blue)))/(Red + NIR + gamma*(Red - Blue))**2)) + ((np.conj(Red) - np.conj(NIR) + np.conj(gamma)*(np.conj(Red) - np.conj(Blue)))/(np.conj(Red) + np.conj(NIR) + np.conj(gamma)*(np.conj(Red) - np.conj(Blue)))**2 + 1/(np.conj(Red) + np.conj(NIR) + np.conj(gamma)*(np.conj(Red) - np.conj(Blue))))*(varNirBand*(1/(Red + NIR + gamma*(Red - Blue)) + (Red - NIR + gamma*(Red - Blue))/(Red + NIR + gamma*(Red - Blue))**2) - covarNirRedBand*((gamma + 1)/(Red + NIR + gamma*(Red - Blue)) - ((gamma + 1)*(Red - NIR + gamma*(Red - Blue)))/(Red + NIR + gamma*(Red - Blue))**2) + covarNirBlueBand*(gamma/(Red + NIR + gamma*(Red - Blue)) - (gamma*(Red - NIR + gamma*(Red - Blue)))/(Red + NIR + gamma*(Red - Blue))**2))

def getSavi(NIR,Red,L):

    return ((NIR-Red)/(NIR+Red+L))*(1+L)

def getSaviError(NIR,Red,varNirBand,varRedBand,covarNirRedBand,L):

    return - ((np.conj(L) + 1)/(np.conj(L) + np.conj(Red) + np.conj(NIR)) - ((np.conj(L) + 1)*(np.conj(Red) - np.conj(NIR)))/(np.conj(L) + np.conj(Red) + np.conj(NIR))**2)*(covarNirRedBand*((L + 1)/(L + Red + NIR) + ((Red - NIR)*(L + 1))/(L + Red + NIR)**2) - varRedBand*((L + 1)/(L + Red + NIR) - ((Red - NIR)*(L + 1))/(L + Red + NIR)**2)) - ((np.conj(L) + 1)/(np.conj(L) + np.conj(Red) + np.conj(NIR)) + ((np.conj(L) + 1)*(np.conj(Red) - np.conj(NIR)))/(np.conj(L) + np.conj(Red) + np.conj(NIR))**2)*(covarNirRedBand*((L + 1)/(L + Red + NIR) - ((Red - NIR)*(L + 1))/(L + Red + NIR)**2) - varNirBand*((L + 1)/(L + Red + NIR) + ((Red - NIR)*(L + 1))/(L + Red + NIR)**2))

def getNmdi(NmdiBand1,NmdiBand2,NmdiBand3):

    return (NmdiBand1 - (NmdiBand2-NmdiBand3))/(NmdiBand1 + (NmdiBand2-NmdiBand3))

def getNmdiError(NmdiBand1,NmdiBand2,NmdiBand3,varNmdi1Band,varNmdi2Band,varNmdi3Band,covarNmdi1BandNmdi2Band,covarNmdi1BandNmdi3Band,covarNmdi2BandNmdi3Band):

    return ((np.conj(NmdiBand1) - np.conj(NmdiBand2) + np.conj(NmdiBand3))/(np.conj(NmdiBand1) + np.conj(NmdiBand2) - np.conj(NmdiBand3))**2 + 1/(np.conj(NmdiBand1) + np.conj(NmdiBand2) - np.conj(NmdiBand3)))*(covarNmdi1BandNmdi3Band*(1/(NmdiBand1 + NmdiBand2 - NmdiBand3) - (NmdiBand1 - NmdiBand2 + NmdiBand3)/(NmdiBand1 + NmdiBand2 - NmdiBand3)**2) - covarNmdi2BandNmdi3Band*(1/(NmdiBand1 + NmdiBand2 - NmdiBand3) + (NmdiBand1 - NmdiBand2 + NmdiBand3)/(NmdiBand1 + NmdiBand2 - NmdiBand3)**2) + varNmdi3Band*(1/(NmdiBand1 + NmdiBand2 - NmdiBand3) + (NmdiBand1 - NmdiBand2 + NmdiBand3)/(NmdiBand1 + NmdiBand2 - NmdiBand3)**2)) - ((np.conj(NmdiBand1) - np.conj(NmdiBand2) + np.conj(NmdiBand3))/(np.conj(NmdiBand1) + np.conj(NmdiBand2) - np.conj(NmdiBand3))**2 + 1/(np.conj(NmdiBand1) + np.conj(NmdiBand2) - np.conj(NmdiBand3)))*(covarNmdi1BandNmdi2Band*(1/(NmdiBand1 + NmdiBand2 - NmdiBand3) - (NmdiBand1 - NmdiBand2 + NmdiBand3)/(NmdiBand1 + NmdiBand2 - NmdiBand3)**2) + covarNmdi2BandNmdi3Band*(1/(NmdiBand1 + NmdiBand2 - NmdiBand3) + (NmdiBand1 - NmdiBand2 + NmdiBand3)/(NmdiBand1 + NmdiBand2 - NmdiBand3)**2) - varNmdi2Band*(1/(NmdiBand1 + NmdiBand2 - NmdiBand3) + (NmdiBand1 - NmdiBand2 + NmdiBand3)/(NmdiBand1 + NmdiBand2 - NmdiBand3)**2)) - ((np.conj(NmdiBand1) - np.conj(NmdiBand2) + np.conj(NmdiBand3))/(np.conj(NmdiBand1) + np.conj(NmdiBand2) - np.conj(NmdiBand3))**2 - 1/(np.conj(NmdiBand1) + np.conj(NmdiBand2) - np.conj(NmdiBand3)))*(covarNmdi1BandNmdi3Band*(1/(NmdiBand1 + NmdiBand2 - NmdiBand3) + (NmdiBand1 - NmdiBand2 + NmdiBand3)/(NmdiBand1 + NmdiBand2 - NmdiBand3)**2) - covarNmdi1BandNmdi2Band*(1/(NmdiBand1 + NmdiBand2 - NmdiBand3) + (NmdiBand1 - NmdiBand2 + NmdiBand3)/(NmdiBand1 + NmdiBand2 - NmdiBand3)**2) + varNmdi1Band*(1/(NmdiBand1 + NmdiBand2 - NmdiBand3) - (NmdiBand1 - NmdiBand2 + NmdiBand3)/(NmdiBand1 + NmdiBand2 - NmdiBand3)**2))

def getLai(Savi,A0,A1,A2):

    return (-1.0/A2) *np.log((A0 - Savi) / (A1))

def getLaiError(Savi,varSavi,A0,A1,A2):

    return varSavi/(A2*np.conj(A2)*(np.conj(A0) - np.conj(Savi))*(A0 - Savi))

def getFpar(NIR,Red,A0,A1,A2,A,B,C):

    return -C*(A*np.exp((B*np.log((A0 + ((3*Red)/2 - (3*NIR)/2)/(Red + NIR + 1/2))/A1))/A2) - 1)

def getFparError(NIR,Red,varNir,varRed,covarNirRed,A0,A1,A2,A,B,C):

    return (A**2*B**2*C**2*varRed*np.exp((2*B*np.log((A0 + ((3*Red)/2 - (3*NIR)/2)/(Red + NIR + 1/2))/A1))/A2)*(3/(2*(Red + NIR + 1/2)) - ((3*Red)/2 - (3*NIR)/2)/(Red + NIR + 1/2)**2)**2)/(A2**2*(A0 + ((3*Red)/2 - (3*NIR)/2)/(Red + NIR + 1/2))**2) + (A**2*B**2*C**2*varNir*np.exp((2*B*np.log((A0 + ((3*Red)/2 - (3*NIR)/2)/(Red + NIR + 1/2))/A1))/A2)*(3/(2*(Red + NIR + 1/2)) + ((3*Red)/2 - (3*NIR)/2)/(Red + NIR + 1/2)**2)**2)/(A2**2*(A0 + ((3*Red)/2 - (3*NIR)/2)/(Red + NIR + 1/2))**2) - (2*A**2*B**2*C**2*covarNirRed*np.exp((2*B*np.log((A0 + ((3*Red)/2 - (3*NIR)/2)/(Red + NIR + 1/2))/A1))/A2)*(3/(2*(Red + NIR + 1/2)) + ((3*Red)/2 - (3*NIR)/2)/(Red + NIR + 1/2)**2)*(3/(2*(Red + NIR + 1/2)) - ((3*Red)/2 - (3*NIR)/2)/(Red + NIR + 1/2)**2))/(A2**2*(A0 + ((3*Red)/2 - (3*NIR)/2)/(Red + NIR + 1/2))**2);

def generateWaterIndices(h5_file,outFolder):
     #To get output filename
     
    #print('Working on: '+h5_file)
     
    h5_filebasename = os.path.basename(h5_file)
    h5_filebasename = h5_filebasename.replace('DP1','DP2')
    #Get metadata for output tif
    reflArray, metadata, wavelengths = h5refl2array(h5_file,'Reflectance',onlyMetadata=True)

    #Get bounds for output
    ul_x, lr_y, lr_x, ul_y,nodata, data_layer = getCurrentFileExtentsAndData(h5_file,Band=None)

    #Set bounds for output
    Bounds = np.array((ul_x, lr_y, lr_x, ul_y),dtype=np.float)

    #Get all required resampled bands
    resampledMsi1Ndii1Band = reflectanceResampling(h5_file ,811.5,826.5)
    nodataIndexes = np.where(resampledMsi1Ndii1Band == metadata['noDataVal'])
    resampledMsi1Ndii1Band = resampledMsi1Ndii1Band / metadata['scaleFactor']
    resampledMsi2Band = reflectanceResampling(h5_file ,1591.5,1606.5) / metadata['scaleFactor']
    resampledNdii2Band = reflectanceResampling(h5_file ,1640.5,1656.5) / metadata['scaleFactor']
    resampledNdwi1Band = reflectanceResampling(h5_file ,849.5,864.5) / metadata['scaleFactor']
    resampledNdwi2Band = reflectanceResampling(h5_file ,1232.5,1247.5) / metadata['scaleFactor']
    resampledNmdi1Band = reflectanceResampling(h5_file ,841.0,876.0) / metadata['scaleFactor']
    resampledNmdi2Band = reflectanceResampling(h5_file ,1628.0,1652.0) / metadata['scaleFactor']
    resampledNmdi3Band = reflectanceResampling(h5_file ,2105.0,2155.0) / metadata['scaleFactor']
    resampledWbi1Band = reflectanceResampling(h5_file ,892.5,907.5) / metadata['scaleFactor']
    resampledWbi2Band = reflectanceResampling(h5_file ,962.5,977.5) / metadata['scaleFactor']

    resampledWaterBand1 = reflectanceResampling(h5_file ,849,851)/metadata['scaleFactor']
    resampledWaterBand2 = reflectanceResampling(h5_file ,1599,1601)/metadata['scaleFactor']

    varMsi1Ndii1Band = (resampledMsi1Ndii1Band*0.05)**2
    varMsi2Band = (resampledMsi2Band*0.05)**2
    varNdii2Band = (resampledNdii2Band*0.05)**2
    varNdwi1Band = (resampledNdwi1Band*0.05)**2
    varNdwi2Band = (resampledNdwi2Band*0.05)**2
    varNmdi1Band = (resampledNmdi1Band*0.05)**2
    varNmdi2Band = (resampledNmdi2Band*0.05)**2
    varNmdi3Band = (resampledNmdi3Band*0.05)**2
    varWbi1Band = (resampledWbi1Band*0.05)**2
    varWbi2Band = (resampledWbi2Band*0.05)**2

    covarWbi1BandWbi2Band = 0
    covarMsi1Ndii1BandMsi2Band = 0
    covarNmdi1BandNmdi2Band = 0
    covarNmdi1BandNmdi3Band = 0
    covarNmdi2BandNmdi3Band = 0
    covarNdwi1BandNdwi2Band = 0
    covarNdii1BandNdii2Band = 0
    covarMsi1Ndii1BandNdii2Band = 0
    #WBI calcs

    WI_list = []
    WI_name_list = []

    WI_list.append(getSimpleRatio(resampledWbi1Band,resampledWbi2Band))
    WI_list.append(getSimpleRatioError(resampledWbi1Band,resampledWbi2Band,varWbi1Band,varWbi2Band,covarWbi1BandWbi2Band)**0.5)
    WI_name_list.append(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','WBI.tif')))
    WI_name_list.append(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','WBI_error.tif')))

    WI_list.append(getNmdi(resampledNmdi1Band,resampledNmdi2Band,resampledNmdi3Band))
    WI_list.append(getNmdiError(resampledNmdi1Band,resampledNmdi2Band,resampledNmdi3Band,varNmdi1Band,varNmdi2Band,varNmdi3Band,covarNmdi1BandNmdi2Band,covarNmdi1BandNmdi3Band,covarNmdi2BandNmdi3Band)**0.5)
    WI_name_list.append(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','NMDI.tif')))
    WI_name_list.append(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','NMDI_error.tif')))


    WI_list.append(getGeneralIndexRatio(resampledNdwi1Band,resampledNdwi2Band))
    WI_list.append(getGeneralIndexRatioError(resampledNdwi1Band,resampledNdwi2Band,varNdwi1Band,varNdwi2Band,covarNdwi1BandNdwi2Band)**0.5)
    WI_name_list.append(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','NDWI.tif')))
    WI_name_list.append(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','NDWI_error.tif')))
    #NDII calcs

    WI_list.append(getGeneralIndexRatio(resampledMsi1Ndii1Band,resampledNdii2Band))
    WI_list.append(getGeneralIndexRatioError(resampledMsi1Ndii1Band,resampledNdii2Band,varMsi1Ndii1Band,varNdii2Band,covarMsi1Ndii1BandNdii2Band)**0.5)
    WI_name_list.append(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','NDII.tif')))
    WI_name_list.append(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','NDII_error.tif')))

    #MSI calcs

    WI_list.append(getSimpleRatio(resampledMsi2Band,resampledMsi1Ndii1Band))
    WI_list.append(getSimpleRatioError(resampledMsi2Band,resampledMsi1Ndii1Band,varMsi2Band,varMsi1Ndii1Band,covarMsi1Ndii1BandMsi2Band)**0.5)
    WI_name_list.append(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','MSI.tif')))
    WI_name_list.append(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','MSI_error.tif')))

    for WI, WI_name in zip(WI_list,WI_name_list):
        WI[nodataIndexes] = metadata['noDataVal']
        WI[np.isinf(WI)] = metadata['noDataVal']
        waterIndexes = np.where((resampledWaterBand1<0.01)*(resampledWaterBand2<0.005))
        WI[waterIndexes[0],waterIndexes[1]] = metadata['noDataVal']
        writeRasterToTif(WI_name,WI,Bounds,metadata['EPSG'],metadata['noDataVal'],metadata['res']['pixelWidth'],metadata['res']['pixelHeight'])

    return WI_name_list

def generateVegIndices(h5_file,outFolder):

    #To get output filename
    h5_filebasename = os.path.basename(h5_file)
    h5_filebasename = h5_filebasename.replace('DP1','DP2')
    #Get metadata for output tif
    reflArray, metadata, wavelengths = h5refl2array(h5_file,'Reflectance',onlyMetadata=True)

    #Get bounds for output
    ul_x, lr_y, lr_x, ul_y,nodata, data_layer = getCurrentFileExtentsAndData(h5_file,Band=None)

    #Set bounds for output
    Bounds = np.array((ul_x, lr_y, lr_x, ul_y),dtype=np.float)

    #Get all required resampled bands
    resampledRedBand = reflectanceResampling(h5_file,635.5,670)
    nodataIndexes = np.where(resampledRedBand == metadata['noDataVal'])
    resampledRedBand = resampledRedBand / metadata['scaleFactor']
    resampledNirBand = reflectanceResampling(h5_file,850.0,880.0)/metadata['scaleFactor']
    resampledBlueBand = reflectanceResampling(h5_file,459.0,479.0)/metadata['scaleFactor']
    resampledPri1Band = reflectanceResampling(h5_file,523.5,538.5)/metadata['scaleFactor']
    resampledPri2Band = reflectanceResampling(h5_file,562.5,577.5)/metadata['scaleFactor']

    resampledWaterBand1 = reflectanceResampling(h5_file,849,851)/metadata['scaleFactor']
    resampledWaterBand2 = reflectanceResampling(h5_file,1599,1601)/metadata['scaleFactor']

    covarNirRedBand = 0
    covarNirBlueBand = 0
    covarRedBlueBand = 0
    covarPri1Pri2Band = 0
    varRedBand = (resampledRedBand*0.05)**2
    varNirBand = (resampledNirBand*0.05)**2
    varBlueBand = (resampledBlueBand*0.05)**2
    varPri1Band = (resampledPri1Band*0.05)**2
    varPri2Band = (resampledPri2Band*0.05)**2

    VI_list = []
    VI_name_list = []

    #NDVI Calcs
    VI_list.append(getGeneralIndexRatio(resampledNirBand,resampledRedBand))
    VI_list.append(getGeneralIndexRatioError(resampledNirBand,resampledRedBand,varNirBand,varRedBand,covarNirRedBand)**0.5)
    VI_name_list.append(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','NDVI.tif')))
    VI_name_list.append(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','NDVI_error.tif')))

    #EVI Calcs
    G=2.5
    C1=6.0
    C2=7.5
    L_EVI=1

    VI_list.append(getEvi(resampledNirBand,resampledRedBand,resampledBlueBand,G,C1,C2,L_EVI))
    VI_list.append(getEviError(resampledNirBand,resampledRedBand,resampledBlueBand,varNirBand,varRedBand,varBlueBand,covarNirRedBand,covarNirBlueBand,covarRedBlueBand,G,C1,C2,L_EVI)**0.5)
    VI_name_list.append(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','EVI.tif')))
    VI_name_list.append(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','EVI_error.tif')))

    #ARVI Calcs
    gamma=1
    VI_list.append(getArvi(resampledNirBand,resampledRedBand,resampledBlueBand,gamma))
    VI_list.append(getArviError(resampledNirBand,resampledRedBand,resampledBlueBand,varNirBand,varRedBand,varBlueBand,covarNirRedBand,covarNirBlueBand,covarRedBlueBand,gamma)**0.5)
    VI_name_list.append(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','ARVI.tif')))
    VI_name_list.append(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','ARVI_error.tif')))

    #PRI Calcs
    VI_list.append(getGeneralIndexRatio(resampledPri1Band,resampledPri2Band))
    VI_list.append(getGeneralIndexRatioError(resampledPri1Band,resampledPri2Band,varPri1Band,varPri2Band,covarPri1Pri2Band)**0.5)
    VI_name_list.append(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','PRI.tif')))
    VI_name_list.append(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','PRI_error.tif')))

    #SAVI
    L_SAVI = 0.5
    VI_list.append(getSavi(resampledNirBand,resampledRedBand,L_SAVI))
    VI_list.append(getSaviError(resampledNirBand,resampledRedBand,varNirBand,varRedBand,covarNirRedBand,L_SAVI)**0.5)
    VI_name_list.append(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','SAVI.tif')))
    VI_name_list.append(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','SAVI_error.tif')))

    for VI, VI_name in zip(VI_list,VI_name_list):
        VI[nodataIndexes] = metadata['noDataVal']
        VI[np.isinf(VI)] = metadata['noDataVal']
        waterIndexes = np.where((resampledWaterBand1<0.01)*(resampledWaterBand2<0.005))
        VI[waterIndexes[0],waterIndexes[1]] = metadata['noDataVal']
        writeRasterToTif(VI_name,VI,Bounds,metadata['EPSG'],metadata['noDataVal'],metadata['res']['pixelWidth'],metadata['res']['pixelHeight'])

    return VI_name_list


def generateFpar(h5_file,outFolder):

    #To get output filename
    h5_filebasename = os.path.basename(h5_file)
    h5_filebasename = h5_filebasename.replace('DP1','DP2')
    #Get metadata for output tif
    reflArray, metadata, wavelengths = h5refl2array(h5_file,'Reflectance',onlyMetadata=True)

    #Get bounds for output
    ul_x, lr_y, lr_x, ul_y, nodata, data_layer = getCurrentFileExtentsAndData(h5_file,Band=None)

    #Set bounds for output
    Bounds = np.array((ul_x, lr_y, lr_x, ul_y),dtype=np.float)

    resampledRedBand = reflectanceResampling(h5_file,628.0,648.0)
    nodataIndexes = np.where(resampledRedBand == metadata['noDataVal'])
    resampledRedBand = resampledRedBand / metadata['scaleFactor']
    resampledNirBand = reflectanceResampling(h5_file,840.0,860.0)/metadata['scaleFactor']

    covarNirRedBand = 0
    varRedBand = (resampledRedBand*0.05)**2
    varNirBand = (resampledNirBand*0.05)**2

    A0 = 0.82;
    A1 = 0.78;
    A2 = 0.60;
    A = 1;
    B = 0.4;
    C = 1;

    fPar = getFpar(resampledNirBand,resampledRedBand,A0,A1,A2,A,B,C)
    varfPar = getFparError(resampledNirBand,resampledRedBand,varNirBand,varRedBand,covarNirRedBand,A0,A1,A2,A,B,C)**0.5

    fPar[nodataIndexes] = metadata['noDataVal']
    varfPar[nodataIndexes] = metadata['noDataVal']

    writeRasterToTif(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','fPAR.tif')),fPar,Bounds,metadata['EPSG'],metadata['noDataVal'],metadata['res']['pixelWidth'],metadata['res']['pixelHeight'])
    writeRasterToTif(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','fPAR_error.tif')),varfPar,Bounds,metadata['EPSG'],metadata['noDataVal'],metadata['res']['pixelWidth'],metadata['res']['pixelHeight'])

    #return fPar,varfPar

def generateLai(h5_file,outFolder):

    #To get output filename
    h5_filebasename = os.path.basename(h5_file)
    h5_filebasename = h5_filebasename.replace('DP1','DP2')
    #Get metadata for output tif
    reflArray, metadata, wavelengths = h5refl2array(h5_file,'Reflectance',onlyMetadata=True)

    #Get bounds for output
    ul_x, lr_y, lr_x, ul_y, nodata, data_layer = getCurrentFileExtentsAndData(h5_file,Band=None)

    #Set bounds for output
    Bounds = np.array((ul_x, lr_y, lr_x, ul_y),dtype=np.float)

    resampledRedBand = reflectanceResampling(h5_file,635.0,670.0)
    nodataIndexes = np.where(resampledRedBand == metadata['noDataVal'])
    resampledRedBand = resampledRedBand / metadata['scaleFactor']
    resampledNirBand = reflectanceResampling(h5_file,850.0,880.0)/metadata['scaleFactor']

    covarNirRedBand = 0
    varRedBand = (resampledRedBand*0.05)**2
    varNirBand = (resampledNirBand*0.05)**2

    A0 = 0.82
    A1  = 0.78
    A2  = 0.60

    #SAVI
    L_Savi = 0.5
    Savi = getSavi(resampledNirBand,resampledRedBand,L_Savi)
    varSavi = getSaviError(resampledNirBand,resampledRedBand,varNirBand,varRedBand,covarNirRedBand,L_Savi)

    #The ATCOR LAI Algorithm
    Lai = getLai(Savi,A0,A1,A2)
    varLai = getLaiError(Savi,varSavi,A0,A1,A2)**0.5

    Lai[nodataIndexes] = metadata['noDataVal']
    varLai[nodataIndexes] = metadata['noDataVal']

    writeRasterToTif(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','LAI.tif')),Lai,Bounds,metadata['EPSG'],metadata['noDataVal'],metadata['res']['pixelWidth'],metadata['res']['pixelHeight'])
    writeRasterToTif(os.path.join(outFolder,h5_filebasename.replace('reflectance.h5','LAI_error.tif')),varLai,Bounds,metadata['EPSG'],metadata['noDataVal'],metadata['res']['pixelWidth'],metadata['res']['pixelHeight'])

    return Lai,varLai

def makeDir(directory):
    if not os.path.exists(directory):
        #print('Making directory:',directory)
        os.makedirs(directory)

def collectFilesInPath(Path,Ext=None):
    Path = os.path.normpath(Path)
    Files = []
    if os.path.isdir(Path):
        if Ext is not None:
            Files = [file for file in os.listdir(Path) if file.endswith(Ext)]
            Files = [os.path.join(Path,File) for File in Files]
        else:
            Files = [os.path.join(Path,File) for File in os.listdir(Path)]
    return Files

def collectFilesInPathIncludingSubfolders(Path,Ext=None):

    Path = os.path.normpath(Path)

    Files = []

    # Walk through the directory and its subfolders
    for root, _, files in os.walk(Path):
        for file in files:
            if file.endswith(Ext):
                # Create the full path of the matching file
                fullPath = os.path.join(root, file)
                Files.append(fullPath)

    return Files

def collectDirectoriesInPath(Path):
    directories = []
    Path = os.path.normpath(Path)
    for root, dirs, _ in os.walk(Path):
        for dir in dirs:
            full_path = os.path.join(root, dir)
            directories.append(full_path)

    return directories

def unzipFiles(zip_file):
    with zipfile.ZipFile(zip_file,"r") as zip_ref:
        zip_ref.extractall(os.path.dirname(zip_file))

def zipFiles(ZipFileList,OutFile):
    with zipfile.ZipFile(OutFile,"w") as zipMe:
        for File in ZipFileList:
            zipMe.write(File, compress_type=zipfile.ZIP_DEFLATED,arcname=os.path.basename(File))

def zipFolder(folderPath, outFile):
    with zipfile.ZipFile(outFile, "w") as zipMe:
        for folderName, subfolders, filenames in os.walk(folderPath):
            for filename in filenames:
                filePath = os.path.join(folderName, filename)
                # Create relative path for files to maintain the directory structure
                relativePath = os.path.relpath(filePath, start=folderPath)
                zipMe.write(filePath, arcname=relativePath, compress_type=zipfile.ZIP_DEFLATED)

def getCurrentFileExtentsAndData(File,Band=1):

    FileName, FileExtension = os.path.splitext(File)

    if FileExtension == '.tif':
        src = rio.open(File)
        ul_x, lr_y, lr_x, ul_y = src.bounds
        nodata = src.nodata
        if Band == 'all':
            data_layer = src.read()
        else:
            data_layer = (src.read(Band))

        ul_y, lr_y, ul_x, lr_x = np.max((ul_y,lr_y)),np.min((ul_y,lr_y)),np.min((ul_x,lr_x)),np.max((ul_x,lr_x))

    if FileExtension == '.h5':
        if Band is None:
            reflArray, metadata, wavelengths = h5refl2array(File,'Reflectance',onlyMetadata = True)
        else:
            reflArray, metadata, wavelengths = h5refl2array(File,'Reflectance')
    
        ul_x = metadata['ext_dict']['xMin']
        lr_x = metadata['ext_dict']['xMax']
        lr_y = metadata['ext_dict']['yMin']
        ul_y = metadata['ext_dict']['yMax']
        nodata = metadata['noDataVal']
        if Band is None:
            data_layer = []
        else:
            data_layer = reflArray[:,:,Band]

    return  ul_x, lr_y, lr_x, ul_y, nodata, data_layer

def reprojectTif(inFile,outCrs):

    #dst_crs = 'EPSG:32613' # CRS for web meractor
    src = rio.open(inFile)
    outTransform, outWidth, outHeight = calculate_default_transform(
        src.crs, outCrs, src.width, src.height, *src.bounds)
    outMetadata = src.meta.copy()
    outMetadata.update({
        'driver': 'GTiff',
        'crs': outCrs,
        'transform': outTransform,
        'width': outWidth,
        'height': outHeight
    })

    outImageReprojected = np.empty((src.count, outHeight, outWidth), dtype=src.dtypes[0])

    rio.warp.reproject(
    source=rio.band(src,1),
    destination=outImageReprojected,
    src_crs=src.crs,
    src_transform=src.transform,
    dst_crs=outCrs,
    dst_transform=outTransform,
    resampling=Resampling.nearest
    )

    src.close()

    return outImageReprojected,outMetadata

def extractcellsToListRemoveNodata(inputGeotif):
    # Open the input GeoTIFF file
    with rio.open(inputGeotif) as src:
        # Get the size of the raster
        rows, cols = src.shape

        # Read all the raster values
        raster_data = src.read(1)  # Assuming it's a single-band raster

        # Open CSV file for writing
        tabulatedData = []

        # Loop through each cell and write to CSV
        for row in range(rows):
            for col in range(cols):
                cell_value = raster_data[row, col]
                if cell_value == src.nodata:
                    continue
                # Convert row and column indices to geographic coordinates
                x, y = src.xy(row, col)

                tabulatedData.append([x, y, cell_value])

    return tabulatedData

# Example usage
# input_geotiff_path = 'path/to/your/input/file.tif'
# output_csv_path = 'path/to/save/csv/output.csv'

def batchAssignEpsgToTif(inFolder,outFolder,EPSGCode):

    tifFiles = [inFolder+'/'+file for file in os.listdir(inFolder) if file.endswith('.tif')]
    with Pool(processes=30) as pool:
        pool.map(partial(assignEpsgToTif, outFolder=outFolder,EPSGCode=EPSGCode),tifFiles)
    # for tifFile in tifFiles:
#        print(inFolder+'/'+tifFile)
#        assignEpsgToTif(outFolder+'/'+tifFile,inFolder+'/'+tifFile,EPSGCode)

def assignEpsgToTif(baseFile,outFolder,EPSGCode):

    outFile = outFolder+'/'+os.path.basename(baseFile)

    with rio.open(baseFile) as inRaster:
        RasterData = inRaster.read()
        NODATA = int(inRaster.nodata)
        transform = inRaster.transform

    if os.path.isfile(baseFile):
        os.remove(baseFile)

    NewDataset = rio.open(outFile,'w',driver='GTiff',compress='LZW',NUM_THREADS='ALL_CPUS',height=RasterData.shape[1],width=RasterData.shape[2],count=RasterData.shape[0],dtype=RasterData.dtype,crs=str('EPSG:'+str(EPSGCode)),nodata=NODATA,transform=transform)
    NewDataset.write(RasterData)
    NewDataset.close()

def fixNoDataValueInTiff(InFile,OutFile):

    with rio.open(InFile) as src:
        ul_x, lr_y, lr_x, ul_y = src.bounds

        if src.crs.is_epsg_code:
            EPSGCode = int(src.crs['init'].lstrip('epsg:'))

        AffineTransform = src.transform

        RasterCellHeight = np.abs(AffineTransform[4])
        RasterCellWidth = np.abs(AffineTransform[0])
        MosaicExtents = np.array((ul_x, lr_y, lr_x, ul_y),dtype=np.float)
        NoData = src.nodata
        OutRaster = src.read(1)

    src.close()

    OutRaster[OutRaster==0] = -9999

    writeRasterToTif(OutFile,OutRaster,MosaicExtents,EPSGCode,NoData,RasterCellWidth,RasterCellHeight)

def writeRasterToTifTransferMetadata(OutRaster,OutFile,BaseFile):

    with rio.open(BaseFile) as src:
        ul_x, lr_y, lr_x, ul_y = src.bounds

        if src.crs.is_epsg_code:
            EPSGCode = int(src.crs['init'].lstrip('epsg:'))

        AffineTransform = src.transform

        RasterCellHeight = np.abs(AffineTransform[4])
        RasterCellWidth = np.abs(AffineTransform[0])
        MosaicExtents = np.array((ul_x, lr_y, lr_x, ul_y),dtype=np.float)
        NoData = src.nodata

    writeRasterToTif(OutFile,OutRaster,MosaicExtents,EPSGCode,NoData,RasterCellWidth,RasterCellHeight)


def writeRasterToTif(OutRaster,Mosaic,Bounds,EPSG,NODATA,RasterCellWidth,RasterCellHeight):
    bandsShape = Mosaic.shape

    if 'Radiance.tif' in OutRaster:
        print("Changing Radiance to int")
        #nodataIndex = np.where(Mosaic==NODATA)
        Mosaic[Mosaic==NODATA] = 0
        Mosaic = Mosaic * 1000
        Mosaic = Mosaic.astype(np.int16)
        Mosaic[Mosaic==0] = NODATA
    if len(bandsShape)==2:
        bandCount = 1
        Mosaic = Mosaic[:][..., np.newaxis]
    else:
        bandCount = bandsShape[2]

    #Transform = Affine.translation(float(Bounds[0]) - RasterCellWidth / 2, float(Bounds[3]) - RasterCellHeight / 2) * Affine.scale( RasterCellWidth,  -RasterCellHeight)
    Transform = Affine.translation(float(Bounds[0]), float(Bounds[3])) * Affine.scale( RasterCellWidth,  -RasterCellHeight)
    
    # Ensure proper data type for TIFF writing
    # Fix for "SampleFormat=IEEEFP and BitsPerSample=8" error
    if Mosaic.dtype == np.uint8 or str(Mosaic.dtype) == 'uint8':
        # If it's uint8 but might be incorrectly flagged as float, ensure it's properly typed
        output_dtype = np.uint8
    elif Mosaic.dtype in [np.float16, np.float32, np.float64]:
        # Convert float data to appropriate integer type to avoid TIFF format issues
        if 'Radiance' in OutRaster and Mosaic.dtype != np.int16:
            # Already converted to int16 above
            output_dtype = Mosaic.dtype
        else:
            # For other float data, convert to uint16 or float32
            output_dtype = np.float32
    else:
        output_dtype = Mosaic.dtype
    
    NewDataset = rio.open(OutRaster,'w',driver='GTiff',compress='LZW',NUM_THREADS='ALL_CPUS',height=Mosaic.shape[0],width=Mosaic.shape[1],count=Mosaic.shape[2],dtype=output_dtype,crs=str('EPSG:'+str(EPSG)),nodata=NODATA,transform=Transform,overwrite=True,BIGTIFF='YES')
    #NewDataset.write(Mosaic)
    NewDataset.write(np.moveaxis(Mosaic,[0,1,2],[1,2,0]))

    NewDataset.close()

def writeRasterToCog(inFile):

    outFile = inFile.replace('.tif','_COG.tif')

    inRaster = gdal.Open(inFile)
    numBands = inRaster.RasterCount

    driver = gdal.GetDriverByName('MEM')
    outdata = driver.Create('', inRaster.RasterXSize, inRaster.RasterYSize, numBands,gdal.GDT_Int16)
    print('Writing bands for COG')

    data = inRaster.ReadAsArray()
    #outdata.WriteArray(data)
    #outdata.SetNoDataValue(-9999)

    for i in range(numBands):
        #band = inRaster.GetRasterBand(i+1)
        #arr = band.ReadAsArray()
        outdata.GetRasterBand(i + 1).WriteArray(data[i,:,:])
        outdata.GetRasterBand(i + 1).SetNoDataValue(-9999)

    outdata.SetGeoTransform(inRaster.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(inRaster.GetProjection())##sets same projection as input

    data = None
    print('Building overview for COG')
    outdata.BuildOverviews("NEAREST", [2, 4, 8, 16, 32, 64])
    #outdata.BuildOverviews("NEAREST", [8, 16, 32, 64])
    driver = gdal.GetDriverByName('GTiff')
    outdata2 = driver.CreateCopy(outFile, outdata,options=["COPY_SRC_OVERVIEWS=YES","TILED=YES","COMPRESS=LZW","BIGTIFF=YES"])

    outdata = None
    outdata2 = None

def smoothImage(inputImage, windowSize):

    originalImageShape = inputImage.shape

    if len(originalImageShape) == 3:
        inputImage = np.squeeze(inputImage)

    kernel = np.ones((windowSize, windowSize), dtype=np.float32) / (windowSize * windowSize)
    smoothedImage = convolve(inputImage.astype(np.float32), kernel)

    if len(originalImageShape) == 3:
        if originalImageShape[0] == 1:
            smoothedImage = smoothedImage[None,:,:]
        if originalImageShape[2] == 1:
            smoothedImage = smoothedImage[:,:,None]


    return smoothedImage.astype(inputImage.dtype)

def smoothImageIgnoreNans(image, window_size):
    """
    Smooths an image using a moving average window of arbitrary size,
    ignoring NaN values.

    Args:
        image (numpy.ndarray): Input image (grayscale or color) with NaN values.
        window_size (int): Size of the moving average window (should be an odd integer).

    Returns:
        numpy.ndarray: Smoothed image.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd integer.")

    # Create a copy of the input image to store the smoothed result
    smoothed_image = np.copy(image)

    # Iterate over the image and apply the moving average, skipping NaN values
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if np.isnan(smoothed_image[i, j] ):
                continue
            window = image[
                max(0, i - window_size // 2):min(image.shape[0], i + window_size // 2 + 1),
                max(0, j - window_size // 2):min(image.shape[1], j + window_size // 2 + 1)
            ]
            smoothed_image[i, j] = np.nanmean(window)

    return smoothed_image

def stretchRaster(RasterArray,NoDataValue,PercentileCutOffs):

    RasterArray[RasterArray==NoDataValue] = np.nan
    LowerValue = np.nanpercentile(RasterArray,PercentileCutOffs[0])
    UpperValue = np.nanpercentile(RasterArray,PercentileCutOffs[1])
    StretchedRasterArray = exposure.rescale_intensity(RasterArray,in_range=(LowerValue,UpperValue),out_range=np.uint8)
    StretchedRasterArray[np.isnan(StretchedRasterArray)] = NoDataValue

    return StretchedRasterArray

def convertEnvi2tif(InputEnviFile,outFile='Same',Bands='All',ScaleFactor=1,NoDataOverRide='None',CorrectNoData='None'):

    if outFile == 'Same':
        filename, file_extension = os.path.splitext(InputEnviFile)
        if file_extension == '':
            outFile = InputEnviFile + '.tif'
        else:
            outFile = InputEnviFile.replace(file_extension, '.tif')

    print(outFile)
    envi_filename, envi_file_extension = os.path.splitext(os.path.basename(InputEnviFile))

    ip_class = evFile.envi_read(InputEnviFile.replace('.hdr',''))
    raw_hdrdata, raw_data = ip_class.read_data()

    [inX,inY,inZ] = raw_data.shape

    if inY == int(raw_hdrdata['bands']):
        raw_data = np.moveaxis(raw_data,[0,1,2],[1,0,2])

    if inZ == int(raw_hdrdata['bands']):
        raw_data = np.moveaxis(raw_data,[0,1,2],[2,0,1])

    if Bands != 'All':
        raw_data = raw_data[Bands,:,:]

    map_info_split = raw_hdrdata['map info'].split(',')

    EPSG = CRS.from_wkt(raw_hdrdata['coordinate system string']).to_epsg()

    Bounds = np.array((float(map_info_split[3]), float(map_info_split[4])+float(raw_hdrdata['samples']), float(map_info_split[3])+float(raw_hdrdata['lines']), float(map_info_split[4])),dtype=np.float)

    if 'data ignore value' in raw_hdrdata.keys():
        no_data = (float(raw_hdrdata['data ignore value']))

    if CorrectNoData != 'None':
        no_data = CorrectNoData

    no_data_indexes = np.where(raw_data==no_data)
    raw_data = raw_data/ScaleFactor

    if NoDataOverRide != 'None':
        no_data = NoDataOverRide
    else:
        no_data = -9999

    raw_data[no_data_indexes]=no_data

    writeRasterToTif(outFile,np.moveaxis(raw_data,[0,1,2],[2,0,1]),Bounds,EPSG,no_data, float(map_info_split[5]),float(map_info_split[6]))

    return raw_data, raw_hdrdata

def deleteFiles(filesToDelete,extensionsToDelete):

        for file in filesToDelete:
            for extension in extensionsToDelete:
                if file.endswith(extension):
                    os.remove(file)
                    break

def clipRasterbyVectorFile(rasterFile,vectorFile,bufferDistance=0):

    boundingBox = getBoundingBox(vectorFile,bufferDistance)

    outImage,outMetadata = clipRasterbyBbox(rasterFile,boundingBox)

    return outImage,outMetadata

def getBoundingBox(inputBoundaryFile,bufferDistance):

    gdf = gpd.read_file(inputBoundaryFile)

    # Get the buffered extent
    bufferedExtent = gdf.geometry.buffer(distance=bufferDistance)  # Adjust the buffer distance as needed

    # Get the bounding box from the buffered extent
    return bufferedExtent.total_bounds

def clipRasterbyBbox(inputGtifFile,boundingBox):

    src = rio.open(inputGtifFile)
    # Create a bounding box geometry
    bboxGeometry = box(*boundingBox)

    # Crop the GeoTIFF using the bounding box
    outImage, outTransform = mask(src, [bboxGeometry], crop=True)

    # Update metadata for the cropped GeoTIFF
    outMetadata = src.meta
    outMetadata.update({
        'driver': 'GTiff',
        'height': outImage.shape[1],
        'width': outImage.shape[2],
        'transform': outTransform
    })

    src.close()

    return outImage,outMetadata

def clipRaster(rasterFile,shpFiles):

    rasterClipped = rxr.open_rasterio(rasterFile, masked=True)
    for shpFile in shpFiles:
        cropExtent = gpd.read_file(shpFile)
        rasterClipped = rasterClipped.rio.clip(cropExtent.geometry.apply(mapping),invert=True)

    return rasterClipped

def clipRasterByRaster(largeRaster,clipRasterFile,outputFile):

    ul_x, lr_y, lr_x, ul_y, nodata, data_layer = getCurrentFileExtentsAndData(clipRasterFile)

    clipWidth = lr_x - ul_x
    clipHeight = ul_y - lr_y
    with rio.open(largeRaster) as src:
        # Calculate the window to clip

        transform = src.transform

        colStart, rowStart = rio.transform.rowcol(transform, ul_x, ul_y)

        window = Window(rowStart, colStart, clipWidth, clipHeight)
        #print(window)
        # Read the data from the specified window
        clipped_data = src.read(window=window)

        # Update metadata (profile) for the clipped raster
        profile = src.profile
        profile.update({
            'height': clipped_data.shape[1],
            'width': clipped_data.shape[2],
            'transform': src.window_transform(window),
            'driver': 'GTiff',  # Specify the output raster format if needed
        })

    # Write the clipped data to a new raster file
    with rio.open(outputFile, 'w', **profile) as dst:
        dst.write(clipped_data)
    #return clipped_data

def process_geotiff(input_file, output_file):
    # Open the input GeoTIFF file
    #gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK','SAMPLE_FORMAT=INT')
    dataset = gdal.Open(input_file, gdal.GA_ReadOnly)

    if dataset is None:
        raise ValueError("Could not open the input GeoTIFF file.")

    # Read the pixel data
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    num_bands = dataset.RasterCount
    data = dataset.ReadAsArray()

    #ENTERT DESIRED PROCESSING FUNCTION HERE
    processed_data = data
    # Create a new GeoTIFF file for writing
    driver = gdal.GetDriverByName('GTiff')
    output_dataset = driver.Create(output_file, width, height, num_bands, gdal.GDT_Byte)

    if output_dataset is None:
        raise ValueError("Could not create the output GeoTIFF file.")

    # Set the geotransform and projection information from the input file
    output_dataset.SetGeoTransform(dataset.GetGeoTransform())
    output_dataset.SetProjection(dataset.GetProjection())

    # Write the processed data to the output file
    for i in range(num_bands):
        output_dataset.GetRasterBand(i + 1).WriteArray(processed_data[i])

    # Close the datasets
    dataset = None
    output_dataset = None


def readEnviRaster(enviFile,justMetadata = False,bands = None):

    if enviFile.endswith('.hdr'):
        enviFile = enviFile.replace('.hdr','')

    datasetEnvi = gdal.Open(enviFile,gdal.GA_ReadOnly)

    if justMetadata:
        rasterData = []
    else:
        
        if bands is None:
            rasterData = datasetEnvi.ReadAsArray()
        else:
            
            rasterData = []
            for band in bands:
                readBand = datasetEnvi.GetRasterBand(band)
                rasterData.append(readBand.ReadAsArray())
            rasterData = np.dstack(rasterData)
            rasterData = np.moveaxis(rasterData,[0,1,2],[1,2,0])
        if len(rasterData.shape) == 2:
            rasterData = rasterData[:,:,None]

    metadata = {
    "geotransform": datasetEnvi.GetGeoTransform(),
    "projection": datasetEnvi.GetProjection(),
    "data_type": datasetEnvi.GetRasterBand(1).DataType,
    "data_ignore_value": datasetEnvi.GetRasterBand(1).GetNoDataValue(),
    "num_bands": datasetEnvi.RasterCount,
    "width": datasetEnvi.RasterXSize,
    "height": datasetEnvi.RasterYSize
    # Add more metadata fields as needed
    }

    datasetEnvi = None

    return rasterData,metadata

def getRasterGdalDtype(inDtype):

#There is more...we just don't use them
    if inDtype == np.float32: 
        gdalDtype = gdal.GDT_Float32    
    elif inDtype ==  np.float64:
        gdalDtype = gdal.GDT_Float64
    elif inDtype == np.int8 or inDtype == np.uint8:
        gdalDtype = gdal.GDT_Byte
    elif inDtype == np.int16:
        gdalDtype = gdal.GDT_Int16
    elif inDtype == np.int32:
        gdalDtype = gdal.GDT_Int32
    elif inDtype == np.uint16:
        gdalDtype = gdal.GDT_UInt16
    elif inDtype == np.uint32:
        gdalDtype = gdal.GDT_UInt32
        
    return gdalDtype
    
def convertGtiffToEnvi(tifFile,outFile,filePathToEnviProjCs):
    
    with rio.open(tifFile) as src:
        # Calculate the window to clip
        raster = src.read()
        transform = src.transform
        crs = src.crs
        epsg = crs.to_epsg()
        
        enviProjCs = parseEnviProjCs(filePathToEnviProjCs,epsg)
        
        gdalDtype = getRasterGdalDtype(raster.dtype)
        
        metadata = {
        "geotransform": (transform.c, transform.a,-0.0,transform.f,0.0,transform.e),
        "projection": enviProjCs,
        "data_type": gdalDtype,
        "data_ignore_value": float(src.nodata),
        "num_bands": int(src.count),
        "width": int(src.width),
        "height": int(src.height)
        # Add more metadata fields as needed
        }

    writeEnviRaster(outFile,np.squeeze(raster),metadata)
 
def writeEnviRaster(outEnviFile,raster,metadata,needsBandMetadata=False):

    if outEnviFile.endswith('.hdr'):
        outEnviFile = outEnviFile.replace('.hdr','')

    if len(raster.shape) == 2:
        raster = raster[:,:,None]
        numBands = 1
    else:
        numBands = raster.shape[2]

    driver = gdal.GetDriverByName("ENVI")
    outputDatasetEnvi = driver.Create(outEnviFile, int(metadata["width"]), int(metadata["height"]),
                          numBands, metadata["data_type"])

    # Set the metadata for the output dataset
    outputDatasetEnvi.SetGeoTransform(metadata["geotransform"])
    outputDatasetEnvi.SetProjection(metadata["projection"])
    
    for band in range(0, numBands):
        outputBand = outputDatasetEnvi.GetRasterBand(band+1)
        outputBand.WriteArray(raster[:,:,band])
        outputBand.SetNoDataValue(metadata["data_ignore_value"])
        if "band_names" in metadata and not needsBandMetadata:
            outputBand.SetDescription(metadata["band_names"][band])
    bandMetadata={}
    if needsBandMetadata:
        
        bandMetadata = {'wavelength units': 'nanometers', 'wavelength': '{'+','.join(map(str, metadata['wavelengths']))+'}','fwhm': '{'+','.join(map(str, metadata['fwhm']))+'}'}
    
    if 'Classes' in metadata:
        bandMetadata['classes'] = metadata['Classes']
    if 'Class_Lookup' in metadata:
        bandMetadata['class lookup'] = '{'+','.join(map(str,metadata['Class_Lookup']))+'}'
    #if 'Class_Names' in metadata:
        #bandMetadata['class names'] = '{'+','.join(map(str,metadata['Class_Names']))+'}'
    #    bandMetadata['class names'] = '{0: geocoded backg,1: water,2: DDV reference,3: non-reference,4: topogr.shadow}'
    
    outputDatasetEnvi.SetMetadata(bandMetadata, 'ENVI')
    outputDatasetEnvi.FlushCache()
    outputDatasetEnvi = None
    
    #classNamesString = 'class names'+' '
    # Decode the byte strings in the array
    
    if 'Class_Names' in metadata:
        decodedClassNames = [class_name.decode('utf-8') for class_name in metadata['Class_Names']]
    
        # Join the decoded strings into a single comma-delimited string
        decodedClassNamesString = '{'+', '.join(decodedClassNames)+'}'
    
        with open(os.path.splitext(outEnviFile)[0]+'.hdr', 'a') as file:
                file.write('class names = '+decodedClassNamesString)

def calcSlopeAspectEnvi(inEnviFile,outSlopeEnviFile,outAspectEnviFile):

    outputTifFile = inEnviFile.split('.')[0]+'.tif'

    convertEnviToGeotiff(inEnviFile.replace('.hdr','')+'.hdr', outputTifFile)

    outSlopeFile = outputTifFile.replace('.tif','_slope.tif')

    outAspectFile = outputTifFile.replace('.tif','_aspect.tif')

    calcSlopeGdal(outputTifFile, outSlopeFile)

    calcAspectGdal(outputTifFile, outAspectFile )

    ul_x, lr_y, lr_x, ul_y, nodata, slope = getCurrentFileExtentsAndData(outSlopeFile)
    ul_x, lr_y, lr_x, ul_y, nodata, aspect = getCurrentFileExtentsAndData(outAspectFile)

    rasterElevationData,metadata = readEnviRaster(inEnviFile)

    if len(rasterElevationData.shape) == 3:
        rasterElevationData = rasterElevationData[:,:,0]

    slope[rasterElevationData==float(metadata["data_ignore_value"])] = metadata["data_ignore_value"]
    aspect[rasterElevationData==float(metadata["data_ignore_value"])] = metadata["data_ignore_value"]

    writeEnviRaster(outSlopeEnviFile,slope,metadata)
    writeEnviRaster(outAspectEnviFile,aspect,metadata)

    os.remove(outputTifFile)
    os.remove(outSlopeFile)
    os.remove(outAspectFile)

def calcSlopeGdal(inputFile,outSlopeFile):

    gdal.DEMProcessing(outSlopeFile, inputFile, "slope",options=["-alg", "Horn", "-compute_edges"])

def calcAspectGdal(inputFile,outAspectFile):

     gdal.DEMProcessing(outAspectFile, inputFile, "aspect",options=["-alg", "Horn", "-compute_edges"])

def convertEnviToGeotiff(inputFile, outputFile, bandsToExport=None):
    # Open the ENVI file

    if inputFile.endswith('.hdr'):
        inputFile = inputFile.replace('.hdr','')

    datasetEnvi = gdal.Open(inputFile)

    if datasetEnvi is None:
        raise ValueError("Could not open the input ENVI file.")

    # Get the GeoTIFF driver
    driverGeotiff = gdal.GetDriverByName('GTiff')

    # Get the number of bands in the input dataset
    numBands = datasetEnvi.RasterCount

    # Determine the bands to export
    if bandsToExport is None:
        bandsToExport = list(range(1, numBands + 1))
    else:
        for band in bandsToExport:
            if band < 1 or band > numBands:
                raise ValueError(f"Invalid band number: {band}. Band number must be between 1 and {numBands}.")

    # Get the dimensions of the input dataset
    width = datasetEnvi.RasterXSize
    height = datasetEnvi.RasterYSize

    # Create the output GeoTIFF file
    datasetGeotiff = driverGeotiff.Create(outputFile, width, height, len(bandsToExport), gdal.GDT_Float32)

    if datasetGeotiff is None:
        raise ValueError("Could not create the output GeoTIFF file.")

    # Copy geotransform and projection from the input dataset to the output dataset
    datasetGeotiff.SetGeoTransform(datasetEnvi.GetGeoTransform())
    datasetGeotiff.SetProjection(datasetEnvi.GetProjection())

    # Read and write the specific bands to the output GeoTIFF
    for i, band in enumerate(bandsToExport):
        bandData = datasetEnvi.GetRasterBand(band).ReadAsArray()
        datasetGeotiff.GetRasterBand(i + 1).WriteArray(bandData.astype(np.float32))

        nodataValue = datasetEnvi.GetRasterBand(band).GetNoDataValue()
        if nodataValue is not None:
            datasetGeotiff.GetRasterBand(i + 1).SetNoDataValue(nodataValue)

    # Close the datasets
    datasetEnvi = None
    datasetGeotiff = None

    print("Conversion complete for " + inputFile)

def getMosaic(Files,MosaicType,MosaicExtents,Band,resX=1,resY=1):

    FileName, FileExtension = os.path.splitext(Files[0])

    NorthSouthExtent = int(np.ceil(MosaicExtents[3]-MosaicExtents[1])/resY)
    EastWestExtent = int(np.ceil(MosaicExtents[2]-MosaicExtents[0])/resX)

    FullExtent = np.empty((NorthSouthExtent,EastWestExtent))
    FullExtent[:] = np.nan

    if MosaicType == 'last_in':
        FullExtentTemp = np.copy(FullExtent)
    if MosaicType == 'average':
        mosaicMask = np.dstack(FullExtent,FullExtent)
    if MosaicType == 'diff_overlap':
        FullExtent = np.dstack((FullExtent,FullExtent))
        mosaicMask = np.empty_like(FullExtent)
        mosaicMask[:] = np.nan
    if MosaicType == 'variance_sum':
        mosaicMask = np.dstack((FullExtent,FullExtent))
    if MosaicType == 'mask':
        mosaicMask = np.dstack((FullExtent,FullExtent))

    FileCount=0

    for File in Files:
        
        if os.path.getsize(File) < 10:
            continue

        ul_x, lr_y, lr_x, ul_y, nodata, data_layer = getCurrentFileExtentsAndData(File,Band=Band)

        rows = int(np.floor((ul_y-lr_y)/resY))
        columns = int(np.floor((lr_x-ul_x)/resX))
        start_index_x = int(np.floor((MosaicExtents[3]-ul_y)/resX))
        start_index_y = int(np.floor((ul_x-MosaicExtents[0])/resY))

        if MosaicType == 'last_in':
            data_layer = data_layer.astype(np.float32)
            data_layer[data_layer == nodata] = np.nan
            FullExtentTemp[start_index_x:start_index_x+rows,start_index_y:start_index_y+columns] = data_layer
            FullExtent[~np.isnan(FullExtentTemp)] = FullExtentTemp[~np.isnan(FullExtentTemp)]
            FullExtent[np.isnan(FullExtent)] = nodata

        if MosaicType == 'average':
            FullExtent[0,start_index_x:start_index_x+rows,start_index_y:start_index_y+columns] = data_layer
            FullExtent[2,:,:] = np.nansum(FullExtent,axis=2)

            mosaicMask[0,start_index_x:start_index_x+rows,start_index_y:start_index_y+columns] = data_layer
            mosaicMask[0,~np.isnan(mosaicMask[0,:,:])] = 1
            mosaicMask[2,:,:] = np.nansum(mosaicMask,axis=2)

        if MosaicType == 'tiles':
            FullExtent[start_index_x:start_index_x+rows,start_index_y:start_index_y+columns] = data_layer

        if MosaicType == 'diff_overlap':
            data_layer[data_layer == nodata] = np.nan
            FullExtent[start_index_x:start_index_x+rows,start_index_y:start_index_y+columns,0] = data_layer
            FullExtent[mosaicMask==1] = FullExtent[mosaicMask==1]*-1

            mosaicMask[start_index_x:start_index_x+rows,start_index_y:start_index_y+columns,0] = data_layer
            mosaicMask[~np.isnan(mosaicMask[:,:,0]),0] = 1

            mosaicMask[:,:,1] = np.nansum(mosaicMask,axis=2)
            mosaicMask[:,:,0] = mosaicMask[:,:,0]*np.nan

            FullExtent[np.squeeze(mosaicMask[:,:,1])>= 3,0] = 0

            FullExtent[:,:,1] = np.nansum(FullExtent,axis=2)
            FullExtent[mosaicMask==1] = np.abs(FullExtent[mosaicMask==1])
            FullExtent[:,:,0] = FullExtent[:,:,0]*np.nan

        if MosaicType == 'variance_sum':
            FullExtent[0,start_index_x:start_index_x+rows,start_index_y:start_index_y+columns] = data_layer**2
            FullExtent[mosaicMask==1] = FullExtent[mosaicMask==1]

            mosaicMask[0,start_index_x:start_index_x+rows,start_index_y+1:start_index_y+columns] = data_layer
            mosaicMask[0,~np.isnan[mosaicMask[0,:,:]]] = 1

            mosaicMask[1,:,:] = np.nansum(mosaicMask,axis=2)
            mosaicMask[0,:,:] = mosaicMask[0,:,:]*np.nan

            FullExtent[0,mosaicMask[1,:,:]>= 3] = 0

            FullExtent[1,:,:] = np.nansum(FullExtent,axis=2)
            FullExtent[mosaicMask==1] = FullExtent[mosaicMask==1]
            FullExtent[0,:,:] = FullExtent[0,:,:]*np.nan

        if MosaicType == 'mask':
            mosaicMask[0,start_index_x:start_index_x+rows,start_index_y:start_index_y+columns] = data_layer
            mosaicMask[0,~np.isnan(mosaicMask[0,:,:])] = 1
            mosaicMask[1,:,:] = np.nansum(mosaicMask,axis=2)

        if MosaicType == 'stack':
            FullExtent[FileCount,start_index_x+1:start_index_x+rows,start_index_y+1:start_index_y+columns] = data_layer
            FileCount += 1

    if MosaicType == 'average':
        FullExtent = np.squeeze((FullExtent[1,:,:]/mosaicMask[1,:,:]))

    if MosaicType == 'diff_overlap':
        mosaicMask[mosaicMask <= 1] = np.nan
        mosaicMask[mosaicMask > 1] = 1
        FullExtent = FullExtent*mosaicMask
        FullExtent = np.squeeze(FullExtent[:,:,1])

    if MosaicType == 'variance_sum':
        mosaicMask[mosaicMask <= 1] = np.nan
        mosaicMask[mosaicMask > 1] = 1
        FullExtent = FullExtent*mosaicMask
        FullExtent = np.squeeze(FullExtent[1,:,:])**0.5

    if MosaicType == 'mask':
        FullExtent = mosaicMask
        FullExtent = FullExtent.astype(np.short)

    #if MosaicType == 'tiles' or MosaicType == 'last_in':
    #    FullExtent = FullExtent.astype(self.MosaicDataType)

    #if FileExtension == '.tif' :
    #    FullExtent = np.fliplr(FullExtent)

    return FullExtent


def getMapExtentsForMosiac(Files):
    FileName, FileExtension = os.path.splitext(Files[0])
    
    if FileExtension == '.tif':
        RasterIoFilesToMosaicInfo = []
        for File in Files:
            if os.path.getsize(File) < 10:
                continue
            try:
                src = rio.open(File)
                RasterIoFilesToMosaicInfo.append(src)
                src.close()
            except rio.errors.RasterioIOError as e:
                if "SampleFormat=IEEEFP and BitsPerSample=8" in str(e):
                    print(f"\nERROR: Invalid TIFF format in file: {File}")
                    print(f"This file has an incompatible format (8-bit with IEEE floating point).")
                    print(f"The file needs to be regenerated with proper data type.")
                    print(f"Full error: {e}\n")
                    # Try to identify which processing step created this file
                    if "RGB_radiance_mosaic" in File:
                        print("This appears to be a mosaic file. The issue likely occurred during TIFF creation.")
                        print("The writeRasterToTif function has been updated to prevent this in future runs.")
                    raise RuntimeError(f"Cannot continue due to invalid TIFF format in {File}. Please delete this file and re-run the processing.")
                else:
                    raise
        xs = []
        ys = []
        for src in RasterIoFilesToMosaicInfo:
            left, bottom, right, top = src.bounds
            xs.extend([left, right])
            ys.extend([bottom, top])
            ExtentW, ExtentS, ExtentE, ExtentN = min(xs), min(ys), max(xs), max(ys)

        if RasterIoFilesToMosaicInfo[0].crs.is_epsg_code:
            EPSGCode = int(RasterIoFilesToMosaicInfo[0].crs['init'].lstrip('epsg:'))
        AffineTransform = RasterIoFilesToMosaicInfo[0].transform

        RasterCellHeight = np.abs(AffineTransform[4])
        RasterCellWidth = np.abs(AffineTransform[0])
        MosaicExtents = np.array((ExtentW, ExtentS, ExtentE, ExtentN),dtype=np.float)
        NoData = src.nodata
        numBands = src.count


    if FileExtension == '.h5':
        NeonH5FilesToMosaicInfo = []
        for File in Files:
            dataArray, metadata, wavelengths = h5refl2array(File,'Reflectance',onlyMetadata = True)
            NeonH5FilesToMosaicInfo.append(metadata)
        xs = []
        ys = []
        for Metadata in NeonH5FilesToMosaicInfo:
            left, bottom, right, top = Metadata['ext_dict']['xMin'], Metadata['ext_dict']['yMin'], Metadata['ext_dict']['xMax'], Metadata['ext_dict']['yMax']
            xs.extend([left, right])
            ys.extend([bottom, top])
            ExtentW, ExtentS, ExtentE, ExtentN = min(xs), min(ys), max(xs), max(ys)

        EPSGCode,MosaicExtents,RasterCellHeight,RasterCellWidth,NoData = getGtifMetadataFromH5Metadata(Metadata)

        numBands = len(wavelengths)

    return MosaicExtents,numBands,RasterCellHeight,RasterCellWidth,NoData,EPSGCode

def getMosaicAndExtents(Files,mosaicType = 'tiles'):

    MosaicExtents,numBands,RasterCellHeight,RasterCellWidth,Nodata,EPSGCode = getMapExtentsForMosiac(Files)

    Mosaic = np.empty((int(np.ceil(MosaicExtents[3]-MosaicExtents[1])/RasterCellHeight),int(np.ceil(MosaicExtents[2]-MosaicExtents[0])/RasterCellHeight),numBands))
    Mosaic[:] = np.nan

    for Band in np.arange(numBands)+1:
        Mosaic[:,:,int(Band-1)] = getMosaic(Files,mosaicType,MosaicExtents,int(Band),RasterCellWidth,RasterCellHeight)

    Mosaic[np.isnan(Mosaic)] = Nodata

    return Mosaic,MosaicExtents,EPSGCode,Nodata,RasterCellWidth,RasterCellHeight

def generateMosaic(Files,outFile,mosaicType = 'tiles',EPSGoveride = None):
    
    os.makedirs(os.path.dirname(outFile),exist_ok=True)

    Mosaic,MosaicExtents,EPSGCode,Nodata,RasterCellWidth,RasterCellHeight = getMosaicAndExtents(Files,mosaicType)
    
    if EPSGoveride is not None:
        EPSGCode = EPSGoveride

    writeRasterToTif(outFile,Mosaic,MosaicExtents,EPSGCode,Nodata,RasterCellWidth,RasterCellHeight)

def plotGtifHelper(gtifFileAndCmap):
    
    gtifFile,cmapIn = gtifFileAndCmap
    
    #print(gtifFile)
    
    rasterVariable = os.path.basename(gtifFile).split('_')[-2]
    
    plot_geotiff(gtifFile, title='', cmap=cmapIn, save_path=gtifFile.replace('.tif','.png'), nodata_color='black',variable='')

def processSpectrometerKmlsHelper(flightline,rawKmlDir,outputKmlDir):
    
    flightline.generateKmls(rawKmlDir,outputKmlDir)

def processRadianceHelper(flightline,rawSpectrometerDir,outputRadianceDir):

    flightline.processRadiance(rawSpectrometerDir,outputRadianceDir)
    

def processOrthoRadianceHelper(flightline,rawSpectrometerDir,radianceDir,payload,leapSeconds,sbetTrajectoryFile,geoidFile,demFile,sapFile):

    flightline.processOrthoRadiance(rawSpectrometerDir,radianceDir,payload,leapSeconds,sbetTrajectoryFile,geoidFile,demFile,sapFile)

def processRdnOrtTifsHelper(flightline,radianceDir,outputDir):

    flightline.convertRdnOrtToTif(radianceDir,outputDir)                                                             
def processClipTopoForAtcorHelper(flightline,radianceDir,aigDsm):

    flightline.processClippedTopoForAtcor(radianceDir,aigDsm)

def processSmoothElevationeHelper(flightline,radianceDir):

    flightline.processSmoothElevation(radianceDir)

def processSlopeAspectHelper(flightline,radianceDir):

    flightline.processSlopeAspect(radianceDir)

def processRadianceH5Helper(flightline,radianceDir,sbetTrajectoryFile,radianceH5Dir,metadataXml,spectrometerFlightLog,scriptsFile):

    flightline.generateRadianceH5(radianceDir,sbetTrajectoryFile,radianceH5Dir,metadataXml,spectrometerFlightLog,scriptsFile)

def processReflectanceHelper(flightline,radianceDir,reflectanceDir,sbetTrajectoryFile,payload):

    flightline.processReflectance(radianceDir,reflectanceDir,sbetTrajectoryFile,payload)

def processReflectanceH5Helper(flightline,radianceDir, reflectanceProcessingDir, sbetTrajectoryFile, reflectanceH5Dir, metadataXML, spectrometerFlightLog, scriptsFile,isBrdf):

    flightline.generateReflectanceH5(radianceDir,reflectanceProcessingDir,sbetTrajectoryFile,reflectanceH5Dir,metadataXML,spectrometerFlightLog,scriptsFile,isBrdf)

def processVegetationIndicesHelper(flightline,reflectanceDir,outputDir):

    flightline.generateVegetationIndices(reflectanceDir,outputDir)

def processWaterIndicesHelper(flightline,reflectanceDir,outputDir):

    flightline.generateWaterIndices(reflectanceDir,outputDir)
    
def zipVegetationIndicesHelper(flightline,inputDir,outputDir):
    
    flightline.zipL2VegIndices(inputDir,outputDir)

def zipWaterIndicesHelper(flightline,inputDir,outputDir):
    
    flightline.zipL2WaterIndices(inputDir,outputDir)

def processFparHelper(flightline,reflectanceDir,outputDir):

    flightline.generateFpar(reflectanceDir,outputDir)

def processLaiHelper(flightline,reflectanceDir,outputDir):

    flightline.generateLai(reflectanceDir,outputDir)

def processAlbedoHelper(flightline,reflectanceDir,outputDir):

    flightline.generateAlbedo(reflectanceDir,outputDir)

def processH5RgbTifHelper(flightline,H5Dir,outputDir,raster):

    flightline.generateTifFromH5(H5Dir,outputDir,raster)
     
def generateWaterMaskFlightlineHelper(flightline,reflectanceH5Dir,outputDir):
    
    flightline.generateWaterMask(reflectanceH5Dir,outputDir)
    
def generateReflectancePngHelper(flightline,reflectanceH5Dir,outputDir):
    
    flightline.generateReflectancePlots(reflectanceH5Dir,outputDir)
    
def convertH5ToEnviRadianceHelper(flightline,radianceH5Folder,outputRadianceDir,filePathToEnviProjCs, spatialIndexesToRead ,bandIndexesToRead ):
    
    flightline.convertRadianceH5ToEnviRadiance(radianceH5Folder,outputRadianceDir,filePathToEnviProjCs, spatialIndexesToRead = spatialIndexesToRead, bandIndexesToRead = bandIndexesToRead)   

def resampleH5SpectrumHelper(flightline,inputDir,outputDir,outputType,filePathToEnviProjCs):
    
    flightline.resampleSpectrum(inputDir,outputDir,filePathToEnviProjCs,outputType=outputType)

def processReflectanceProductQa(product,siteFolder,flightday,metadataXML,site,productLevel,previousYearSiteVisit):

    matlabEng = matlab.engine.start_matlab()
    matlabEng.cd(nisSpectrometerQaCodeDir, nargout=0)
    matlabEng.generateSpectrometerProductQA(str(siteFolder),str(flightday),str(product),str(metadataXML),str(site),matlab.double(productLevel),str(previousYearSiteVisit))

    matlabEng.quit()

def pulsewavesTranslateHelper(plsFile,pulseTranslateCommandDict,commandOutputDir):

    pulseTranslateCommandDict["i"] = plsFile      
    executeLastoolsCommand('pulse2pulse',pulseTranslateCommandDict,outputDir=commandOutputDir)
    
def zipPulsewavesHelper(plsFile,pulseZipCommandDict,commandOutputDir):

    pulseZipCommandDict["i"] = plsFile
    pulseZipCommandDict["o"] = plsFile.replace('.pls','.plz')           
    executeLastoolsCommand('pulse2pulse',pulseZipCommandDict,outputDir=commandOutputDir)
    os.remove(plsFile)
    os.remove(plsFile.replace('.pls','.wvs'))

def processColorizeLazHelper(files,commandOutputDir):

    lazFile,imageFile = files

    lastoolsCommandColorDict = {"i": lazFile,"odir":os.path.dirname(lazFile), "image":imageFile,"odix": "_colorized","olaz":""}

    executeLastoolsCommand('lascolor',lastoolsCommandColorDict,outputDir=commandOutputDir)

    os.remove(lazFile)

def convertRdnOrtForAtcorHelper(flightline,radianceDir,reflectanceDir):
    
    flightline.convertRdnOrtForAtcor(radianceDir,reflectanceDir)
    
def generateScaFileForAtcorHelper(flightline,radianceDir):

    flightline.generateScaFileForAtcor(radianceDir)
    
def fixNansInAtcorRasterHelper(flightline,reflectanceDir):
    
    flightline.fixNansInAtcorRaster(reflectanceDir)
    
def createMissingDDVHelper(flightline,reflectanceDir):
    
    flightline.createMissingDDV(reflectanceDir)
    
# def H5WriterFunctionHelper(iterableFiles,radOrt,sbetFile, outputDir, metadataXML, NISlog, ScriptsFile):
    
#     reflectanceEnviFile, elevationEnviFile,shadowEnviFile = iterableFiles
    
#     H5WriterFunction(reflectanceEnviFile,elevationEnviFile,shadowEnviFile,radOrt,sbetFile, outputDir, metadataXML, NISlog, ScriptsFile)
    
def H5WriterFunctionHelper(iterableFiles,radOrt,sbetFile, outputDir, metadataXML, NISlog, ScriptsFile):
    reflectanceEnviFile, elevationEnviFile,shadowEnviFile = iterableFiles
    def write_h5(*args, **kwargs):
        H5WriterFunction(*args, **kwargs)
    qa_check_and_retry(write_h5, qa_check_h5_file, reflectanceEnviFile, elevationEnviFile, shadowEnviFile, radOrt, sbetFile, outputDir, metadataXML, NISlog, ScriptsFile, max_retries=3)
    
    
def zipFilesHelper(zipFileInfo):
    
    ZipFileList,ZipFile = zipFileInfo
    
    zipFiles(ZipFileList,ZipFile)
    
def generateWaterMaskHelper(files,nirBand,swirBand,nirThreshold,swirThreshold):
    
    h5File,outFile = files
    
    generateWaterMask(h5File,outFile,nirBand,swirBand,nirThreshold,swirThreshold)

def generateReflectancePlotsHelper(files,nirBand,swirBand,nirThreshold,swirThreshold):
    
    h5File,outFile = files
    
    generateReflectancePlots(h5File,outFile,nirBand,swirBand,nirThreshold,swirThreshold)


class YearSiteVisitClass:

    def __init__(self,YearSiteVisit,skipMissions = False, skipFlightlines = False):

        self.YearSiteVisit=YearSiteVisit

        self.year=YearSiteVisit.split('_')[0]
        self.siteString=YearSiteVisit.split('_')[1]
        self.visit=YearSiteVisit.split('_')[2]
        self.site = siteClass(self.siteString)
        currentFile  = os.path.abspath(os.path.join(__file__))
        self.DatabaseLookupDir = os.path.join(currentFile.lower().split('gold_pipeline')[0], 'Gold_Pipeline', 'ProcessingPipelines', 'res', 'FlightLogDatabases', self.year)
        self.legacyFlightDateFile = os.path.join(currentFile.lower().split('gold_pipeline')[0], 'Gold_Pipeline', 'ProcessingPipelines', 'res', 'Lookups', 'FlightDates.csv')
                                       
        if skipMissions:
            self.missions = []
        else:
            self.getMissionListAndPayload()
            self.getMissions(skipFlightlines=skipFlightlines)
            self.payload = payloadClass(self.year,self.campaigns[0])
        
        self.nodata = -9999
        self.tileSize = 1000
        self.RasterCellSize = 1

        # TODO: Lookup Domain from lookup db, to avoid having to update multiple lookups
        DomainLookup = os.path.join(currentFile.lower().split('gold_pipeline')[0], 'Gold_Pipeline', 'ProcessingPipelines', 'NIS', 'res', 'Lookups', 'HDF5_lookup.csv')
        self.filePathToEnviProjCs = 'C:/Program Files/Harris/ENVI56/IDL88/resource/pedata/predefined/EnviPEProjcsStrings.txt'
        self.lidarValidationDataFolder = os.path.join(currentFile.lower().split('gold_pipeline')[0], 'Gold_Pipeline', 'ProcessingPipelines', 'Lidar', 'res', 'Validation', 'SiteGPS')
        self.lidarValidationFile = [os.path.join(self.lidarValidationDataFolder,file) for file in os.listdir(self.lidarValidationDataFolder) if self.site.site in file]
        self.getLidarProcessingParameters()
        
        self.geoidFileDir = os.path.join(currentFile.lower().split('gold_pipeline')[0], 'Gold_Pipeline', 'ProcessingPipelines', 'NIS', 'res', 'Geoid12A')
        self.usgsDemFileDir = 'S:/external/USGS_DEM/DEMs/'
        with open(DomainLookup) as CsvFile:
            DomainLookupContents = csv.reader(CsvFile,delimiter=',')
            for Row in DomainLookupContents:
                if Row[1] == self.siteString:
                    self.Domain=Row[2]
                    self.utmZone = Row[3]
                    self.epsgCode = 32600+int(self.utmZone)  #This only works for northern hemisphere UTM zones
        self.BaseDir=os.path.join('D:\\',self.year,'FullSite',self.Domain,self.YearSiteVisit)
        self.lidarFilePrefix = 'NEON_'+self.Domain+'_'+self.site.site
        self.l1ProductBaseName = self.lidarFilePrefix+'_DP1_'
        self.l3ProductBaseName = self.lidarFilePrefix+'_DP3_'
        self.qaProductBaseName = self.lidarFilePrefix+'_DPQA_'
        self.dl = DownloadClass()
        
        self.getProductDirs()
        self.getProductFiles()
        self.getGeoidFile()
        self.cameraParamsFile = os.path.join(self.CameraProcessingDir,self.YearSiteVisit+'_site_parameters.txt')

        self.obtainedPreviousSiteVisit = False

        self.SpectrometerProducts = ['Radiance','Reflectance','VegIndices','WaterIndices','Albedo','FPAR','LAI']
        self.vegIndices = ['ARVI','NDVI','SAVI','EVI','PRI']
        self.waterIndices = ['WBI','NMDI','NDWI','NDII','MSI']
        self.L3lidarProducts = ['DTMGtif','DSMGtif','CanopyHeightModelGtif','SlopeGtif','AspectGtif']

        self.spectrometerAncillaryProductQaList = ['Acquisition_Date','Weather_Quality_Indicator','Dark_Dense_Vegetation_Classification','Path_Length','Smooth_Surface_Elevation','Slope','Aspect','Water_Vapor_Column','Illumination_Factor','Sky_View_Factor']

        self.spectrometerProductQaList= ['NDVI','EVI','ARVI','PRI','SAVI','MSI','WBI','NDII','NDWI','NMDI','Albedo','LAI','fPAR']
        
        self.spectrometerMosaicColorMaps = {"Aspect":"gist_rainbow","Elevation":"gist_earth","Slope":"jet","Water_Vapor_Column":"Blues","Illumination_Factor":"binary",
                                            "Sky_View_Factor":"cool","Path_Length":"Wistia","ReflectanceRms":"Spectral","MaxDiffWavelength":"hsv","MaxDiff":"inferno"}

        self.nirBand = 850
        self.swirBand = 1600
        self.nirThreshold = 100
        self.swirThreshold = 50 
        self.MosaicList = []
        self.targetWavelengths = np.arange(380,2510,5)

    def getLidarProcessingParameters(self):

        self.lidarCores=30 # number of computer cores
        self.productStepSize = 1
        self.killTriangles=250 #this tells how much of a distance we are willing to interpolate across holes in DSM and DEM.
        self.overlapGridStep=5 # grid cell size in meters for computing overlap/difference values
        self.tileBuffer=25 #value in m for buffer around tiles
        self.lidarProcessingTileSize=250 #value in metres for size of tiles. If the following error warning appears during execution of any step "cannot alloc enough TIN triangles to triangulate X(a number) points", reduce the tilesize
        self.chmInterval=0.1 #value for the interval of the pit-free CHM height bins in meters
        self.classifyPlanar=0.1
        self.chmKillTriangles=2
        self.noiseStep=4
        self.noiseIsolated=5
        self.intensityKillTriangle=5
        self.thinSubcircle=0.1
        self.thinStep=0.5
        self.lasGroundGridStep=2 # grid cell size in meters for performing ground point classification
        self.lasGroundSearchParameter='extra_fine'
        self.groundClassId = 2
        self.unclassifiedClassId = 1
        self.highVegetationClassId = 5
        self.modelKeyPointClassId = 8
        self.buildingClassId = 6
        self.lasGroundOffset=0.15
        self.boundaryThinGridStep = 10
        self.boundaryConcavity = 30
        self.scanAngleLimit = 18.5
        self.usgsBuffer = 4500 #units of meters
        self.usgsCsvName = 'usgsReprojectClipDem.csv'
        self.usgsDecimate = 2
        self.lidarNoiseClass = 7
        self.usgsFilterUpperLimit = 100
        self.usgsFilterLowerLimit = -25
        self.lidarBandingFile = 'test_banding.txt'
        self.lasoverlapOutputFilename = 'lasoverlap_step'+str(self.overlapGridStep)+'_max.tif'
        self.lasoverlapHistogramName = 'lasoverlap_step'+str(self.overlapGridStep)+'_diff_histogram.png'
        self.lasoverlapAbsHistogramName = 'lasoverlap_step'+str(self.overlapGridStep)+'_abs_diff_histogram.png'
        self.lasoverlapCumulativeHistogramName = 'lasoverlap_step'+str(self.overlapGridStep)+'_abs_diff_cumulative_histogram.png'
        self.lidarFilterWindowSize = 3
        self.mosaicColorMaps = {"Aspect":"gist_rainbow","CHM":"Greens","DSM":"terrain","DTM":"gist_earth","Slope":"jet",
                                "EdgeLongestAll":"brg","EdgeLongestGround":"brg",
                                "PointDensityAll":"viridis","PointDensityGround":"viridis",
                                "Intensity":"Greys","HorizontalUncertainty":"inferno","VerticalUncertainty":"inferno",
                                "RiseTime":"jet","FallTime":"jet","Peaks":"jet","SSP":"jet","SmoothDtm":"gist_earth"}
          
        self.lasToolFileListExt = 'LasToolsFileList.txt'

    def getWaveformLidarProcessingParameters(self):
        
        self.pulsesToSaveInMemory = 5000000

    def getStartingLidarFiles(self):

        self.getProductFiles()

        self.startingfileListLas = makeFileListForLastools(self.LidarL1UnclassifiedLasDir,'lasFileList.txt','.las')

        self.startingfileListExtraLas = os.path.join(self.LidarL1UnclassifiedLasDir,'lasFileListWithExtra.txt')

        write_file_paths_to_text(self.LidarL1UnclassifiedLasDir, '.las', self.startingfileListExtraLas)

    def getMissionListAndPayload(self):

        campaigns = []
        self.MissionList = []
        self.campaigns = []
        dbFiles = collectFilesInPath(self.DatabaseLookupDir,'.accdb')


        if int(self.year) <= 2021 or self.siteString == 'PUUM':
            
            legacyMissionsCampaignDf = pd.read_csv(self.legacyFlightDateFile)
            matchedValues = legacyMissionsCampaignDf[legacyMissionsCampaignDf['YearSiteVisit'] == self.YearSiteVisit][['FlightDate', 'Campaign']]
            
            for index,row in matchedValues.iterrows():
                
                self.MissionList.append(str(row['FlightDate']))
                self.campaigns.append(row['Campaign'])
            
        else:                                                         
            missionCounter = 1
            for dbFile in dbFiles:

                tableNames = ['tblFlightLog','tblMission','lkpSite']      
                flightLigDfs = accessDbToDataframes(dbFile,tableNames)
                flightLogDf = flightLigDfs[0] 
                missionDf = flightLigDfs[1]
                siteLookupDf = flightLigDfs[2]

                siteLookupDf = siteLookupDf.dropna(subset=['SiteName'])
                siteIds = siteLookupDf[siteLookupDf['SiteName'].str.contains(str(self.site.site))]['ID']
      
                repeatMission = False
                for siteId in siteIds:
                    missionIds = flightLogDf[flightLogDf['StudySites'].str.contains(str(siteId))]['MID']

                    if missionIds.empty:
                        continue
                    campaigns = list(flightLogDf[flightLogDf['StudySites'].str.contains(str(siteId))]['CampaignID'])

                    for campaign in campaigns:
                        self.campaigns.append(campaign.split('_')[-1])

                    for missionId in missionIds:

                        mission = list(missionDf[missionDf['ID']==int(missionId)]['MissionID'])[0]
                        
                        for existingmission in self.MissionList:
                            if mission == existingmission:
                                repeatMission = True
                                
                        if repeatMission:
                            repeatMission = False
                            continue
        
                        self.MissionList.append(mission)
                        print('Mission '+str(missionCounter)+' is '+str(list(missionDf[missionDf['ID']==int(missionId)]['MissionID'])[0]))
                        print('Campaign '+str(missionCounter)+' is '+str(self.campaigns[missionCounter-1]))
                        missionCounter += 1
                campaigns = []
        if not self.MissionList:
            print('No missions match year site combination')
    
    def getProductDirs(self):

        os.makedirs(self.BaseDir,exist_ok=True)

        for mission in self.missions:

            mission.getProductDirs(self.BaseDir)

        self.L1ProductDir = os.path.join(self.BaseDir,'L1')
        self.L2ProductDir = os.path.join(self.BaseDir,'L2')
        self.L3ProductDir = os.path.join(self.BaseDir,'L3')
        self.ProcessingDir = os.path.join(self.BaseDir,'Processing')
        self.InternalDir = os.path.join(self.BaseDir,'Internal')
        self.QaProductsDir = os.path.join(self.BaseDir,'QA')
        self.MetadataProductsDir = os.path.join(self.BaseDir,'Metadata')

        self.L1SpectrometerProductDir = os.path.join(self.L1ProductDir,'Spectrometer')

        self.SpectrometerL1RadianceDir = os.path.join(self.L1SpectrometerProductDir,'Radiance')

        self.SpectrometerL3Dir = os.path.join(self.L3ProductDir,'Spectrometer')
        self.SpectrometerL3AlbedoDir=os.path.join(self.SpectrometerL3Dir,'Albedo')
        self.SpectrometerL3BiomassDir=os.path.join(self.SpectrometerL3Dir,'Biomass')
        self.SpectrometerL3FparDir=os.path.join(self.SpectrometerL3Dir,'FPAR')
        self.SpectrometerL3LaiDir=os.path.join(self.SpectrometerL3Dir,'LAI')
        self.SpectrometerL3VegIndicesDir=os.path.join(self.SpectrometerL3Dir,'VegIndices')
        self.SpectrometerL3WaterIndicesDir=os.path.join(self.SpectrometerL3Dir,'WaterIndices')
        self.SpectrometerL3ReflectanceDir=os.path.join(self.SpectrometerL3Dir,'Reflectance')

        self.LidarL1Dir = os.path.join(self.L1ProductDir,'DiscreteLidar')
        self.LidarL1TileDir = os.path.join(self.LidarL1Dir,'ClassifiedPointCloud')
        self.LidarL1UnclassifiedLasDir = os.path.join(self.LidarL1Dir,'Las')
        self.LidarL1UnclassifiedExtraLasDir = os.path.join(self.LidarL1UnclassifiedLasDir,'extra')
        self.LidarL1UnclassifiedLazDir = os.path.join(self.LidarL1Dir,'Laz')
        self.LidarL1ClassifiedLasDir = os.path.join(self.LidarL1Dir,'ClassifiedPointCloud')

        self.LidarL3Dir=os.path.join(self.L3ProductDir,'DiscreteLidar')
        self.LidarL3DtmDir=os.path.join(self.LidarL3Dir,'DTMGtif')
        self.LidarL3DsmDir=os.path.join(self.LidarL3Dir,'DSMGtif')
        self.LidarL3ChmDir=os.path.join(self.LidarL3Dir,'CanopyHeightModelGtif')
        self.LidarL3SlopeDir=os.path.join(self.LidarL3Dir,'SlopeGtif')
        self.LidarL3AspectDir=os.path.join(self.LidarL3Dir,'AspectGtif')

        self.WaveformLidarL1Dir=os.path.join(self.L1ProductDir,'WaveformLidar')
        self.PulswavesL1Dir=os.path.join(self.WaveformLidarL1Dir,'Pulsewaves')

        self.CameraL3Dir = os.path.join(self.L3ProductDir,'Camera')
        self.CameraL3ImagesDir=os.path.join(self.CameraL3Dir,'Mosaic')
        self.CameraProcessingDir = os.path.join(self.ProcessingDir,'Camera')
        self.CameraProcessingMosaicDir = os.path.join(self.CameraProcessingDir,'Mosaic')
        self.CameraProcessingSummaryFilesDir = os.path.join(self.CameraProcessingDir,'Summary_files')

        self.SpectrometerQADir = os.path.join(self.QaProductsDir,'Spectrometer')
        self.SpectrometerL3QADir=os.path.join(self.SpectrometerQADir ,'L3QA')
        self.SpectrometerQADateDir = os.path.join(self.SpectrometerQADir ,'DateMosaic')
        self.SpectrometerL3QaReflectanceDir=os.path.join(self.SpectrometerQADir ,'SampleReflectanceMosaic')
        self.SpectrometerL3QaRmsDir=os.path.join(self.SpectrometerQADir ,'RmsMosaic')
        self.SpectrometerL3QaMaxWavelengthDir=os.path.join(self.SpectrometerQADir ,'MaxDiffWavelengthMosaic')
        self.SpectrometerL3QaMaxDiffDir = os.path.join(self.SpectrometerQADir ,'MaxDiffReflectanceMosaic')

        self.spectrometerProcessingDir = os.path.join(self.ProcessingDir,'Spectrometer')
        self.spectrometerProcessingSpectralResamplingDir = os.path.join(self.spectrometerProcessingDir,'SpectralResampledTiles')
        self.SpectrometerQaL3HistsDir = os.path.join(self.SpectrometerQADir,'L3Hists')
        self.SpectrometerQaDifferenceMosaicDir = os.path.join(self.SpectrometerQADir,'ReflectanceDifferenceMosaic')
        self.SpectrometerQaTempAncillaryRastersDir = os.path.join(self.SpectrometerQADir,'ReflectanceAncillaryRastersMosaic')
        self.SpectrometerQaL3PcaDir = os.path.join(self.SpectrometerQADir,'PCA')

        self.DiscreteLidarProcessingDir = os.path.join(self.ProcessingDir,'DiscreteLidar')
        self.stripAlignDiscreteLidarProcessingDir = os.path.join(self.DiscreteLidarProcessingDir,'StripAlign')
        self.TempStripAlignDiscreteLidarProcessingDir = os.path.join(self.stripAlignDiscreteLidarProcessingDir,'Temp_Files')
        self.LidarLastoolsProcessingDir = os.path.join(self.DiscreteLidarProcessingDir,'LASTOOLS')
        self.LidarLastoolsProcessingCommandOutputDir = os.path.join(self.LidarLastoolsProcessingDir,'CommandOutput')
        self.LidarLastoolsProcessingAsciiDir = os.path.join(self.LidarLastoolsProcessingDir,'ASCII')
        self.LidarLastoolsProcessingTempTilesDir = os.path.join(self.LidarLastoolsProcessingDir,'TempTiles')
        self.LidarLastoolsProcessingTempTilesAllPointsDir = os.path.join(self.LidarLastoolsProcessingDir,'TempTilesAllPoints')
        self.LidarLastoolsProcessingTempTilesNoisePointsDir = os.path.join(self.LidarLastoolsProcessingDir,'TempTilesNoisePoints')
        self.LidarLastoolsProcessingTempTilesNormHeightDir = os.path.join(self.LidarLastoolsProcessingDir,'TempTilesNormHeight')
        self.LidarLastoolsProcessingTempTilesNormHeightNoBufferDir = os.path.join(self.LidarLastoolsProcessingDir,'TempTilesNormHeightNoBuffer')
        self.LidarLastoolsProcessingTempTilesSortPointsDir = os.path.join(self.LidarLastoolsProcessingDir,'TempTilesSort')
        self.LidarLastoolsProcessingTempTilesGroundDir = os.path.join(self.LidarLastoolsProcessingDir,'TempTilesGround')
        self.LidarLastoolsProcessingTempTilesClassifiedNoNoiseDir = os.path.join(self.LidarLastoolsProcessingDir,'TempTilesClassifiedNoNoise')
        self.LidarLastoolsProcessingTempTilesClassifiedNoBufferDir = os.path.join(self.LidarLastoolsProcessingDir,'TempTilesClassifiedNoNoiseNoBuffer')
        self.LidarLastoolsProcessingTempTilesClassifiedIntermediateDir = os.path.join(self.LidarLastoolsProcessingDir,'IntermediateClassifiedTiles')
        self.LidarLastoolsProcessingChmProcessingDir = os.path.join(self.LidarLastoolsProcessingDir,'CanopyHeightModelGtif')
        self.LidarLastoolsProcessingChmProcessingCombinedDir = os.path.join(self.LidarLastoolsProcessingDir,'CanopyHeightModelGtifCombined')
        self.LidarLastoolsProcessingChmProcessingGrndDir = os.path.join(self.LidarLastoolsProcessingChmProcessingDir,'Grnd')
        self.LidarLastoolsProcessingTempTilesNormHeightThinDir = os.path.join(self.LidarLastoolsProcessingDir,'TempTilesNormHeightThin')
        self.LidarLastoolsProcessingChmProcessingStandardDir =  os.path.join(self.LidarLastoolsProcessingChmProcessingDir,'standard')
        self.LidarLastoolsProcessingLasReoffsetDir = os.path.join(self.LidarLastoolsProcessingDir,'LasReoffset')
        self.LidarLastoolsProcessingLasReoffsetExtraDir = os.path.join(self.LidarLastoolsProcessingLasReoffsetDir,'extra')
        self.LidarLastoolsProcessingLasDenoiseDir = os.path.join(self.LidarLastoolsProcessingDir,'LasDenoise')
        self.LidarLastoolsProcessingLasDenoiseExtraDir = os.path.join(self.LidarLastoolsProcessingLasDenoiseDir,'extra')
        self.LidarLastoolsProcessingUsgsDir = os.path.join(self.LidarLastoolsProcessingDir,'USGSDEM')
        self.LidarLastoolsProcessingLasFilteredDir = os.path.join(self.LidarLastoolsProcessingDir,'LasFiltered')
        self.LidarLastoolsProcessingLasFilteredExtraDir = os.path.join(self.LidarLastoolsProcessingLasFilteredDir,'extra')
        self.LidarLastoolsProcessingDtmBbGtifDir = os.path.join(self.LidarLastoolsProcessingDir,'DTMGtifWBB')
        self.LidarLastoolsProcessingDtmGtifDir = os.path.join(self.LidarLastoolsProcessingDir,'DTMGtif')
        self.LidarLastoolsProcessingDsmSmallTileGtifDir = os.path.join(self.LidarLastoolsProcessingDir,'DSMGtifOriginalTile')
        self.LidarLastoolsProcessingDsmGtifDir = os.path.join(self.LidarLastoolsProcessingDir,'DSMGtif')
        self.LidarL3SlopeProcessingBbDir = os.path.join(self.LidarLastoolsProcessingDir,'SlopeGtifWBB')
        self.LidarL3AspectProcessingBbDir = os.path.join(self.LidarLastoolsProcessingDir,'AspectGtifWBB')
        self.LidarL3SlopeProcessingDir = os.path.join(self.LidarLastoolsProcessingDir,'SlopeGtif')
        self.LidarL3AspectProcessingDir = os.path.join(self.LidarLastoolsProcessingDir,'AspectGtif')
        self.lidarProcessingLastoolsUncertaintyDir = os.path.join(self.LidarLastoolsProcessingDir,'UncertaintyLas')
        self.LidarFilterDtmProcessingDir = os.path.join(self.LidarLastoolsProcessingDir,'FilterDtm')

        self.LidarInternalDir = os.path.join(self.InternalDir,'DiscreteLidar')
        self.LidarInternalLazStandardDir = os.path.join(self.LidarInternalDir,'LazStandard')
        self.LidarInternalAigDsmDir = os.path.join(self.LidarInternalDir,'AIGDSM')
        self.LidarInternalChmStatsDir = os.path.join(self.LidarInternalDir,'CHMStats')
        self.LidarInternalPitFreeChmDir = os.path.join(self.LidarInternalDir,'CanopyHeightModelNonPitFree')
        self.LidarInternalScriptsDir = os.path.join(self.LidarInternalDir,'LAStoolsScripts')
        self.LidarInternalIntensityImageDir = os.path.join(self.LidarInternalDir,'IntensityImage')
        self.LidarInternalLazDir = os.path.join(self.LidarInternalDir,'OriginalLaz')
        self.LidarInternalLasDir = os.path.join(self.LidarInternalDir,'OriginalLas')
        self.LidarInternalLazExtraDir = os.path.join(self.LidarInternalLazDir,'extra')
        self.LidarInternalLasExtraDir = os.path.join(self.LidarInternalLasDir,'extra')
        self.LidarInternalLasOverageDir = os.path.join(self.LidarInternalDir,'LasOverage')                                                                                
        self.LidarInternalFilteredDtmDir = os.path.join(self.LidarInternalDir,'FilteredDTMGtif')
        
        self.LidarQaDir = os.path.join(self.QaProductsDir,'DiscreteLidar')
        self.LidarInternalValidationDir = os.path.join(self.LidarQaDir,'Validation')  
        self.LidarQABandingDir = os.path.join(self.LidarQaDir,'Banding')
        self.LidarQALasInfoDir = os.path.join(self.LidarQaDir,'LasInfo')
        self.LidarQALasQCDir = os.path.join(self.LidarQaDir,'LasQC')
        self.LidarQAProcessingReportDir = os.path.join(self.LidarQaDir,'ProcessingReport')
        self.LidarInternalValidationDir = os.path.join(self.LidarQaDir,'Validation')                                                                                    

        self.LidarL1QADir = os.path.join(self.LidarQaDir,'L1QA')
        self.LidarL3QADir = os.path.join(self.LidarQaDir,'L3QA')

        self.LidarQAPointDensityDir = os.path.join(self.LidarQaDir,'PointDensity')
        self.LidarQAPointDensityGroundDir = os.path.join(self.LidarQAPointDensityDir,'PointDensityGroundRaster')
        self.LidarQAPointDensityAllDir = os.path.join(self.LidarQAPointDensityDir,'PointDensityAllRaster')
        self.LidarQALongestEdgeGroundDir = os.path.join(self.LidarQAPointDensityDir,'EdgeLongestGroundPoints')
        self.LidarQALongestEdgeAllDir = os.path.join(self.LidarQAPointDensityDir,'EdgeLongestAllPoints')
        self.LidarQAShortestEdgeGroundDir = os.path.join(self.LidarQAPointDensityDir,'EdgeShortestGroundPoints')
        self.LidarQAShortestEdgeAllDir = os.path.join(self.LidarQAPointDensityDir,'EdgeShortestAllPoints')

        self.LidarQaUncertaintyDir = os.path.join(self.LidarQaDir,'Uncertainty')
        self.LidarQaUncertaintyCsvDir = os.path.join(self.LidarQaUncertaintyDir,'UncertaintyCsv')
        self.LidarQAUncertaintyTilesDir = os.path.join(self.LidarQaUncertaintyDir,'UncertaintyTiles')
        self.LidarQAHorzUncertaintyDir = os.path.join(self.LidarQaUncertaintyDir,'UncertaintyHorzRaster')
        self.LidarQAVertUncertaintyDir = os.path.join(self.LidarQaUncertaintyDir,'UncertaintyVertRaster')

        self.LidarMetadataDir = os.path.join(self.MetadataProductsDir,'DiscreteLidar')
        self.LidarMetadataKmlDir = os.path.join(self.LidarMetadataDir,'FlightlineBoundary')
        self.LidarMetadataTileDir = os.path.join(self.LidarMetadataDir,'TileBoundary')
        self.LidarMetadataTileKmlDir = os.path.join(self.LidarMetadataTileDir,'kmls')
        self.LidarMetadataTileShpDir = os.path.join(self.LidarMetadataTileDir,'shps')
        self.LidarMetadataUncertaintyLazDir = os.path.join(self.LidarMetadataDir,'Uncertainty')

        self.waveformLidarInternalDir = os.path.join(self.InternalDir,'WaveformLidar')
        self.LidarInternalPulsewavesDir=os.path.join(self.waveformLidarInternalDir,'PulsewavesOriginal')
        self.LidarInternalPulsewavesProcessingDir = os.path.join(self.waveformLidarInternalDir,'PulsewavesIntermediateLaz')
        self.LidarPulsewavesPointDensityDir = os.path.join(self.LidarInternalPulsewavesProcessingDir,'PointDensity','PointsPerSquareMeter')
        self.LidarPulsewavesLongestEdgeDir = os.path.join(self.LidarInternalPulsewavesProcessingDir,'PointDensity','EdgeLongestAllPoints')
        self.LidarPulsewavesRiseFallMosaicDir = os.path.join(self.LidarInternalPulsewavesProcessingDir,'RiseFallMosaic')
        self.LidarPulsewavesPeakSSPMosaicDir = os.path.join(self.LidarInternalPulsewavesProcessingDir,'PeakSSPMosaic')
        self.LidarPulsewavesDSMMosaicDir = os.path.join(self.LidarInternalPulsewavesProcessingDir,'DSMElevationMosaic')
        self.LidarPulsewavesDTMMosaicDir = os.path.join(self.LidarInternalPulsewavesProcessingDir,'LastElevationMosaic')

        self.waveformLidarQADir = os.path.join(self.QaProductsDir,'WaveformLidar')
        self.waveformLidarQAProcessingReportDir = os.path.join(self.waveformLidarQADir ,'PulsewavesProcessingReport')

    def getProductFiles(self):

        SpectrometerL1ReflectanceFiles = []
        SpectrometerL1ReflectanceProcessingFiles = []
        SpectrometerL1RadianceFiles = []
        SpectrometerL1RadianceProcessingFiles = []
        SpectrometerL2AlbedoFiles = []
        SpectrometerL2BiomassFiles = []
        SpectrometerL2FparFiles = []
        SpectrometerL2LaiFiles = []
        SpectrometerL2VegIndicesFiles = []
        SpectrometerL2VegIndicesTifFiles = []
        SpectrometerL2WaterIndicesTifFiles = []
        SpectrometerL2VegIndicesZipFiles = []
        SpectrometerL2WaterIndicesZipFiles = []
        SpectrometerL3WaterIndicesTifFiles = []
        SpectrometerL3WaterIndicesTifFiles = []
        SpectrometerL2QAFiles = []
        SpectrometerMetadataRgbFiles = []
        CameraL1ImagesFiles = []
        CameraL3ImagesFiles = []

        for mission in self.missions:
            mission.getProductFiles()

        self.LidarL1UnclassifiedLasFiles=collectFilesInPath(self.LidarL1UnclassifiedLasDir,'.las')
        self.LidarL1UnclassifiedExtraLasFiles=collectFilesInPath(self.LidarL1UnclassifiedExtraLasDir,'.las')
        self.LidarL1ClassifiedLasFiles=collectFilesInPath(self.LidarL1ClassifiedLasDir,'.laz')
        self.LidarL1PulsewavesPlzZipFiles=collectFilesInPath(self.PulswavesL1Dir,'.plz')
        self.LidarL1PulsewavesWvzZipFiles=collectFilesInPath(self.PulswavesL1Dir,'.wvz')
        
        self.LidarL1PulsewavesFiles=collectFilesInPath(self.PulswavesL1Dir,'.pls')

        self.LidarL1FlightlineFiles = collectFilesInPath(self.LidarL1UnclassifiedLazDir,'.laz')
        self.LidarL1TilesFiles = collectFilesInPath(self.LidarL1TileDir,'.laz')
        self.LidarL1TilesLasFiles = collectFilesInPath(self.LidarL1TileDir,'.las')

        self.LidarL3DtmFiles=collectFilesInPath(self.LidarL3DtmDir,'.tif')
        self.LidarL3DsmFiles=collectFilesInPath(self.LidarL3DsmDir,'.tif')
        self.LidarL3ChmFiles=collectFilesInPath(self.LidarL3ChmDir,'.tif')
        self.LidarL3SlopeFiles=collectFilesInPath(self.LidarL3SlopeDir,'.tif')
        self.LidarL3AspectFiles=collectFilesInPath(self.LidarL3AspectDir,'.tif')
        self.LidarLastoolsProcessingDsmGtifFiles = collectFilesInPath(self.LidarLastoolsProcessingDsmGtifDir,'.tif')

        self.LidarInternalAigDsmFiles=collectFilesInPath(self.LidarInternalAigDsmDir)

        for file in self.LidarInternalAigDsmFiles:
            if file.endswith('dem_vf_bf'):
                self.internalAigDsmFile = file
            if file.endswith('sap_vf_bf'):
                self.internalAigSapFile = file

        self.LidarProcessingAsciiFiles=collectFilesInPath(self.LidarLastoolsProcessingAsciiDir,'.txt')

        self.LidarInternalPulsewavesFiles = collectFilesInPath(self.LidarInternalPulsewavesDir,'.pls') 

        self.LidarInternalChmStatsFiles = collectFilesInPath(self.LidarInternalChmStatsDir,Ext=None)
        self.LidarInternalPitFreeChmFiles = collectFilesInPath(self.LidarInternalPitFreeChmDir,Ext=None)
        self.LidarInternalScritpsFiles = collectFilesInPath(self.LidarInternalScriptsDir,Ext=None)
        self.LidarInternalFilteredDtmFiles = collectFilesInPath(self.LidarInternalFilteredDtmDir,Ext='.tif')
        self.LidarInternalLazStandardFiles = collectFilesInPath(self.LidarInternalLazStandardDir,Ext='.laz')

        self.LidarMetadataHorizontalUncertaintyLazFiles = collectFilesInPath(self.LidarMetadataUncertaintyLazDir, 'horizontal_uncertainty.laz')
        self.LidarMetadataVerticalUncertaintyLazFiles = collectFilesInPath(self.LidarMetadataUncertaintyLazDir, 'vertical_uncertainty.laz')
        self.LidarMetadataTileShpFiles = collectFilesInPath(self.LidarMetadataTileShpDir,Ext=None)
        self.LidarMetadataTileKmlFiles = collectFilesInPath(self.LidarMetadataTileKmlDir,'.kml')
        self.LidarMetadataFlightlineBoundaryFiles = collectFilesInPath(self.LidarMetadataKmlDir,Ext='reoffset_boundary.kml')
        

        self.LidarQAPointDensityGroundFiles = collectFilesInPath(self.LidarQAPointDensityGroundDir,'.tif')
        self.LidarQAPointDensityAllFiles = collectFilesInPath(self.LidarQAPointDensityAllDir,'.tif')
        self.LidarQALongestEdgeGroundFiles = collectFilesInPath(self.LidarQALongestEdgeGroundDir,'.tif')
        self.LidarQALongestEdgeAllFiles = collectFilesInPath(self.LidarQALongestEdgeAllDir,'.tif')
        self.LidarQAShortestEdgeGroundFiles = collectFilesInPath(self.LidarQAShortestEdgeGroundDir,'.tif')
        self.LidarQAShortestEdgeAllFiles = collectFilesInPath(self.LidarQAShortestEdgeAllDir,'.tif')

        self.LidarQAProcessingReportFiles = collectFilesInPath(self.LidarQAProcessingReportDir,'mosaic.tif')
        self.LidarQAProcessingReportDifferenceFiles = collectFilesInPath(self.LidarQAProcessingReportDir,'difference.tif')

        self.LidarInternalIntensityFiles = collectFilesInPath(self.LidarInternalIntensityImageDir,'intensity.tif')
        self.LidarInternalAigDsmFiles = collectFilesInPath(self.LidarInternalAigDsmDir,'vf_bf.hdr')
        self.LidarInternalLidarChmBilFile = collectFilesInPath(self.LidarInternalPitFreeChmDir,'.bil')
        if self.LidarInternalLidarChmBilFile:
            self.LidarInternalLidarChmBilFile = self.LidarInternalLidarChmBilFile[0]

        self.LidarInternalLidarChmIntervalFile = collectFilesInPath(self.LidarInternalPitFreeChmDir,'non_pit_free_increments.txt')
        if self.LidarInternalLidarChmIntervalFile:
            self.LidarInternalLidarChmIntervalFile=self.LidarInternalLidarChmIntervalFile[0]

        self.LidarQAHorzUncertaintyFiles = collectFilesInPath(self.LidarQAHorzUncertaintyDir,'.tif')
        self.LidarQAVertUncertaintyFiles = collectFilesInPath(self.LidarQAVertUncertaintyDir,'.tif')

        self.LidarProcessingChmLevelFiles = collectFilesInPathIncludingSubfolders(self.LidarLastoolsProcessingChmProcessingDir,Ext='.tif')
        self.LidarLastoolsProcessingDtmBbGtifFiles = collectFilesInPath(self.LidarLastoolsProcessingDtmBbGtifDir,Ext='.tif')
        self.LidarLastoolsProcessingDtmGtifFiles =  collectFilesInPath(self.LidarLastoolsProcessingDtmGtifDir,Ext='.tif')
        self.LidarLastoolsProcessingDsmGtifFiles =  collectFilesInPath(self.LidarLastoolsProcessingDsmGtifDir,Ext='.tif')
        self.LidarLastoolsProcessingChmProcessingCombinedFiles =  collectFilesInPath(self.LidarLastoolsProcessingChmProcessingCombinedDir,Ext='.tif')
        self.LidarLastoolsProcessingDsmSmallTileGtifFiles = collectFilesInPath(self.LidarLastoolsProcessingDsmSmallTileGtifDir,Ext='.tif')
        self.LidarLastoolsProcessingTempTilesClassifiedNoBufferFiles = collectFilesInPathIncludingSubfolders(self.LidarLastoolsProcessingTempTilesClassifiedNoBufferDir,Ext='.laz')
        self.LidarLastoolsProcessingTempTilesGroundFiles = collectFilesInPathIncludingSubfolders(self.LidarLastoolsProcessingTempTilesGroundDir,Ext='.laz')                                                                                                                                                            
        self.LidarFilterDtmProcessingFiles =  collectFilesInPath(self.LidarFilterDtmProcessingDir,Ext='.tif')
        self.LidarL3SlopeProcessingBbFiles = collectFilesInPath(self.LidarL3SlopeProcessingBbDir,Ext='.tif')
        self.LidarL3AspectProcessingBbFiles = collectFilesInPath(self.LidarL3AspectProcessingBbDir,Ext='.tif')
        self.LidarL3SlopeProcessingFiles = collectFilesInPath(self.LidarL3SlopeProcessingDir,Ext='.tif')
        self.LidarL3AspectProcessingFiles = collectFilesInPath(self.LidarL3AspectProcessingDir,Ext='.tif')
        
        self.WaveformLidarPointDensityFiles = collectFilesInPath(self.LidarPulsewavesPointDensityDir,Ext='.tif')
        self.WaveformLidarLongestEdgeFiles = collectFilesInPath(self.LidarPulsewavesLongestEdgeDir,Ext='.tif')
        self.WaveformLidarPeaksFiles = collectFilesInPath(self.LidarPulsewavesPeakSSPMosaicDir,Ext='peaks.tif')
        self.WaveformLidarSSPFiles = collectFilesInPath(self.LidarPulsewavesPeakSSPMosaicDir,Ext='SSP.tif')
        self.WaveformLidarRiseTimeFiles = collectFilesInPath(self.LidarPulsewavesRiseFallMosaicDir,Ext='rise_time.tif')
        self.WaveformLidarFallTimeFiles = collectFilesInPath(self.LidarPulsewavesRiseFallMosaicDir,Ext='fall_time.tif')
        self.WaveformLidarDSMFiles = collectFilesInPath(self.LidarPulsewavesDSMMosaicDir,Ext='DSM.tif')
        self.WaveformLidarDTMFiles = collectFilesInPath(self.LidarPulsewavesDTMMosaicDir,Ext='.tif')
        self.WaveformLidarQAMosaicFiles = collectFilesInPath(self.waveformLidarQAProcessingReportDir,Ext='mosaic.tif')
        self.WaveformLidarQADifferenceFiles = collectFilesInPath(self.waveformLidarQAProcessingReportDir,'difference.tif')

        self.CameraL3ImagesFiles=collectFilesInPath(self.CameraL3ImagesDir,'.tif')
        self.CameraProcessingMosaicFiles = collectFilesInPath(self.CameraProcessingMosaicDir,'')

        self.SpectrometerL3AlbedoFiles=collectFilesInPath(self.SpectrometerL3AlbedoDir,'.tif')
        self.SpectrometerL3BiomassFiles=collectFilesInPath(self.SpectrometerL3BiomassDir,'.tif')
        self.SpectrometerL3FparFiles=collectFilesInPath(self.SpectrometerL3FparDir,'.tif')
        self.SpectrometerL3FparErrorFiles=collectFilesInPath(self.SpectrometerL3FparDir,'error.tif')
        self.SpectrometerL3LaiFiles=collectFilesInPath(self.SpectrometerL3LaiDir,'.tif')
        self.SpectrometerL3LaiErrorFiles=collectFilesInPath(self.SpectrometerL3LaiDir,'error.tif')
        self.SpectrometerL3VegIndicesZipFiles=collectFilesInPath(self.SpectrometerL3VegIndicesDir,'.zip')
        self.SpectrometerL3WaterIndicesZipFiles=collectFilesInPath(self.SpectrometerL3WaterIndicesDir,'.zip')
        self.SpectrometerL3VegIndicesTifFiles=collectFilesInPath(self.SpectrometerL3VegIndicesDir,'.tif')
        self.SpectrometerL3VegIndicesErrorTifFiles=collectFilesInPath(self.SpectrometerL3VegIndicesDir,'error.tif')
        self.SpectrometerL3WaterIndicesTifFiles=collectFilesInPath(self.SpectrometerL3WaterIndicesDir,'.tif')
        self.SpectrometerL3WaterIndicesErrorTifFiles=collectFilesInPath(self.SpectrometerL3WaterIndicesDir,'error.tif')

        self.SpectrometerL3ReflectanceFiles=collectFilesInPath(self.SpectrometerL3ReflectanceDir,'.h5')
        self.SpectrometerL3ReflectanceEnviFiles=collectFilesInPath(self.SpectrometerL3ReflectanceDir,'Reflectance')
        self.SpectrometerL3ElevationEnviFiles=collectFilesInPath(self.SpectrometerL3ReflectanceDir,'Smooth_Surface_Elevation')
        self.SpectrometerL3ShadowEnviFiles=collectFilesInPath(self.SpectrometerL3ReflectanceDir,'Cast_Shadow')
        self.SpectrometerL3ReflectanceProcessingFiles = collectFilesInPath(self.SpectrometerL3ReflectanceDir,'')

        self.SpectrometerQaDifferenceMosaicFiles = collectFilesInPath(self.SpectrometerQaDifferenceMosaicDir,'.tif')

        self.SpectrometerL3QAFiles = collectFilesInPath(self.SpectrometerL3QADir)
        self.SpectrometerL3HistsFiles = collectFilesInPath(self.SpectrometerQaL3HistsDir,'.tif')
        self.SpectrometerQADateFiles = collectFilesInPath(self.SpectrometerQADateDir,'.tif')

        self.SpectrometerL3QaRmsFiles=collectFilesInPath(self.SpectrometerL3QaRmsDir,'.tif')
        self.SpectrometerL3QaMaxWavelengtFiles=collectFilesInPath(self.SpectrometerL3QaMaxWavelengthDir,'.tif')
        self.SpectrometerL3QaWaterMaskFiles = collectFilesInPath(self.SpectrometerL3QaReflectanceDir,'.tif')  
        self.SpectrometerL3QaMaxDiffFiles = collectFilesInPath(self.SpectrometerL3QaMaxDiffDir,'.tif')
        self.SpectrometerQaL3PcaFiles = collectFilesInPath(self.SpectrometerQaL3PcaDir,'.tif')
        self.rgbMosaicFile = collectFilesInPath(self.SpectrometerQaL3HistsDir,'_RGB_mosaic.tif')
        if self.rgbMosaicFile:
            self.rgbMosaicFile=self.rgbMosaicFile[0]
            
        self.nirgbMosaicFile = collectFilesInPath(self.SpectrometerQaL3HistsDir,'NIRGB_mosaic.tif')
        if self.nirgbMosaicFile:
            self.nirgbMosaicFile=self.nirgbMosaicFile[0]

        self.weatherMosaicFile = collectFilesInPath(self.SpectrometerQaL3HistsDir,'Weather_Quality_Indicator_mosaic.tif')
        if self.weatherMosaicFile:
            self.weatherMosaicFile=self.weatherMosaicFile[0]
        
        self.redMosaicFile = collectFilesInPath(self.SpectrometerQaL3HistsDir,'_red_mosaic.tif')
        if self.redMosaicFile:
            self.redMosaicFile=self.redMosaicFile[0]
        
        self.greenMosaicFile = collectFilesInPath(self.SpectrometerQaL3HistsDir,'_green_mosaic.tif')
        if self.greenMosaicFile:
            self.greenMosaicFile=self.greenMosaicFile[0]
    
        self.blueMosaicFile = collectFilesInPath(self.SpectrometerQaL3HistsDir,'_blue_mosaic.tif')
        if self.blueMosaicFile:
            self.blueMosaicFile=self.blueMosaicFile[0]
        
        self.nirMosaicFile = collectFilesInPath(self.SpectrometerQaL3HistsDir,'_NIR_mosaic.tif')
        if self.nirMosaicFile:
            self.nirMosaicFile=self.nirMosaicFile[0]

        self.rmsMosaicFile = collectFilesInPath(self.SpectrometerQaL3HistsDir,'ReflectanceRMS_mosaic.tif')
        if self.rmsMosaicFile:
            self.rmsMosaicFile=self.rmsMosaicFile[0]
    
        self.maxDiffMosaicFile = collectFilesInPath(self.SpectrometerQaL3HistsDir,'MaxDiff_mosaic.tif')
        if self.maxDiffMosaicFile:
            self.maxDiffMosaicFile=self.maxDiffMosaicFile[0]
        
        self.maxDiffWavelengthMosaicFile = collectFilesInPath(self.SpectrometerQaL3HistsDir,'MaxDiffWavelength_mosaic.tif')
        if self.maxDiffWavelengthMosaicFile:
            self.maxDiffWavelengthMosaicFile=self.maxDiffWavelengthMosaicFile[0]
        
        self.DdvMosaicFile = collectFilesInPath(self.SpectrometerQaL3HistsDir,'Dense_Vegetation_Classification_mosaic.tif')
        if self.DdvMosaicFile:
            self.DdvMosaicFile=self.DdvMosaicFile[0]
        
        self.waterMaskMosaicFile = collectFilesInPath(self.SpectrometerQaL3HistsDir,'water_mask_mosaic.tif')
        if self.waterMaskMosaicFile:
            self.waterMaskMosaicFile=self.waterMaskMosaicFile[0]
        
        self.acquisitionDateMosaicFile = collectFilesInPath(self.SpectrometerQaL3HistsDir,'Acquisition_Date_mosaic.tif')
        if self.acquisitionDateMosaicFile:
            self.acquisitionDateMosaicFile=self.acquisitionDateMosaicFile[0]

    def getPreviousYearSiteVisit(self):

        print('Looking up previous year site visit...')
        if self.visit == '1':
            self.previousYearSiteVisit = None
        else:
            self.previousSiteVisit = previousSiteVisit(self.YearSiteVisit)
    
            print('Previous site visit is '+self.previousSiteVisit)
    
            self.previousYearSiteVisit = self.dl.get_previous_visit(self.YearSiteVisit)
            self.obtainedPreviousSiteVisit = True
            
            print('Previous year site visit is '+self.previousYearSiteVisit)

    def getMatchingReflectanceFilesForPreviousVisit(self):
        
        previousReflectanceH5Files = collectFilesInPath(self.previousL3ReflectanceDir,Ext='.h5')
        
        ysvMatchingFiles = []
        previousYsvMatchingFiles= []
        
        for h5File in self.SpectrometerL3ReflectanceFiles:
            
            baseNameSplit = os.path.basename(h5File).split('_')
            
            cornerCoordinate = baseNameSplit[4]+'_'+baseNameSplit[5]
            
            for previousH5File in previousReflectanceH5Files:
                
                if cornerCoordinate in previousH5File:
                    ysvMatchingFiles.append(h5File)
                    previousYsvMatchingFiles.append(previousH5File)
                    break
                
        return ysvMatchingFiles,previousYsvMatchingFiles

    def downloadPreviousYearSiteVisitSpectrometer(self):

        self.previousL3HistsDir = self.SpectrometerQaL3HistsDir.replace(self.YearSiteVisit,self.previousYearSiteVisit).replace(self.year,self.previousYearSiteVisit[0:4])

        self.previousL3ReflectanceDir = self.SpectrometerL3ReflectanceDir.replace(self.YearSiteVisit,self.previousYearSiteVisit).replace(self.year,self.previousYearSiteVisit[0:4])

        print(self.previousL3HistsDir)

        print(self.previousL3ReflectanceDir)

        self.dl.download_aop_object(self.previousL3HistsDir)

        self.dl.download_aop_object(self.previousL3ReflectanceDir)

        self.previousL3HistFiles = collectFilesInPath(self.previousL3HistsDir,Ext='.tif')

        self.previousL3ReflectanceFiles = collectFilesInPath(self.previousL3ReflectanceDir,Ext='.h5')

    def downloadPreviousYearSiteVisitLidar(self):

        self.previousLidarProcessingReportDir = self.LidarQAProcessingReportDir.replace(self.YearSiteVisit,self.previousYearSiteVisit).replace(self.year,self.previousYearSiteVisit[0:4])

        self.dl.download_aop_object(self.previousLidarProcessingReportDir)

        self.previousLidarProcessingReportFiles = collectFilesInPath(self.previousLidarProcessingReportDir,Ext='.tif')

    def getMissions(self,skipFlightlines=False):

        self.missions = []

        for mission,payload in zip(self.MissionList,self.campaigns):

            mission = missionClass(mission,payload,self.site,skipFlightlines=skipFlightlines)

            #Make sure there is at least one valid flightline in the mission
            if not skipFlightlines:
                if mission.FlightLines:
                    self.missions.append(mission)
            else:
                self.missions.append(mission)
        for mission in self.missions:
            print('Final mission is: '+mission.missionId)
  
    def keepMission(self,missionIdentifiers):
        
        tempMissions = []
        tempCampaigns = []
        
        missionIdentifiers = list(set(missionIdentifiers))
        for missionIdentifier in missionIdentifiers:
            for mission,campaign in zip(self.missions,self.campaigns):
                if missionIdentifier in mission.flightday:
                    tempMissions.append(mission)
                    tempCampaigns.append(campaign)
                    print('Keeping '+mission.flightday)
                    
        self.missions = tempMissions
        self.campaigns = tempCampaigns
        
        self.payload = payloadClass(self.year,self.campaigns[0])
        
    def removeMission(self,missionIdentifiers):
        
        missionIdentifiers = list(set(missionIdentifiers))
        for missionIdentifier in missionIdentifiers:
            missionIndex = 0
            for mission in self.missions:
                if missionIdentifier in mission.flightday: 
                    removedMission = self.missions.pop(missionIndex)
                    print('Removed '+removedMission.flightday)
                missionIndex+=1

    def getGeoidFile(self):

        if self.Domain=='D18' or self.Domain=='D19':
            self.geoidFile = 'g2012aa0_envi'
        elif self.Domain=='D04':
            self.geoidFile = 'g2012ap0_envi'
        else:
            self.geoidFile = 'g2012au0_envi'

    def generateMosaicFpar(self):

        os.makedirs(self.SpectrometerL3FparDir,exist_ok=True)

        with Pool(processes=30) as pool:
            pool.map(partial(generateFpar, outFolder=self.SpectrometerL3FparDir),self.SpectrometerL3ReflectanceFiles)
        self.getProductFiles()
        
    def generateMosaicLai(self):

        os.makedirs(self.SpectrometerL3LaiDir,exist_ok=True)

        with Pool(processes=30) as pool:
            pool.map(partial(generateLai, outFolder=self.SpectrometerL3LaiDir),self.SpectrometerL3ReflectanceFiles)
        self.getProductFiles()
       
    def generateMosaicVegIndices(self):

        os.makedirs(self.SpectrometerL3VegIndicesDir,exist_ok=True)

        with Pool(processes=30) as pool:
            pool.map(partial(generateVegIndices, outFolder=self.SpectrometerL3VegIndicesDir),self.SpectrometerL3ReflectanceFiles)
        self.getProductFiles()
        self.zipL3Files('VegIndices')
        
    def generateMosaicWaterIndices(self):

        os.makedirs(self.SpectrometerL3WaterIndicesDir,exist_ok=True)

        with Pool(processes=30) as pool:
            pool.map(partial(generateWaterIndices, outFolder=self.SpectrometerL3WaterIndicesDir),self.SpectrometerL3ReflectanceFiles)
        self.getProductFiles()
        self.zipL3Files('WaterIndices')
        
    def generateMosaicWaterMask(self):

        os.makedirs(self.SpectrometerL3QaReflectanceDir,exist_ok=True)
        outTifs = []
        for h5File in self.SpectrometerL3ReflectanceFiles:
            outTifs.append(os.path.join(self.SpectrometerL3QaReflectanceDir,os.path.basename(h5File).replace('reflectance.h5','water_mask.tif')))
        
        inH5sAndOutTifsFiles = zip(self.SpectrometerL3ReflectanceFiles, outTifs)
        
        with Pool(processes=30) as pool:
            processFunction = partial(generateWaterMaskHelper,nirBand=self.nirBand,swirBand=self.swirBand,nirThreshold=self.nirThreshold,swirThreshold=self.swirThreshold)
            pool.map(processFunction,inH5sAndOutTifsFiles)
            

    def generateReflectancePlots(self):
        
        os.makedirs(self.SpectrometerL3QaReflectanceDir,exist_ok=True)
        
        outPngs = []
        
        for h5File in self.SpectrometerL3ReflectanceFiles:
            outPngs.append(os.path.join(self.SpectrometerL3QaReflectanceDir,os.path.basename(h5File).replace('reflectance.h5','')))
        
        inH5sAndOutTifsFiles = zip(self.SpectrometerL3ReflectanceFiles, outPngs)
        
        with Pool(processes=30) as pool:
            processFunction = partial(generateReflectancePlotsHelper,nirBand=self.nirBand,swirBand=self.swirBand,nirThreshold=self.nirThreshold,swirThreshold=self.swirThreshold)
            pool.map(processFunction,inH5sAndOutTifsFiles)

    def unzipL3VegetationIndices(self):
        for zip_file in self.SpectrometerL3VegIndicesZipFiles:
            unzipFiles(zip_file)

    def unzipL3WaterIndices(self):

        for zip_file in self.SpectrometerL3WaterIndicesZipFiles:
            unzipFiles(zip_file)

    def deleteFiles(self,Files,Product):
        for File in Files:
            if Product in File:
                os.remove(File)

    def zipL3Files(self,Type):

        if Type=='VegIndices':
            #ZipFiles = self.SpectrometerL2VegIndicesZipFiles
            FilesToZip = self.SpectrometerL3VegIndicesTifFiles
            #for fileToZip in FilesToZip:
            #    print('Files to Zip: '+FileToZip)

        if Type == 'WaterIndices':
            #ZipFiles = self.SpectrometerL2WaterIndicesZipFiles
            FilesToZip = self.SpectrometerL3WaterIndicesTifFiles

        tileCoords = []
        ZipFile = []
        ZipFileListList = []
        #print(FilesToZipInMission)
        for File in FilesToZip:
            FileBase=os.path.basename(File)
            SplitFileBase=FileBase.split('_')
            tileCoords.append(SplitFileBase[4]+'_'+SplitFileBase[5])
            uniqueTileCoordsList = list(set(list(tileCoords)))
        #print(uniqueNisTimeList)
        for tileCoord in uniqueTileCoordsList:
            ZipFileList = []
            for FileToZip in FilesToZip:
                FileBase=os.path.basename(FileToZip)
                if tileCoord == FileBase.split('_')[4]+'_'+FileBase.split('_')[5]:
                    ZipFileList.append(FileToZip)
            if Type == 'VegIndices':
                ZipFile.append(ZipFileList[0].replace(ZipFileList[0].split('_')[-1],'VegIndices.zip'))
            if Type == 'WaterIndices':
                ZipFile.append(ZipFileList[0].replace(ZipFileList[0].split('_')[-1],'WaterIndices.zip'))
            ZipFileListList.append(ZipFileList)

        with Pool(processes=30) as pool:
            pool.map(zipFilesHelper, zip(ZipFileListList, ZipFile))

    def zipLidarTileBoundaryFolder(self):
        ZipFolderName = os.path.join(self.LidarMetadataDir, self.YearSiteVisit + '_TileBoundary.zip')
        print('Creating Zip Folder: ',ZipFolderName)
        zipFolder(self.LidarMetadataTileDir,ZipFolderName)
        print('Rearranging Tile Boundary folders')
        shutil.move(self.LidarMetadataTileKmlDir, self.LidarMetadataDir)
        shutil.move(self.LidarMetadataTileShpDir, self.LidarMetadataDir)
        shutil.move(ZipFolderName,self.LidarMetadataTileDir)

    def writeMosaicFromProductClass(self,outFile,Mosaic):

        Mosaic = Mosaic.astype(self.MosaicDataType)
        writeRasterToTif(outFile,Mosaic,self.MosaicExtents,self.EPSGCode,self.NoData,self.RasterCellWidth,self.RasterCellHeight)

    def convertH5sToTifs(self):

        for SpectrometerL1ReflectanceFile in self.SpectrometerL1ReflectanceFiles:
            getAllRastersFromH5(SpectrometerL1ReflectanceFile)

        for SpectrometerL1RadianceFile in self.SpectrometerL1RadianceFiles:
            getAllRastersFromH5(SpectrometerL1RadianceFile)

    def generateReflectanceAncillaryRasterMosaics(self):

        self.getProductFiles()

        os.makedirs(self.SpectrometerQaTempAncillaryRastersDir,exist_ok=True)

        generateAncillaryRastersFromH5s(self.spectrometerAncillaryProductQaList,self.SpectrometerL3ReflectanceFiles,self.SpectrometerQaTempAncillaryRastersDir,self.SpectrometerQaL3HistsDir,self.YearSiteVisit,'tiles')

    def generateReflectanceRgbRasterMosaics(self):
        
        os.makedirs(self.SpectrometerQaTempAncillaryRastersDir,exist_ok=True)
        
        generateRgbAndNirTifMosaicFromH5s(self.SpectrometerL3ReflectanceFiles,self.SpectrometerQaL3HistsDir,self.YearSiteVisit,'tiles')

    def generateRgbDistributionPlot(self):
 
        self.getProductFiles()
        
        generateRgbCumulativeDistribution(self.rgbMosaicFile,os.path.join(self.SpectrometerQaL3HistsDir,self.YearSiteVisit))
        
    def generateWaterMaskQaMosaic(self):
        
        generateMosaic(self.SpectrometerL3QaWaterMaskFiles,os.path.join(self.SpectrometerQaL3HistsDir,self.YearSiteVisit+'_'+'water_mask'+'_mosaic.tif'))

    def generateVegIndicesQaMosaics(self):
        
        for vegIndex in self.vegIndices:
            vegIndexFiles = collectFilesInPath(self.SpectrometerL3VegIndicesDir,vegIndex+'.tif')
            generateMosaic(vegIndexFiles,self.SpectrometerQaL3HistsDir+'\\'+self.YearSiteVisit+'_'+vegIndex+'_mosaic.tif')

    def generateVegIndicesDifferenceQaMosaics(self):
         
        for vegIndex in self.vegIndices:
            vegIndexFiles = collectFilesInPath(self.SpectrometerL3VegIndicesDir,vegIndex+'.tif')
            for previousFile in self.previousL3HistFiles:
                if vegIndex in previousFile and (vegIndex+'_mosaic' in previousFile or vegIndex+'_tiles' in previousFile or vegIndex+'.tif' in previousFile):
                     currentVegIndexPreviousFile = previousFile
                     differenceArray, profile = differenceGeotiffFiles(os.path.join(self.SpectrometerQaL3HistsDir,self.YearSiteVisit+'_'+vegIndex+'_mosaic.tif'),currentVegIndexPreviousFile)

                     outputFile = os.path.join(self.SpectrometerQaL3HistsDir,self.YearSiteVisit+'_'+self.previousYearSiteVisit+'_'+vegIndex+'_difference_mosaic.tif')
                     with rio.open(outputFile, "w", **profile) as dst:
                         dst.write(differenceArray)
                     break

    def generateWaterIndicesDifferenceQaMosaics(self):
         
        for waterIndex in self.waterIndices:
            waterIndexFiles = collectFilesInPath(self.SpectrometerL3VegIndicesDir,waterIndex+'.tif')
            for previousFile in self.previousL3HistFiles:
                if waterIndex in previousFile and (waterIndex+'_mosaic' in previousFile or waterIndex+'_tiles' in previousFile or waterIndex+'.tif' in previousFile):
                     currentWaterIndexPreviousFile = previousFile
                     differenceArray, profile = differenceGeotiffFiles(os.path.join(self.SpectrometerQaL3HistsDir,self.YearSiteVisit+'_'+waterIndex+'_mosaic.tif'),currentWaterIndexPreviousFile)

                     outputFile = os.path.join(self.SpectrometerQaL3HistsDir,self.YearSiteVisit+'_'+self.previousYearSiteVisit+'_'+waterIndex+'_difference_mosaic.tif')
                     with rio.open(outputFile, "w", **profile) as dst:
                         dst.write(differenceArray)
                     break                

    def generateVegIndicesErrorQaMosaics(self):
        
        for vegIndex in self.vegIndices:
            vegIndexFiles = collectFilesInPath(self.SpectrometerL3VegIndicesDir,vegIndex+'_error.tif')
            generateMosaic(vegIndexFiles,self.SpectrometerQaL3HistsDir+'\\'+self.YearSiteVisit+'_'+vegIndex+'_error_mosaic.tif')

    def generateWaterIndicesQaMosaics(self):
        
        for waterIndex in self.waterIndices:

            waterIndexFiles = collectFilesInPath(self.SpectrometerL3WaterIndicesDir,waterIndex+'.tif')
            generateMosaic(waterIndexFiles,self.SpectrometerQaL3HistsDir+'\\'+self.YearSiteVisit+'_'+waterIndex+'_mosaic.tif')

    def generateWaterIndicesErrorQaMosaics(self):
        
        for waterIndex in self.waterIndices:

            waterIndexFiles = collectFilesInPath(self.SpectrometerL3WaterIndicesDir,waterIndex+'_error.tif')
            generateMosaic(waterIndexFiles,self.SpectrometerQaL3HistsDir+'\\'+self.YearSiteVisit+'_'+waterIndex+'_error_mosaic.tif')

    def generateAlbedoQaMosaic(self):
        
        generateMosaic(self.SpectrometerL3AlbedoFiles,self.SpectrometerQaL3HistsDir+'\\'+self.YearSiteVisit+'_'+'Albedo'+'_mosaic.tif')
        
    def generateAlbedoDifferenceQaMosaic(self):
         
        for previousFile in self.previousL3HistFiles:
            if 'Albedo_mosaic' in previousFile or 'Albedo_tiles' in previousFile or 'albedo.tif' in previousFile:
                 currentPreviousFile = previousFile
                 differenceArray, profile = differenceGeotiffFiles(os.path.join(self.SpectrometerQaL3HistsDir,self.YearSiteVisit+'_Albedo_mosaic.tif'),currentPreviousFile)

                 outputFile = os.path.join(self.SpectrometerQaL3HistsDir,self.YearSiteVisit+'_'+self.previousYearSiteVisit+'_Albedo_difference_mosaic.tif')
                 with rio.open(outputFile, "w", **profile) as dst:
                     dst.write(differenceArray)
                 break

    def generateFparQaMosaic(self):
        
        generateMosaic(self.SpectrometerL3FparFiles,self.SpectrometerQaL3HistsDir+'\\'+self.YearSiteVisit+'_'+'Fpar'+'_mosaic.tif')
    
    def generateFparErrorQaMosaic(self):
        
        generateMosaic(self.SpectrometerL3FparErrorFiles,self.SpectrometerQaL3HistsDir+'\\'+self.YearSiteVisit+'_'+'Fpar'+'_error_mosaic.tif')
        
    def generateFparDifferenceQaMosaic(self):
         
        for previousFile in self.previousL3HistFiles:
            if 'Fpar_mosaic' in previousFile or 'fPAR_tiles' in previousFile or 'fPAR.tif' in previousFile:
                 matchedPreviousFile = previousFile
                 differenceArray, profile = differenceGeotiffFiles(os.path.join(self.SpectrometerQaL3HistsDir,self.YearSiteVisit+'_Fpar_mosaic.tif'),matchedPreviousFile)

                 outputFile = os.path.join(self.SpectrometerQaL3HistsDir,self.YearSiteVisit+'_'+self.previousYearSiteVisit+'_Fpar_difference_mosaic.tif')
                 with rio.open(outputFile, "w", **profile) as dst:
                     dst.write(differenceArray)
                 break                     

    def generateLaiQaMosaic(self):
        
        generateMosaic(self.SpectrometerL3LaiFiles,self.SpectrometerQaL3HistsDir+'\\'+self.YearSiteVisit+'_'+'Lai'+'_mosaic.tif')

    def generateLaiErrorQaMosaic(self):
        
        generateMosaic(self.SpectrometerL3LaiErrorFiles,self.SpectrometerQaL3HistsDir+'\\'+self.YearSiteVisit+'_'+'Lai'+'_error_mosaic.tif')
        
    def generateLaiDifferenceQaMosaic(self):
         
        for previousFile in self.previousL3HistFiles:
            if 'Lai_mosaic' in previousFile or 'LAI_tiles' in previousFile or 'LAI.tif' in previousFile:
                 matchedPreviousFile = previousFile
                 differenceArray, profile = differenceGeotiffFiles(os.path.join(self.SpectrometerQaL3HistsDir,self.YearSiteVisit+'_Lai_mosaic.tif'),matchedPreviousFile)

                 outputFile = os.path.join(self.SpectrometerQaL3HistsDir,self.YearSiteVisit+'_'+self.previousYearSiteVisit+'_Lai_difference_mosaic.tif')
                 with rio.open(outputFile, "w", **profile) as dst:
                     dst.write(differenceArray)
                 break

    def generateReflectanceRmsQaMosaic(self):
        
        generateMosaic(self.SpectrometerL3QaRmsFiles,self.SpectrometerQaL3HistsDir+'\\'+self.YearSiteVisit+'_'+'ReflectanceRMS'+'_mosaic.tif')

    def generateReflectanceMaxDiffWavelengthQaMosaic(self):
        
        generateMosaic(self.SpectrometerL3QaMaxWavelengtFiles,self.SpectrometerQaL3HistsDir+'\\'+self.YearSiteVisit+'_'+'MaxDiffWavelength'+'_mosaic.tif')

    def generateReflectanceMaxDiffQaMosaic(self):
        
        generateMosaic(self.SpectrometerL3QaMaxDiffFiles,self.SpectrometerQaL3HistsDir+'\\'+self.YearSiteVisit+'_'+'MaxDiff'+'_mosaic.tif')

    def generateSpectrometerProductL3QaMosaics(self):
        
        self.generateVegIndicesQaMosaics()
        self.generateWaterIndicesQaMosaics()
        self.generateAlbedoQaMosaic()
        self.generateFparQaMosaic()
        self.generateLaiQaMosaic() 

    def generateSpectrometerProductErrorL3QaMosaics(self):
        
        self.generateVegIndicesErrorQaMosaics()
        self.generateWaterIndicesErrorQaMosaics()
        self.generateFparErrorQaMosaic()
        self.generateLaiErrorQaMosaic()

    def generateReflectanceDifferenceQaMosaics(self):
        
        self.generateReflectanceRmsQaMosaic()
        self.generateReflectanceMaxDiffWavelengthQaMosaic()
        self.generateReflectanceMaxDiffQaMosaic()

    def generateSpectrometerL3DifferenceQaMosaics(self):
       
        self.generateLaiDifferenceQaMosaic()
        self.generateFparDifferenceQaMosaic()
        self.generateAlbedoDifferenceQaMosaic()
        self.generateWaterIndicesDifferenceQaMosaics()
        self.generateVegIndicesDifferenceQaMosaics()

    def getSpectrometerQaReportMaps(self):

        self.getProductFiles()
        
        getSpectrometerQaReportMaps(self.SpectrometerL3HistsFiles,self.spectrometerMosaicColorMaps,'viridis','cividis','RdYlBu')

    def getSpectrometerQaReportHistograms(self):

        self.getProductFiles()
        
        for mosaicFile in self.SpectrometerL3HistsFiles:
            
            if 'Weather_Quality_Indicator' in mosaicFile or 'water_mask' in mosaicFile or '_RGB_' in mosaicFile:
                continue

            rasterVariable = os.path.basename(mosaicFile).split(self.YearSiteVisit)[-1]
            rasterVariable = os.path.basename(rasterVariable).split(self.previousYearSiteVisit)[-1]
            rasterVariable = rasterVariable.split('mosaic')[0]
            rasterVariable = rasterVariable.replace('_',' ')
            
            plotGtifHistogram(mosaicFile,mosaicFile.replace('.tif','_histogram.png'),rasterVariable)

    def getSpectrometerQaReportSummaryFiles(self):

        self.getProductFiles()
        
        if self.previousYearSiteVisit is not None:
        
            getSpectrometerQaReportSummaryFiles(self.SpectrometerQaL3HistsDir,self.YearSiteVisit+'_'+self.previousYearSiteVisit,self.spectrometerProductQaList,previousYear=True)
        
        else:
            
            getSpectrometerQaReportSummaryFiles(self.SpectrometerQaL3HistsDir,self.YearSiteVisit,self.spectrometerProductQaList,previousYear=True, firstYear = True)

    def getSpectrometerAncillaryQaReportSummaryFiles(self):

        self.getProductFiles()
        
        getSpectrometerAncillaryQaReportSummaryFiles(self.SpectrometerQaL3HistsDir,self.YearSiteVisit,self.spectrometerAncillaryProductQaList)

    def getSpectrometerReflectanceDifferenceSummaryFiles(self):

        self.getProductFiles()
        
        getSpectrometerReflectanceDifferenceSummaryFiles(self.SpectrometerQaL3HistsDir,self.rmsMosaicFile,self.maxDiffMosaicFile,self.maxDiffWavelengthMosaicFile)

    def getSpectrometerRGBQaReportSummaryFiles(self):

        self.getProductFiles()
        
        getSpectrometerRGBQaReportSummaryFiles(self.SpectrometerQaL3HistsDir,self.redMosaicFile,self.greenMosaicFile,self.blueMosaicFile,self.nirMosaicFile)

    def getRGBPngMap(self):
        
        self.getProductFiles()
        
        print(self.rgbMosaicFile)
        
        plot_multiband_geotiff(self.rgbMosaicFile, title='', stretch='linear5',save_path=self.rgbMosaicFile.replace('.tif','.png'), nodata_color='black',variable='')

    def getDdvPngMap(self):
        
        self.getProductFiles()
        
        DdvClassLabels = ['NoData','Water','DDV', 'NonRef','Shadow']
        DdvClassColors=['White','Blue','Green','Grey','Black']
        plot_classified_geotiff(self.DdvMosaicFile,DdvClassLabels,DdvClassColors, title='', save_path=self.DdvMosaicFile.replace('.tif','.png'), variable='DDV')

    def getWaterMaskPngMap(self):
        
        self.getProductFiles()
        
        waterMaskClassLabels = ['Non-water','Water']
        waterMaskClassColors=['Black','White']
        plot_classified_geotiff(self.waterMaskMosaicFile,waterMaskClassLabels,waterMaskClassColors,title='', save_path=self.waterMaskMosaicFile.replace('.tif','.png'), variable='Water Mask')

    def getNIRGBPngMap(self):
        
        self.getProductFiles()
        
        plot_multiband_geotiff(self.nirgbMosaicFile, title='', stretch='linear5',save_path=self.nirgbMosaicFile.replace('.tif','.png'), nodata_color='black',variable='')

    def getWeatherPngMap(self):
        
        self.getProductFiles()
        
        with rio.open(self.weatherMosaicFile) as src:
            weatherData = src.read()
            metadata = src.meta.copy()
            
        redBand = np.zeros_like(weatherData)
        greenBand = np.zeros_like(weatherData)
        blueBand = np.zeros_like(weatherData)
        
        greenColor = (55/255,200/255,55/255)
        yellowColor = (200/255,200/255,25/255)
        redColor = (200/255,25/255,25/255)
        
        redBand[weatherData==1] = 55
        greenBand[weatherData==1] = 200
        blueBand[weatherData==1] = 55
        
        redBand[weatherData==2] = 200
        greenBand[weatherData==2] = 200
        blueBand[weatherData==2] = 25
        
        redBand[weatherData==3] = 200
        greenBand[weatherData==3] = 25
        blueBand[weatherData==3] = 25
        
        weatherStack = [redBand,greenBand,blueBand]
        metadata['count'] = len(weatherStack)
        metadata['nodata'] = 0
        metadata['dtype'] = np.uint8
        weatherStack = np.vstack(weatherStack).astype(np.uint8)
        
        weatherOutFile = self.weatherMosaicFile.replace('mosaic','RGB_mosaic')
        
        weatherClassLabels = ['<10% Cloud Cover','10-50% Clound Cover','>50% Cloud Cover']
        
        with rio.open(weatherOutFile, 'w', **metadata) as dst:
            dst.write(weatherStack)

        plot_multiband_geotiff(weatherOutFile, title='', stretch=None,save_path=weatherOutFile.replace('.tif','.png'), nodata_color='black',variable='',classLabels=weatherClassLabels,classColors=[greenColor,yellowColor,redColor])
    
    def getAcquisitionDatePngMap(self):

        self.getProductFiles()

        plot_dates_geotiff(self.acquisitionDateMosaicFile, title='', save_path=self.acquisitionDateMosaicFile.replace('.tif','.png'))

    #TODO: THIS CAN BE IMPROVED WITH MP
    def getPcaFromReflectance(self,numOutputComponents):

        os.makedirs(self.SpectrometerQaL3PcaDir,exist_ok=True)   

        sampleNormalizedDataCube = []
        
        sampleFactor = 2

        for h5File in self.SpectrometerL3ReflectanceFiles:
            
            print('Working on PCA for: '+h5File)

            reflectanceArray,metadata,wavelengths = h5refl2array(h5File, 'Reflectance', onlyMetadata = False)
            
            if np.any(np.isnan(reflectanceArray)):
                print('We have Nans reflectance')
            
            normalizedDataCube = getCleanNormalizedDataCube(reflectanceArray,metadata,wavelengths)
            
            normalizedDataCube = normalizedDataCube.astype(np.float32)
            
            normalizedDataCube[normalizedDataCube == metadata['noDataVal']] = np.nan
            
            sampleNormalizedDataCube.append(normalizedDataCube[::sampleFactor,::sampleFactor,:])
            
        dataCubeForPca = prepDataForPca(np.hstack(sampleNormalizedDataCube))    
        
        pca, pcaComponents, pcaExplainedVariance = performPca(dataCubeForPca)

        for h5File in self.SpectrometerL3ReflectanceFiles:
        
            reflectanceArray,metadata,wavelengths = h5refl2array(h5File, 'Reflectance', onlyMetadata = False)
            
            normalizedDataCube = getCleanNormalizedDataCube(reflectanceArray,metadata,wavelengths)
            
            normalizedDataCube = normalizedDataCube.astype(np.float32)
            
            normalizedDataCube[normalizedDataCube == metadata['noDataVal']] = np.nan
            
            normalizedDataCube = prepDataForPca(normalizedDataCube,findingPcaComponents = False)
                 
            normalizedDataCube[np.isnan(normalizedDataCube)] = metadata['noDataVal']
            
            pcaComponents = applyPca(pca,normalizedDataCube)
    
            if numOutputComponents > reflectanceArray.shape[2]:
                numOutputComponents =  reflectanceArray.shape[2]
    
            #normalizedPcaComponents = normalizeBands(pcaComponents)
            #normalizedPcaComponents[np.isnan(normalizedPcaComponents)] = metadata['noDataVal']

            outputFile = os.path.join(self.SpectrometerQaL3PcaDir,os.path.basename(h5File).replace('bidirectional_reflectance.h5','PCA.tif')) 

            pcaComponents[normalizedDataCube==metadata['noDataVal']]=metadata['noDataVal']
            
            EPSGCode,MosaicExtents,RasterCellHeight,RasterCellWidth,NoData = getGtifMetadataFromH5Metadata(metadata)

            writeRasterToTif(outputFile,pcaComponents[:,:,0:numOutputComponents],MosaicExtents,EPSGCode,NoData,RasterCellWidth,RasterCellHeight)

    def getPcaMosaic(self):

        self.getProductFiles()

        generateMosaic(self.SpectrometerQaL3PcaFiles,os.path.join(self.SpectrometerQaL3HistsDir,self.YearSiteVisit+'_'+'PCA'+'_mosaic.tif'))

    #def generateLidarL3Mosaics(self):

    #    self.getProductFiles()

    #   os.makedirs(self.LidarQAProcessingReportDir,exist_ok=True)
      
    def generateLidarDtmMosaic(self):
        
        generateMosaic(self.LidarL3DtmFiles,os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_DTM_mosaic.tif'))
        
        self.MosaicList.append(os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_DTM_mosaic.tif'))
    
    def generateLidarDsmMosaic(self):
        
        generateMosaic(self.LidarL3DsmFiles,os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_DSM_mosaic.tif'))
        
        self.MosaicList.append(os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_DSM_mosaic.tif'))
        
    def generateLidarChmMosaic(self):
        
        generateMosaic(self.LidarL3ChmFiles,os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_CHM_mosaic.tif'))
        
        self.MosaicList.append(os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_CHM_mosaic.tif'))
        
    def generateLidarSlopeMosaic(self):
        
        generateMosaic(self.LidarL3SlopeFiles,os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_Slope_mosaic.tif'))
        
        self.MosaicList.append(os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_Slope_mosaic.tif'))

    def generateLidarAspectMosaic(self):
        
        generateMosaic(self.LidarL3SlopeFiles,os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_Aspect_mosaic.tif'))
        
        self.MosaicList.append(os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_Aspect_mosaic.tif'))

    def generateLidarL3Mosaics(self):

        self.getProductFiles()

        self.generateLidarDtmMosaic()
        
        self.generateLidarDsmMosaic()
        
        self.generateLidarChmMosaic()
        
        self.generateLidarSlopeMosaic()
        
        self.generateLidarAspectMosaic()
        
    def generateLidarFilterDtmMosaic(self):
        
        generateMosaic(self.LidarFilterDtmProcessingFiles,os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_SmoothDtm_mosaic.tif'),EPSGoveride = self.epsgCode)        

    def generateLidarIntensityMosaics(self):

        self.getProductFiles()

        try: 
            generateMosaic(self.LidarInternalIntensityFiles,self.LidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_Intensity_mosaic.tif')
            self.MosaicList.append(self.LidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_Intensity_mosaic.tif')
        except:
            print('Could not make mosaic for '+self.LidarInternalIntensityImageDir+'\\'+self.YearSiteVisit+'_Intensity_mosaic.tif')

    def generatePointDensityMosaics(self):

        self.getProductFiles()

        try:
            if not os.path.exists(os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_PointDensityGround_mosaic.tif')):
            
                generateMosaic(self.LidarQAPointDensityGroundFiles,self.LidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_PointDensityGround_mosaic.tif')
                self.MosaicList.append(os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_PointDensityGround_mosaic.tif'))
            else:
                self.MosaicList.append(os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_PointDensityGround_mosaic.tif'))
        except:
            print('Could not make mosaic for '+self.LidarQAPointDensityGroundDir+'\\'+self.YearSiteVisit+'_PointDensityGround_mosaic.tif')

        try:
            
            if not os.path.exists(os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_PointDensityAll_mosaic.tif')):
                generateMosaic(self.LidarQAPointDensityAllFiles,self.LidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_PointDensityAll_mosaic.tif')
                self.MosaicList.append(os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_PointDensityAll_mosaic.tif'))
            else:
                self.MosaicList.append(os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_PointDensityAll_mosaic.tif'))
        except:
            print('Could not make mosaic for '+self.LidarQAPointDensityAllDir+'\\'+self.YearSiteVisit+'_PointDensityAll_mosaic.tif')


        try:
            if not os.path.exists(os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_EdgeLongestGround_mosaic.tif')): 
                generateMosaic(self.LidarQALongestEdgeGroundFiles,self.LidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_EdgeLongestGround_mosaic.tif')
                self.MosaicList.append(os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_EdgeLongestGround_mosaic.tif'))
            else:
                self.MosaicList.append(os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_EdgeLongestGround_mosaic.tif'))
        except:
            print('Could not make mosaic for '+self.LidarQALongestEdgeGroundDir+'\\'+self.YearSiteVisit+'_EdgeLongestGround_mosaic.tif')

        try:
            if not os.path.exists(os.path.join(self.LidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_EdgeLongestAll_mosaic.tif')):
                generateMosaic(self.LidarQALongestEdgeAllFiles,self.LidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_EdgeLongestAll_mosaic.tif')
                self.MosaicList.append(self.LidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_EdgeLongestAll_mosaic.tif')
            else:
                self.MosaicList.append(self.LidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_EdgeLongestAll_mosaic.tif')
        except:
            print('Could not make mosaic for '+self.LidarQALongestEdgeAllDir+'\\'+self.YearSiteVisit+'_EdgeLongestAll_mosaic.tif')
       
    def generateLidarUncertaintyMosaics(self):

        self.getProductFiles()

        try:
            generateMosaic(self.LidarQAHorzUncertaintyFiles,self.LidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_HorizontalUncertainty_mosaic.tif')
            self.MosaicList.append(self.LidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_HorizontalUncertainty_mosaic.tif')
        except:
            print('Could not make mosaic for '+self.LidarQAHorzUncertaintyDir+'\\'+self.YearSiteVisit+'_HorizontalUncertainty_mosaic.tif')

        try:
            generateMosaic(self.LidarQAVertUncertaintyFiles,self.LidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_VerticalUncertainty_mosaic.tif')
            self.MosaicList.append(self.LidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_VerticalUncertainty_mosaic.tif')
        except:
            print('Could not make mosaic for '+self.LidarQAVertUncertaintyDir+'\\'+self.YearSiteVisit+'_VerticalUncertainty_mosaic.tif')

    def generateWaveformQaMosaics(self):
        self.getProductDirs()
        self.getProductFiles()

        try:
            generateMosaic(self.WaveformLidarPointDensityFiles,self.waveformLidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_PointDensityAll_mosaic.tif')
            self.MosaicList.append(self.waveformLidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_HorizontalUncertainty_mosaic.tif')
        except:
            print('Could not make mosaic for '+self.WaveformLidarPointDensityDir+'\\'+self.YearSiteVisit+'_PointDensityAll_mosaic.tif')

        try:
            generateMosaic(self.WaveformLidarLongestEdgeFiles,self.waveformLidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_EdgeLongestAll_mosaic.tif')
            self.MosaicList.append(self.waveformLidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_EdgeLongestAll_mosaic.tif')
        except:
            print('Could not make mosaic for '+self.WaveformLidarLongestEdgeDir+'\\'+self.YearSiteVisit+'_EdgeLongestAll_mosaic.tif')

        try:
            generateMosaic(self.WaveformLidarPeaksFiles,self.waveformLidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_Peaks_mosaic.tif')
            self.MosaicList.append(self.waveformLidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_Peaks_mosaic.tif')
        except:
            print('Could not make mosaic for '+self.WaveformLidarPeaksDir+'\\'+self.YearSiteVisit+'_Peaks_mosaic.tif')

        try:
            generateMosaic(self.WaveformLidarSSPFiles,self.waveformLidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_SSP_mosaic.tif')
            self.MosaicList.append(self.waveformLidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_SSP_mosaic.tif')
        except:
            print('Could not make mosaic for '+self.WaveformLidarSSPFiles+'\\'+self.YearSiteVisit+'_SSP_mosaic.tif')

        try:
            generateMosaic(self.WaveformLidarRiseTimeFiles,self.waveformLidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_RiseTime_mosaic.tif')
            self.MosaicList.append(self.waveformLidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_RiseTime_mosaic.tif')
        except:
            print('Could not make mosaic for '+self.WaveformLidarRiseTimeDir+'\\'+self.YearSiteVisit+'_RiseTime_mosaic.tif')

        try:
            generateMosaic(self.WaveformLidarFallTimeFiles,self.waveformLidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_FallTime_mosaic.tif')
            self.MosaicList.append(self.waveformLidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_FallTime_mosaic.tif')
        except:
            print('Could not make mosaic for '+self.WaveformLidarRiseTimeDir+'\\'+self.YearSiteVisit+'_FallTime_mosaic.tif')

        try:
            generateMosaic(self.WaveformLidarDSMFiles,self.waveformLidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_DSM_mosaic.tif')
            self.MosaicList.append(self.waveformLidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_DSM_mosaic.tif')
        except:
            print('Could not make mosaic for '+self.WaveformLidarDSMDir+'\\'+self.YearSiteVisit+'_DSM_mosaic.tif')

        try:
            generateMosaic(self.WaveformLidarDTMFiles,self.waveformLidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_DTM_mosaic.tif')
            self.MosaicList.append(self.waveformLidarQAProcessingReportDir+'\\'+self.YearSiteVisit+'_DTM_mosaic.tif')
        except:
            print('Could not make mosaic for '+self.WaveformLidarDTMDir+'\\'+self.YearSiteVisit+'_DTM_mosaic.tif')

    def getLidarValidationData(self):
        
        # if there is no lidar validation file, return
        if not self.lidarValidationFile:
            print('WARNING: No lidar validation data for site',self.YearSiteVisit)
            return
        else:
        
            self.lidarValidationData = genfromtxt(self.lidarValidationFile, delimiter=',')
            
            self.lidarValidationHorzCoords = list(zip(self.lidarValidationData[:, 1], self.lidarValidationData[:, 2]))
            
            self.lidarValidatioVertCoords = self.lidarValidationData[:,3]

    def getPulsewavesQaReportMaps(self):

        self.getProductFiles()

        for mosaicFile in self.WaveformLidarQAMosaicFiles:

            rasterVariable = os.path.basename(mosaicFile).split('_')[-2]

            plot_geotiff(mosaicFile , title='', cmap=self.mosaicColorMaps[rasterVariable], save_path=mosaicFile.replace('.tif','.png'), nodata_color='black',variable=rasterVariable)

    def getPulsewavesQaReportHistograms(self):

        self.getProductFiles()

        for mosaicFile in self.WaveformLidarQAMosaicFiles:

            rasterVariable = os.path.basename(mosaicFile).split('_')[-2]

            plotGtifHistogram(mosaicFile,mosaicFile.replace('.tif','_hist.png'),rasterVariable)

    def getWaveformLidarDifferenceMosaics(self):
    # function to difference the waveform and discrete-derived DTM and DSM mosaics
        self.getProductFiles()
        self.downloadLidarQa()

        # for mosaicFile in self.LidarQAProcessingReportFiles:
        for mosaicFile in self.WaveformLidarQAMosaicFiles:
            if 'DTM' in mosaicFile or 'DSM' in mosaicFile:
                print('mosaicFile: ',mosaicFile)
                rasterVariable = os.path.basename(mosaicFile).split('_')[-2]

                foundMatch = False
                for discreteMosaicFile in self.LidarQAProcessingReportFiles:

                    if rasterVariable.lower() in os.path.basename(discreteMosaicFile).replace('_','').lower():
                        foundMatch = True
                        break

                if not foundMatch:
                    print(f"Could not find {discreteMosaicFile}")
                    continue

                print(rasterVariable)
                print(mosaicFile)
                print(discreteMosaicFile)

                differenceRaster, profile = differenceGeotiffFiles(mosaicFile,discreteMosaicFile)

                outputFile = os.path.join(self.waveformLidarQAProcessingReportDir,self.YearSiteVisit+'_'+rasterVariable+'_difference.tif')
                with rio.open(outputFile, "w", **profile) as dst:
                    dst.write(differenceRaster)

    def getPulsewavesQaReportDifferenceMaps(self):

        self.getProductFiles()

        for mosaicFile in self.WaveformLidarQADifferenceFiles:

            rasterVariable = os.path.basename(mosaicFile).split('_')[-2] + '_Difference'

            plot_geotiff(mosaicFile , title='', cmap="RdBu", save_path=mosaicFile.replace('.tif','.png'), nodata_color='black', variable=rasterVariable, vmin=-2, vmax=2)

    def getPulsewavesQaReportDifferenceHistograms(self):

        self.getProductFiles()

        for mosaicFile in self.WaveformLidarQADifferenceFiles:

            rasterVariable = os.path.basename(mosaicFile).split('_')[-2] + '_Difference'

            plotGtifHistogram(mosaicFile,mosaicFile.replace('.tif','_hist.png'),rasterVariable,xlims=(-1,1))

    def generatePulsewavesQAReport(self):

        currentDir = os.getcwd()

        # run Lidar\src\L1\WaveformQA\pulsewaves_qaqc_md.py script to generate WaveformLidar QA markdown file
        os.chdir(lidarSrcL1WaveformDir)
        
        os.system(f"python pulsewaves_qaqc_md.py {self.YearSiteVisit}")
        
        os.chdir(currentDir)

    def getMinimumNoiseFraction(self,numOutputComponents):

        for mnfDirectory in self.SpectrometerL1QaMnfDir:

            os.makedirs(mnfDirectory,exist_ok=True)

        for reflectanceH5File in self.SpectrometerL1ReflectanceFiles:

            dataArray,reflectanceMetadata,wavelengths = h5refl2array(reflectanceH5File, 'Reflectance', onlyMetadata = False)

            badBandMask = findBadBands(wavelengths,(np.array((1800,1960)),np.array((1337,1430)),np.array((300,400),dtype=np.float32),np.array((2450,2600),dtype=np.float32))).astype(np.float32)

            badBandMask[badBandMask==0] = np.nan

            dataArray = dataArray.astype(np.float32)

            dataArray[dataArray==reflectanceMetadata['noDataVal']] = np.nan

            normalizedDataCube = normalizeSpectraSumSquares(np.multiply(dataArray,badBandMask[None,None,:]))

            normalizedDataCube[np.isnan(normalizedDataCube)] = 0

            mnfDataCube = performMnf(normalizedDataCube)

            mnfDataCube[np.isnan(mnfDataCube)] = reflectanceMetadata['noDataVal']

            outputFile = reflectanceH5File.replace('\\L1\\','\\QA\\')
            outputFile = outputFile.replace('\\ReflectanceH5\\','\\MNF\\')
            outputFile = outputFile.replace('.h5','_MNF.tif')

            if numOutputComponents > dataArray.shape[2]:
                numOutputComponents =  dataArray.shape[2]

            mnfDataCube[np.isnan(dataArray)]= reflectanceMetadata['noDataVal']

            writeRasterToTif(outputFile,mnfDataCube[:,:,0:numOutputComponents],np.array(reflectanceMetadata['extent'],dtype=np.float)[0],reflectanceMetadata['EPSG'],reflectanceMetadata['noDataVal'], reflectanceMetadata['res']['pixelWidth'],reflectanceMetadata['res']['pixelHeight'])

    def downloadAllSpectrometerProducts(self):

        print('Downloading all spectrometer '+'products'+' for '+self.YearSiteVisit)
        for mission in self.missions:
            mission.downloadAllL1SpectrometerProducts()
            mission.downloadAllL2SpectrometerProducts()
        self.downloadAllL3SpectrometerProducts()

    def downloadAllProducts(self):

        print('Downloading all products for '+self.YearSiteVisit)

        self.downloadAllCameraProducts()
        self.downloadAllLidarProducts()
        self.downloadAllSpectrometerProducts()

    def downloadAllCameraProducts(self):

        print('Downloading all camera products for '+self.YearSiteVisit)
        for mission in self.missions:
            mission.downloadL1Camera()

        self.downloadL3Camera()

    def downloadL3Camera(self):

        print('Downloading spectrometer '+'L3'+' '+'camera'+' for '+self.YearSiteVisit)

        self.dl.download_aop_object(self.CameraL3ImagesDir,Ext='.tif')
        
        self.getProductFiles()

    def downloadL3ReflectanceH5s(self):

        print('Downloading '+'L3 '+'Reflectance '+ 'for '+self.YearSiteVisit)

        self.dl.download_aop_object(self.SpectrometerL3ReflectanceDir,Ext='.h5')
        self.getProductFiles()

    def downloadL3Albedo(self):

        print('Downloading '+'L3 '+'Albedo '+ 'for '+self.YearSiteVisit)
        self.dl.download_aop_object(self.SpectrometerL3AlbedoDir,Ext='.tif')
        self.getProductFiles()


    def downloadL3Fpar(self):

        print('Downloading '+'L3 '+'FPAR '+ 'for '+self.YearSiteVisit)
        self.dl.download_aop_object(self.SpectrometerL3FparDir,Ext='.tif')
        self.getProductFiles()

    def downloadL3Lai(self):

        print('Downloading '+'L3 '+'LAI '+ 'for '+self.YearSiteVisit)

        self.dl.download_aop_object(self.SpectrometerL3LaiDir,ext='.tif')
        self.getProductFiles()

    def downloadL3VegetationIndices(self):

        print('Downloading '+'L3 '+'Vegetation Indices '+ 'for '+self.YearSiteVisit)

        if not self.SpectrometerL3VegIndicesTifFiles:
            self.dl.download_aop_object(self.SpectrometerL3VegIndicesDir,Ext='zip')
            self.getProductFiles()
            self.unzipL3VegetationIndices()
            self.getProductFiles()
            for zipFile in self.SpectrometerL3VegIndicesZipFiles:
                os.remove(zipFile)
            self.getProductFiles()

    def downloadL3WaterIndices(self):

        print('Downloading '+'L3 '+'Water Indices '+ 'for '+self.YearSiteVisit)

        if not self.SpectrometerL3WaterIndicesTifFiles:

            self.dl.download_aop_object(self.SpectrometerL3WaterIndicesDir,Ext='zip')
            self.getProductFiles()
            self.unzipL3WaterIndices()
            self.getProductFiles()
            for zipFile in self.SpectrometerL3WaterIndicesZipFiles:
                os.remove(zipFile)
            self.getProductFiles()

    def downloadAllL3SpectrometerProducts(self):

        print('Downloading all '+'L3 '+'spectrometer products '+ 'for '+self.YearSiteVisit)

        self.downloadL3ReflectanceH5s()
        self.downloadL3Albedo()
        self.downloadL3Fpar()
        self.downloadL3Lai()
        self.downloadL3VegetationIndices()
        self.downloadL3WaterIndices()

        self.getProductFiles()

    def downloadL3SpectrometerQaMosaics(self,product=None):
        
        print('Downloading all '+'L3 '+'spectrometer QA mosaics '+ 'for '+self.YearSiteVisit)

        if product is None:

            self.dl.download_aop_object(self.SpectrometerQaL3HistsDir,Ext='.tif')
        else:
            
            if int(self.year) > 2020:
            
                productExtension = product+'_mosaic.tif'
            else:
                productExtension = product+'.tif'
            
            self.dl.download_aop_object(self.SpectrometerQaL3HistsDir,Ext=productExtension)

        self.getProductFiles()
        
    def downloadInternalLidarAigDsm(self):

        print('Downloading all lidar AIGDSM '+self.YearSiteVisit)

        self.dl.download_aop_object(os.path.join(self.LidarInternalAigDsmDir))
        self.getProductFiles()

    def downloadAllLidarProducts(self):

        print('Downloading all lidar products for '+self.YearSiteVisit)

        self.downloadL1LidarProducts()
        self.downloadL3LidarProducts()

    def downloadL1UnclassifiedPointCloud(self):

         print('Downloading Unclassified flightlines for '+self.YearSiteVisit)

         self.dl.download_aop_object(self.LidarL1UnclassifiedLazDir,Ext='laz')

         self.getProductFiles()

    def downloadL1ClassifiedPointCloud(self):

        print('Downloading Classified point cloud tiles for '+self.YearSiteVisit)

        self.dl.download_aop_object(self.LidarL1ClassifiedLasDir,Ext='.laz')

        self.getProductFiles()

    def downloadL1LidarProducts(self):

        print('Downloading all lidar L1 products for '+self.YearSiteVisit)

        self.downloadL1UnclassifiedPointCloud()
        self.downloadL1ClassifiedPointCloud()

    def downloadL3LidarDtm(self):
        print('Downloading tiles for '+self.YearSiteVisit+' DTMGtif')

        self.dl.download_aop_object(self.LidarL3DtmDir,Ext='tif')

        self.getProductFiles()

    def downloadL3LidarDsm(self):
        print('Downloading tiles for '+self.YearSiteVisit+' DSMGtif')

        self.dl.download_aop_object(self.LidarL3DsmDir,Ext='tif')

        self.getProductFiles()

    def downloadL3LidarChm(self):
        print('Downloading tiles for '+self.YearSiteVisit+' CanopyHeightModelGtif')

        self.dl.download_aop_object(self.LidarL3ChmDir,Ext='tif')

        self.getProductFiles()

    def downloadL3LidarSlope(self):
        print('Downloading tiles for '+self.YearSiteVisit+' SlopeGtif')

        self.dl.download_aop_object(self.LidarL3SlopeDir,Ext='tif')

        self.getProductFiles()

    def downloadL3LidarAspect(self):
        print('Downloading tiles for '+self.YearSiteVisit+' AspectGtif')

        self.dl.download_aop_object(self.LidarL3AspectDir,Ext='tif')

        self.getProductFiles()

    def downloadLidarMetadataTileBoundary(self):
        print('Downloading '+self.YearSiteVisit+' Metadata/DiscreteLidar/TileBoundary/kmls')
        self.dl.download_aop_object(self.LidarMetadataKmlDir)
        
        print('Downloading '+self.YearSiteVisit+' Metadata/DiscreteLidar/TileBoundary/shps')
        self.dl.download_aop_object(self.LidarMetadataTileShpDir)

        self.getProductFiles()

    def downloadLidarQa(self):

        print('Downloading lidar QA for '+self.YearSiteVisit)

        self.dl.download_aop_object(self.LidarQAProcessingReportDir)

        self.getProductFiles()

    def downloadLidarL3QaPointDensity(self):

        self.dl.download_aop_object(self.LidarQAPointDensityGroundDir,Ext='tif')
        self.dl.download_aop_object(self.LidarQAPointDensityAllDir,Ext='tif')
        self.dl.download_aop_object(self.LidarQALongestEdgeGroundDir,Ext='tif')
        self.dl.download_aop_object(self.LidarQALongestEdgeAllDir,Ext='tif')

        self.getProductFiles()

    def downloadLidarL3QaUncertainty(self,doLazTiles = False):

        self.dl.download_aop_object(self.LidarQAHorzUncertaintyDir,Ext='tif')
        self.dl.download_aop_object(self.LidarQAVertUncertaintyDir,Ext='tif')

        if doLazTiles:
            self.dl.download_aop_object(self.LidarQAUncertaintyTilesDir,ext='laz')

        self.getProductFiles()

    def downloadL3LidarProducts(self):

        print('Downloading all L3 lidar products for '+self.YearSiteVisit)

        self.downloadL3LidarDtm()
        self.downloadL3LidarDsm()
        self.downloadL3LidarChm()
        self.downloadL3LidarSlope()
        self.downloadL3LidarAspect()

        self.getProductFiles()

    def lidarPreprocessing(self):

        self.getStartingLidarFiles()

        os.makedirs(self.LidarInternalLazDir,exist_ok=True)
        os.makedirs(self.LidarInternalLazExtraDir,exist_ok=True)

        os.makedirs(self.LidarInternalLasDir,exist_ok=True)
        os.makedirs(self.LidarInternalLasExtraDir,exist_ok=True)

        for lasFile in self.LidarL1UnclassifiedLasFiles:
            shutil.move(lasFile, os.path.join(self.LidarInternalLasDir,os.path.basename(lasFile)))
        for lasFile in self.LidarL1UnclassifiedExtraLasFiles:
            shutil.move(lasFile, os.path.join(self.LidarInternalLasExtraDir,os.path.basename(lasFile)))

        fileListLas = makeFileListForLastools(self.LidarInternalLasDir,'Las'+self.lasToolFileListExt,'.las')
        lastoolsCommandLasZipDict = {"lof": fileListLas,"odir":self.LidarInternalLazDir,"olaz":"","cores":str(self.lidarCores)}
        executeLastoolsCommand('las2las',lastoolsCommandLasZipDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

        fileListExtraLas = makeFileListForLastools(self.LidarInternalLasExtraDir,'ExtraLas'+self.lasToolFileListExt,'.las')
        lastoolsCommandExtraLasZipDict = {"lof": fileListExtraLas,"odir":self.LidarInternalLazExtraDir,"olaz":"","cores":str(self.lidarCores)}
        executeLastoolsCommand('las2las',lastoolsCommandExtraLasZipDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

        lastoolsCommandAdjustGpsTimeDict = {"lof":fileListLas,"odir":self.LidarL1UnclassifiedLasDir,"olas":"","cores":str(self.lidarCores),"adjusted_to_week":""}
        if self.payload.lidarSensorManufacturer == 'riegl':
            lastoolsCommandAdjustGpsTimeDict["set_version"] = "1.4"

        executeLastoolsCommand('las2las',lastoolsCommandAdjustGpsTimeDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

        lastoolsCommandAdjustGpsTimeExtraDict = {"lof":fileListExtraLas,"odir":self.LidarL1UnclassifiedExtraLasDir,"olas":"","cores":str(self.lidarCores),"adjusted_to_week":""}
        if self.payload.lidarSensorManufacturer == 'riegl':
            lastoolsCommandAdjustGpsTimeDict["set_version"] = "1.4"

        executeLastoolsCommand('las2las',lastoolsCommandAdjustGpsTimeExtraDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def runLidarOverlap(self):

        os.makedirs(self.LidarQALasQCDir,exist_ok=True)
        
        # overlap for all files used in L3 processing (not including extra files)
        lastoolsCommandOverlapDict = {"lof": self.startingfileListLas,"o":os.path.join(self.LidarQALasQCDir,"lasoverlap_step"+str(self.overlapGridStep)+".png"),"cores":str(self.lidarCores),"step": str(self.overlapGridStep),"files_are_flightlines" : "","last_only":"","no_diff":""}
        executeLastoolsCommand('lasoverlap',lastoolsCommandOverlapDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

        #overlap for all files, including the extra files
        lastoolsCommandOverlapAllDict = lastoolsCommandOverlapDict
        lastoolsCommandOverlapAllDict["lof"] = self.startingfileListExtraLas
        lastoolsCommandOverlapAllDict["o"] = os.path.join(self.LidarQALasQCDir,"lasoverlap_all_files_step"+str(self.overlapGridStep)+".png")

        executeLastoolsCommand('lasoverlap',lastoolsCommandOverlapAllDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def runLasInfo(self):
        # run lasinfo on all las files
        os.makedirs(self.LidarQALasInfoDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarLastoolsProcessingLasFilteredDir,'Filtered'+self.lasToolFileListExt,'.laz')
        fileListExtra = makeFileListForLastools(self.LidarLastoolsProcessingLasFilteredExtraDir,'FilteredExtra'+self.lasToolFileListExt,'.laz')

        lastoolsCommandLasInfoDict = {"lof": fileList,"odir":self.LidarQALasInfoDir,"cores":str(self.lidarCores),"compute_density" : "","otxt":""}
        executeLastoolsCommand('lasinfo',lastoolsCommandLasInfoDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

        lastoolsCommandLasInfoDict["lof"] = fileListExtra        
        executeLastoolsCommand('lasinfo',lastoolsCommandLasInfoDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)
        
        #run lasinfo on all las files, after dropping noise class

        lastoolsCommandLasInfoDict["lof"] = fileList      

        lastoolsCommandLasInfoDict["drop_class"] = self.lidarNoiseClass
        lastoolsCommandLasInfoDict["odix"] = '_denoise'

        executeLastoolsCommand('lasinfo',lastoolsCommandLasInfoDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)
        
        lastoolsCommandLasInfoDict["lof"] = fileListExtra
        
        executeLastoolsCommand('lasinfo',lastoolsCommandLasInfoDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def runLasBoundary(self):

        os.makedirs(self.LidarMetadataKmlDir,exist_ok=True)

        lastoolsCommandBoundaryDict = {"lof": self.startingfileListLas,"o":os.path.join(self.LidarMetadataKmlDir,self.YearSiteVisit+"_full_boundary.kml"),"cores":str(self.lidarCores),
                                       "thin_with_grid": str(self.boundaryThinGridStep),"concavity" : str(self.boundaryConcavity),"last_only":"","merged":"","disjoint":"","holes":""}

        executeLastoolsCommand('lasboundary',lastoolsCommandBoundaryDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

        lastoolsCommandBoundaryDict["lof"] = self.startingfileListExtraLas
        lastoolsCommandBoundaryDict["o"] = os.path.join(self.LidarMetadataKmlDir,self.YearSiteVisit+"_full_boundary_all_files.kml")

        executeLastoolsCommand('lasboundary',lastoolsCommandBoundaryDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

        lastoolsCommandBoundaryDict["o"] = os.path.join(self.LidarMetadataKmlDir,self.YearSiteVisit+"_full_boundary_all_files.shp")
        executeLastoolsCommand('lasboundary',lastoolsCommandBoundaryDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def applyEllipsoidCorrection(self):

        os.makedirs(self.LidarLastoolsProcessingLasReoffsetDir,exist_ok=True)
        
        os.makedirs(self.LidarL1UnclassifiedExtraLasDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarL1UnclassifiedLasDir,'UnclassifiedLaz'+self.lasToolFileListExt,'.las')

        fileListExtra = makeFileListForLastools(self.LidarL1UnclassifiedExtraLasDir,'UnclassifiedExtraLaz'+self.lasToolFileListExt,'.las')

        lastoolsCommandLasReoffsetDict = {"lof": fileList,"odir":self.LidarLastoolsProcessingLasReoffsetDir,"cores":str(self.lidarCores),"odix" : "_reoffset","translate_z":str(self.site.nad83wgs84offset),"olas":"","auto_reoffset":"","olas":""}

        executeLastoolsCommand('las2las',lastoolsCommandLasReoffsetDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

        os.makedirs(self.LidarLastoolsProcessingLasReoffsetExtraDir,exist_ok=True)

        lastoolsCommandLasReoffsetDict = {"lof": fileListExtra,"odir":self.LidarLastoolsProcessingLasReoffsetExtraDir,"cores":str(self.lidarCores),"odix" : "_reoffset","translate_z":str(self.site.nad83wgs84offset),"olas":"","auto_reoffset":"","olas":""}

        executeLastoolsCommand('las2las',lastoolsCommandLasReoffsetDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def zipLasFiles(self):

        os.makedirs(self.LidarL1UnclassifiedLazDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarLastoolsProcessingLasReoffsetDir,'ReoffsetLas'+self.lasToolFileListExt,'.las')

        fileListExtra = makeFileListForLastools(self.LidarLastoolsProcessingLasReoffsetExtraDir,'ReoffsetExtraLas'+self.lasToolFileListExt,'.las')

        lastoolsCommandLasZipDict = {"lof": fileList,"odir":self.LidarL1UnclassifiedLazDir,"cores":str(self.lidarCores),"olaz" : ""}

        executeLastoolsCommand('las2las',lastoolsCommandLasZipDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

        lastoolsCommandLasZipDict["lof"] = fileListExtra

        executeLastoolsCommand('las2las',lastoolsCommandLasZipDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)
    def unzipClassifiedLasTileFiles(self):

        os.makedirs(self.LidarL1UnclassifiedLazDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarL1TileDir,'ReoffsetLas'+self.lasToolFileListExt,'.laz')

        lastoolsCommandLasUnZipDict = {"lof": fileList,"odir":self.LidarL1TileDir,"cores":str(self.lidarCores),"olas" : ""}

        executeLastoolsCommand('las2las',lastoolsCommandLasUnZipDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def runLasNoise(self):

        os.makedirs(self.LidarLastoolsProcessingLasDenoiseDir,exist_ok=True)
        os.makedirs(self.LidarLastoolsProcessingLasDenoiseExtraDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarLastoolsProcessingLasReoffsetDir,'ReoffsetLas'+self.lasToolFileListExt,'.las')
        
        fileListExtra = makeFileListForLastools(self.LidarLastoolsProcessingLasReoffsetExtraDir,'ReoffsetLasExtra'+self.lasToolFileListExt,'.las')
        lastoolsCommandLasNoiseDict = {"lof": fileList,"odir":self.LidarLastoolsProcessingLasDenoiseDir,"cores":str(self.lidarCores),"step":str(self.noiseStep),
                                     "isolated":str(self.noiseIsolated),"odix":'_denoise',"olaz":"","keep_scan_angle":str(self.scanAngleLimit*-1) +' '+str(self.scanAngleLimit)}

        if self.payload.lidarSensorManufacturer == 'optech':
            del lastoolsCommandLasNoiseDict["keep_scan_angle"]

        executeLastoolsCommand('lasnoise',lastoolsCommandLasNoiseDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)
        
        lastoolsCommandLasNoiseDict["lof"] = fileListExtra
        lastoolsCommandLasNoiseDict["odir"] = self.LidarLastoolsProcessingLasDenoiseExtraDir

        executeLastoolsCommand('lasnoise',lastoolsCommandLasNoiseDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def prepareUsgsDem(self):

        os.makedirs(self.LidarLastoolsProcessingUsgsDir,exist_ok=True)

        inUsgsDemFile = os.path.join(self.usgsDemFileDir,self.Domain+'_'+self.site.site+'_dem_clip.tif')

        outReprojectUsgsDem = os.path.join(self.LidarLastoolsProcessingUsgsDir,self.YearSiteVisit+'_'+'usgsReprojectDem.tif')

        outRaster,outMetadata = reprojectTif(inUsgsDemFile,'EPSG:'+str(self.epsgCode))

        with rio.open(outReprojectUsgsDem , 'w', **outMetadata) as dest:
            dest.write(outRaster)

        outClipUsgsDemFile = os.path.join(self.LidarLastoolsProcessingUsgsDir,self.YearSiteVisit+'_'+'usgsClipDem.tif')

        outRaster,outMetadata = clipRasterbyVectorFile(outReprojectUsgsDem,os.path.join(self.LidarMetadataKmlDir,self.YearSiteVisit+"_full_boundary_all_files.shp"),bufferDistance=self.usgsBuffer)
        #outRaster[outRaster==outMetadata["nodata"]] = np.median(outRaster[outRaster!=outMetadata["nodata"]])
        with rio.open(outClipUsgsDemFile, 'w', **outMetadata) as dest:
            dest.write(smoothImage(outRaster, 3))

        tabulatedData = extractcellsToListRemoveNodata(outClipUsgsDemFile)

        outReprojectClipUsgsCsv = os.path.join(self.LidarLastoolsProcessingUsgsDir,self.YearSiteVisit+'_'+self.usgsCsvName)

        with open(outReprojectClipUsgsCsv , 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            csv_writer.writerows(tabulatedData)

        lastoolsCommandLasTxtDict = {"i": outReprojectClipUsgsCsv,"odir":self.LidarLastoolsProcessingUsgsDir,"olaz" : "", "keep_every_nth":str(self.usgsDecimate),
                                     "parse":"xyz","epsg":str(self.epsgCode),"olaz":""}

        executeLastoolsCommand('txt2las',lastoolsCommandLasTxtDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def getUsgsTif(self):


        outReprojectClipUsgsLaz = os.path.join(self.LidarLastoolsProcessingUsgsDir,self.YearSiteVisit+'_'+self.usgsCsvName).replace('.csv','.laz')

        lastoolsCommandLasUsgsDict = {"i": outReprojectClipUsgsLaz,"odir":self.LidarLastoolsProcessingUsgsDir,"step":self.productStepSize,"otif" : "","nodata":str(self.nodata)}

        executeLastoolsCommand('blast2dem',lastoolsCommandLasUsgsDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def runLasNoiseFilterUsgsDem(self):
        
        #USE PYTHON MP HERE, MULTI CORE IN LASTOOLS NOT WORKING

        os.makedirs(self.LidarLastoolsProcessingLasFilteredDir,exist_ok=True)
        os.makedirs(self.LidarLastoolsProcessingLasFilteredExtraDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarLastoolsProcessingLasDenoiseDir,'DenoiseLaz'+self.lasToolFileListExt,'.laz')
        
        fileListExtra = makeFileListForLastools(self.LidarLastoolsProcessingLasDenoiseExtraDir,'DenoiseLazExtra'+self.lasToolFileListExt,'.laz')

        outReprojectClipUsgsLaz = os.path.join(self.LidarLastoolsProcessingUsgsDir,self.YearSiteVisit+'_'+self.usgsCsvName).replace('.csv','.laz')

        lastoolsCommandLasHeightDict = {"lof": fileList,"odir":self.LidarLastoolsProcessingLasFilteredDir,"cores":str(self.lidarCores),"olaz" : "", "odix":"_filtered",
                                     "ground_points":outReprojectClipUsgsLaz,"classify_above":str(self.usgsFilterUpperLimit)+' '+str(self.lidarNoiseClass),"classify_below":str(self.usgsFilterLowerLimit)+' '+str(self.lidarNoiseClass)}

        executeLastoolsCommand('lasheight64',lastoolsCommandLasHeightDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)
        
        lastoolsCommandLasHeightDict["lof"] = fileListExtra
        lastoolsCommandLasHeightDict["odir"] = self.LidarLastoolsProcessingLasFilteredExtraDir

        executeLastoolsCommand('lasheight64',lastoolsCommandLasHeightDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def tileLaz(self,reTile=False):
        
    
        if reTile:
            
            os.makedirs(self.LidarLastoolsProcessingTempTilesGroundDir,exist_ok=True)
            
            fileList = makeFileListForLastools(self.LidarL1ClassifiedLasDir,'ClassifiedLaz'+self.lasToolFileListExt,'.laz')
            
        else:

            os.makedirs(self.LidarLastoolsProcessingTempTilesAllPointsDir,exist_ok=True)

            fileList = makeFileListForLastools(self.LidarLastoolsProcessingLasFilteredDir,'FilteredLas'+self.lasToolFileListExt,'.laz')

        lastoolsCommandTileDict = {"lof": fileList,"odir":self.LidarLastoolsProcessingTempTilesAllPointsDir,"tile_size": str(self.lidarProcessingTileSize),
                                        "buffer":str(self.tileBuffer),"files_are_flightlines":"","extra_pass":"","o":self.YearSiteVisit,"olaz":""}
        
        if reTile:
            
            del lastoolsCommandTileDict['files_are_flightlines']
            
            lastoolsCommandTileDict["odir"] = self.LidarLastoolsProcessingTempTilesGroundDir
            

        executeLastoolsCommand('lastile64',lastoolsCommandTileDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def dropNoisePointsFromTiles(self):

        os.makedirs(self.LidarLastoolsProcessingTempTilesDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarLastoolsProcessingTempTilesAllPointsDir,'TempTilesAll'+self.lasToolFileListExt,'.laz')

        lastoolsCommandDropNoiseDict = {"lof":fileList,"odir":self.LidarLastoolsProcessingTempTilesDir,"cores":str(self.lidarCores),"drop_class": str(self.lidarNoiseClass),"olaz":""}

        executeLastoolsCommand('las2las',lastoolsCommandDropNoiseDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def keepNoisePointsFromTiles(self):

        os.makedirs(self.LidarLastoolsProcessingTempTilesNoisePointsDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarLastoolsProcessingTempTilesAllPointsDir,'TempTilesAll'+self.lasToolFileListExt,'.laz')

        lastoolsCommandKeepNoiseDict = {"lof":fileList,"odir":self.LidarLastoolsProcessingTempTilesNoisePointsDir,"keep_class": str(self.lidarNoiseClass),"odix":"_only_noise","olaz":"","remove_buffer":""}

        executeLastoolsCommand('lastile',lastoolsCommandKeepNoiseDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def sortTiles(self):

        os.makedirs(self.LidarLastoolsProcessingTempTilesSortPointsDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarLastoolsProcessingTempTilesDir,'TempTiles'+self.lasToolFileListExt,'.laz')

        lastoolsCommandSortDict = {"lof":fileList,"odir":self.LidarLastoolsProcessingTempTilesSortPointsDir,"cores":str(self.lidarCores),"olaz":""}

        executeLastoolsCommand('lassort',lastoolsCommandSortDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def runGroundClassification(self):

        os.makedirs(self.LidarLastoolsProcessingTempTilesGroundDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarLastoolsProcessingTempTilesSortPointsDir,'TempTilesSort'+self.lasToolFileListExt,'.laz')

        lastoolsCommandGroundDict = {"lof":fileList,"odir":self.LidarLastoolsProcessingTempTilesGroundDir,"cores":str(self.lidarCores),"step":str(self.lasGroundGridStep),
                                   "offset":str(self.lasGroundOffset),"compute_height":"",self.lasGroundSearchParameter:"","olaz":""}

        executeLastoolsCommand('lasground',lastoolsCommandGroundDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def runOverage(self):

        os.makedirs(self.LidarInternalLasOverageDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarLastoolsProcessingTempTilesSortPointsDir,'TempTilesSort'+self.lasToolFileListExt,'.laz')

        lastoolsCommandOverageDict = {"lof":fileList,"odir":self.LidarInternalLasOverageDir,"step":str(self.productStepSize),"odix":"_overage",
                                   "olaz":"","remove_overage":""}

        executeLastoolsCommand('lasoverage',lastoolsCommandOverageDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def getIntensityRaster(self):

        os.makedirs(self.LidarInternalIntensityImageDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarInternalLasOverageDir,'OverageLaz'+self.lasToolFileListExt,'.laz')

        lastoolsCommandOverageDict = {"lof":fileList,"odir":self.LidarInternalIntensityImageDir,"cores":str(self.lidarCores),"step":str(self.productStepSize),"odix":"_intensity",
                                   "kill":str(self.intensityKillTriangle),"nodata":str(self.nodata),"otif":"","intensity":"","use_tile_bb":""}

        executeLastoolsCommand('las2dem',lastoolsCommandOverageDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def runOverlapAnalysis(self):

        os.makedirs(self.LidarQALasQCDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarLastoolsProcessingTempTilesGroundDir,'TempTilesGround'+self.lasToolFileListExt,'.laz')

        lastoolsCommandOverlapDict = {"lof":fileList,"odir":self.LidarQALasQCDir,"step":str(self.overlapGridStep),"o":self.lasoverlapOutputFilename,"odir":self.LidarQALasQCDir,
                                   "keep_class":str(self.groundClassId),"use_bb":"","merged":"","no_over":"","values":""}

        executeLastoolsCommand('lasoverlap',lastoolsCommandOverlapDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def getOverlapPlots(self):

        gtifPath = os.path.join(self.LidarQALasQCDir,self.lasoverlapOutputFilename.replace('.tif','_diff.tif'))

        mapPlotOutFile = os.path.splitext(gtifPath)[0]+".png"

        plot_geotiff(gtifPath, title='Elevation difference for '+self.YearSiteVisit, cmap='viridis', save_path=mapPlotOutFile)

        with rio.open(gtifPath) as src:
            # Read the raster data
            raster_data = src.read(1, masked=True)  # Use masked=True to handle nodata values

        outPath = os.path.join(self.LidarQALasQCDir,self.lasoverlapHistogramName)

        plot_histogram(raster_data, variable_name='Elevation Difference', abs_value=False, bins='fd', cumulative=False, title=None, xlabel='Elevation Difference (m)', save_path=outPath)

        outPath = os.path.join(self.LidarQALasQCDir,self.lasoverlapAbsHistogramName)

        plot_histogram(raster_data, variable_name='Elevation Difference', abs_value=True, bins='fd', cumulative=False, title=None, xlabel='Elevation Difference (m)', save_path=outPath)

        outPath = os.path.join(self.LidarQALasQCDir,self.lasoverlapCumulativeHistogramName)

        plot_histogram(raster_data, variable_name='Elevation Difference', abs_value=True, bins='fd', cumulative=True, title=None, xlabel='Elevation Difference (m)', save_path=outPath)

    def getDtmRasters(self):

        os.makedirs(self.LidarLastoolsProcessingDtmBbGtifDir,exist_ok=True)

        os.makedirs(self.LidarLastoolsProcessingDtmGtifDir,exist_ok=True)

        os.makedirs(self.LidarL3DtmDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarLastoolsProcessingTempTilesGroundDir,'TempTilesGround'+self.lasToolFileListExt,'.laz')

        lastoolsCommandDtmDict = {"lof":fileList,"odir":self.LidarLastoolsProcessingDtmBbGtifDir,"cores":str(self.lidarCores),"step":str(self.productStepSize),"kill":str(self.killTriangles),"odix":"_DTM_w_bb",
                                   "keep_class":str(self.groundClassId),"elevation":"","otif":""}

        executeLastoolsCommand('las2dem',lastoolsCommandDtmDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

        lastoolsCommandDtmDict["use_tile_bb"] = ""
        lastoolsCommandDtmDict["odix"] = "_DTM"
        lastoolsCommandDtmDict["odir"] = self.LidarLastoolsProcessingDtmGtifDir

        executeLastoolsCommand('las2dem',lastoolsCommandDtmDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def runPointCloudClassification(self):

        os.makedirs(self.LidarLastoolsProcessingTempTilesClassifiedNoNoiseDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarLastoolsProcessingTempTilesGroundDir,'TempTilesGround'+self.lasToolFileListExt,'.laz')

        lastoolsCommandClassifyDict = {"lof":fileList,"odir":self.LidarLastoolsProcessingTempTilesClassifiedNoNoiseDir,"cores":str(self.lidarCores),"planar":str(self.classifyPlanar),"olaz":""}

        executeLastoolsCommand('lasclassify',lastoolsCommandClassifyDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def getDsmRasters(self):

        os.makedirs(self.LidarLastoolsProcessingDsmSmallTileGtifDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarLastoolsProcessingTempTilesClassifiedNoNoiseDir,'TempTilesClassifiedNoNoise'+self.lasToolFileListExt,'.laz')
        
        lastoolsCommandDtmDict = {"lof":fileList,"odir":self.LidarLastoolsProcessingDsmSmallTileGtifDir,"cores":str(self.lidarCores),"step":str(self.productStepSize),"kill":str(self.killTriangles),"odix":"_DSM",
                                   "keep_class":str(self.unclassifiedClassId)+' '+str(self.groundClassId)+' '+str(self.highVegetationClassId)+' '+str(self.buildingClassId),"elevation":"","otif":"","use_tile_bb":"","first_only":""}

        executeLastoolsCommand('las2dem',lastoolsCommandDtmDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def combineDtmAndDsm(self):

        self.getProductFiles()

        os.makedirs(self.LidarL3DsmDir,exist_ok=True)


        for dsmFile in self.LidarLastoolsProcessingDsmGtifFiles:

            dtmFile = os.path.join(self.LidarL3DtmDir,os.path.basename(dsmFile).replace('DSM','DTM'))

            if os.path.isfile(dtmFile):

                maxRaster, metadata = getMaxTwoRasters(dsmFile,dtmFile)

                with rio.open(os.path.join(self.LidarL3DsmDir,os.path.basename(dsmFile)), 'w', **metadata) as dst:
                # Write the combined data to the output file
                    dst.write(maxRaster, 1)

    def createAigDsm(self):

        self.getProductFiles()

        inReprojectClipUsgsTif = os.path.join(self.LidarLastoolsProcessingUsgsDir,self.YearSiteVisit+'_'+self.usgsCsvName).replace('.csv','.tif')

        os.makedirs(self.LidarInternalAigDsmDir,exist_ok=True)

        mosaicAigDsmFile = os.path.join(self.LidarInternalAigDsmDir,self.YearSiteVisit+'_dem_vf_bf')

        filesToMosaic = self.LidarL3DsmFiles
        filesToMosaic.append(inReprojectClipUsgsTif)
        filesToMosaic.reverse()

        Mosaic,MosaicExtents,EPSGCode,Nodata,RasterCellWidth,RasterCellHeight = getMosaicAndExtents(filesToMosaic,'last_in')

        Mosaic[Mosaic==Nodata] = np.median(Mosaic[Mosaic!=Nodata])

        enviProjCs = parseEnviProjCs(self.filePathToEnviProjCs,int(self.epsgCode))

        metadata = {
        "geotransform": (MosaicExtents[0], RasterCellWidth,-0.0,MosaicExtents[3],0.0,-1*RasterCellHeight),
        "projection": enviProjCs,
        "data_type": int(6),
        "data_ignore_value": float(Nodata),
        "num_bands": int(1),
        "width": int(Mosaic.shape[1]),
        "height": int(Mosaic.shape[0])
        # Add more metadata fields as needed
        }

        writeEnviRaster(mosaicAigDsmFile,Mosaic,metadata)

    def runLidarPointDensity(self):

        os.makedirs(self.LidarQAPointDensityAllDir,exist_ok=True)

        os.makedirs(self.LidarQAPointDensityGroundDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarL1TileDir,'L1Tiles'+self.lasToolFileListExt,'.laz')

        lastoolsCommandPointDensityAllDict = {"lof": fileList,"odir":self.LidarQAPointDensityAllDir, "odix": "_point_density_all", "step": str(self.RasterCellSize),"point_density_32bit" : "","otif":"","cores":str(self.lidarCores)}

        lastoolsCommandPointDensityGroundDict = {"lof": fileList,"odir":self.LidarQAPointDensityGroundDir, "odix": "_point_density_ground", "step": str(self.RasterCellSize),"point_density_32bit" : "","otif":"","keep_class":str(self.groundClassId),"cores":str(self.lidarCores)}

        executeLastoolsCommand('lasgrid',lastoolsCommandPointDensityAllDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

        executeLastoolsCommand('lasgrid',lastoolsCommandPointDensityGroundDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def runLidarTriangularEdges(self):

        os.makedirs(self.LidarQALongestEdgeGroundDir,exist_ok=True)
        os.makedirs(self.LidarQALongestEdgeAllDir,exist_ok=True)
        os.makedirs(self.LidarQAShortestEdgeGroundDir,exist_ok=True)
        os.makedirs(self.LidarQAShortestEdgeAllDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarLastoolsProcessingTempTilesGroundDir,'TempTilesGround'+self.lasToolFileListExt,'.laz')

        lastoolsCommandTriangleEdgeAll = {"lof": fileList,"odir":self.LidarQALongestEdgeAllDir,"kill":str(self.killTriangles),"step":str(self.RasterCellSize),"odix": "_edge_longest_all_points","cores":str(self.lidarCores),"otif": "","edge_longest":"","use_tile_bb":"","extra_pass":""}
        executeLastoolsCommand('las2dem',lastoolsCommandTriangleEdgeAll,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

        del lastoolsCommandTriangleEdgeAll["edge_longest"]
        lastoolsCommandTriangleEdgeAll["edge_shortest"] = ""
        lastoolsCommandTriangleEdgeAll["odir"] = self.LidarQAShortestEdgeAllDir
        lastoolsCommandTriangleEdgeAll["odix"] = "_edge_shortest_all_points"
        executeLastoolsCommand('las2dem',lastoolsCommandTriangleEdgeAll,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

        lastoolsCommandTriangleEdgeGround = {"lof": fileList,"odir":self.LidarQALongestEdgeGroundDir,"kill":str(self.killTriangles),"step":str(self.RasterCellSize),"odix": "_edge_longest_ground_points","cores":str(self.lidarCores),"keep_class":str(self.groundClassId),"otif": "","edge_longest":"","use_tile_bb":"","extra_pass":""}
        executeLastoolsCommand('las2dem',lastoolsCommandTriangleEdgeGround,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

        del lastoolsCommandTriangleEdgeGround["edge_longest"]
        lastoolsCommandTriangleEdgeGround["edge_shortest"] = ""
        lastoolsCommandTriangleEdgeGround["odir"] = self.LidarQAShortestEdgeGroundDir
        lastoolsCommandTriangleEdgeGround["odix"] = "_edge_shortest_ground_points"
        executeLastoolsCommand('las2dem',lastoolsCommandTriangleEdgeGround,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def createLidarFlightlineBoundaries(self):

        filelist = makeFileListForLastools(self.LidarL1UnclassifiedLazDir,'L1UnclassifiedFlightlines'+self.lasToolFileListExt,'.laz')

        os.makedirs(self.LidarMetadataKmlDir,exist_ok=True)

        lastoolsCommandBoundaryDict = {"lof": filelist,"odir":self.LidarMetadataKmlDir,"odix":"_boundary","thin_with_grid":"5",
                                    "concavity":"10","cores":str(self.lidarCores),"last_only":"","disjoint":"","holes":"","okml":""}

        executeLastoolsCommand('lasboundary',lastoolsCommandBoundaryDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def createLidarTileBoundaries(self):

        os.makedirs(self.LidarMetadataTileDir,exist_ok=True)
        os.makedirs(self.LidarMetadataTileKmlDir,exist_ok=True)
        os.makedirs(self.LidarMetadataTileShpDir,exist_ok=True)

        filelist = makeFileListForLastools(self.LidarL1TileDir,'L1Tiles'+self.lasToolFileListExt,'.laz')

        lastoolsCommandBoundaryDict = {"lof": filelist,"odir":self.LidarMetadataTileKmlDir,"cores":str(self.lidarCores),"okml":""}

        executeLastoolsCommand('lasboundary',lastoolsCommandBoundaryDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

        lastoolsCommandBoundaryDict = {"lof": filelist,"odir":self.LidarMetadataTileShpDir,"cores":str(self.lidarCores),"epsg":str(self.epsgCode),"oshp":""}

        executeLastoolsCommand('lasboundary',lastoolsCommandBoundaryDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def mergeLidarTileBoundaries(self):

        shapefile_merge(self.LidarMetadataTileShpDir,self.YearSiteVisit)

    def getLidarAscii(self):

        filelistAllLazLines  = makeFileListForLastools(self.LidarL1UnclassifiedLazDir,'L1UnclassifiedLaz'+self.lasToolFileListExt,'.laz')

        os.makedirs(self.LidarLastoolsProcessingAsciiDir,exist_ok=True)

        lastoolsCommandAsciiDict = {"lof": filelistAllLazLines,"odir":self.LidarLastoolsProcessingAsciiDir,"parse": "xyztia","sep" : "comma","cores":str(self.lidarCores)}

        executeLastoolsCommand('las2txt',lastoolsCommandAsciiDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def calcLidarUncertainty(self):

        self.getProductFiles()
     
        for mission in self.missions:
            mission.generateLidarUncertainty()

    def zipUncertaintyLasFiles(self):

        os.makedirs(self.LidarMetadataUncertaintyLazDir,exist_ok=True)

        fileList = makeFileListForLastools(self.lidarProcessingLastoolsUncertaintyDir,'UncertaintyLas'+self.lasToolFileListExt,'.las')

        lastoolsCommandLasZipDict = {"lof": fileList,"odir":self.LidarMetadataUncertaintyLazDir,"cores":str(self.lidarCores),"olaz" : ""}

        executeLastoolsCommand('las2las',lastoolsCommandLasZipDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def calcUncertainty(self):

        self.getProductFiles()

        with Pool(processes=15) as pool:
            uncertaintyFunc = partial(calculateUncertainty, outputDir=self.LidarQaUncertaintyCsvDir,campaign=self.payload.payloadId,sensor=self.payload.lidarSensorManufacturer+self.payload.lidarSensorModel,year=self.year,utmZone = self.utmZone)
            pool.map(uncertaintyFunc,self.LidarProcessingAsciiFiles)

    def convertUncertaintyCsvToLaz(self):

        filelistAllCsvLines  = makeFileListForLastools(self.LidarQaUncertaintyCsvDir,'UncertaintyCsv'+self.lasToolFileListExt,'.csv')

        os.makedirs(self.LidarMetadataUncertaintyLazDir,exist_ok=True)

        lastoolsCommandHorzUncerDict = {"lof": filelistAllCsvLines,"odir":self.LidarMetadataUncertaintyLazDir,"parse": "xysz","set_scale" : "0.001 0.001 0.001","olas":"","odix":"_horizontal_uncertainty","utm":str(self.utmZone)+'N',"cores":str(self.lidarCores)}
        lastoolsCommandVertUncerDict = {"lof": filelistAllCsvLines,"odir":self.LidarMetadataUncertaintyLazDir,"parse": "xyz","set_scale" : "0.001 0.001 0.001","olas":"","odix":"_vertical_uncertainty","utm":str(self.utmZone)+'N',"cores":str(self.lidarCores)}

        executeLastoolsCommand('txt2las',lastoolsCommandHorzUncerDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)
        executeLastoolsCommand('txt2las',lastoolsCommandVertUncerDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def tileUncertaintyLaz(self):

        os.makedirs(self.LidarQAUncertaintyTilesDir,exist_ok=True)
        
        filelistAllHorzUncerLines  = makeFileListForLastools(self.LidarMetadataUncertaintyLazDir,'HorizontalUncertaintyLaz'+self.lasToolFileListExt,'horizontal_uncertainty.laz')
        filelistAllVertUncerLines  = makeFileListForLastools(self.LidarMetadataUncertaintyLazDir,'VericalUncertaintyLaz'+self.lasToolFileListExt,'vertical_uncertainty.laz')

        lastoolsCommandUncerDict = {"lof": filelistAllHorzUncerLines,"odir":self.LidarQAUncertaintyTilesDir,"tile_size": str(self.tileSize),"files_are_flightlines":"","extra_pass":"","odix":"_horizontal_uncertainty","o":self.YearSiteVisit,"olaz":""}
   
        executeLastoolsCommand('lastile64',lastoolsCommandUncerDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

        lastoolsCommandUncerDict["lof"] = filelistAllVertUncerLines
        lastoolsCommandUncerDict["odix"] = "_vertical_uncertainty"

        executeLastoolsCommand('lastile64',lastoolsCommandUncerDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def createUncertaintyRaster(self):

        os.makedirs(self.LidarQAHorzUncertaintyDir,exist_ok=True)
        os.makedirs(self.LidarQAVertUncertaintyDir,exist_ok=True)

        filelistAllHorzUncertaintyLazTiles  = makeFileListForLastools(self.LidarQAUncertaintyTilesDir,'HorizontalUncertaintyLazTiles'+self.lasToolFileListExt,'horizontal_uncertainty.laz')
        filelistAllVertUncertaintyLazTiles  = makeFileListForLastools(self.LidarQAUncertaintyTilesDir,'VerticalUncertaintyLazTiles'+self.lasToolFileListExt,'vertical_uncertainty.laz')

        lastoolsCommandHorzUncertaintyDict = {"lof": filelistAllHorzUncertaintyLazTiles,"odir":self.LidarQAHorzUncertaintyDir,"step": str(self.RasterCellSize),"highest" : "","otif":"","cores":str(self.lidarCores)}
        lastoolsCommandVertUncertaintyDict = {"lof": filelistAllVertUncertaintyLazTiles,"odir":self.LidarQAVertUncertaintyDir,"step": str(self.RasterCellSize),"highest" : "","otif":"","cores":str(self.lidarCores)}

        executeLastoolsCommand('lasgrid',lastoolsCommandHorzUncertaintyDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)
        executeLastoolsCommand('lasgrid',lastoolsCommandVertUncertaintyDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def extractHeightChm(self):

        os.makedirs(self.LidarInternalChmStatsDir,exist_ok=True)

        self.getProductFiles()

        extract_height_chm(self.LidarInternalPitFreeChmDir,os.path.basename(self.LidarInternalLidarChmBilFile),self.LidarInternalChmStatsDir,0.1,self.payload.lidarSensorManufacturer+self.payload.lidarSensorModel)

        self.getProductFiles()

        with open(self.LidarInternalLidarChmIntervalFile, "r") as finInputLevelFile:
            intervals = finInputLevelFile.read()
            self.intervalsList = intervals.split('\n')

    def normalizeHeights(self):

        fileList = makeFileListForLastools(self.LidarLastoolsProcessingTempTilesClassifiedNoNoiseDir,'TempTileNoNoise'+self.lasToolFileListExt,'.laz')

        os.makedirs(self.LidarLastoolsProcessingTempTilesNormHeightDir, exist_ok=True)

        lastoolsCommandNormHeightDict = {"lof": fileList,"odir":self.LidarLastoolsProcessingTempTilesNormHeightDir,"replace_z": "","olaz" : "","keep_class":"1 2 5","cores":str(self.lidarCores)}

        executeLastoolsCommand('lasheight',lastoolsCommandNormHeightDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def gridStandardChm(self):

        fileListTiles = makeFileListForLastools(self.LidarLastoolsProcessingTempTilesNormHeightNoBufferDir,'TempTilesNoNoiseNoBuffer'+self.lasToolFileListExt,'.laz')

        os.makedirs(self.LidarInternalPitFreeChmDir, exist_ok=True)

        lastoolsCommandGridStandardChmDict = {"lof": fileListTiles,"odir":self.LidarInternalPitFreeChmDir,"step": str(self.RasterCellSize),"elevation" : "","merged":"",
                                               "highest":"","obil":"","o":self.YearSiteVisit+"_CHM_non_pit_free.bil","nodata":str(self.nodata),"drop_z_below":"0"}

        executeLastoolsCommand('lasgrid',lastoolsCommandGridStandardChmDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

        lastoolsCommandGridStandardChmDict["o"] = self.YearSiteVisit+"_CHM_non_pit_free.tif"
        del lastoolsCommandGridStandardChmDict["obil"]
        lastoolsCommandGridStandardChmDict["otif"] = ""

        executeLastoolsCommand('lasgrid',lastoolsCommandGridStandardChmDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def removeTileBufferNormalizeTiles(self):

        os.makedirs(self.LidarLastoolsProcessingTempTilesNormHeightNoBufferDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarLastoolsProcessingTempTilesNormHeightDir,'TempTilesNormHeight'+self.lasToolFileListExt,'.laz')

        lastoolsCommandUnTileDict = {"lof": fileList,"odir":self.LidarLastoolsProcessingTempTilesNormHeightNoBufferDir,"cores":str(self.lidarCores),"remove_buffer": "","olaz":""}

        executeLastoolsCommand('lastile',lastoolsCommandUnTileDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def gridStandardChmGrnd(self):

        fileListTiles = makeFileListForLastools(self.LidarLastoolsProcessingTempTilesNormHeightNoBufferDir,'TempTilesNormHeightNoBuffer'+self.lasToolFileListExt,'.laz')

        os.makedirs(self.LidarLastoolsProcessingChmProcessingGrndDir, exist_ok=True)

        lastoolsCommandGridStandardChmnDict = {"lof": fileListTiles,"odir":self.LidarLastoolsProcessingChmProcessingGrndDir,"step": str(self.RasterCellSize),"elevation" : "","use_tile_bb":"","kill":"250",
                                               "drop_z_above":"0.1","otif":"","odix":"_grnd_CHM","keep_class":"2","keep_last":"","cores":str(self.lidarCores)}

        executeLastoolsCommand('las2dem',lastoolsCommandGridStandardChmnDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def thinLazFiles(self):

        fileListToThin = makeFileListForLastools(self.LidarLastoolsProcessingTempTilesNormHeightDir,'TempTilesNormHeight'+self.lasToolFileListExt,'.laz')

        os.makedirs(self.LidarLastoolsProcessingTempTilesNormHeightThinDir, exist_ok=True)

        lastoolsCommandThinDict = {"lof": fileListToThin,"odir":self.LidarLastoolsProcessingTempTilesNormHeightThinDir,"step": str(self.thinStep),"subcircle" : str(self.thinSubcircle),"highest":"","olaz":"",
                                               "odix":"_thin","cores":str(self.lidarCores)}

        executeLastoolsCommand('lasthin',lastoolsCommandThinDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def createStandardChm(self):

        fileListTempTilesNormHeight = makeFileListForLastools(self.LidarLastoolsProcessingTempTilesNormHeightDir,'TempTilesNormHeight'+self.lasToolFileListExt,'.laz')

        os.makedirs(self.LidarLastoolsProcessingChmProcessingStandardDir, exist_ok=True)

        lastoolsCommandGridStandardChmnDict = {"lof": fileListTempTilesNormHeight,"odir":self.LidarLastoolsProcessingChmProcessingStandardDir,"step": str(self.RasterCellSize),"nodata":str(self.nodata),"elevation":"","otif":"",
                                               "odix":"_standard_CHM","highest":"","use_tile_bb":"","drop_z_below":"0","cores":str(self.lidarCores)}

        executeLastoolsCommand('lasgrid',lastoolsCommandGridStandardChmnDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def createChmLevels(self):

        fileListThinNormHeight = makeFileListForLastools(self.LidarLastoolsProcessingTempTilesNormHeightThinDir,'TempTilesNormHeightThis'+self.lasToolFileListExt,'.laz')


        lastoolsCommandGridChmLevelsDict = {"lof": fileListThinNormHeight,"odir":self.LidarLastoolsProcessingChmProcessingDir,"step": str(self.RasterCellSize),"nodata":str(self.nodata),"elevation":"","otif":"",
                                               "odix":"_CHM","kill":str(self.chmKillTriangles),"use_tile_bb":"","cores":str(self.lidarCores),"drop_z_below":"0"}

        for interval in self.intervalsList:

            outFolder = os.path.join(self.LidarLastoolsProcessingChmProcessingDir,'Below_'+interval)
            os.makedirs(outFolder, exist_ok=True)

            lastoolsCommandGridChmLevelsDict["odir"] = outFolder
            lastoolsCommandGridChmLevelsDict["odix"] = '_'+str(interval)+'_CHM'
            lastoolsCommandGridChmLevelsDict["drop_z_below"] = str(interval)

            executeLastoolsCommand('las2dem',lastoolsCommandGridChmLevelsDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def reTileChm(self):

        self.getProductFiles()
        os.makedirs(self.LidarL3ChmDir, exist_ok=True)
        tiles, tileExtents = retileTiles(self.LidarLastoolsProcessingChmProcessingCombinedFiles,self.tileSize)

        for tile, tileExtent in zip(tiles,tileExtents):

            outFile = os.path.join(self.LidarL3ChmDir,self.l3ProductBaseName+str(tileExtent['ext_dict']['xMin'])+'_'+str(tileExtent['ext_dict']['yMin'])+'_'+'CHM'+'.tif')
            MosaicExtents = np.array((tileExtent['ext_dict']['xMin'],tileExtent['ext_dict']['yMin'],tileExtent['ext_dict']['xMax'],tileExtent['ext_dict']['yMax']))
            writeRasterToTif(outFile,tile,MosaicExtents,self.epsgCode,self.nodata,self.RasterCellSize,self.RasterCellSize)

    def reTileDtm(self):

        self.getProductFiles()

        os.makedirs(self.LidarL3DtmDir, exist_ok=True)

        tiles, tileExtents = retileTiles(self.LidarLastoolsProcessingDtmGtifFiles,self.tileSize)

        for tile, tileExtent in zip(tiles,tileExtents):

            outFile = os.path.join(self.LidarL3DtmDir,self.l3ProductBaseName+str(tileExtent['ext_dict']['xMin'])+'_'+str(tileExtent['ext_dict']['yMin'])+'_'+'DTM'+'.tif')
            MosaicExtents = np.array((tileExtent['ext_dict']['xMin'],tileExtent['ext_dict']['yMin'],tileExtent['ext_dict']['xMax'],tileExtent['ext_dict']['yMax']))
            writeRasterToTif(outFile,tile,MosaicExtents,self.epsgCode,self.nodata,self.RasterCellSize,self.RasterCellSize)

    def reTileDsm(self):

        self.getProductFiles()

        os.makedirs(self.LidarLastoolsProcessingDsmGtifDir, exist_ok=True)

        tiles, tileExtents = retileTiles(self.LidarLastoolsProcessingDsmSmallTileGtifFiles,self.tileSize)

        for tile, tileExtent in zip(tiles,tileExtents):

            outFile = os.path.join(self.LidarLastoolsProcessingDsmGtifDir,self.l3ProductBaseName+str(tileExtent['ext_dict']['xMin'])+'_'+str(tileExtent['ext_dict']['yMin'])+'_'+'DSM'+'.tif')
            MosaicExtents = np.array((tileExtent['ext_dict']['xMin'],tileExtent['ext_dict']['yMin'],tileExtent['ext_dict']['xMax'],tileExtent['ext_dict']['yMax']))
            writeRasterToTif(outFile,tile,MosaicExtents,self.epsgCode,self.nodata,self.RasterCellSize,self.RasterCellSize)

    def combineChmLevels(self):

        os.makedirs(self.LidarLastoolsProcessingChmProcessingCombinedDir,exist_ok = True)

        self.getProductFiles()

        stackedChmArrays,metaDataFileList = combineChms(self.LidarProcessingChmLevelFiles)

        for stackedChmArray, metaDataFileName in zip(stackedChmArrays,metaDataFileList):

            metaDataFileNameSplit = os.path.basename(metaDataFileName).split('_')

            outFile = os.path.join(self.LidarLastoolsProcessingChmProcessingCombinedDir,self.l3ProductBaseName+metaDataFileNameSplit[3]+'_'+metaDataFileNameSplit[4]+'_CHM.tif')

            writeRasterToTifTransferMetadata(stackedChmArray,outFile,metaDataFileName)

    def calcAigDsmSlopeAspect(self):

        outputTifFile = self.internalAigDsmFile+'.tif'

        outputSmoothedTifFile = self.internalAigDsmFile+'_smooth.tif'

        convertEnviToGeotiff(self.internalAigDsmFile+'.hdr', outputTifFile)

        ul_x, lr_y, lr_x, ul_y, nodata,dem = getCurrentFileExtentsAndData(outputTifFile,Band=1)

        smoothedImage = smoothImage(dem, 3)

        writeRasterToTifTransferMetadata(smoothedImage,outputSmoothedTifFile,outputTifFile)

        outSlopeFile = outputTifFile.replace('dem','slope')

        outAspectFile = outputTifFile.replace('dem','aspect')

        calcSlopeGdal(outputSmoothedTifFile,outSlopeFile)

        calcAspectGdal(outputSmoothedTifFile,outAspectFile)

        ul_x, lr_y, lr_x, ul_y, nodata,slope = getCurrentFileExtentsAndData(outSlopeFile)
        ul_x, lr_y, lr_x, ul_y, nodata,aspect = getCurrentFileExtentsAndData(outAspectFile)

        slopeAspect = np.moveaxis(np.stack((slope,aspect)),[0,1,2],[2,0,1])

        rasterData,metadata = readEnviRaster(self.internalAigDsmFile,justMetadata = True)

        writeEnviRaster(self.internalAigDsmFile.replace('dem','sap'),slopeAspect,metadata)

        os.remove(outputTifFile)
        os.remove(outSlopeFile)
        os.remove(outAspectFile)

    def removeTileBuffer(self):

        os.makedirs(self.LidarLastoolsProcessingTempTilesClassifiedNoBufferDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarLastoolsProcessingTempTilesClassifiedNoNoiseDir,'TempTilesClassifiedNoNoise'+self.lasToolFileListExt,'.laz')

        lastoolsCommandUnTileDict = {"lof": fileList,"odir":self.LidarLastoolsProcessingTempTilesClassifiedNoBufferDir,"cores":str(self.lidarCores),"remove_buffer": "","olaz":""}

        executeLastoolsCommand('lastile',lastoolsCommandUnTileDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def addNoiseBackToTiles(self):

        self.getProductFiles()
        os.makedirs(self.LidarLastoolsProcessingTempTilesClassifiedIntermediateDir,exist_ok=True)

        for lazFile in self.LidarLastoolsProcessingTempTilesClassifiedNoBufferFiles:
            noiseTileName = os.path.basename(lazFile).replace('.laz','_only_noise.laz')
            lastoolsCommandAddNoiseDict = {"i": lazFile+' '+os.path.join(self.LidarLastoolsProcessingTempTilesNoisePointsDir,noiseTileName),
                                         "odir":self.LidarLastoolsProcessingTempTilesClassifiedIntermediateDir,"o":lazFile,"olaz":""}

            executeLastoolsCommand('lasmerge',lastoolsCommandAddNoiseDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def reTilePointCloudTiles(self):

        os.makedirs(self.LidarL1TileDir,exist_ok=True)

        fileList = makeFileListForLastools(self.LidarLastoolsProcessingTempTilesClassifiedIntermediateDir,'TempTilesClassifiedIntermediate'+self.lasToolFileListExt,'.laz')

        lastoolsCommandRetileDict = {"lof": fileList,"odir":self.LidarL1TileDir,"tile_size":self.tileSize,"o":self.YearSiteVisit,"olaz":""}

        executeLastoolsCommand('lastile64',lastoolsCommandRetileDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def runLasControl(self):

        if self.lidarValidationFile:

            os.makedirs(self.LidarInternalValidationDir,exist_ok=True)

            fileList = makeFileListForLastools(self.LidarL1ClassifiedLasDir,'L1ClassifiedTiles'+self.lasToolFileListExt,'.laz')

            lastoolsCommandControlDict = {"lof": fileList,"cp_out":os.path.join(self.LidarInternalValidationDir,self.YearSiteVisit+'_lidar_validation_lascontrol.txt'),"cp":self.lidarValidationFile[0],"keep_class":str(self.groundClassId)+' '+str(self.modelKeyPointClassId),"parse":"sxyz"}

            executeLastoolsCommand('lascontrol',lastoolsCommandControlDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def getDataForBandingAnalysis(self):

        os.makedirs(self.LidarQABandingDir,exist_ok=True)

        lastoolsCommandControlDict = {"i":self.bandingFile,"odir":self.LidarQABandingDir,"o":self.lidarBandingFile,"parse":"xyzid","sep":"comma"}

        executeLastoolsCommand('las2txt',lastoolsCommandControlDict,outputDir = self.LidarLastoolsProcessingCommandOutputDir)

    def getPlotsForBandingAnalysis(self):

        bandingData = np.loadtxt(os.path.join(self.LidarQABandingDir,self.lidarBandingFile), delimiter=',')

        intensityDirectionOne = bandingData[bandingData[:,4]==0,3]
        intensityDirectionTwo = bandingData[bandingData[:,4]==1,3]

        outFile = os.path.join(self.LidarQABandingDir,os.path.basename(self.bandingFile)+'_hist.png')

        plot_histograms_matplotlib(intensityDirectionOne, intensityDirectionTwo,'Port to starboard', 'Starboard to port','Banding Analysis',save_path=outFile)

    def writeBandingDataTextFile(self):

        bandingData = np.loadtxt(os.path.join(self.LidarQABandingDir,self.lidarBandingFile), delimiter=',')

        intensityDirectionOne = bandingData[bandingData[:,4]==0,3]
        intensityDirectionTwo = bandingData[bandingData[:,4]==1,3]

        outFile = os.path.join(self.LidarQABandingDir,os.path.basename(self.bandingFile)+'_output_stats.txt')

        write_stats_to_file(intensityDirectionOne, intensityDirectionTwo, 'Port to starboard', 'Starboard to port', outFile)

    def smoothBufferedTiles(self):

        self.getProductFiles()
        os.makedirs(self.LidarInternalFilteredDtmDir,exist_ok=True)

        with Pool(processes=25) as pool:
            processFunction = partial(smoothGeotiffFile,outGeotiffDir= self.LidarInternalFilteredDtmDir, windowSize=self.lidarFilterWindowSize)
            pool.map(processFunction,self.LidarLastoolsProcessingDtmBbGtifFiles)

        #for dtmFile in self.LidarLastoolsProcessingDtmBbGtifFiles:

        #    smoothGeotiffFile(dtmFile, os.path.join(self.LidarInternalFilteredDtmDir,os.path.basename(dtmFile).replace('.tif','_filtered.tif')), self.lidarFilterWindowSize)

    def createSlopeLidarProduct(self):

        self.getProductFiles()

        os.makedirs(self.LidarL3SlopeProcessingBbDir,exist_ok=True)

        for dtmFile in self.LidarInternalFilteredDtmFiles:

            calcSlopeGdal(dtmFile,os.path.join(self.LidarL3SlopeProcessingBbDir,os.path.basename(dtmFile).replace('.tif','_slope.tif')))

    def createAspectLidarProduct(self):

        self.getProductFiles()

        os.makedirs(self.LidarL3AspectProcessingBbDir,exist_ok=True)

        for dtmFile in self.LidarInternalFilteredDtmFiles:

            calcAspectGdal(dtmFile,os.path.join(self.LidarL3AspectProcessingBbDir,os.path.basename(dtmFile).replace('.tif','_aspect.tif')))

    def removeSlopeBuffers(self):

        self.getProductFiles()
        
        os.makedirs(self.LidarL3SlopeProcessingDir,exist_ok=True)
        
        batchRemoveBuffers(self.LidarL3SlopeProcessingBbFiles,self.LidarL3SlopeProcessingDir,self.tileBuffer)

    def removeAspectBuffers(self):

        self.getProductFiles()
        
        os.makedirs(self.LidarL3AspectProcessingDir,exist_ok=True)
        
        batchRemoveBuffers(self.LidarL3AspectProcessingBbFiles,self.LidarL3AspectProcessingDir,self.tileBuffer)

    def removeFilterDtmBuffers(self):

        self.getProductFiles()
        
        os.makedirs(self.LidarFilterDtmProcessingDir,exist_ok=True)
  
        batchRemoveBuffers(self.LidarInternalFilteredDtmFiles,self.LidarFilterDtmProcessingDir,self.tileBuffer)
        
    def retileSlope(self):

        self.getProductFiles()

        os.makedirs(self.LidarL3SlopeDir,exist_ok=True)

        tiles, tileExtents = retileTiles(self.LidarL3SlopeProcessingFiles,self.tileSize)

        for tile, tileExtent in zip(tiles,tileExtents):

            outFile = os.path.join(self.LidarL3SlopeDir,self.l3ProductBaseName+str(tileExtent['ext_dict']['xMin'])+'_'+str(tileExtent['ext_dict']['yMin'])+'_'+'Slope'+'.tif')
            MosaicExtents = np.array((tileExtent['ext_dict']['xMin'],tileExtent['ext_dict']['yMin'],tileExtent['ext_dict']['xMax'],tileExtent['ext_dict']['yMax']))
            writeRasterToTif(outFile,tile,MosaicExtents,self.epsgCode,self.nodata,self.RasterCellSize,self.RasterCellSize)

    def retileAspect(self):

        self.getProductFiles()

        os.makedirs(self.LidarL3AspectDir,exist_ok=True)

        tiles, tileExtents = retileTiles(self.LidarL3AspectProcessingFiles,self.tileSize)

        for tile, tileExtent in zip(tiles,tileExtents):

            outFile = os.path.join(self.LidarL3AspectDir,self.l3ProductBaseName+str(tileExtent['ext_dict']['xMin'])+'_'+str(tileExtent['ext_dict']['yMin'])+'_'+'Aspect'+'.tif')
            MosaicExtents = np.array((tileExtent['ext_dict']['xMin'],tileExtent['ext_dict']['yMin'],tileExtent['ext_dict']['xMax'],tileExtent['ext_dict']['yMax']))
            writeRasterToTif(outFile,tile,MosaicExtents,self.epsgCode,self.nodata,self.RasterCellSize,self.RasterCellSize)

    def getLidarQaReportMaps(self):

        self.getProductFiles()

        for mosaicFile in self.LidarQAProcessingReportFiles:

            rasterVariable = os.path.basename(mosaicFile).split('_')[-2]

            plot_geotiff(mosaicFile , title='', cmap=self.mosaicColorMaps[rasterVariable], save_path=mosaicFile.replace('.tif','.png'), nodata_color='black',variable=rasterVariable)

    def getLidarQaReportHistograms(self):

        self.getProductFiles()

        for mosaicFile in self.LidarQAProcessingReportFiles:

            rasterVariable = os.path.basename(mosaicFile).split('_')[-2]

            plotGtifHistogram(mosaicFile,mosaicFile.replace('.tif','_hist.png'),rasterVariable)

    def getLidarDifferencePreviousYearSiteVisit(self):

        self.getProductFiles()

        for mosaicFile in self.LidarQAProcessingReportFiles:

            rasterVariable = os.path.basename(mosaicFile).split('_')[-2]

            foundMatch = False
            for previousMosaicFile in self.previousLidarProcessingReportFiles:

                if rasterVariable.lower() in os.path.basename(previousMosaicFile).replace('_','').lower():
                    foundMatch = True
                    break
            if not foundMatch:
                print("Could not find matching file to "+os.path.basename(mosaicFile))
                continue

            print(rasterVariable)
            print(mosaicFile)
            print(previousMosaicFile)

            differenceRaster,profile = differenceGeotiffFiles(mosaicFile,previousMosaicFile)

            outputFile = os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_'+self.previousYearSiteVisit+'_'+rasterVariable+'_difference.tif')
            with rio.open(outputFile, "w", **profile) as dst:
                dst.write(differenceRaster)

            #plotGtifHistogram(mosaicFile,mosaicFile.replace('.tif','_hist.png'),rasterVariable)

    def getLidarQaReportDifferenceMaps(self):

        self.getProductFiles()

        for mosaicFile in self.LidarQAProcessingReportDifferenceFiles:

            rasterVariable = os.path.basename(mosaicFile).split('_')[-2] + '_Difference'

            # plot_geotiff(mosaicFile , title='', cmap=self.mosaicColorMaps[rasterVariable], save_path=mosaicFile.replace('.tif','.png'), nodata_color='black',variable=rasterVariable)
            plot_geotiff(mosaicFile , title='', cmap="RdBu", save_path=mosaicFile.replace('.tif','.png'), nodata_color='black',variable=rasterVariable)

    def getLidarQaReportDifferenceHistograms(self):

        self.getProductFiles()

        for mosaicFile in self.LidarQAProcessingReportDifferenceFiles:

            rasterVariable = os.path.basename(mosaicFile).split('_')[-2] + '_Difference'

            plotGtifHistogram(mosaicFile,mosaicFile.replace('.tif','_hist.png'),rasterVariable)

    def runL3LidarQA(self):
        # os.makedirs(self.LidarL3QADir,exist_ok=True)
        currentDir = os.getcwd()

        # run lidar_l3_qaqc.py script to generate L3 QA markdown file
        os.chdir(lidarSrcL3Dir)

        os.system(f"python lidar_l3_qaqc.py {self.YearSiteVisit}")

        os.chdir(currentDir)

    def getMosaicQaPngMaps(self):
        
        print('Getting PNG maps for product mosaics for '+self.YearSiteVisit)
        self.getSpectrometerQaReportMaps()
        
        print('Getting reflectance RGB map for '+self.YearSiteVisit)
        self.getRGBPngMap()
        
        print('Getting reflectance NIRGB map for '+self.YearSiteVisit)
        self.getNIRGBPngMap()
        
        print('Getting weather mask map for '+self.YearSiteVisit)
        self.getWaterMaskPngMap()
        
        print('Getting weather png map for '+self.YearSiteVisit)
        self.getWeatherPngMap()
        
        print('Getting DDV png map for '+self.YearSiteVisit)
        self.getDdvPngMap()
        
        print('Getting Acquisition Date png map for '+self.YearSiteVisit)
        self.getAcquisitionDatePngMap()

    def getSpectrometerQaReports(self):

        print('Getting summary files for product mosaics for '+self.YearSiteVisit)
        self.getSpectrometerQaReportSummaryFiles()
        
        print('Getting ancillary raster summary files for product mosaics for '+self.YearSiteVisit)
        self.getSpectrometerAncillaryQaReportSummaryFiles()
        
        if self.previousYearSiteVisit is not None:
        
            print('Getting reflectance difference raster summary files for product mosaics for '+self.YearSiteVisit)
            self.getSpectrometerReflectanceDifferenceSummaryFiles()
      
        print('Getting reflectance RGB raster summary files for product mosaics for '+self.YearSiteVisit)
        self.getSpectrometerRGBQaReportSummaryFiles()    

    def removeLastoolsFolder(self):
        
        if os.path.exists(self.LidarLastoolsProcessingDir):
        
            shutil.rmtree(self.LidarLastoolsProcessingDir)

    def removeLasFileDir(self):
        
        if os.path.exists(self.LidarL1UnclassifiedLasDir):
        
            shutil.rmtree(self.LidarL1UnclassifiedLasDir)

    def cleanInternalLidarDir(self):
        
        tfwFiles = collectFilesInPathIncludingSubfolders(self.LidarInternalIntensityImageDir,'.tfw')
        
        kmlFiles = collectFilesInPathIncludingSubfolders(self.LidarInternalIntensityImageDir,'.kml')
        
        allFiles = tfwFiles+kmlFiles

        for file in allFiles:
            
            os.remove(file)

    def cleanQaLidarDir(self):
        
        self.cleanPointDensityDir()
        
        self.cleanUncertaintyDir()

    def cleanPointDensityDir(self):
        
        tfwFiles = collectFilesInPathIncludingSubfolders(self.LidarQAPointDensityDir,'.tfw')
        
        kmlFiles = collectFilesInPathIncludingSubfolders(self.LidarQAPointDensityDir,'.kml')
        
        allFiles = tfwFiles+kmlFiles

        for file in allFiles:
            
            os.remove(file)

    def cleanUncertaintyDir(self):
        
        tfwFiles = collectFilesInPathIncludingSubfolders(self.LidarQaUncertaintyDir,'.tfw')
        
        kmlFiles = collectFilesInPathIncludingSubfolders(self.LidarQaUncertaintyDir,'.kml')
        
        allFiles = tfwFiles+kmlFiles

        for file in allFiles:
            
            os.remove(file)

    def cleanLasToolsFileList(self):
        
        lasToolsFiles = collectFilesInPathIncludingSubfolders(self.BaseDir,self.lasToolFileListExt)
        
        for file in lasToolsFiles:
            os.remove(file)

    def deleteRecordsFile(self):
        
        recordsFiles = [os.path.join(self.BaseDir,file) for file in os.listdir(self.BaseDir) if file.endswith('record_names.csv')]
        
        for file in recordsFiles:
            os.remove(file)

    def translateWaveformLidar(self):
                        
        os.makedirs(self.PulswavesL1Dir,exist_ok=True)

        pulseTranslateCommandDict = {"i": "","odir":self.PulswavesL1Dir,"translate_z": str(self.site.nad83wgs84offset),"opls":""}
        
        # with Pool(processes=15) as pool:
        #     processFunction = partial(pulsewavesTranslateHelper,pulseTranslateCommandDict=pulseTranslateCommandDict,commandOutputDir=self.LidarLastoolsProcessingCommandOutputDir)
        #     pool.map(processFunction,self.LidarInternalPulsewavesFiles)
        
        for file in self.LidarInternalPulsewavesFiles:
        
            pulseTranslateCommandDict["i"] = file      
            executeLastoolsCommand('pulse2pulse',pulseTranslateCommandDict,outputDir=self.LidarLastoolsProcessingCommandOutputDir)
            
            
        
    def zipWaveformPulsewavesFiles(self):

        os.makedirs(self.PulswavesL1Dir,exist_ok=True)

        pulseZipCommandDict = {"i": "","odir":self.PulswavesL1Dir,"o":"","oplz":""}
        
        # with Pool(processes=15) as pool:
        #     processFunction = partial(zipPulsewavesHelper,pulseZipCommandDict=pulseZipCommandDict,commandOutputDir=self.LidarLastoolsProcessingCommandOutputDir)
        #     pool.map(processFunction,self.LidarL1PulsewavesFiles)
        
        for file in self.LidarL1PulsewavesFiles:
            zipPulsewavesHelper(file,pulseZipCommandDict,self.LidarLastoolsProcessingCommandOutputDir)
        
        
    def processWaveformLidarToDiscrete(self):

        self.getProductFiles()        
        self.getWaveformLidarProcessingParameters()
        os.makedirs(self.LidarInternalPulsewavesProcessingDir,exist_ok=True)
        currentDir = os.getcwd()
        os.chdir(lidarSrcL1WaveformDir)
        for plsFile in self.LidarL1PulsewavesFiles:
            
            finPlsFile = open(plsFile,"rb")
            finPlsFile.seek(184,0)
            numberOfPulses = np.fromfile(finPlsFile,np.int64,1)[0]
            
            print('Working on: '+plsFile)
            
            finPlsFile.close()
            numTotalBlocksToProcess = np.floor(numberOfPulses/self.pulsesToSaveInMemory)
            if (numTotalBlocksToProcess>0):
                remainderPulses = np.floor(numberOfPulses - numTotalBlocksToProcess*self.pulsesToSaveInMemory)
                remainderPulsesPerBlock = np.floor(remainderPulses/numTotalBlocksToProcess)
                pointsPerBlock = int(self.pulsesToSaveInMemory +remainderPulsesPerBlock)
            else:
                pointsPerBlock = numberOfPulses
                numTotalBlocksToProcess = 1
            
            for blockCountNum in range(0,int(numTotalBlocksToProcess)):
                startPulse = str(pointsPerBlock*blockCountNum) 
                endPulse = str(pointsPerBlock*(blockCountNum+1))
                
                if self.payload.lidarSensorManufacturer == 'riegl':
                    os.system("python PulsewavesLMGPU_function_call_riegl.py "+plsFile+" "+ os.path.join(self.LidarInternalPulsewavesProcessingDir,'')+" "+startPulse+" "+endPulse)
                if self.payload.lidarSensorManufacturer == 'optech':
                    os.system("python PulsewavesLMGPU_function_call_optech.py "+plsFile+" "+ os.path.join(self.LidarInternalPulsewavesProcessingDir,'')+" "+startPulse+" "+endPulse)
                 
        os.chdir(currentDir)

    def qaWaveformLidar(self):
        
        WaveformQa(os.path.join(self.BaseDir,''),self.YearSiteVisit,self.payload.lidarSensorManufacturer)

    def processNeonProducts(self):
        
        self.processSbets()
        
        self.processL1Lidar()
        
        self.processL3Lidar()
        
        self.processSpectrometer()
        
        self.processCamera()
        IDL.run('.RESET_SESSION')
        IDL.exit()
        
        self.processWaveformLidar()

    def processSbets(self):

        for mission in self.missions:
            print('Running SBET processing for '+mission.missionId)

            mission.processSbet()

    def processL1RieglLidar(self):
        for mission in self.missions:
            print('Running L1 lidar processing for '+mission.missionId)
            mission.runWavexWorkflow()

        print('Moving WavEx Outfiles to the Internal Directory for '+ self.YearSiteVisit)
        self.moveAllWavexOutfiles()

        print('Running Lidar strip align for '+ self.YearSiteVisit)
        self.runStripAlign()
        
        self.organizeLidarFilesForProcessing()
        self.runL1LidarQA()

    def processL1OptechLidar(self):
        for mission in self.missions:
            print('Running L1 lidar processing for '+mission.missionId)
            mission.runOptechWorkflow()
            print('Moving Optech Outfiles to the Internal Directory for '+mission.missionId)
            mission.moveAllOptechOutfiles()

        print('Renaming LazStandard Files')
        self.renameOptechLazStandardFiles()

        print('Running Lidar strip align for '+ self.YearSiteVisit)
        self.runStripAlign()
        
        print('Organizing Lidar Files for L3 Processing')
        self.organizeLidarFilesForProcessing()
        
        print('Running L1 Lidar QA')
        self.runL1LidarQA()

    def processL1Lidar(self):
        print('Campaign:',self.campaigns[0])
        if self.campaigns[0].startswith('P3'):
            self.processL1RieglLidar()
            print('Running processL1RieglLidar')
        else:
            print('Running processL1OptechLidar')
            self.processL1OptechLidar()


    # def processL1Lidar(self):

    #     for mission in self.missions:
    #         print('Running L1 lidar processing for '+mission.missionId)
    #         if self.campaigns[0] == 'P3':
    #             mission.runWavexWorkflow()
    #         else:
    #             print('Running runOptechWorkflow')
    #             mission.runOptechWorkflow()
        
    #     print('Running Lidar strip align for '+ self.YearSiteVisit)
        
    #     self.moveAllWavexOutfiles()
        
    #     self.runStripAlign()
        
    #     self.organizeLidarFilesForProcessing()
    #     self.runL1LidarQA()

    def processWaveformLidar(self):
        
        self.translateWaveformLidar()
        
        self.processWaveformLidarToDiscrete()
        
        self.qaWaveformLidar()
        
        self.zipWaveformPulsewavesFiles()
        
        self.renameWaveformLidarFiles()

    def processSpectrometer(self):

        self.processSpectrometerMissions()
        self.processSpectrometerMosaic()
        print('Finished processSpectrometerProducts '+self.YearSiteVisit)

    def reapplyBrdfCorrection(self):
        self.reapplyBrdf = True
        for mission in self.missions:
            mission.reapplyBrdfCorrectionWorkflow()
        self.processSpectrometerMosaicProducts()

    def processSpectrometerMissions(self):

        self.downloadInternalLidarAigDsm()
        aigdsmSapExists=False
        for file in self.LidarInternalAigDsmFiles:
            if file.endswith('sap_vf_bf.hdr'):
                aigdsmSapExists = True
        
        if aigdsmSapExists:
            pass
        else:
            self.calcAigDsmSlopeAspect()
            self.getProductFiles()
        for mission in self.missions:
            mission.spectrometerProcessWorkflow(os.path.join(self.geoidFileDir,self.geoidFile),self.internalAigDsmFile,self.internalAigSapFile,self.spectrometerProductQaList,self.Domain,self.visit)

    def processSpectrometerMosaic(self):

        self.processMosaic()
        self.processMosaicSpectrometerProducts()
        self.processMosaicQa()
        self.cleanMosaicProducts()
        print('Finished processSpectrometerMosaicProducts '+self.YearSiteVisit)

    def processMosaic(self):

        print('Running mosaics for '+self.YearSiteVisit)
        self.processMosaicTiles()
        print('Running L3 reflectance H5 for '+self.YearSiteVisit)
        self.processMosaicH5ReflectanceWriter()
        self.getProductFiles()
        self.cleanupL3H5()

    def processMosaicSpectrometerProducts(self):

        print('Running L3 FPAR mosaics for '+self.YearSiteVisit)
        self.generateMosaicFpar()
        print('Running L3 LAI mosaics for '+self.YearSiteVisit)
        self.generateMosaicLai()
        print('Running L3 Veg Indices mosaics for '+self.YearSiteVisit)
        self.generateMosaicVegIndices()
        print('Running L3 Water Indices mosaics for '+self.YearSiteVisit)
        self.generateMosaicWaterIndices()
        print('Adding no data to L3 Albedo mosaics for '+self.YearSiteVisit)
        self.addNodataToAlbedo()
        self.getProductFiles()

    def processMosaicQa(self):
        
        self.getProductFiles()

        print('Getting previous Year Site Visit Spectrometer data for '+self.YearSiteVisit)

        self.getPreviousYearSiteVisit()
        if self.previousYearSiteVisit is not None:

            self.downloadPreviousYearSiteVisitSpectrometer()
        
        print('Generating water mask and Reflectance plots')
        
        self.generateReflectancePlots()
        self.generateMosaicWaterMask()

        print('Getting data product mosaics for '+self.YearSiteVisit)
        self.generateSpectrometerProductL3QaMosaics()
        
        print('Getting data product error mosaics for '+self.YearSiteVisit)
        self.generateSpectrometerProductErrorL3QaMosaics()
        
        if self.previousYearSiteVisit is not None:
       
            print('Getting data product difference mosaics for '+self.YearSiteVisit)
            self.generateSpectrometerL3DifferenceQaMosaics()
        
            print('Running reflectance difference analysis for '+self.YearSiteVisit)
            self.generateMosaicReflectanceDifferenceProductQa()
        
            print('Getting reflectance difference mosaics for '+self.YearSiteVisit)
            self.generateReflectanceDifferenceQaMosaics()
        
        print('Getting water mask mosaic for '+self.YearSiteVisit)
        self.generateWaterMaskQaMosaic()
        
        print('Getting reflectance ancillary data product mosaics for '+self.YearSiteVisit)
        self.generateReflectanceAncillaryRasterMosaics()
        
        print('Getting reflectance RGB data product mosaics for '+self.YearSiteVisit)
        self.generateReflectanceRgbRasterMosaics()
        
        print('Getting reflectance RGB data histogram plot for '+self.YearSiteVisit)
        self.generateRgbDistributionPlot()
        
        print('Getting PNG map plots for '+self.YearSiteVisit)
        self.getMosaicQaPngMaps()
        
        print('Getting PNG histograms for product mosaics for '+self.YearSiteVisit)
        self.getSpectrometerQaReportHistograms()
        
        print('Getting report summaries for '+self.YearSiteVisit)
        self.getSpectrometerQaReports()
        
        print('Generating spectrometer L3 QA PDF report for '+self.YearSiteVisit)
        self.generateMosaicQaPdf()

        print('Generating spectrometer L3 QA Markdown and HTML reports for '+self.YearSiteVisit)
        self.generateMosaicQaHtml()
        print('Finished '+self.YearSiteVisit)

    def processMosaicTiles(self):

        os.makedirs(self.SpectrometerL3ReflectanceDir,exist_ok=True)

        #from idlpy import IDL

        IDL.run('cd, "' + nisSpectrometerMosaicDir + '"')
        IDL.run('.compile ' + '"' + 'generateReflectanceAlbedoMosaic.pro' + '"')

        IDL.yearSiteVisitBaseFolder = str(self.BaseDir+'/')
        IDL.payload = str(self.payload.payloadId)

        crossStrips = ''
        for mission in self.missions:
            for line in mission.FlightLines:
                if line.nisCrossStrip:
                    crossStrips += str(line.bidirectionalReflectanceSpectrometerH5File)
                    crossStrips +=','
        crossStrips = crossStrips[0:-1]
        if crossStrips == '':
            crossStrips = 'None'
        IDL.crossStrips = str(crossStrips)
        IDL.doReflectance = 1


        IDL.run('generateReflectanceAlbedoMosaic(yearSiteVisitBaseFolder,payload,crossStrips,doReflectance)',stdout=True)
        IDL.close()
        IDL.Heap_GC()
        #IDL.exit()

        self.getProductFiles()

    def processMosaicH5ReflectanceWriter(self,isBrdf=True):

        allIterableFiles = zip(self.SpectrometerL3ReflectanceEnviFiles, self.SpectrometerL3ElevationEnviFiles,self.SpectrometerL3ShadowEnviFiles)
        allIterableFiles = list(allIterableFiles)
        print(self.missions[0].SpectrometerL1RadianceProcessingDir+os.path.sep)
        # with Pool(processes=5) as pool:
        #     processFunction = partial(H5WriterFunctionHelper,radOrt=os.path.join(self.SpectrometerL1RadianceDir,''),sbetFile=self.missions[0].sbetTrajectoryFile,outputDir=self.SpectrometerL3ReflectanceDir+os.path.sep,metadataXML=self.missions[0].metadataXml,NISlog=self.missions[0].spectrometerFlightLog,ScriptsFile=self.missions[0].scriptsFile)
        #     pool.map(processFunction,allIterableFiles)
        for reflectanceL3File,elevationL3File,shadowL3File in allIterableFiles:
            print("working on "+reflectanceL3File)
            if os.path.exists(os.path.join(self.SpectrometerL3ReflectanceDir,reflectanceL3File.replace('_Reflectance','_bidirectional_reflectance.h5'))):
                print('Skipping: '+os.path.join(self.SpectrometerL3ReflectanceDir,reflectanceL3File.replace('_Reflectance','_bidirectional_reflectance.h5')))
            else:
                print('Working on: '+os.path.join(self.SpectrometerL3ReflectanceDir,reflectanceL3File.replace('_Reflectance','_bidirectional_reflectance.h5')))
                H5WriterFunction(reflectanceL3File,elevationL3File,shadowL3File,rdn_ort=os.path.join(self.SpectrometerL1RadianceDir,''),sbet_file=self.missions[0].sbetTrajectoryFile, dataDirOut=self.SpectrometerL3ReflectanceDir+os.path.sep, FlightMetaDataXML=self.missions[0].metadataXml, NISlog=self.missions[0].spectrometerFlightLog,  ScriptsFile=self.missions[0].scriptsFile)

        self.getProductFiles()

    # def generateMosaicReflectanceQa(self):

    #     os.makedirs(self.SpectrometerL3QaReflectanceDir,exist_ok=True)

    #     pool = Pool(processes=30)
    #     processFunction = partial(generateReflectanceQa,outputDir=self.SpectrometerL3QaReflectanceDir)
    #     pool.map(processFunction,self.SpectrometerL3ReflectanceFiles)
    #     pool.close()

    def getReflectanceDifferenceTileTifs(self):
        
        self.getProductFiles()
        
        os.makedirs(self.SpectrometerQaDifferenceMosaicDir,exist_ok=True)
        
        currentTileFiles,previousTileFiles = getMatchingTileFiles(self.SpectrometerL3ReflectanceFiles,self.previousL3ReflectanceFiles)
                
        getOverlapDifferenceTif(currentTileFiles,previousTileFiles,self.SpectrometerQaDifferenceMosaicDir)

    def getReflectanceMeanDifferenceTiles(self):
        
        self.wavelengthSampleCount,self.meanDifferenceSpectra = getReflectanceMeanDifference(self.SpectrometerQaDifferenceMosaicFiles)
    
    def getTileReflectanceDifferenceSummaryStats(self):
        
        reflectanceArray, metadata, wavelengths = h5refl2array(self.SpectrometerL3ReflectanceFiles[0], 'Reflectance',onlyMetadata = True)
        
        rmsArrays,maxIndicesArrays,maxDiffArrays,self.maxDifferenceWavelengths,self.sumWavelengthDifferenceSquared,self.sumWavelengthVariance = getFlightlineReflectanceDifferenceSummaryStats(self.SpectrometerQaDifferenceMosaicFiles,self.meanDifferenceSpectra,wavelengths)

        return rmsArrays,maxIndicesArrays,maxDiffArrays

    def getTileRmsAndMaxWavelgthErrorRasters(self,rmsArrays,maxIndicesArrays,maxDiffArrays):
        
        filebasename = []
                
        os.makedirs(self.SpectrometerL3QaRmsDir,exist_ok=True)
        os.makedirs(self.SpectrometerL3QaMaxWavelengthDir,exist_ok=True)
        os.makedirs(self.SpectrometerL3QaMaxDiffDir,exist_ok=True)
        
        for file in self.SpectrometerQaDifferenceMosaicFiles:
            fileSplit = os.path.basename(file).split('_')
            filebasename.append(self.YearSiteVisit+'_'+self.previousYearSiteVisit+'_'+fileSplit[1]+'_'+fileSplit[2])
        
        getRmsAndMaxWavelgthErrorRasters(rmsArrays,maxIndicesArrays,maxDiffArrays,self.SpectrometerQaDifferenceMosaicFiles,self.SpectrometerL3QaRmsDir,self.SpectrometerL3QaMaxWavelengthDir,self.SpectrometerL3QaMaxDiffDir,filebasename)

        rmsArrays = []
        maxIndicesArrays = []

    def getWavelengthDifferenceSummaryPlots(self):
        
        scaleReflectanceFactor = 100
        
        reflectanceArray, metadata, wavelengths = h5refl2array(self.SpectrometerL3ReflectanceFiles[0], 'Reflectance',onlyMetadata = True)
        
        outputFileBaseName = self.YearSiteVisit+'_'+self.previousYearSiteVisit
        
        generateReflectanceDifferenceLinePlots(wavelengths,self.meanDifferenceSpectra,self.wavelengthStandardDeviation,self.wavelengthTotalRms,self.maxDifferenceWavelengthsTotal,scaleReflectanceFactor,self.SpectrometerQaL3HistsDir,outputFileBaseName)

    def generateMosaicReflectanceDifferenceProductQa(self):
        
        self.getReflectanceDifferenceTileTifs()
        
        self.getProductFiles()
        
        self.getReflectanceMeanDifferenceTiles()
        
        rmsArrays,maxIndicesArrays,maxDiffArrays = self.getTileReflectanceDifferenceSummaryStats()
        
        self.getTileRmsAndMaxWavelgthErrorRasters(rmsArrays,maxIndicesArrays,maxDiffArrays)

        self.getProductFiles()

        self.wavelengthStandardDeviation = (np.sum(self.sumWavelengthVariance,axis=0)[:,None]/np.sum(self.wavelengthSampleCount,axis=0)[:,None])**0.5
        
        self.wavelengthTotalRms = (np.sum(self.sumWavelengthDifferenceSquared,axis=0)[:,None]/np.sum(self.wavelengthSampleCount,axis=0)[:,None])**0.5
        
        self.maxDifferenceWavelengthsTotal = np.nanmax(self.maxDifferenceWavelengths,axis=0)[:,None]
        
        self.getWavelengthDifferenceSummaryPlots()

    def generateDateRasterMosaic(self):

        os.makedirs(self.SpectrometerQADateDir,exist_ok=True)
        'Running date rasters'
        #for h5File in self.SpectrometerL3ReflectanceFiles:
        #    generateDateTifs(h5File,self.SpectrometerQADateDir)

        with Pool(processes=15) as pool:
            processFunction = partial(generateDateTifs,outFolder=self.SpectrometerQADateDir)
            pool.map(processFunction,self.SpectrometerL3ReflectanceFiles)
        self.getProductFiles()
        generateMosaic(self.SpectrometerQADateFiles,os.path.join(self.SpectrometerQADateDir,self.YearSiteVisit+'_DateRasterMosaic.tif'))
        self.getProductFiles()

        self.dateRasterMosaic= os.path.join(self.SpectrometerQADateDir,self.YearSiteVisit+'_DateRasterMosaic.tif')
        
    def resampleSpectrum(self,outputType='ENVI'):
          
        os.makedirs(self.spectrometerProcessingSpectralResamplingDir,exist_ok=True)
        
        with Pool(processes=15) as pool:
            processFunction = partial(generateResampleH5Spectrum,targetWavelengths=self.targetWavelengths,outFolder=self.spectrometerProcessingSpectralResamplingDir,outputType=outputType,filePathToEnviProjCs=self.filePathToEnviProjCs)
            pool.map(processFunction,self.SpectrometerL3ReflectanceFiles)


    def cleanMosaicProducts(self):
        
        self.getProductFiles()
        
        self.cleanVegIndexTifs()
        
        self.cleanWaterIndexTifs()
        
        self.cleanSpectrometerProcessingFolder()

    def cleanVegIndexTifs(self):
        
        for tifFile in self.SpectrometerL3VegIndicesTifFiles:
            os.remove(tifFile)

    def cleanWaterIndexTifs(self):
        
        for tifFile in self.SpectrometerL3WaterIndicesTifFiles:
            os.remove(tifFile)

    def cleanSpectrometerProcessingFolder(self):
        
        if os.path.exists(self.spectrometerProcessingDir):
        
            shutil.rmtree(self.spectrometerProcessingDir)

    def getSamplesFromDateRaster(self,coordinates):

        self.generateDateRasterMosaic()

        rasterValues = getRasterValuesAtCoordinates(self.dateRasterMosaic, coordinates)

        return rasterValues

    def addNodataToAlbedo(self):

        for albedoFile in self.SpectrometerL3AlbedoFiles:

            AddNoDataToTIFF(albedoFile)

    def generateMosaicQaPdf(self):

        matlabEng = matlab.engine.start_matlab()
        matlabEng.cd(nisSpectrometerQaCodeDir, nargout=0)
        
        if self.previousYearSiteVisit is not None:
            matlabEng.generateL3QAPdf(str(self.BaseDir+os.path.sep),str(self.YearSiteVisit),str(self.previousYearSiteVisit))
        else:
            matlabEng.generateL3QAPdf(str(self.BaseDir+os.path.sep),str(self.YearSiteVisit),'first')
        matlabEng.quit()

    def generateMosaicQaHtml(self):
        # function to generate NIS L3 QA documents in markdown and html formats
        # requires the following folders / contents to be downloaded:
            # SpectrometerL3ReflectanceDir (L3\Spectrometer\Reflectance)
            # SpectrometerQaL3HistsDir (QA\Spectrometer\L3Hists)

        currentDir = os.getcwd()
        print(f'Changing directories to {nisSpectrometerQaCodeDir}')
        os.chdir(nisSpectrometerQaCodeDir)

        if self.previousYearSiteVisit is not None:
            print('Running generateL3QAMarkdown.py with the following inputs:')
            print('year_site_visit:', self.YearSiteVisit)
            print('previous_visit:',self.previousYearSiteVisit)
            os.system(f"python generateL3QAMarkdown.py {self.YearSiteVisit} {self.previousYearSiteVisit}")
        else:
            print('No previous site visit found!')
            print('Running generateL3QAMarkdown.py with the following inputs:')
            print('year_site_visit:', self.YearSiteVisit)
            os.system(f"python generateL3QAMarkdown.py {self.YearSiteVisit}")

        os.chdir(currentDir)

    def cleanupL3H5(self):

        self.getProductFiles()

        deleteExtensionListReflectance = ['_Aerosol_Optical_Thickness', '_Aerosol_Optical_Thickness.hdr',
                                          '_Aspect', '_Aspect.hdr',
                                          '_Azimuth', '_Azimuth.hdr',
                                          '_BRDF_Mask', '_BRDF_Mask.hdr',
                                          '_Cast_Shadow', '_Cast_Shadow.hdr',
                                          '_Dark_Dense_Vegetation_Classification', '_Dark_Dense_Vegetation_Classification.hdr',
                                          '_Haze_Cloud_Water_Map', '_Haze_Cloud_Water_Map.hdr',
                                          '_Illumination_Factor', '_Illumination_Factor.hdr',
                                          '_Path_Length', '_Path_Length.hdr',
                                          '_Reflectance', '_Reflectance.hdr',
                                          '_Rules', '_Rules.hdr',
                                          '_Sky_View_Factor', '_Sky_View_Factor.hdr',
                                          '_Slope', '_Slope.hdr',
                                          '_Smooth_Surface_Elevation', '_Smooth_Surface_Elevation.hdr',
                                          '_Visibility_Index_Map', '_Visibility_Index_Map.hdr',
                                          '_Weather_Quality_Indicator', '_Weather_Quality_Indicator.hdr',
                                          '_Water_Vapor_Column', '_Water_Vapor_Column.hdr',
                                          '_Zenith', '_Zenith.hdr']

        deleteFiles(self.SpectrometerL3ReflectanceProcessingFiles,deleteExtensionListReflectance)

    def renameOptechLazStandardFiles(self):

        lazFiles = collectFilesInPath(self.LidarInternalLazStandardDir,'.laz')
        # plsFiles = collectFilesInPath(self.LidarInternalLazStandardDir,'.pls')
        # wvsFiles = collectFilesInPath(self.LidarInternalLazStandardDir,'.wvs')
        
        # allFilesToMove = lazFiles+plsFiles+wvsFiles
        
        for file in lazFiles:
            # print('OLD NAME: ',file)
            # print('NEW NAME: ',os.path.basename(file).replace(f'_{self.year}',f'_{self.YearSiteVisit}_{self.year}'))
            shutil.move(file,os.path.join(self.LidarInternalLazStandardDir,os.path.basename(file).replace('L00','L0').replace(f'_{self.year}',f'_{self.YearSiteVisit}_{self.year}')))

    def moveAllWavexOutfiles(self):
        
        os.makedirs(self.LidarInternalLazStandardDir,exist_ok=True)
    
        lazFiles = collectFilesInPath(self.missions[0].wavexOutputDir,'.laz')
        plsFiles = collectFilesInPath(self.missions[0].wavexOutputDir,'.pls')
        wvsFiles = collectFilesInPath(self.missions[0].wavexOutputDir,'.wvs')
        
        allFilesToMove = lazFiles+plsFiles+wvsFiles
        
        for file in allFilesToMove:
            shutil.move(file,os.path.join(self.LidarInternalLazStandardDir,os.path.basename(file)))

    def runStripAlign(self):
        
        os.makedirs(self.LidarL1UnclassifiedLasDir,exist_ok=True)
        os.makedirs(self.TempStripAlignDiscreteLidarProcessingDir,exist_ok=True)
        
        sbet_list=[]
        gpsdate_list=[]
        laz_groups_command=''
        po_groups_command=''
        
        for mission in self.missions:
        
            gps_date = mission.flightday[0:4]+'-'+mission.flightday[4:6]+'-'+mission.flightday[6:8]
            gpsdate_list.append(gps_date)
            sbet_list.append(mission.sbetTrajectoryFile)
            laz_groups_command = laz_groups_command + '-i *'+mission.flightday+ '_s.laz '
            po_groups_command = po_groups_command + '-po '+mission.sbetTrajectoryFile + ' '

        stripalign_cmd ="C:\\Bayes\\StripAlign.exe -align"+ \
        " -I "+ self.LidarInternalLazStandardDir + ' ' + \
        laz_groups_command + \
        po_groups_command + \
        '-gps_date '+' '.join(gpsdate_list) + \
        " -O "+self.stripAlignDiscreteLidarProcessingDir + \
        " -T "+self.TempStripAlignDiscreteLidarProcessingDir + \
        " -large -olas -opls -tiff -dz16"
        #print(stripalign_cmd)
        check_call(stripalign_cmd)
        
    def organizeLidarFilesForProcessing(self):

        os.makedirs(self.LidarL1UnclassifiedExtraLasDir,exist_ok=True)

        stripAlignLasFiles = collectFilesInPath(self.stripAlignDiscreteLidarProcessingDir,'las')
        stripAlignPlsFiles = collectFilesInPath(self.stripAlignDiscreteLidarProcessingDir,'pls')
        internalWvsFiles = collectFilesInPath(self.LidarInternalLazStandardDir,'wvs')

        if len(stripAlignPlsFiles) == 0 and len(internalWvsFiles) == 0:
            print('No waveform files found, moving las files to L1/DiscreteLidar/Las folder')
            for stripAlignLasFile in stripAlignLasFiles:
                print('New Las Location:',os.path.join(self.LidarL1UnclassifiedLasDir,os.path.basename(stripAlignLasFile).replace('_s.','_r.')))
                shutil.move(stripAlignLasFile, os.path.join(self.LidarL1UnclassifiedLasDir,os.path.basename(stripAlignLasFile).replace('_s.','_r.')))
        else:
            print('Moving las, pls, and wvs files to L1/DiscreteLidar/Las folder')
            os.makedirs(self.LidarInternalPulsewavesDir,exist_ok=True)
            for stripAlignLasFile,stripAlignPlsFile,internalWvsFile in zip(stripAlignLasFiles,stripAlignPlsFiles,internalWvsFiles):
                shutil.move(stripAlignLasFile, os.path.join(self.LidarL1UnclassifiedLasDir,os.path.basename(stripAlignLasFile).replace('_s.','_r.')))
                shutil.move(stripAlignPlsFile, os.path.join(self.LidarInternalPulsewavesDir,os.path.basename(stripAlignPlsFile).replace('_s.','_r.')))
                shutil.move(internalWvsFile, os.path.join(self.LidarInternalPulsewavesDir,os.path.basename(internalWvsFile).replace('_s.','_r.')))

        print('Determining extra lines')
        self.getProductFiles()

        numberMissionLines = []
        crossStrips = []
        for mission in self.missions:
            numberMissionLines.append(int(len(mission.FlightLines)))

            for flightline in mission.FlightLines:
                if flightline.nisCrossStrip:
                    crossStrips.append(flightline.lineNumber.split('-')[0])
                    
        for crossStrip in crossStrips:
            for lasFile in self.LidarL1UnclassifiedLasFiles:
                if crossStrip in os.path.basename(lasFile):
                    shutil.move(lasFile,os.path.join(self.LidarL1UnclassifiedExtraLasDir,os.path.basename(lasFile)))
            self.getProductFiles()
            
        flightDayWithMaxLines = self.missions[np.where(np.array(numberMissionLines)==np.max(np.array(numberMissionLines)))[0][0]].flightday

        allLineNumbers = []

        for lasFile in self.LidarL1UnclassifiedLasFiles:
            lineSplit = os.path.basename(lasFile).split('_')[0].split('-')
            allLineNumbers.append(lineSplit[0]) 
                 
        allUniqueNumbers = list(set(allLineNumbers))    

        linesToMove = []

        for lineNumber in allUniqueNumbers:

            repeatLines =  [line for line in self.LidarL1UnclassifiedLasFiles if lineNumber in os.path.basename(line)]

            if len(repeatLines) == 1:
                continue
            else:

                fileSizes = []
                
                for repeatLine in repeatLines:
                    fileSizes.append(os.path.getsize(repeatLine))
                    
                relativeFileSizes = np.array(fileSizes)/np.max(np.array(fileSizes))
        
                indexSmallFiles = np.where(relativeFileSizes < 0.9)
                
                if indexSmallFiles[0].size > 0:
                
                    for indexSmallFile in indexSmallFiles:
                        linesToMove.append(repeatLines[indexSmallFile[0]])

        for lineToMove in linesToMove:
            shutil.move(lineToMove,os.path.join(self.LidarL1UnclassifiedExtraLasDir,os.path.basename(lineToMove)))

        linesToMove = []
        self.getProductFiles()
        
        for lineNumber in allUniqueNumbers:
            
            repeatLines =  [line for line in self.LidarL1UnclassifiedLasFiles if lineNumber in os.path.basename(line)]
            
            if len(repeatLines) == 1:
                continue
            else:
                for repeatLine in repeatLines:
                    flightDay = os.path.basename(repeatLine).split('_')[1]
                    if flightDay != flightDayWithMaxLines:
                        linesToMove.append(repeatLine)
        
        for lineToMove in linesToMove:
            shutil.move(lineToMove,os.path.join(self.LidarL1UnclassifiedExtraLasDir,os.path.basename(lineToMove)))
            
        linesToMove = []
        self.getProductFiles()

        for lineNumber in allUniqueNumbers:

            repeatLines =  [line for line in self.LidarL1UnclassifiedLasFiles if lineNumber in os.path.basename(line)]
            
            if len(repeatLines) == 1:
                continue
            else:
                for repeatLine in repeatLines:
                    lineSplit = os.path.basename(lasFile).split('_')[0].split('-')
                    lineNumber = lineSplit[0]  
                    lineRepeat = lineSplit[1]
                    
                    if int(lineRepeat) > 1:
                        linesToMove.append(repeatLine)

        for lineToMove in linesToMove:
            shutil.move(lineToMove,os.path.join(self.LidarL1UnclassifiedExtraLasDir,os.path.basename(lineToMove)))

    def runL1LidarQA(self):
        os.makedirs(self.LidarL1QADir,exist_ok=True)
        currentDir = os.getcwd()
        
        print('Payload: ',self.payload.payload)
        
        # generate coverage plots
        os.chdir(lidarLibPythonDir)
        os.system(f"python make_lidar_coverage_plots.py {self.YearSiteVisit}")
        
        # run Riegl or Optech QAQC script to generate L1 Lidar QA markdown file
        if self.payload.payload == 'P3':
            os.chdir(lidarSrcL1RieglDir)
            os.system(f"python riegl_bayes_qaqc_md.py {self.YearSiteVisit}")
            os.chdir(currentDir)
        elif self.payload.payload == 'P1':
            print('Generating Optech L1 QAQC Markdown')
            os.chdir(lidarSrcL1OptechDir)
            print('Current Directory: ',os.getcwd())
            os.system(f"python optech_galaxy_lms47_qaqc_md.py {self.YearSiteVisit}")
            os.chdir(currentDir)

    def writeCameraInputFile(self):
        
        os.makedirs(self.CameraProcessingDir,exist_ok=True)
                
        cameraParameterDict = {}
        
        allMissions = []
        
        for mission in self.missions:
            allMissions.append(mission.missionId)
            
        allMissions = sorted(allMissions, key=lambda x: (int(x.split('_')[0]), x.split('_')[1])) 
           
        cameraParameterDict['fullSite'] = self.YearSiteVisit
        cameraParameterDict['flights'] = allMissions
        cameraParameterDict['site'] = self.site.site
        cameraParameterDict['cameraModel'] = self.payload.cameraGeolocationModel
    
        if self.payload.payload == 'P3':
    
            cameraParameterDict['timeOffset'] = self.payload.cameraTimingOffset
            
        writeCameraSummaryFile(self.cameraParamsFile, cameraParameterDict)

    def cleanCameraProcessingTemp(self):

        if os.path.exists(os.path.join(self.CameraProcessingDir,'temp')):
            shutil.rmtree(os.path.join(self.CameraProcessingDir,'temp'))

    def cleanCameraMosaicFolder(self):
            
        for file in self.CameraProcessingMosaicFiles:
            if file.endswith('.sav'):
                continue
            else:
                os.remove(file)

    def moveSummaryFiles(self):
        
        shutil.move(self.cameraParamsFile,os.path.join(self.CameraProcessingSummaryFilesDir,os.path.basename(self.cameraParamsFile)))
        shutil.move(os.path.join(self.CameraProcessingDir,'rawFileList.txt'),os.path.join(self.CameraProcessingSummaryFilesDir,'rawFileList.txt'))

    def processCamera(self):
        
        print('Writing camera summary input file for '+self.YearSiteVisit)    
        self.writeCameraInputFile()
        
        print('Downloading camera raw files for '+self.YearSiteVisit)
        
        for mission in self.missions:
            
            #mission.downloadCamera()
            mission.downloadCameraHkQa()
                   
        print('Running camera preprocessing for '+self.YearSiteVisit)    
        self.runCameraPreprocessing()
        
        for mission in self.missions:
            
            print('Running camera ortho for '+mission.missionId)    
            mission.runCameraOrtho()
            
        print('Running camera ortho check for '+self.YearSiteVisit)
        self.runCameraOrthoCheck()
        
        print('Running camera mosaic for '+self.YearSiteVisit)
        self.runCameraMosaic()
        
        print('Running point cloud colorization for '+self.YearSiteVisit)
        self.runColorizePointCloud()
        
        if os.path.exists(os.path.join(self.L1ProductDir,'GPSIMU')):
            shutil.rmtree(os.path.join(self.L1ProductDir,'GPSIMU'))
            
        for mission in self.missions:
            
            print('Cleaning camera ortho for '+mission.missionId)    
            mission.cleanCameraOrthoFolder()
        
        print('Cleaning camera directories for '+self.YearSiteVisit)
        self.cleanCameraProcessingTemp()
        self.cleanCameraMosaicFolder()
        self.moveSummaryFiles()

    def runCameraPreprocessing(self):
        
        IDL.run('.RESET_SESSION')
        IDL.run('cd, "' + self.BaseDir + '"')
        IDL.run('.compile ' + '"' + 'run_camera_setup.pro' + '"')

        IDL.siteParametersFile = str(self.cameraParamsFile)

        IDL.run('run_camera_setup(siteParametersFile)',stdout=True)

    def runCameraOrthoCheck(self):

        IDL.run('cd, "' + self.BaseDir + '"')
        IDL.run('.compile ' + '"' + 'run_post_ortho_error_check.pro' + '"')
        IDL.run('run_post_ortho_error_check',stdout=True)

    def runCameraMosaic(self):
        #from idlpy import IDL

        IDL.run('cd, "' + self.BaseDir + '"')
        IDL.run('.compile ' + '"' + 'run_aig_dms_mosaic.pro' + '"')
        IDL.run('run_aig_dms_mosaic',stdout=True)

    def runColorizePointCloud(self):
        
        lazList = []
        imageList = []
        for lazFile in self.LidarL1ClassifiedLasFiles:
            lazFileSplit = os.path.splitext(os.path.basename(lazFile))[0].split('_')
            tileRef = lazFileSplit[4]+'_'+lazFileSplit[5]
            for imageFile in self.CameraL3ImagesFiles:
                if tileRef in os.path.basename(imageFile):
                    lazList.append(lazFile)
                    imageList.append(imageFile)
                    
        with Pool(processes=15) as pool:

            processFunction = partial(processColorizeLazHelper,commandOutputDir=self.LidarLastoolsProcessingCommandOutputDir)

            pool.map(processFunction,zip(lazList,imageList))
                
        #for files in zip(lazList,imageList):
            
        #    processColorizeLazHelper(files,self.LidarLastoolsProcessingCommandOutputDir)


    def renameDiscreteLidarFiles(self):

        self.getProductFiles()
        
        renameLidarFiles(self.LidarL1FlightlineFiles,self.LidarL1UnclassifiedLazDir,self.l1ProductBaseName,fileSuffix='unclassified_point_cloud',flightline=True)
        
        renameLidarFiles(self.LidarMetadataHorizontalUncertaintyLazFiles,self.LidarMetadataUncertaintyLazDir,self.qaProductBaseName,fileSuffix='horizontal_uncertainty',flightline=True)
        
        renameLidarFiles(self.LidarMetadataVerticalUncertaintyLazFiles,self.LidarMetadataUncertaintyLazDir,self.qaProductBaseName,fileSuffix='vertical_uncertainty',flightline=True)
        
        renameLidarFiles(self.LidarL1TilesFiles,self.LidarL1ClassifiedLasDir,self.l1ProductBaseName,fileSuffix='classified_point_cloud',tiles=True)
        
        renameLidarFiles(self.LidarMetadataTileKmlFiles,self.LidarMetadataTileKmlDir,self.qaProductBaseName,fileSuffix='boundary',tiles=True)
        
        renameLidarFiles(self.LidarMetadataTileShpFiles,self.LidarMetadataTileShpDir,self.qaProductBaseName,fileSuffix='boundary',tiles=True)
        
        renameLidarFiles(self.LidarMetadataFlightlineBoundaryFiles,self.LidarMetadataKmlDir,self.qaProductBaseName,fileSuffix='boundary',flightline=True)

    def renameWaveformLidarFiles(self):
        
        self.getProductFiles()
        
        renameLidarFiles(self.LidarL1PulsewavesPlzZipFiles,self.PulswavesL1Dir,self.l1ProductBaseName,flightline=True)
        renameLidarFiles(self.LidarL1PulsewavesWvzZipFiles,self.PulswavesL1Dir,self.l1ProductBaseName,flightline=True)

    def processL3Lidar(self):

        print('Running lidar preprocessing for '+self.YearSiteVisit)
        self.lidarPreprocessingWorkflow()

        print('Running lidar noise filtering workflow for '+self.YearSiteVisit)
        self.lidarNoiseFilteringWorkflow()

        print('Running lidar tiling workflow for '+self.YearSiteVisit)
        self.lidarTilingWorkflow()

        print('Running lidar ground classification for '+self.YearSiteVisit)
        self.runGroundClassification()

        print('Running lidar intensity workflow for '+self.YearSiteVisit)
        self.lidarIntensityImageWorkflow()

        print('Running lidar overlap workflow for '+self.YearSiteVisit)
        self.lidarOverlapWorkflow()

        print('Running lidar DTM workflow for '+self.YearSiteVisit)
        self.lidarDtmWorkflow()

        print('Running lidar point cloud classification for '+self.YearSiteVisit)
        self.runPointCloudClassification()

        print('Running lidar DSM workflow for '+self.YearSiteVisit)
        self.lidarDsmWorkflow()

        print('Running lidar AIGDSM workflow for '+self.YearSiteVisit)
        self.aigDsmWorkflow()

        print('Running lidar Canopy Height Model workflow for '+self.YearSiteVisit)
        self.lidarChmWorkflow()

        print('Running lidar slope / aspect workflow for '+self.YearSiteVisit)
        self.lidarSlopeAspectWorkflow()

        print('Running lidar Uncertainty workflow for '+self.YearSiteVisit)
        self.lidarUncertaintyWorkflow()

        print('Running lidar classified point cloud buffer removal workflow for '+self.YearSiteVisit)
        self.classifiedTiledPointCloudWorkflow()

        print('Running lidar point density workflow for '+self.YearSiteVisit)
        self.lidarPointDensityWorkflow()

        print('Running lidar boundary workflow for '+self.YearSiteVisit)
        self.lidarBoundaryWorkflow()

        print('Running lidar validation workflow for '+self.YearSiteVisit)
        self.runLidarValidation()
        
        if self.payload.lidarSensorManufacturer == 'optech':

            print('Running lidar banding analysis workflow for '+self.YearSiteVisit)
            self.getProductFiles()

            self.bandingFile = self.LidarL1FlightlineFiles[0]

            self.bandingAnalysisWorkflow()

        print('Running lidar QA workflow for '+self.YearSiteVisit)
        self.lidarQaWorkflow()

        print('Running post lidar cleanup for '+self.YearSiteVisit)
        self.postLidarCleanup()

    def lidarPreprocessingWorkflow(self):

        print('Running lidar preprocessing for '+self.YearSiteVisit)
        self.lidarPreprocessing()

        print('Running lidar flightline overlap for '+self.YearSiteVisit)
        self.runLidarOverlap()

        print('Running lidar boundaries for '+self.YearSiteVisit)
        self.runLasBoundary()

        print('Running lidar ellipsoid correction for '+self.YearSiteVisit)
        self.applyEllipsoidCorrection()

        print('Running lidar zipping flightlines for '+self.YearSiteVisit)
        self.zipLasFiles()

    def lidarNoiseFilteringWorkflow(self):

        print('Running lidar noise classification for '+self.YearSiteVisit)
        self.runLasNoise()

        print('Running lidar USGS dem prep for '+self.YearSiteVisit)
        self.prepareUsgsDem()

        print('Running lidar noise filtering from USGS dem for '+self.YearSiteVisit)
        self.runLasNoiseFilterUsgsDem()

        print('Running lidar info for '+self.YearSiteVisit)
        self.runLasInfo()

    def lidarTilingWorkflow(self):

        print('Running lidar point cloud tiling for '+self.YearSiteVisit)
        self.tileLaz()

        print('Running lidar noise drop from tiles for '+self.YearSiteVisit)
        self.dropNoisePointsFromTiles()

        print('Running lidar noise point tile generation for '+self.YearSiteVisit)
        self.keepNoisePointsFromTiles()

        print('Running lidar tile point sorting for '+self.YearSiteVisit)
        self.sortTiles()

    def lidarIntensityImageWorkflow(self):

        print('Running lidar overage calculations for '+self.YearSiteVisit)
        self.runOverage()
        
        print('Running lidar intensity image generation preprocessing for '+self.YearSiteVisit)
        self.getIntensityRaster()

    def lidarOverlapWorkflow(self):

        print('Running lidar overlap analysis for '+self.YearSiteVisit)
        self.runOverlapAnalysis()

        print('Running lidar overlap plots for '+self.YearSiteVisit)
        self.getOverlapPlots()

    def lidarDtmWorkflow(self):

        print('Running lidar dtm generation for '+self.YearSiteVisit)
        self.getDtmRasters()

        print('Running lidar DTM retiling for '+self.YearSiteVisit)
        self.reTileDtm()

    def lidarDsmWorkflow(self):

        print('Running lidar dsm generation for '+self.YearSiteVisit)
        self.getDsmRasters()

        print('Running lidar dsm retiling for '+self.YearSiteVisit)
        self.reTileDsm()

        print('Running lidar dsm / dtm max height generation for '+self.YearSiteVisit)
        self.combineDtmAndDsm()

    def aigDsmWorkflow(self):

        print('Running lidar USGS tif generation for '+self.YearSiteVisit)
        self.getUsgsTif()

        print('Running lidar AIGDSM generation for '+self.YearSiteVisit)
        self.createAigDsm()

    def classifiedTiledPointCloudWorkflow(self):

        print('Running lidar point cloud tile buffer removal for '+self.YearSiteVisit)
        self.removeTileBuffer()

        print('Running lidar tile addition of noise points '+self.YearSiteVisit)
        self.addNoiseBackToTiles()

        print('Running lidar point cloud tile retiling '+self.YearSiteVisit)
        self.reTilePointCloudTiles()

    def bandingAnalysisWorkflow(self):

        print('Running lidar banding analysis for '+self.YearSiteVisit)
        self.getDataForBandingAnalysis()

        print('Running lidar banding analysis plots for '+self.YearSiteVisit)
        self.getPlotsForBandingAnalysis()

        print('Running lidar banding stats summary file for '+self.YearSiteVisit)
        self.writeBandingDataTextFile()

    def lidarUncertaintyWorkflow(self):

        print('Getting lidar simulated uncertainty for '+self.YearSiteVisit)
        self.calcLidarUncertainty()
        
        print('Getting lidar zip files for '+self.YearSiteVisit)
        self.zipUncertaintyLasFiles()

        print('Getting lidar uncertainty tiles for '+self.YearSiteVisit)
        self.tileUncertaintyLaz()

        print('Getting lidar uncertainty raster tiles for '+self.YearSiteVisit)
        self.createUncertaintyRaster()

    def lidarChmWorkflow(self):

        t1 = time.time()

        print('Running lidar height normalization for '+self.YearSiteVisit)
        self.normalizeHeights()

        print('Running lidar normalization buffer removal for '+self.YearSiteVisit)
        self.removeTileBufferNormalizeTiles()

        print('Running lidar standard (pitty) Canopy Height Model '+self.YearSiteVisit)
        self.gridStandardChm()

        print('Running lidar gridding for standard ground for '+self.YearSiteVisit)
        self.gridStandardChmGrnd()

        print('Running lidar point cloud thinning for '+self.YearSiteVisit)
        self.thinLazFiles()

        print('Running lidar standard CHM generation '+self.YearSiteVisit)
        self.createStandardChm()

        print('Running CHM height extraction for '+self.YearSiteVisit)
        self.extractHeightChm()

        print('Running lidar CHM levels for '+self.YearSiteVisit)
        self.createChmLevels()

        print('Running lidar CHM level integration for '+self.YearSiteVisit)
        self.combineChmLevels()

        print('Running lidar CHM retiling for '+self.YearSiteVisit)
        self.reTileChm()
    
    
    def lidarSlopeAspectWorkflow(self):

        print('Getting lidar smoothed dtm for '+self.YearSiteVisit)
        self.smoothBufferedTiles()

        print('Getting lidar slope for '+self.YearSiteVisit)
        self.createSlopeLidarProduct()

        print('Getting lidar aspect for '+self.YearSiteVisit)
        self.createAspectLidarProduct()

        print('Getting lidar slope with buffers removed for '+self.YearSiteVisit)
        self.removeSlopeBuffers()

        print('Getting lidar aspect with buffers removed for '+self.YearSiteVisit)
        self.removeAspectBuffers()

        print('Getting lidar retiled slope for '+self.YearSiteVisit)
        self.retileSlope()

        print('Getting lidar retiled aspect for '+self.YearSiteVisit)
        self.retileAspect()

    def lidarPointDensityWorkflow(self):

        print('Getting lidar point density for '+self.YearSiteVisit)
        self.runLidarPointDensity()

        print('Getting lidar triangular edges for '+self.YearSiteVisit)
        self.runLidarTriangularEdges()

    def lidarBoundaryWorkflow(self):

        print('Getting lidar flightline boundaries for '+self.YearSiteVisit)
        self.createLidarFlightlineBoundaries()

        print('Getting lidar tile boundaries for '+self.YearSiteVisit)
        self.createLidarTileBoundaries()

        print('Getting merged tile boundaries for '+self.YearSiteVisit)
        self.mergeLidarTileBoundaries()
        
    def runLidarValidation(self):
        
        print('Running lascontrol for '+self.YearSiteVisit)        
        self.runLasControl()
        
        self.getLidarValidationFromRaster()
        
    def getLidarValidationFromRaster(self):
        
        self.getProductFiles()
        if not self.lidarValidationFile:
            
            print('WARNING: No lidar validation data for site',self.YearSiteVisit)
            return
        else:   
            
            if int(self.year) >= 2024: 
            
                self.removeFilterDtmBuffers()
                
            else:
                for file in self.LidarInternalFilteredDtmFiles:
                    
                    os.makedirs(self.LidarFilterDtmProcessingDir,exist_ok=True)
                
                    shutil.copy(file,os.path.join(self.LidarFilterDtmProcessingDir,os.path.basename(file)))
                    
            self.getProductFiles()
            
            self.generateLidarFilterDtmMosaic()
            
            self.getLidarValidationData()
            
            os.makedirs(self.LidarInternalValidationDir,exist_ok=True)
            
            vertCoordsRaster = np.array(getRasterValuesAtCoordinates(os.path.join(self.LidarQAProcessingReportDir,self.YearSiteVisit+'_SmoothDtm_mosaic.tif'),self.lidarValidationHorzCoords))
            
            verticalError = np.array(vertCoordsRaster) - self.lidarValidationVertCoords
            
            verticalError[vertCoordsRaster==self.nodata] = np.nan
            
            verticalError[np.abs(verticalError-np.mean(verticalError))>3*np.nanstd(verticalError)] = np.nan
            
            outputPngFile = os.path.join(self.LidarInternalValidationDir,self.YearSiteVisit+'_lidarValidationFromRaster.png')
            
            makeBarChart(verticalError,outputFile =outputPngFile,title=None,xlabel='GPS validation Point',ylabel='Vertical Difference (m)')
            
            meanResiduals = np.nanmean(verticalError)
            
            absMeanResiduals = np.nanmean(np.abs(verticalError))
            
            stdResiduals = np.nanstd(verticalError)
            
            rmsResiduals =(np.nansum(verticalError**2)/len(verticalError[~np.isnan(verticalError)]))**0.5 
            
            print('Mean value is '+str(meanResiduals))
            
            print('Standard Deviation is '+str(stdResiduals))
            
            print('Root Mean Square is '+str(rmsResiduals))
            
            outputResidualsTextFile = os.path.join(self.LidarInternalValidationDir,self.YearSiteVisit+'_lidarResiduals.txt')
            
            outputResidualsSummaryStatsTextFile = os.path.join(self.LidarInternalValidationDir,self.YearSiteVisit+'_lidarResidualsSummaryStats.txt')
            
            np.savetxt(outputResidualsTextFile,np.hstack((np.array(self.lidarValidationHorzCoords),verticalError[:,None])),delimiter=',', header = 'Lidar residuals (m)')
                        
            np.savetxt(outputResidualsSummaryStatsTextFile,[np.array((meanResiduals,absMeanResiduals,stdResiduals,rmsResiduals))],delimiter=',', header = 'Mean Error, Absolute Mean Error, Standard Deviation Error, RMS Error')

        
    def getLidarValidationData(self):
        
        if not self.lidarValidationFile:
            print('WARNING: No lidar validation file found for site',self.YearSiteVisit)

        else:
            
            self.lidarValidationData = np.genfromtxt(self.lidarValidationFile[0], delimiter=',')
            
            self.lidarValidationHorzCoords = list(zip(self.lidarValidationData[:, 1], self.lidarValidationData[:, 2]))
            
            self.lidarValidationVertCoords = self.lidarValidationData[:,3]

    def lidarQaWorkflow(self):

        print('Getting lidar product mosaics for '+self.YearSiteVisit)
        self.generateLidarL3Mosaics()

        print('Getting lidar intensity mosaic for '+self.YearSiteVisit)
        self.generateLidarIntensityMosaics()

        print('Getting lidar point density mosaics for '+self.YearSiteVisit)
        self.generatePointDensityMosaics()

        print('Getting lidar uncertainty mosaics for '+self.YearSiteVisit)
        self.generateLidarUncertaintyMosaics()

        print('Getting lidar QA mosaics maps for '+self.YearSiteVisit)
        self.getLidarQaReportMaps()

        print('Getting lidar QA mosaics histograms for '+self.YearSiteVisit)
        self.getLidarQaReportHistograms()

        print('Getting previous Year Site Visit for QA for '+self.YearSiteVisit)
        self.getPreviousYearSiteVisit()

        if self.previousYearSiteVisit is not None:
            print('Downloading previous Year Site Visit for QA for '+self.YearSiteVisit)
            self.downloadPreviousYearSiteVisitLidar()
        
            print('Getting lidar product difference to previous Year Site Visit for '+self.YearSiteVisit)
            self.getLidarDifferencePreviousYearSiteVisit()

            print('Getting lidar report difference maps for '+self.YearSiteVisit)
            self.getLidarQaReportDifferenceMaps()
    
            print('Getting lidar report difference map histograms for '+self.YearSiteVisit)
            self.getLidarQaReportDifferenceHistograms()

        print('Making lidar L3 QA report (markdown) for '+self.YearSiteVisit)
        self.runL3LidarQA()

    def postLidarCleanup(self):
        
        print('Cleaning Lastools processing dir for '+self.YearSiteVisit)
        self.removeLastoolsFolder()
        
        print('Cleaning Las dir for for '+self.YearSiteVisit)
        self.removeLasFileDir()
            
        print('Cleaning Internal Lidar dir for '+self.YearSiteVisit)
        self.cleanInternalLidarDir()
           
        print('Cleaning QA Lidar dir for '+self.YearSiteVisit)
        self.cleanQaLidarDir()
        
        print('Cleaning lidar records for '+self.YearSiteVisit)
        self.deleteRecordsFile()
        
        print('Cleaning Lastools file lists for '+self.YearSiteVisit)
        self.cleanLasToolsFileList()
        
        print('Renaming lidar files for '+self.YearSiteVisit)  
        self.renameDiscreteLidarFiles()
        
        print('Zipping lidar tile boundary files for '+self.YearSiteVisit)  
        self.zipLidarTileBoundaryFolder()

    def waveformLidarQaWorkflow(self):
        print('Getting pulsewaves QA mosaics for '+self.YearSiteVisit)
        self.generateWaveformQaMosaics()

        print('Getting pulsewaves QA mosaics maps for '+self.YearSiteVisit)
        self.getPulsewavesQaReportMaps()

        print('Getting pulsewaves QA mosaics histograms for '+self.YearSiteVisit)
        self.getPulsewavesQaReportHistograms()
        
        print('Getting pulsewaves DTM and DSM difference mosaics for '+self.YearSiteVisit)
        self.getWaveformLidarDifferenceMosaics()
        
        print('Getting pulsewaves DTM and DSM difference maps for '+self.YearSiteVisit)
        self.getPulsewavesQaReportDifferenceMaps()

        print('Getting pulsewaves DTM and DSM difference histograms for '+self.YearSiteVisit)
        self.getPulsewavesQaReportDifferenceHistograms()

        print('Generating pulsewaves QA markdown file for '+self.YearSiteVisit)
        self.generatePulsewavesQAReport()


class missionClass:
    def __init__(self,missionId,payloadCampaign,site,skipFlightlines=False,dailyBaseDir=None,rawDir=None,doWaveform=True):
        self.missionId=missionId+'_'+payloadCampaign
        self.missionSite=site
        self.payloadCampaign = payloadCampaign
        self.year=missionId[0:4]
        self.flightday=missionId[0:10]
        self.payloadId=self.payloadCampaign[0:2]
        self.campaign=self.payloadCampaign[2:4]
        self.payload = payloadClass(self.year,self.payloadCampaign)
        self.doWaveform = doWaveform
        self.filePathToEnviProjCs = 'C:/Program Files/Harris/ENVI56/IDL88/resource/pedata/predefined/EnviPEProjcsStrings.txt'

        if dailyBaseDir is None:
            self.dailyBaseDir = os.path.join('D:',os.sep,self.year,'Daily')
        else:
            self.dailyBaseDir = dailyBaseDir

        if rawDir is None:
            self.rawDir = os.path.join('D:\\Raw\\',self.year,self.payloadId,self.campaign,self.missionId,'L0')
        else:
            self.rawDir = rawDir

        self.dl = DownloadClass()
        self.getRawMissionDirs()
        self.getDailyDirs()
        self.rinexObsExt = self.year[2:4]+'o'
        self.getMissionFiles()
        if not skipFlightlines:
            self.getMissionFlightLines()
        self.getLeapSeconds()
        self.convertToRinexPath = r"C:\Program Files (x86)\Trimble\convertToRINEX\convertToRinex.exe"
        self.gpsAntennaLookup = os.path.join(sbetPipelineBase,'res','permanent_gps_lookup.csv')
        self.template_posbat_file = "template_smartbase.posbat"
        self.reapplyBrdf = False
        
        self.getRawDbTables()
        
        self.SpectrometerProducts = ['Radiance','Reflectance','VegIndices','WaterIndices','Albedo','FPAR','LAI']
        self.vegIndices = ['ARVI','NDVI','SAVI','EVI','PRI']
        self.waterIndices = ['WBI','NMDI','NDWI','NDII','MSI']
        self.spectrometerProductQaList= ['NDVI','EVI','ARVI','PRI','SAVI','MSI','WBI','NDII','NDWI','NMDI','Albedo','LAI','fPAR']
        self.spectrometerAncillaryProductQaList = ['Dark_Dense_Vegetation_Classification','Path_Length','Smooth_Surface_Elevation','Slope','Aspect','Water_Vapor_Column','Illumination_Factor','Sky_View_Factor']
        self.spectrometerMosaicColorMaps = {"Aspect":"gist_rainbow","Elevation":"gist_earth","Slope":"jet","Water_Vapor_Column":"Blues","Illumination_Factor":"binary",
                                            "Sky_View_Factor":"cool","Path_Length":"Wistia","ReflectanceRms":"Spectral","MaxDiffWavelength":"hsv","MaxDiff":"inferno"} 

    def getDailyDirs(self):

        self.sbetDir=os.path.join(self.dailyBaseDir,self.missionId)
        self.sbetL1Dir = os.path.join(self.sbetDir,'L1')
        self.sbetProcessingDir = os.path.join(self.sbetDir,'Processing')
        self.sbetProcessingGpsImuDir = os.path.join(self.sbetProcessingDir,'GPSIMU')
        self.sbetGpsImuDir = os.path.join(self.sbetL1Dir,'GPSIMU')
        self.sbetBasestationDir = os.path.join(self.sbetL1Dir,'Basestation')
        self.sbetBasestationRinexDir = os.path.join(self.sbetBasestationDir,'Rinex')
        self.sbetBasestationTrimbleDir = os.path.join(self.sbetBasestationDir,'Trimble')
        self.sbetQaDir = os.path.join(self.sbetDir,'QA')
        self.sbetQaGpsImuDir = os.path.join(self.sbetQaDir,'GPSIMU')

    def getRawMissionDirs(self):

        self.groundGpsDir = os.path.join('D:\\','GPS')

        self.rawAncillaryDir=os.path.join(self.rawDir,'Ancillary')
        self.rawFlightLogDir=os.path.join(self.rawAncillaryDir,'FlightLogs')
        self.rawFlightPlansDir = os.path.join(self.rawAncillaryDir,'FlightPlans')
        self.rawFlightPlansKmlDir = os.path.join(self.rawFlightPlansDir,'KMLs')
        self.rawFlightPlansXmlDir = os.path.join(self.rawFlightPlansDir,'XMLs')
        self.flightDbDir = os.path.join(self.rawAncillaryDir,self.payloadId+'FlightLogDB')
        self.flightTrackingDir = os.path.join(self.rawAncillaryDir,self.payloadId+'FlightLogDB','FlightTracking')

        self.rawWaveformDir=os.path.join(self.rawDir,'WaveformLidar')
        self.rawCameraDir = os.path.join(self.rawDir,'Camera')
        self.rawSpectrometerDir = os.path.join(self.rawDir,'Spectrometer')
        if self.payloadId == 'P3' or (self.payloadId == 'P2' and int(self.year) > 2024):
            self.rawGpsDir = os.path.join(self.rawDir,'GPSIMU')
        else:
            self.rawGpsDir = os.path.join(self.rawDir,'GPSIMU','POS')

        self.rawQaChecksDir = os.path.join('D:\\Raw\\',self.year,self.payloadId,self.campaign,self.missionId,'QA_checks')
        currentFile = os.path.abspath(os.path.join(__file__))
        self.scriptsDir = os.path.join(currentFile.lower().split('gold_pipeline')[0], 'Gold_Pipeline', 'ProcessingPipelines', 'NIS', 'res', 'Scripts')

        if self.payloadId=='P3':
            self.rawDiscreteLidarDir=os.path.join(self.rawWaveformDir,'03_RIEGL_RAW','01_SDF','Q780')
            self.rawWaveformLidarDir = self.rawDiscreteLidarDir
        else:
            self.rawDiscreteLidarDir=os.path.join(self.rawDir,'DiscreteLidar')
                                     
            self.rawWaveformLidarDir=os.path.join(self.rawWaveformDir,'DISK1')
            
    def getProductDirs(self,productBaseDir):

        self.productBaseDir = productBaseDir
        self.L1ProductDir =  os.path.join(self.productBaseDir,'L1')
        self.L2ProductDir =  os.path.join(self.productBaseDir,'L2')
        self.internalDir =  os.path.join(self.productBaseDir,'Internal')
        self.metadataProductDir =  os.path.join(self.productBaseDir,'Metadata')
        self.ProcessingProductDir =  os.path.join(self.productBaseDir,'Processing')
        self.QaDir =  os.path.join(self.productBaseDir,'QA')
        
        self.L1LidarProductDir =  os.path.join(self.L1ProductDir,'L1','DiscreteLidar')
        self.L1LidarLasDir =  os.path.join(self.productBaseDir,'L1','DiscreteLidar','Las')
        self.L1LidarExtraLasDir =  os.path.join(self.productBaseDir,'L1','DiscreteLidar','Las','extra')
        
        self.L1SpectrometerProductDir = os.path.join(self.L1ProductDir,'Spectrometer')
        self.SpectrometerL1RadianceProcessingDir=os.path.join(self.L1SpectrometerProductDir,'Radiance',self.flightday)
        self.SpectrometerL1ReflectanceDir=os.path.join(self.L1SpectrometerProductDir,'ReflectanceH5',self.flightday)
        self.SpectrometerL1DirectionalReflectanceDir=os.path.join(self.L1SpectrometerProductDir,'DirectionalReflectanceH5',self.flightday)
        self.SpectrometerL1BidirectionalReflectanceDir=os.path.join(self.L1SpectrometerProductDir,'BidirectionalReflectanceH5',self.flightday)
        self.SpectrometerL1RadianceDir=os.path.join(self.L1SpectrometerProductDir,'RadianceH5',self.flightday)
        self.SpectrometerL1ReflectanceProcessingDir=os.path.join(self.L1SpectrometerProductDir,'Reflectance',self.flightday)

        self.L2SpectrometerProductDir = os.path.join(self.L2ProductDir,'Spectrometer')
        self.SpectrometerL2AlbedoDir=os.path.join(self.L2SpectrometerProductDir,'Albedo',self.flightday)
        self.SpectrometerL2BiomassDir=os.path.join(self.L2SpectrometerProductDir,'Biomass',self.flightday)
        self.SpectrometerL2FparDir=os.path.join(self.L2SpectrometerProductDir,'FPAR',self.flightday)
        self.SpectrometerL2LaiDir=os.path.join(self.L2SpectrometerProductDir,'LAI',self.flightday)
        self.SpectrometerL2VegIndicesDir=os.path.join(self.L2SpectrometerProductDir,'VegIndices',self.flightday)
        self.SpectrometerL2WaterIndicesDir=os.path.join(self.L2SpectrometerProductDir,'WaterIndices',self.flightday)

        self.QaSpectrometerDir = os.path.join(self.QaDir,'Spectrometer')
        self.SpectrometerQaRadianceDir = os.path.join(self.QaSpectrometerDir,'Radiance',self.flightday)
        self.SpectrometerL2QADir=os.path.join(self.QaSpectrometerDir,'L2QA',self.flightday)
        self.SpectrometerL2QaReflectanceDir=os.path.join(self.QaSpectrometerDir,'SampleReflectance',self.flightday)
        self.SpectrometerL2QaReflectanceProductDir=os.path.join(self.QaSpectrometerDir,'L2Hists',self.flightday)
        self.SpectrometerL1QaPcaDir=os.path.join(self.QaSpectrometerDir,'PCA',self.flightday)
        self.SpectrometerL1QaMnfDir=os.path.join(self.QaSpectrometerDir,'MNF',self.flightday)
        self.SpectrometerOrthoRadianceQaDir=r=os.path.join(self.QaSpectrometerDir,'OrthoRadiance',self.flightday)
        self.SpectrometerRadianceQaDir=os.path.join(self.QaSpectrometerDir,'L1QA',self.flightday)
        self.SpectrometerMetadataRgbRadianceTifsDir = os.path.join(self.QaSpectrometerDir,'RgbRadiance',self.flightday)
        self.SpectrometerQaFlightlineDifferenceDir = os.path.join(self.QaSpectrometerDir,'ReflectanceDifferenceFlightline',self.flightday)
        self.SpectrometerL2QaFlightlineRmsDir = os.path.join(self.QaSpectrometerDir ,'RmsFlightline',self.flightday)
        self.SpectrometerL2QaFlightlineMaxIndicesDir = os.path.join(self.QaSpectrometerDir,'MaxIndicesFlightline',self.flightday)
        self.SpectrometerL2QaFlightlineMaxDiffDir =  os.path.join(self.QaSpectrometerDir,'MaxDiffFlightline',self.flightday)
        self.SpectrometerQaTempAncillaryRastersDir = os.path.join(self.QaSpectrometerDir,'ReflectanceAncillaryRastersFlightline',self.flightday)
        
        self.metadataSpectrometerDir = os.path.join(self.metadataProductDir,'Spectrometer')
        self.SpectrometerMetadataRgbTifsDir=os.path.join(self.metadataSpectrometerDir,'RGBTifs',self.flightday)
        self.SpectrometerMetadataKmlsDir =os.path.join(self.metadataSpectrometerDir,'FlightlineBoundary',self.flightday)
        
        self.metadataLidarDir = os.path.join(self.metadataProductDir,'DiscreteLidar')
        self.metadataLidarUncertaintyDir = os.path.join(self.metadataLidarDir,'Uncertainty')
        
        self.LidarInternalLazStandardDir = os.path.join(self.internalDir,'DiscreteLidar','LazStandard')

        self.lidarProcessingDir = os.path.join(self.ProcessingProductDir,'DiscreteLidar')
        self.lidarProcessingLastoolsDir = os.path.join(self.lidarProcessingDir,'LASTOOLS')
        self.lidarProcessingLastoolsUncertaintyDir = os.path.join(self.lidarProcessingLastoolsDir,'UncertaintyLas')
        
        self.lmsProcessingDir = os.path.join(self.lidarProcessingDir,'LMS')

        # Riegl L1 Lidar processing sub-directories
        if self.payloadId=='P3':
            self.wavexInputsDir = os.path.join(self.lmsProcessingDir,'WavEx_Inputs')
            self.wavexOutputDir = os.path.join(self.lmsProcessingDir,'WavEx_Outputs')
            self.lidarAtmosphericDir = os.path.join(self.wavexInputsDir,'Atmos')
        
        # Optech Galaxy L1 Lidar processing sub-directories
        elif self.payloadId=='P1' and int(self.year) >= 2021:
            self.lmsMissionProjectDir = os.path.join(self.lmsProcessingDir,'LMS_' + self.flightday)
            self.lmsMissionDataDir = os.path.join(self.lmsMissionProjectDir,self.flightday)
            self.lmsFlightPlanDir = os.path.join(self.lmsMissionProjectDir,'FlightPlan')
            self.lmsInstrumentFileDir = os.path.join(self.lmsMissionProjectDir,'Galaxy5060445') # eg. Galaxy5060445 self.payload.lidarSensorModel + 
            self.lmsReferenceFrameDir = os.path.join(self.lmsMissionProjectDir,'ReferenceFrames')

        self.spectrometerProcessingDir = os.path.join(self.ProcessingProductDir,'Spectrometer')
        self.brdfInputsTempDir = os.path.join(self.spectrometerProcessingDir,'tempBrdfInputs',self.flightday)
        self.spectrallyResampledDir = os.path.join(self.spectrometerProcessingDir,'SpectrallyResampled',self.flightday)
        self.CameraL1Dir=os.path.join(self.L1ProductDir,'Camera')
        self.CameraL1ImagesDir=os.path.join(self.CameraL1Dir,'Images')

        self.cameraProcessingDir = os.path.join(self.ProcessingProductDir,'Camera')
        self.cameraOrthoProcessingDir = os.path.join(self.ProcessingProductDir,'Camera','Orthorectification',self.flightday)

    def getMissionFiles(self):

        self.sbetTrajectoryFile = os.path.join(self.sbetGpsImuDir,'sbet_'+self.flightday+'.out')
        self.sbetTrajectoryErrorFile = os.path.join(self.sbetGpsImuDir,'smrmsg_'+self.flightday+'.out')
        self.sbetTrajectoryMetadataFile = os.path.join(self.sbetGpsImuDir,'piinkaru_'+self.flightday+'.out')
        self.sbetTrajectoryCalibrationFile = os.path.join(self.sbetGpsImuDir,'iincal_'+self.flightday+'.out')
 
        self.metadataXml = collectFilesInPath(self.rawAncillaryDir,Ext='metadata.xml')
        if self.metadataXml:
            self.metadataXml = self.metadataXml[0]
        self.spectrometerFlightLog = collectFilesInPath(self.rawFlightLogDir,Ext='NIS.csv')
        if self.spectrometerFlightLog:
            self.spectrometerFlightLog = self.spectrometerFlightLog[0]
        self.lidarFlightLog = collectFilesInPath(self.rawFlightLogDir,Ext='Lidar.csv')
        if self.lidarFlightLog:
            self.lidarFlightLog = self.lidarFlightLog[0]

        if self.payloadId=='P3':
            self.rawDiscreteLidarFiles = collectFilesInPath(self.rawDiscreteLidarDir,Ext='.sdf')
            self.rawWaveformLidarFiles = self.rawDiscreteLidarFiles
        else:
            self.rawDiscreteLidarFiles = collectFilesInPath(self.rawDiscreteLidarDir,Ext='.range')

            if int(self.year) <= 2015:
                self.rawWaveformLidarFiles = collectFilesInPath(self.rawWaveformLidarDir,Ext='.ix2')
                self.rawWaveformLidarFiles = collectFilesInPath(self.rawWaveformLidarDir,Ext='.df2')
            elif int(self.year) <= 2023 and int(self.year) > 2015:
                self.rawWaveformLidarFiles = collectFilesInPath(self.rawWaveformLidarDir,Ext='.ix3')
                self.rawWaveformLidarFiles = collectFilesInPath(self.rawWaveformLidarDir,Ext='.df3')
            elif int(self.year) >= 2024:
                self.rawWaveformLidarFiles = collectFilesInPath(self.rawWaveformLidarDir,Ext='.ix4')
                self.rawWaveformLidarFiles = collectFilesInPath(self.rawWaveformLidarDir,Ext='.df4')
            
            self.optechFlightPlanFiles = collectFilesInPath(self.rawFlightPlansXmlDir,Ext='.xml')
            # self.optechInstrumentFiles = self.getOptechInstrumentFiles()
            # self.optechGeoReferenceFiles = self.getOptechGeoReferenceFiles()

        self.rawCameraFiles = collectFilesInPathIncludingSubfolders(self.rawCameraDir,Ext='.IIQ')
        self.rawPosFiles = collectFilesInPath(self.rawGpsDir)
        self.rawSpectrometerDataDirectories = collectDirectoriesInPath(self.rawSpectrometerDir)
        self.rawTrimbleGpsFiles = collectFilesInPath(self.sbetBasestationTrimbleDir)
        self.rawRinexGpsFiles = collectFilesInPath(self.sbetBasestationRinexDir)
        self.rawRinexGpsObsFiles = collectFilesInPath(self.sbetBasestationRinexDir,self.rinexObsExt)

        self.scriptsFile = os.path.join(self.scriptsDir,'scripts.txt')
        for directory in self.rawSpectrometerDataDirectories:
            if 'Logs' in directory:
                self.rawSpectrometerDataDirectories.remove(directory)

        self.posbatFile = os.path.join(self.sbetProcessingGpsImuDir,self.missionId+'.posbat')
        self.rppFilesTemp = collectFilesInPath(self.rawWaveformDir,'.rpp')
        self.rppFiles = []
        for rppFile in self.rppFilesTemp:
            if not rppFile.startswith('.') and self.missionSite.site in rppFile:
                self.rppFiles.append(rppFile)

        self.dbFile = collectFilesInPath(self.flightDbDir,'.accdb')
        if self.dbFile:
            self.dbFile = self.dbFile[0]
            
        self.logFiles = collectFilesInPath(self.rawFlightLogDir)

    def getRawDbTables(self):

        if self.dbFile:
            tableNames = ['tblFlightLog','tblFlightLogSpectrometer','tblMission','tblFlightLogLidar','lkpSite']
            
            if int(self.year) > 2021:
                
                
                flightLogDfs = accessDbToDataframes(self.dbFile,tableNames)
                
                self.flightLogDf = flightLogDfs[0]
                self.spectrometerLogDf = flightLogDfs[1]
                self.missionDf = flightLogDfs[2]
                self.lidarLog = flightLogDfs[3]
                self.siteLookupDf = flightLogDfs[4]
                self.siteLookupDf = self.siteLookupDf.dropna(subset=['SiteName'])

    def getProductFiles(self):

        self.SpectrometerL1ReflectanceFiles = collectFilesInPath(self.SpectrometerL1ReflectanceDir,'.h5')
        
        self.SpectrometerL1DirectionalReflectanceFiles = collectFilesInPath(self.SpectrometerL1DirectionalReflectanceDir,'.h5')
        self.SpectrometerL1BidirectionalReflectanceFiles = collectFilesInPath(self.SpectrometerL1BidirectionalReflectanceDir,'.h5')
        
        self.L1LidarLasFiles =  collectFilesInPath(self.L1LidarLasDir ,'.las') + collectFilesInPath(self.L1LidarExtraLasDir ,'.las') 
        self.L1LidarLasFiles = [file for file in self.L1LidarLasFiles if self.flightday in file]
        
        self.SpectrometerL1RadianceFiles = collectFilesInPath(self.SpectrometerL1RadianceDir,'.h5')
        self.SpectrometerL1RadianceProcessingFiles = collectFilesInPath(self.SpectrometerL1RadianceProcessingDir)
        self.SpectrometerL1ReflectanceProcessingFiles = collectFilesInPath(self.SpectrometerL1ReflectanceProcessingDir)
        self.SpectrometerL2AlbedoFiles = collectFilesInPath(self.SpectrometerL2AlbedoDir,'albedo.tif')
        self.SpectrometerL2BiomassFiles = collectFilesInPath(self.SpectrometerL2BiomassDir,'.tif')
        self.SpectrometerL2FparFiles = collectFilesInPath(self.SpectrometerL2FparDir,'fPAR.tif')
        self.SpectrometerL2FparErrorFiles = collectFilesInPath(self.SpectrometerL2FparDir,'error.tif')
        self.SpectrometerL2LaiFiles = collectFilesInPath(self.SpectrometerL2LaiDir,'LAI.tif')
        self.SpectrometerL2LaiErrorFiles = collectFilesInPath(self.SpectrometerL2LaiDir,'error.tif')
        self.SpectrometerL2VegIndicesZipFiles = collectFilesInPath(self.SpectrometerL2VegIndicesDir,'.zip')
        self.SpectrometerL2VegIndicesTifFiles = collectFilesInPath(self.SpectrometerL2VegIndicesDir,'.tif')
        self.SpectrometerL2VegIndicesErrorTifFiles = collectFilesInPath(self.SpectrometerL2VegIndicesDir,'error.tif')
        self.SpectrometerL2WaterIndicesTifFiles = collectFilesInPath(self.SpectrometerL2WaterIndicesDir,'.tif')
        self.SpectrometerL2WaterIndicesErrorTifFiles = collectFilesInPath(self.SpectrometerL2WaterIndicesDir,'error.tif')
        self.SpectrometerL2WaterIndicesZipFiles = collectFilesInPath(self.SpectrometerL2WaterIndicesDir,'.zip')
        self.SpectrometerL2QAFiles = collectFilesInPath(self.SpectrometerL2QADir)
        self.CameraL1ImagesFiles = collectFilesInPath(self.CameraL1ImagesDir,'.tif')
        self.SpectrometerMetadataRgbFiles = collectFilesInPath(self.SpectrometerMetadataRgbTifsDir,'.tif')
        self.SpectrometerMetadataRgbRadianceFiles = collectFilesInPath(self.SpectrometerMetadataRgbRadianceTifsDir,'.tif')
        self.SpectrometerL2QaWaterMaskFiles = collectFilesInPath(self.SpectrometerL2QaReflectanceDir,'.tif')
        
        self.SpectrometerL2QaRmsFiles = collectFilesInPath(self.SpectrometerL2QaFlightlineRmsDir,'.tif')
        self.SpectrometerL2QaMaxWavelengthFiles = collectFilesInPath(self.SpectrometerL2QaFlightlineMaxIndicesDir,'.tif')
        self.SpectrometerL2QaMaxDiffFiles = collectFilesInPath(self.SpectrometerL2QaFlightlineMaxDiffDir,'.tif')
        self.SpectrometerL2HistsFiles =  collectFilesInPath(self.SpectrometerL2QaReflectanceProductDir,'.tif')
        self.SpectrometerOrthoRadianceQaFiles = collectFilesInPath(self.SpectrometerOrthoRadianceQaDir,'.tif')

        self.posPacProcessedFiles = collectFilesInPathIncludingSubfolders(self.sbetProcessingGpsImuDir,Ext='out')
        
        self.orthoProcessingParamsFile = os.path.join(self.cameraOrthoProcessingDir,'dms_ortho_parameters_'+self.flightday+'.txt')
        
        self.orthoProcessingFiles = collectFilesInPath(self.cameraOrthoProcessingDir,'')
        
        self.SpectrometerQaFlightlineDifferenceFiles = collectFilesInPath(self.SpectrometerQaFlightlineDifferenceDir,'.tif')
        
        self.brdfInputsTempFiles =  collectFilesInPath(self.brdfInputsTempDir) 
        

                
        self.rgbMosaicFile = collectFilesInPath(self.SpectrometerL2QaReflectanceProductDir,'_RGB_mosaic.tif')
        if self.rgbMosaicFile:
            self.rgbMosaicFile=self.rgbMosaicFile[0]
        
        self.nirgbMosaicFile = collectFilesInPath(self.SpectrometerL2QaReflectanceProductDir,'_NIRGB_mosaic.tif')
        if self.nirgbMosaicFile:
            self.nirgbMosaicFile=self.nirgbMosaicFile[0]
            
        self.waterMaskMosaicFile = collectFilesInPath(self.SpectrometerL2QaReflectanceProductDir,'_water_mask_mosaic.tif')
        if self.waterMaskMosaicFile:
            self.waterMaskMosaicFile=self.waterMaskMosaicFile[0]
            
        self.DdvMosaicFile = collectFilesInPath(self.SpectrometerL2QaReflectanceProductDir,'_Dark_Dense_Vegetation_Classification_mosaic.tif')
        if self.DdvMosaicFile:
            self.DdvMosaicFile=self.DdvMosaicFile[0]
        
        self.rmsMosaicFile = collectFilesInPath(self.SpectrometerL2QaReflectanceProductDir,'ReflectanceRMS_mosaic.tif')
        if self.rmsMosaicFile:
            self.rmsMosaicFile=self.rmsMosaicFile[0]
    
        self.maxDiffMosaicFile = collectFilesInPath(self.SpectrometerL2QaReflectanceProductDir,'MaxDiff_mosaic.tif')
        if self.maxDiffMosaicFile:
            self.maxDiffMosaicFile=self.maxDiffMosaicFile[0]
            
        self.maxDiffWavelengthMosaicFile = collectFilesInPath(self.SpectrometerL2QaReflectanceProductDir,'MaxDiffWavelength_mosaic.tif')
        if self.maxDiffWavelengthMosaicFile:
            self.maxDiffWavelengthMosaicFile=self.maxDiffWavelengthMosaicFile[0]
        
        self.redMosaicFile = collectFilesInPath(self.SpectrometerL2QaReflectanceProductDir,'_red_mosaic.tif')
        if self.redMosaicFile:
            self.redMosaicFile=self.redMosaicFile[0]
        
        self.greenMosaicFile = collectFilesInPath(self.SpectrometerL2QaReflectanceProductDir,'_green_mosaic.tif')
        if self.greenMosaicFile:
            self.greenMosaicFile=self.greenMosaicFile[0]
    
        self.blueMosaicFile = collectFilesInPath(self.SpectrometerL2QaReflectanceProductDir,'_blue_mosaic.tif')
        if self.blueMosaicFile:
            self.blueMosaicFile=self.blueMosaicFile[0]
        
        self.nirMosaicFile = collectFilesInPath(self.SpectrometerL2QaReflectanceProductDir,'_NIR_mosaic.tif')
        if self.nirMosaicFile:
            self.nirMosaicFile=self.nirMosaicFile[0]

    def getGpsBasestations(self):

        gpsBasestations = []

        if self.dbFile:
        
            missionId = self.missionDf[self.missionDf['MissionID'] == self.flightday]['ID'].item()
            flightIds = self.flightLogDf[self.flightLogDf['MID'] == missionId]['ID']
            
            for flightId in flightIds:
                gpsBasestationsUsed = self.spectrometerLogDf[self.spectrometerLogDf['FLID'] == flightId]['GPSStationsUsed'].item()
                                            
                if gpsBasestationsUsed == '':
                    continue
                                                                
                
                if ',' in gpsBasestationsUsed:
                    gpsBasestationsUsedSplit = gpsBasestationsUsed.split(',')
                elif ';' in gpsBasestationsUsed:
                    gpsBasestationsUsedSplit = gpsBasestationsUsed.split(';')
                else:
                    gpsBasestationsUsedSplit = [gpsBasestationsUsed]
                    
                if gpsBasestationsUsedSplit:
                    for gpsBasestationUsedSplit in gpsBasestationsUsedSplit:
                        gpsBasestations.append(gpsBasestationUsedSplit)

        if gpsBasestations:
            self.gpsBasestations = list(set(gpsBasestations))
        else:
            self.gpsBasestations = None

    def downloadGpsBasestations(self):

        os.makedirs(self.sbetBasestationTrimbleDir,exist_ok=True)
        if self.gpsBasestations is not None:
            for gpsStation in self.gpsBasestations:
    
                if gpsStation.startswith('0'):
    
                    gpsStation = 'AOP_GPS'+gpsStation[0:2]
    
                self.dl.download_gps(gpsStation,self.flightday,self.sbetBasestationTrimbleDir,all_dates=False)

    def convertTrimble2Rinex(self):

        self.getMissionFiles()

        os.makedirs(self.sbetBasestationRinexDir,exist_ok=True)

        for trimbleFile in self.rawTrimbleGpsFiles:
            rinexPath = os.path.join(self.sbetBasestationRinexDir,os.path.splitext(os.path.basename(trimbleFile))[0]+'.'+self.rinexObsExt)

            if os.path.exists(rinexPath):
                continue
            else:
                check_call([self.convertToRinexPath, trimbleFile, "-p", self.sbetBasestationRinexDir])
        self.getMissionFiles()
        [os.rename(f, f.replace('AOP_GPS', 'AG')) for f in self.rawRinexGpsFiles]
        self.getMissionFiles()
        [os.rename(f, f.replace('AOP_', '')) for f in self.rawRinexGpsFiles]

    def addAntennaInfoToRinex(self):

        self.getMissionFiles()

        perm_gps_antennas=pd.read_csv(self.gpsAntennaLookup)
        for rinexObsFile in self.rawRinexGpsObsFiles:

            if os.path.basename(rinexObsFile).startswith('AG'):
                continue

            site=os.path.basename(rinexObsFile).split('_')[0]
            print(site)
            antenna=perm_gps_antennas.loc[perm_gps_antennas['Site']==site,'ANT_TYPE'].iloc[0]
            print(site,': ',antenna)
            for line in fileinput.input([rinexObsFile], inplace=1):
                if fileinput.filelineno() <= 10:
                    line = line.replace('UNKNOWN_EXT',perm_gps_antennas.loc[perm_gps_antennas['Site']==site,'ANT_TYPE'].iloc[0])
                print(line.rstrip('\n'))

    def generatePosbatFile(self):

        os.makedirs(self.sbetProcessingGpsImuDir,exist_ok=True)
        shutil.copy2(os.path.join(sbetPipelineBase,'res',self.template_posbat_file),self.sbetProcessingGpsImuDir)

        if os.path.isfile(self.posbatFile):
            os.remove(self.posbatFile)

        os.rename(os.path.join(self.sbetProcessingGpsImuDir,self.template_posbat_file),self.posbatFile)

        print('First POS file:',self.rawPosFiles[0])
        print('Last POS file:',self.rawPosFiles[-1])

        tree = et.parse(self.posbatFile) #; print(tree)
    #    root = tree.getroot()

        for node in tree.xpath("//Name"):
            #print("Template Project Name:",node.text)
            node.text = self.missionId+'_posbat'
            print("Project Name:",node.text)
        for node in tree.xpath("//Kernel"):
            #print("Template Kernel (POSPac Mission) Name:",node.text)
            node.text = self.flightday
            print("Kernel Name:",node.text)
        for node in tree.xpath("//FirstPosFile"):
            #print("Template FirstPosFile:",node.text)
            node.text = self.rawPosFiles[0] #need to account for sets of POS files
            print("FirstPosFile:",node.text)
        for node in tree.xpath("//LastPosFile"):
            #print("Template LastPosFile:",node.text)
            node.text = self.rawPosFiles[-1] #need to account for sets of POS files
            print("LastPosFile:",node.text)

        for node in tree.xpath("//PriGNSSLeverX"):
            node.text = self.payload.sbetLeverArm[0]
        for node in tree.xpath("//PriGNSSLeverY"):
            node.text = self.payload.sbetLeverArm[1]
        for node in tree.xpath("//PriGNSSLeverZ"):
            node.text = self.payload.sbetLeverArm[2]

    # add single base list (if there are RINEX files)
        print('Adding external basestations')

        if self.rawRinexGpsObsFiles:
            tree.xpath("//Process")[0].append(et.Element("SingleBaseList"))
            i=0
            for rinexFile in self.rawRinexGpsObsFiles:
                rinexCode=os.path.basename(rinexFile).split('_')[0]
                tree.xpath("//SingleBaseList")[0].append(et.Element("BatchBaseStationCoordinateInfo"))
                et.SubElement(tree.xpath("//BatchBaseStationCoordinateInfo")[i],"StationID").text=rinexCode
                et.SubElement(tree.xpath("//BatchBaseStationCoordinateInfo")[i],"DataFile").text=rinexFile
                et.SubElement(tree.xpath("//BatchBaseStationCoordinateInfo")[i],"UseRtxPos").text='true'
                et.SubElement(tree.xpath("//BatchBaseStationCoordinateInfo")[i],"CoordinateQuality").text='SURVEY_ACC'
                i+=1
        tree.write(self.posbatFile)

        for node in tree.xpath("//ReportFile"):
            #print("Template Project Name:",node.text)
            node.text = os.path.join(self.sbetQaDir,self.missionId+'_POSPacQCreport.pdf')

    def runPOSPac(self):

        currentDir = os.getcwd()
        os.chdir(self.sbetProcessingGpsImuDir)
        check_call('C:\\Program Files\\Applanix\\POSPac MMS 9.2\\POSPac.exe -b ' + os.path.basename(self.posbatFile))
        os.chdir(currentDir)

    def prepQaFigures(self):

        reportDir = os.path.join(self.sbetProcessingGpsImuDir,self.missionId,self.missionId+'_posbat',self.flightday,'Report')
        pospacPdf = os.path.join(reportDir,'report_'+self.flightday+'.pdf')
        outPath = os.path.join(reportDir,'pospac_figs')
        os.makedirs(outPath,exist_ok=True)
        pospacReport = PyPDF2.PdfFileReader(open(pospacPdf, "rb"))

        print('Extracting images from qa report pdf')
        fig=1
        for p in range(pospacReport.getNumPages()):
            page = pospacReport.getPage(p)
    #         print(p)
            if '/XObject' in page['/Resources']:
                xObject = page['/Resources']['/XObject'].getObject()
                for obj in xObject:
                    if xObject[obj]['/Subtype'] == '/Image':
                        size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                        data = xObject[obj].getData()
                        if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                            mode = "RGB"
                        else:
                            mode = "P"
                        if '/Filter' in xObject[obj]:
                            if xObject[obj]['/Filter'] == '/FlateDecode':
                                img = Image.frombytes(mode, size, data)
                                img.save(os.path.join(outPath,str(fig) + ".png"))
                                fig+=1
                            else:
                                print("No image found.")
        
        os.makedirs(self.sbetQaGpsImuDir,exist_ok=True)
        shutil.copy2(pospacPdf,os.path.join(self.sbetQaGpsImuDir,'report_'+self.flightday+'.pdf'))
        shutil.copy2(os.path.join(outPath,"1.png"),os.path.join(self.sbetQaGpsImuDir,self.missionId+'_L1_sats.png'))
        shutil.copy2(os.path.join(outPath,"4.png"),os.path.join(self.sbetQaGpsImuDir,self.missionId+'_L2_sats.png'))
        shutil.copy2(os.path.join(outPath,"30.png"),os.path.join(self.sbetQaGpsImuDir,self.missionId+'_Fwd_Rev_Sep.png'))

    def moveSbetOutputs(self):
        
        self.getProductFiles()
        os.makedirs(self.sbetGpsImuDir,exist_ok=True)
        
        for outFile in self.posPacProcessedFiles:
        
            if os.path.basename(outFile).startswith('sbet') or os.path.basename(outFile).startswith('smrmsg') or os.path.basename(outFile).startswith('iincal') or os.path.basename(outFile).startswith('piinkaru'):
                
                shutil.move(outFile,os.path.join(self.sbetGpsImuDir,os.path.basename(outFile)))
         
        self.getProductFiles()           

    def getSbet(self):
        
        self.sbet = applanixSbetClass(self.sbetTrajectoryFile,self.sbetTrajectoryErrorFile,self.sbetTrajectoryMetadataFile,self.sbetTrajectoryCalibrationFile)
        self.sbet.getAllSbetData()
        self.sbet.transformSbetToProjection(self.missionSite.epsg)

    def getLeapSeconds(self):

        self.leapSeconds = getLeapSeconds(int(self.flightday[0:8]))

# WavEx methods (Riegl Lidar L1 Processing)
    def getRppRecords(self):

        self.sdfFilesToProcess = []
        sdfFiles = []
        records = []
        lasdatas = []
        self.record_dicts = []
        os.makedirs(self.productBaseDir,exist_ok=True)

        for rppFile in self.rppFiles:

            rppTree = et.parse(rppFile)
            self.riAlsVersion = rppTree.xpath('/document/header/docinfo/creator/@data')[0]
            sdfFilesRpp=rppTree.xpath('//object[@kind="sdf-file"]/@name')

            #select sdf files from the .rpp file that are in the raw sdf directory for this mission
            filesToProcess = [os.path.join(self.rawWaveformLidarDir,file+'.sdf') for file in sdfFilesRpp if file+'.sdf' in os.listdir(self.rawDiscreteLidarDir)]
            
            if filesToProcess:
                self.sdfFilesToProcess.append(filesToProcess)
            else:
                self.sdfFilesToProcess.append(self.rawDiscreteLidarFiles)

            record = rppTree.xpath('//object[@kind="record"]/@name')
            lasdata = rppTree.xpath('//object[@kind="lasdata"]/@name')
            self.record_dicts.append(dict(zip(record,lasdata)))
            #print(record_dict)

            for record_dict in self.record_dicts:

                with open(os.path.join(self.productBaseDir,self.flightday+'_'+os.path.splitext(os.path.basename(rppFile))[0]+'_record_names.csv'), 'w', newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    for key, value in record_dict.items():
                        writer.writerow([key, value])
                    csv_file.close()

    def getWavexProcessingFiles(self):

        calibFileDir = os.path.join(lidarRieglResDir,'InstrumentFiles',self.year)

        self.calibFile = [os.path.join(calibFileDir,calFile) for calFile in os.listdir(calibFileDir) if 'bayes_cal' in calFile][0]

        self.wktFile = os.path.join(lidarRieglResDir,'WKT_Files',self.missionSite.epsg+'.prj')
        self.wavexProcessingHeight = '1000' #assume height above ground level is 1000m (nominal altitude)

    def createWavexAtmFiles(self):

        os.makedirs(self.lidarAtmosphericDir,exist_ok=True)

        for rppFile in self.rppFiles:
            try:
                success = make_atm_files(self.flightday,rppFile,self.lidarAtmosphericDir)
            except:
                success = False
            if success:
                continue
            else:
                print('Could not generate atmospheric files from telemetry database, attempting to read info from the flight log database.')
                atmDf = self.lidarLog.merge(self.flightLogDf[['ID','MID']],left_on='FLID',right_on='ID')
                tempFlightLog = self.flightLogDf[['ID','CampaignID','MID']]
                if self.payload.payload=='P3':
                    atmDf['CabinAltitude_m'] = np.round(atmDf['CabinAltitude']/3.2808,1)
                    
                    dbMID = self.missionDf[self.missionDf['MissionID']==self.flightday]
                    
                    print(atmDf.loc[atmDf['MID'] == dbMID['ID'].values[0]][['MID','CabinAltitude_m','PilotOutsideAirTempC','CabinRelativeHumidity','CabinPressureMbar']])
                    atm_vals = atmDf.loc[atmDf['MID'] == dbMID['ID'].values[0]][['CabinAltitude_m','PilotOutsideAirTempC','CabinRelativeHumidity','CabinPressureMbar']]
                else:
                    print('WARNING: Payload is not P3, code currently only set up for Payload 3')
                    # atmDf['Time'] = 0
                    # print(atmDf.loc[atmDf['missionID'] == self.missionId][['missionID','Time','PilotOutsideAirTempC','CabinPressureMbar']])
                    # atm_vals = atmDdf.loc[atm_df['missionID'] == self.missionId][['Time','PilotOutsideAirTempC','CabinPressureMbar']]
                # TODO: MAKE THIS A TRY/EXCEPT?
                atm_file = os.path.join(self.lidarAtmosphericDir,self.flightday+'_flightlog.txt')
                with open(atm_file, 'w+') as file:
                    atm_vals.to_csv(file, header=False, index=False,sep = ' ')

    def runWavex(self):

        os.makedirs(self.wavexOutputDir,exist_ok=True)

        for sdfFiles in self.sdfFilesToProcess:

            for sdfFile in sdfFiles: #use [:2] for testing on first two

                sdfFileBasename = os.path.basename(sdfFile)

                if os.path.isfile(os.path.join(self.lidarAtmosphericDir,sdfFileBasename.replace('.sdf','_Q780.txt'))):
                    atm_file = os.path.join(self.lidarAtmosphericDir,sdfFileBasename.replace('.sdf','_Q780.txt'))
                    print('\nRunning wavex on',sdfFileBasename, ' using atm file',os.path.join(self.lidarAtmosphericDir,sdfFileBasename.replace('.sdf','_Q780.txt')))
                else:
                    atm_file = os.path.join(self.lidarAtmosphericDir,self.flightday+'_flightlog.txt')
                    if not os.path.isfile(atm_file):
                        print('WARNING! Could not find any files ending with _flightlog.txt')
                        print('flightlog_files: ',','.join([file for file in os.listdir(self.lidarAtmosphericDir) if 'flightlog' in file]))
        #                 print('Double check atm file '+atm_file+'and press any key to continue')
        #             print('running wavex on',sdf_file_name, ' using atm file',atm_file.split('\\')[-1])

                wavex_cmd ="C:\\Bayes\\WavEx.exe -i " + \
                    sdfFile +" -po "+ self.sbetTrajectoryFile + \
                    " -calib_corr " + self.calibFile + \
                    " -wkt " + self.wktFile + \
                    " -geoid g2012au0" + \
                    " -O " + self.wavexOutputDir + \
                    " -olstat -osdfstat -ounc"

                if int(self.year) >= 2021:
                    wavex_cmd = wavex_cmd + " -utc -t_corr 0.0081"
                if self.doWaveform:
                    wavex_cmd = wavex_cmd + " -outgoing -opls"
                if os.path.isfile(atm_file):
                    wavex_cmd = wavex_cmd + " -atm " + atm_file + " -agl " + self.wavexProcessingHeight
                else:
                    print('WARNING: Skipping atmospheric correction, no data found!')

                #print(wavex_cmd)
                check_call(wavex_cmd)

    def renameWavexOutfiles(self):

        lazFiles = collectFilesInPath(self.wavexOutputDir,'.laz')
        plsFiles = collectFilesInPath(self.wavexOutputDir,'.pls')
        wvsFiles = collectFilesInPath(self.wavexOutputDir,'.wvs')

        allFilesToRename = lazFiles+plsFiles+wvsFiles
        newFiles = []
        rename_dict = {}

        #for file in allFilesToMove:
        #    shutil.move(file,os.path.join(self.lidarInternalLazDir,os.path.basename(file)))
        #    newFiles.append(os.path.join(self.lidarInternalLazDir,os.path.basename(file)))

        print('Renaming all files')

        for record_dict in self.record_dicts:
        #rename lines using line numbers (use L###-1 unless that line name already exists, then use the next integer up)
            for key,value in record_dict.items():

                for file in allFilesToRename:
                    if value.split('_')[1] in file:
                        if '_' in key:
                            line_no = int(key.split('_')[1].replace('Line',''))
                            line_str ='L'+ str(line_no).zfill(3)
                            rpp_line=value.split('_')[1]
                            ext = os.path.splitext(os.path.basename(file))[1]
                            if not any((line_str in line and line.endswith(ext)) for line in allFilesToRename):
                                new_name = rpp_line.replace(rpp_line,line_str)+'-1_'+self.flightday+'_s'+os.path.splitext(file)[1]
                                print(line_str,'does not already exist, renaming',file,'to',new_name)
                                rename_dict[file] = new_name
                            else:
                                matchingLines = [os.path.basename(s) for s in allFilesToRename if (line_str in s) and os.path.splitext(os.path.basename(s))[1] == ext]
                                for matchingLine in matchingLines:
                                    rev = max(int(line.split('_')[0].split('-')[1]) for line in matchingLines)+1
                                    new_name = rpp_line.replace(rpp_line,line_str)+'-'+str(rev)+'_'+self.flightday+'_s'+os.path.splitext(file)[1]
                                    print(line_str,'already exists, renaming',file,'to',new_name)
                                    rename_dict[file] = new_name
                            shutil.move(os.path.join(self.wavexOutputDir,file),os.path.join(self.wavexOutputDir,new_name))
                        else:
                            print('WARNING no line number found in rpp for record',value)
                            new_name =value.split('_')[1]+'_'+self.flightday+'_s'+os.path.splitext(file)[1]    
                            rename_dict[file] = new_name
              
        return rename_dict

# Optech LMS methods (Optech Lidar L1 Processing)

    def moveOptechMissionData(self):
        # move sbet, smrmsg, and range file to the mission folder
        print('Copying sbet and smrmsg from Daily folder to Optech LMS Mission folder')
        print('sbet:',self.sbetTrajectoryFile)
        print('smrmsg:',self.sbetTrajectoryErrorFile)
        print('Lidar Mission Folder:',self.lmsMissionDataDir)
        os.makedirs(self.lmsMissionDataDir,exist_ok=True)
        shutil.copy2(self.sbetTrajectoryFile , self.lmsMissionDataDir)
        shutil.copy2(self.sbetTrajectoryErrorFile , self.lmsMissionDataDir)
        
        print('Moving Lidar range file from Raw folder to Optech LMS Mission folder')
        if len(self.rawDiscreteLidarFiles) > 1:
            print('WARNING: Multiple range files found.')
            # exit
        for range_file in self.rawDiscreteLidarFiles:
            print('range:',range_file)
            shutil.move(range_file,self.lmsMissionDataDir)

    # class TooManyFilesFoundError(Exception):
    #     pass

    def getOptechInstrumentFiles(self,sensor='5060445',lmsv='4.7'):
        # use the instrument file lookup to determine the correct instrument files for the date
        # these are the res, tbl, and lcp files

        instrumentFileDir = os.path.join(lidarOptechResDir,'instrument')
        instrumentLookupDB = os.path.join(instrumentFileDir,'optech_instrument_file_lookup.db')
        
        missionDate='-'.join([self.flightday[0:4], self.flightday[4:6], self.flightday[6:8]])

    # def get_instrument_files(sensor, mission_date, lmsv=lms_version_short, lcp_file=None):
        try:
            with sqlite3.connect(instrumentLookupDB) as conn:
                res_tbl_query = """SELECT res_file, tbl_file FROM res_intensity WHERE sensor=? AND ((? BETWEEN start_date AND end_date) OR (? > start_date AND end_date IS NULL))"""
                # tbl_query = """SELECT tbl_file FROM res_intensity WHERE sensor=? AND ((? BETWEEN start_date AND end_date) OR (? > start_date AND end_date IS NULL))"""
                res_file, tbl_file = conn.cursor().execute(res_tbl_query, (sensor, missionDate, missionDate)).fetchall()[0]
                # print(f'RES: {res_file}\nTBL: {tbl_file}\n')
            
            with sqlite3.connect(instrumentLookupDB) as conn:
                lcp_query = """SELECT lcp_file FROM lcp WHERE sensor=? AND lms_version=? AND ((? BETWEEN start_date AND end_date) OR (? > start_date AND end_date IS NULL))"""
                lcp_file = conn.cursor().execute(lcp_query, (sensor, lmsv, missionDate, missionDate)).fetchall()[0][0]
                # print('LCP: ',lcp_file)
                # print('RES FILE:',res_file)
            
            print(f'Optech Instrument Files: \nRES: {res_file}\nTBL: {tbl_file}\nLCP: {lcp_file}')

            res_fullfile = os.path.join(instrumentFileDir, 'res_intensity', res_file)
            tbl_fullfile = os.path.join(instrumentFileDir, 'res_intensity', tbl_file)
            lcp_fullfile = os.path.join(instrumentFileDir, 'lcp', lcp_file)

            return res_fullfile, tbl_fullfile, lcp_fullfile

        except sqlite3.DatabaseError as e:
            print('Database error: %s', e)
            # logging.error('Database error: %s', e)
            raise
        except Exception as e:
            print('An unexpected error occurred: %s', e)
            # logging.error('An unexpected error occurred: %s', e)
            raise
    
    # Usage of the function should handle the custom TooManyFilesError
    # try:
    #     files = getOptechInstrumentFiles('sensor_name', '2021-06-01')
    # except TooManyFilesError as e:
    #     logging.error(e)

    def getOptechProcessingFiles(self):
        # get the mission folder, georeference (grf) files, and flight plan (?)

        self.resFile, self.tblFile, self.lcpFile = self.getOptechInstrumentFiles()

        # Optech Instrument Files
        print('Copying instrument files to Optech LMS Instrument folder: ',self.lmsInstrumentFileDir)
        os.makedirs(self.lmsInstrumentFileDir,exist_ok=True)
        shutil.copy2(self.resFile , self.lmsInstrumentFileDir)
        shutil.copy2(self.tblFile , self.lmsInstrumentFileDir)
        shutil.copy2(self.lcpFile , self.lmsInstrumentFileDir)

        # Georeference Files
        grfFileDir = os.path.join(lidarOptechResDir,'grf','lms45')
        # copy in reference frame and instrument files
        print('Copying in georeference frame (grf) files to LMS Reference Frame folder: ',self.lmsReferenceFrameDir)
        refFrameFiles = [file for file in os.listdir(grfFileDir) if str(int(self.missionSite.utmZone)).zfill(2) in file]
        print('Reference Frame Files' + ','.join(refFrameFiles))

        os.makedirs(self.lmsReferenceFrameDir,exist_ok=True)
        for grfFile in refFrameFiles:
            if 'Geoid12a' in grfFile:
                shutil.copy2(os.path.join(grfFileDir,grfFile),os.path.join(self.lmsReferenceFrameDir,'out.grf'))
            else:
                shutil.copy2(os.path.join(grfFileDir,grfFile),os.path.join(self.lmsReferenceFrameDir,'prj.grf'))

        # Flight Plan File (.xml)
        os.makedirs(self.lmsFlightPlanDir,exist_ok=True)
        # print('missionSite.site: ',self.missionSite.site)
        flightPlans = [file for file in os.listdir(self.rawFlightPlansXmlDir) if file.endswith('.xml') and self.missionSite.site in file]
        for plan in flightPlans:
            print(f'Copying {plan} to LMS FlightPlan directory')
            shutil.copy2(os.path.join(self.rawFlightPlansXmlDir,plan),self.lmsFlightPlanDir)
        # shutil.copy2(os.path.join(grfFileDir,grfFile),os.path.join(self.lmsReferenceFrameDir,'out.grf'))

    def makeOptechMetFile(self):
        # make met file
        
        self.getSbet()

        sbetTimes = self.sbet.time
        sbetStart = sbetTimes[0]
        sbetEnd = sbetTimes[-1]
        print('SBET START TIME:', sbetStart)
        print('SBET END TIME:', sbetEnd)

        print('Attempting to run make_met_file.py to generate met file from cabin and nidaq telemetry data')
        print('Site: ',self.missionSite.site)
        
        # lmsQaDir = os.path.join(self.LidarQaDir,'LMS')
        # os.makedirs(lmsQaDir,exist_ok=True)
        make_met_file(self.flightday,self.lmsMissionDataDir,self.lmsMissionDataDir,sbetStart,sbetEnd)

        # display contents of the missionSite
        # for i in inspect.getmembers(self.missionSite):
        #     if not i[0].startswith('_'):
        #         # To remove other methods that does not start with a underscore
        #         if not inspect.ismethod(i[1]): 
        #             print(i)

        # met_file = os.path.join(self.lmsMissionDataDir,self.flightday+'.met')

        # if os.path.exists(met_file):
        #     print('SUCCESS! .met file generated from telemetry data!')
        # else:
        #     print('WARNING! Could not generate met file '+ mission.flightday +'.met from telemetry data, looking in flight log database for T & P.')
        #     import make_atm_flightlog_accdb as accdb
        #     accdb.make_optech_met(site.year_site_v,mission.mission_name)

        return

    def runOptechLMS(self):

        lmsSettingsDir = os.path.join(lidarOptechResDir, 'lms47_standard_settings')

        print(f'Running LMS on {self.lmsMissionProjectDir}')
        optech_lms_cmd ="C:\\Optech\\Optech LMS 4.7.1.32246\\manager\\lms-manager.exe -run " + self.lmsMissionProjectDir + ' -settings ' + lmsSettingsDir

        print(optech_lms_cmd)
        check_call(optech_lms_cmd)

    def moveAllOptechOutfiles(self):
        
        os.makedirs(self.LidarInternalLazStandardDir,exist_ok=True)

        lmsOutputDir = os.path.join(self.lmsMissionProjectDir,'LMSProject','Output')
        lazFiles = collectFilesInPath(lmsOutputDir,'.laz')
        # plsFiles = collectFilesInPath(lmsOutputDir,'.pls')
        # wvsFiles = collectFilesInPath(lmsOutputDir,'.wvs')
        
        for file in lazFiles:
            shutil.move(file,os.path.join(self.LidarInternalLazStandardDir,os.path.basename(file)))
        # allFilesToMove = lazFiles+plsFiles+wvsFiles
        # for file in allFilesToMove:
        #     shutil.move(file,os.path.join(self.LidarInternalLazStandardDir,os.path.basename(file)))

# Download Methods
    def downloadAncillary(self):
        
        if not os.path.exists(os.path.join(self.rawAncillaryDir,self.flightday+'_'+self.payload.payloadId +'_metadata.xml')):
            self.dl.download_aop_object(self.rawAncillaryDir,Ext='metadata.xml')
        if not os.path.exists(os.path.join(self.rawFlightLogDir, self.flightday + '_' + self.payload.payloadId + '_NIS.csv')):
            self.dl.download_aop_object(self.rawFlightLogDir)
        self.dl.download_aop_object(self.rawFlightPlansKmlDir,Ext='.kml')
        self.dl.download_aop_object(self.rawFlightPlansXmlDir,Ext='.xml')
        self.dl.download_aop_object(self.flightDbDir,Ext='.accdb')
        self.dl.download_aop_object(self.flightTrackingDir,Ext='.json')
        self.getMissionFiles()

    def downloadGpsImu(self):

        self.dl.download_aop_object(self.rawGpsDir)
        self.getMissionFiles()

    def downloadRawLidar(self):

        if self.payload.payload=='P3':
            self.dl.download_aop_object(self.rawDiscreteLidarDir,Ext='.sdf')
            self.dl.download_aop_object(self.rawWaveformDir,Ext='.rpp')
        else:
            self.dl.download_aop_object(self.rawDiscreteLidarDir,Ext='.range')
            # waveform - leave out downloading the waveform data
            # if int(self.year) <= 2015:
            #     self.dl.download_aop_object(self.rawWaveformLidarDir,Ext='.ix2')
            #     self.dl.download_aop_object(self.rawWaveformLidarDir,Ext='.df2')
            # elif int(self.year) <= 2023 and int(self.year) > 2015:
            #     self.dl.download_aop_object(self.rawWaveformLidarDir,Ext='.ix3')
            #     self.dl.download_aop_object(self.rawWaveformLidarDir,Ext='.df3')
            # elif int(self.year) >= 2024:
            #     self.dl.download_aop_object(self.rawWaveformLidarDir,Ext='.ix4')
            #     self.dl.download_aop_object(self.rawWaveformLidarDir,Ext='.df4')

        self.getMissionFiles()

    def downloadCamera(self):
        
        if self.payload.payload=='P3':
            key = '_'+self.missionSite.site+'_'
        else:
            key=None
            
        self.dl.download_aop_object(self.rawCameraDir,keyword=key)
        self.getMissionFiles()

    def downloadSpectrometer(self):

        for line in self.FlightLines:
            self.dl.download_aop_object(os.path.join(self.rawSpectrometerDir,self.payload.nisId.replace('-','')+'_'+line.nisFlightID[0:8]+'_'+line.nisFilename,'hsi','kml'))
            self.dl.download_aop_object(os.path.join(self.rawSpectrometerDir,self.payload.nisId.replace('-','')+'_'+line.nisFlightID[0:8]+'_'+line.nisFilename,'hsi','raw'))
        self.dl.download_aop_object(os.path.join(self.rawSpectrometerDir,'Logs'))
        self.getMissionFiles()

    def downloadProcessedSbet(self):

        self.dl.download_aop_object(os.path.join(self.sbetGpsImuDir),Ext='.out')
        self.getMissionFiles()

    def downloadSbetHkQa(self):

        self.dl.download_aop_object(os.path.join( self.rawQaChecksDir,'trajectory'),Ext='.out')
        self.dl.download_aop_object(os.path.join( self.rawQaChecksDir,'trajectory'),Ext='.png')

        self.dl.download_aop_object(os.path.join( self.rawQaChecksDir,'trajectory',self.flightday+'_trajectory_QA'),Ext='.log')
        self.dl.download_aop_object(os.path.join( self.rawQaChecksDir,'trajectory',self.flightday+'_trajectory_QA'),Ext='.pospac')
        self.getMissionFiles()

    def downloadCameraHkQa(self):

        self.dl.download_aop_object(os.path.join(self.rawQaChecksDir,self.missionSite.site,'Camera'))
        self.getMissionFiles()

    def downloadLidarHkQa(self):

        self.dl.download_aop_object(os.path.join( self.rawQaChecksDir,self.missionSite.site,'LiDAR','CoverageKMLs'))

        if self.payloadId == 'P3':
            self.dl.download_aop_object(os.path.join( self.rawQaChecksDir,'WLiDARKMLs'))
        self.getMissionFiles()

    def downloadCoverageHkQa(self):

        self.dl.download_aop_object(os.path.join( self.rawQaChecksDir,self.missionSite.site,'Coverage'))
        self.getMissionFiles()

    def downloadSpectrometerHkQa(self):

        self.dl.download_aop_object(os.path.join( self.rawQaChecksDir,self.missionSite.site,'NIS'))
        self.dl.download_aop_object(os.path.join( self.rawQaChecksDir,self.missionSite.site,'NIS','DecimatedKMLs'))
        self.dl.download_aop_object(os.path.join( self.rawQaChecksDir,self.missionSite.site,'NIS','QuickLooks'))
        self.dl.download_aop_object(os.path.join( self.rawQaChecksDir,self.missionSite.site,'NIS','misc'))

    def downloadSpectrometerHkQa(self):

        self.dl.download_aop_object(os.path.join( self.rawQaChecksDir,self.missionSite.site,'WaveformLidar'))

    def downloadAllHkQa(self):

        self.downloadSbetHkQa()
        self.downloadCameraHkQa()
        self.downloadLidarHkQa()
        self.downloadCoverageHkQa()
        self.downloadSpectrometerHkQa()

    def downloadAllRawProductsData(self):

        self.downloadAncillary()
        self.downloadGpsImu()
        self.downloadLidar()
        self.downloadCamera()
        self.downloadSpectrometer()

    def downloadAllRawProductsAndHkQaData(self):

        self.downloadAllRawProductsData()
        self.downloadAllHkQa()

    def downloadSpectrometerRawData(self):

        self.downloadAncillary()
        self.downloadSpectrometer()

    def downloadDataRawSpectrometerProcessing(self):

        self.downloadSpectrometerRawData()
        self.downloadProcessedSbet()

    def getMissionFlightLines(self):

        self.FlightLines = []

        self.downloadAncillary()
        if self.spectrometerFlightLog:
            self.dataCollected = True
        else:
            self.dataCollected = False
            return

        self.nisLineMetadataDf = pd.read_csv(self.spectrometerFlightLog,dtype=str)

        self.lidarCameraLineMetadataDf = pd.read_csv(self.lidarFlightLog,dtype=str)
        self.lidarCameraLineMetadataDf = self.lidarCameraLineMetadataDf.drop(self.lidarCameraLineMetadataDf[self.lidarCameraLineMetadataDf['LineNumber'] == 'AT'].index)
        self.lidarCameraLineMetadataDf = self.lidarCameraLineMetadataDf.drop(self.lidarCameraLineMetadataDf[self.lidarCameraLineMetadataDf['LineNumber'] == 'GT'].index)
        self.lidarCameraLineMetadataDf = self.lidarCameraLineMetadataDf.drop(self.lidarCameraLineMetadataDf[self.lidarCameraLineMetadataDf['LineNumber'] == 'Adhoc'].index)
        self.lidarCameraLineMetadataDf = self.lidarCameraLineMetadataDf.dropna()

        rowCounter = 1
        for index,row in self.nisLineMetadataDf.iterrows():
            nisLineMetaData = row
            if nisLineMetaData.isnull().values.any():
                print('Skipping row '+str(rowCounter)+' due to existence of NULL value in necessary metadata')
                continue
            if nisLineMetaData['LineNumber'] == 'AT' or nisLineMetaData['LineNumber'] == 'GT' or nisLineMetaData['LineNumber'] == 'Adhoc' or nisLineMetaData['DNP'] == 'True' or nisLineMetaData['DNP'] == 'TRUE' or self.missionSite.site not in nisLineMetaData['Site']:
                continue
            
            if nisLineMetaData['Site'] != self.missionSite.site:
                nisLineMetaData['Site'] = self.missionSite.site
            
            lidarLineMetaData = self.lidarCameraLineMetadataDf[(self.lidarCameraLineMetadataDf['LineNumber'].astype(int) == int(nisLineMetaData['LineNumber']))]
            if lidarLineMetaData.shape[0] > 1:
                lidarLineMetaData = lidarLineMetaData.iloc[1]

            nisLineMetaData['LineNumberNew'] = getFltLogLineNumber(self.nisLineMetadataDf.dropna(),self.missionSite.site,str(nisLineMetaData['FlightID'][0:8])+'_'+str(nisLineMetaData['Filename']))

            flightline = LineClass(nisLineMetaData,lidarLineMetaData.squeeze(),self.payload,self.missionSite.site,self.missionSite.domain,self.rawDir)
            #flightline.rawSpectrometerDir = os.path.join(self.rawSpectrometerDir,self.payload.nisId.replace('-','')+'_'+flightline.nisFlightID[0:8]+'_'+flightline.nisFilename,'hsi','raw')
          
            self.FlightLines.append(flightline)
            rowCounter +=1
        self.numFlightlines = len(self.FlightLines)

        
    def keepFlightline(self,lineIdentifiers):
        
        tempFlightlines = []
        lineIdentifiers = list(set(lineIdentifiers))
        for lineIdentifier in lineIdentifiers:
            for flightline in self.FlightLines:
                if lineIdentifier in flightline.nisFilename or lineIdentifier in flightline.lineNumber:
                    tempFlightlines.append(flightline)
                    print('Keeping '+flightline.nisFilename+' / '+flightline.lineNumber)
                    
        self.FlightLines = tempFlightlines
        self.numFlightlines = len(self.FlightLines)
        
        print('Flightlines left are: ')
        
        for flightline in self.FlightLines:
            print(flightline.nisFilename +' '+flightline.lineNumber)
            
            
            
    def removeFlightline(self,lineIdentifiers):
        
        tempFlightlines = []
        lineIdentifiers = list(set(lineIdentifiers))
        for lineIdentifier in lineIdentifiers:
            flightlineIndex = 0
            for flightline in self.FlightLines:
                if lineIdentifier in flightline.nisFilename or lineIdentifier in flightline.lineNumber:
                    removedFlightline = self.FlightLines.pop(flightlineIndex)
                    print('Removed '+removedFlightline.nisFilename+' / '+removedFlightline.lineNumber)
                flightlineIndex+=1
                
        self.numFlightlines = len(self.FlightLines)
                                
    def unzipL2VegetationIndices(self):
        for zip_file in self.SpectrometerL2VegIndicesZipFiles:
            unzipFiles(zip_file)
    
    def generateSpectrometerKmls(self):
                
        os.makedirs(self.SpectrometerMetadataKmlsDir,exist_ok=True)
        with Pool(processes=25) as pool:
            processFunction = partial(processSpectrometerKmlsHelper, rawKmlDir=self.rawSpectrometerDir,outputKmlDir=self.SpectrometerMetadataKmlsDir)
            pool.map(processFunction,self.FlightLines)
        
    def processRadiance(self):

        
        os.makedirs(self.SpectrometerL1RadianceProcessingDir,exist_ok=True)
        os.makedirs(self.SpectrometerQaRadianceDir,exist_ok=True)
        
        numCpus = 25
        
        for flightLine in self.FlightLines:
            
            flightLine.processRadiance(self.rawSpectrometerDir,self.SpectrometerL1RadianceProcessingDir,self.SpectrometerQaRadianceDir,numCpus)


        #Old Matlab implementation
        # if self.numFlightlines <= 15:
        #     numCores = 15
        # else:
        #     numCores = np.ceil(self.numFlightlines/(np.ceil(self.numFlightlines/15)))
            
            
        # with Pool(processes=int(numCores)) as pool:
        #     processFunction = partial(processRadianceHelper, rawSpectrometerDir=self.rawSpectrometerDir,outputRadianceDir=self.SpectrometerL1RadianceProcessingDir)
        #     pool.map(processFunction,self.FlightLines)
          
    def processRadianceQa(self,domain,visit):

        os.makedirs(self.SpectrometerRadianceQaDir,exist_ok=True)
        nisL1qaQcPipeline(self.missionId,domain,self.missionSite.site,visit)

    def processOrthoRadiance(self,geoidFile,demFile,sapFile):

        with Pool(processes=25) as pool:
            processFunction = partial(processOrthoRadianceHelper, rawSpectrometerDir=self.rawSpectrometerDir,radianceDir=self.SpectrometerL1RadianceProcessingDir,payload=self.payload,leapSeconds=self.leapSeconds,sbetTrajectoryFile=self.sbetTrajectoryFile,geoidFile=geoidFile,demFile=demFile,sapFile=sapFile)
            pool.map(processFunction,self.FlightLines)

    def generateEnviRdnOrtRgbNirTifs(self):
        
        os.makedirs(self.SpectrometerOrthoRadianceQaDir,exist_ok=True)
                
        with Pool(processes=15) as pool:
            processFunction = partial(processRdnOrtTifsHelper,radianceDir=self.SpectrometerL1RadianceProcessingDir,outputDir=self.SpectrometerOrthoRadianceQaDir)
            pool.map(processFunction,self.FlightLines)
    
    def generateOrthoQa(self):

        self.generateEnviRdnOrtRgbNirTifs()
        
        self.getProductFiles()
        
        rgbFileName = os.path.join(self.SpectrometerOrthoRadianceQaDir,self.missionSite.site+'_'+self.flightday+'_RGB_radiance_mosaic.tif')
        
        generateMosaic(self.SpectrometerOrthoRadianceQaFiles,rgbFileName,mosaicType = 'last_in')
        
        plot_multiband_geotiff(rgbFileName, title='', stretch='linear5',save_path=rgbFileName.replace('.tif','.png'), nodata_color='black',variable='')

        #os.makedirs(self.SpectrometerOrthoRadianceQaDir,exist_ok=True)
        #matlabEng = matlab.engine.start_matlab()
        #matlabEng.cd(nisSpectrometerQaCodeDir, nargout=0)
        #matlabEng.generateOrthoQA(self.SpectrometerL1RadianceProcessingDir,self.SpectrometerOrthoRadianceQaDir,self.flightday,self.metadataXml,self.missionSite.site,matlab.double(85))
        #matlabEng.quit()
        
    def clipTopoForAtcor(self,aigDsm):

        with Pool(processes=30) as pool:
            processFunction = partial(processClipTopoForAtcorHelper, radianceDir=self.SpectrometerL1RadianceProcessingDir,aigDsm=aigDsm)
            pool.map(processFunction,self.FlightLines)
            
    def processSmoothElevation(self):

        with Pool(processes=30) as pool:
            processFunction = partial(processSmoothElevationeHelper, radianceDir=self.SpectrometerL1RadianceProcessingDir)
            pool.map(processFunction,self.FlightLines)
            
    def processSlopeAspect(self):

        with Pool(processes=30) as pool:
            processFunction = partial(processSlopeAspectHelper, radianceDir=self.SpectrometerL1RadianceProcessingDir)
            pool.map(processFunction,self.FlightLines)

    def processH5RadianceWriter(self):

        os.makedirs(self.SpectrometerL1RadianceDir,exist_ok=True)

        with Pool(processes=5) as pool:
            processFunction = partial(processRadianceH5Helper,radianceDir=self.SpectrometerL1RadianceProcessingDir,sbetTrajectoryFile=self.sbetTrajectoryFile,radianceH5Dir=self.SpectrometerL1RadianceDir,metadataXml=self.metadataXml,spectrometerFlightLog=self.spectrometerFlightLog,scriptsFile=self.scriptsFile)
            pool.map(processFunction,self.FlightLines)
       
        #for flightline in self.FlightLines:
        #    flightline.generateRadianceH5(self.SpectrometerL1RadianceProcessingDir,self.sbetTrajectoryFile,self.SpectrometerL1RadianceDir,self.metadataXml,self.spectrometerFlightLog,self.scriptsFile)

    def convertRdnOrtToEnviBsq(self):

        t0 = time.time()
        
        os.makedirs(self.SpectrometerL1ReflectanceProcessingDir,exist_ok=True)
        if self.numFlightlines <= 10:
            numCores = self.numFlightlines
        else:
            numCores = np.ceil(self.numFlightlines/(np.ceil(self.numFlightlines/10)))

        with Pool(processes=int(numCores)) as pool:
            processFunction = partial(convertRdnOrtForAtcorHelper,radianceDir=self.SpectrometerL1RadianceProcessingDir,reflectanceDir=self.SpectrometerL1ReflectanceProcessingDir)
            pool.map(processFunction,self.FlightLines)

        print('Finished in '+str(round(time.time()-t0,1))+' seconds')

        #for flightline in self.FlightLines:
        #    flightline.convertRdnOrtForAtcor(self.SpectrometerL1RadianceProcessingDir,self.SpectrometerL1ReflectanceProcessingDir)

    def generateScaFile(self):

        with Pool(processes=25) as pool:
            processFunction = partial(generateScaFileForAtcorHelper,radianceDir=self.SpectrometerL1RadianceProcessingDir)
            pool.map(processFunction,self.FlightLines)

        #for flightline in self.FlightLines:
        #    flightline.generateScaFileForAtcor(self.SpectrometerL1RadianceProcessingDir)

    def getFlightlineBeginningEndTimes(self):

        self.getSbet()
        self.sbet.getAllSbetData()
        
        for flightline in self.FlightLines:
            flightline.getBeginningEndFlightTime(self.SpectrometerL1RadianceProcessingDir,self.sbet.gpsDayOfWeek)

    def getFlightlineAverageAltitude(self):
        
        for flightline in self.FlightLines:
            flightline.getLineAverageAltitude(self.sbet)

    def getFlightlineAverageHeading(self):
        
        for flightline in self.FlightLines:
            flightline.getLineAverageHeading(self.sbet)

    def getAverageFlightlineSunZenith(self):

        for flightline in self.FlightLines:
            flightline.getAverageSunZenith(self.SpectrometerL1RadianceProcessingDir)

    def getAverageFlightlineSunAzimuth(self):
        
        for flightline in self.FlightLines:
            flightline.getAverageSunAzimuth(self.SpectrometerL1RadianceProcessingDir)

    def getAverageFlightlineElevation(self):

        for flightline in self.FlightLines:
            flightline.getAverageElevation(self.SpectrometerL1RadianceProcessingDir)

    def createAtcorSensorAndAtmLibs(self):
        
        for flightline in self.FlightLines:
            flightline.createAtcorSensorAndAtmLib(self.payload)

    def cleanupAtcorSensorAndAtmLibs(self):

        for flightline in self.FlightLines:
            flightline.cleanupAtcorSensorAndAtmLib()

    def prepareAtcorInputFiles(self):

        self.convertRdnOrtToEnviBsq()
        self.generateScaFile()

    def getAtcorInputVariables(self):

        self.getSbet()
        self.getFlightlineBeginningEndTimes()
        self.getFlightlineAverageAltitude()
        self.getFlightlineAverageHeading()
        self.getAverageFlightlineSunZenith()
        self.getAverageFlightlineSunAzimuth()
        self.getAverageFlightlineElevation()

    def prepareAtcorInputs(self):

        self.prepareAtcorInputFiles() #This needs to be done before ATCOR
        self.getAtcorInputVariables()
        self.createAtcorSensorAndAtmLibs()

    def postAtcorUpdates(self):

        print('Updating the no data values in ATCOR BSQs for '+self.missionId)
        self.postAtcorFileUpdates()
        print('Checking for missing DDVs for '+self.missionId)
        self.createMissingDDVs() 
        print('Cleaning atcor sensor and atmlib for '+self.missionId)
        self.cleanupAtcorSensorAndAtmLibs()

    def processReflectance(self):

        os.makedirs(self.SpectrometerL1ReflectanceProcessingDir,exist_ok=True)

        with Pool(processes=25) as pool:
            processFunction = partial(processReflectanceHelper,radianceDir=self.SpectrometerL1RadianceProcessingDir,reflectanceDir=self.SpectrometerL1ReflectanceProcessingDir,sbetTrajectoryFile=self.sbetTrajectoryFile,payload=self.payload)
            pool.map(processFunction,self.FlightLines)
            
        #for flightline in self.FlightLines:
        #    flightline.processReflectance(self.SpectrometerL1RadianceProcessingDir,self.SpectrometerL1ReflectanceProcessingDir,self.sbetTrajectoryFile,self.payload)

    def postAtcorFileUpdates(self):
        
        with Pool(processes=15) as pool:
            processFunction = partial(fixNansInAtcorRasterHelper,reflectanceDir=self.SpectrometerL1ReflectanceProcessingDir)
            pool.map(processFunction,self.FlightLines)

    def createMissingDDVs(self):    

        with Pool(processes=25) as pool:
            processFunction = partial(createMissingDDVHelper,reflectanceDir=self.SpectrometerL1ReflectanceProcessingDir)
            pool.map(processFunction,self.FlightLines)
        
        # for flightline in self.FlightLines:
        #     flightline.createMissingDDV(self.SpectrometerL1ReflectanceProcessingDir)

    def processH5ReflectanceWriter(self,isBrdf=False):

        os.makedirs(self.SpectrometerL1ReflectanceDir,exist_ok=True)

        # with Pool(processes=5) as pool:
        #     processFunction = partial(processReflectanceH5Helper,radianceDir=self.SpectrometerL1RadianceProcessingDir,reflectanceProcessingDir=self.SpectrometerL1ReflectanceProcessingDir,sbetTrajectoryFile=self.sbetTrajectoryFile,reflectanceH5Dir=self.SpectrometerL1ReflectanceDir,metadataXML=self.metadataXml,spectrometerFlightLog=self.spectrometerFlightLog,scriptsFile=self.scriptsFile,isBrdf=isBrdf)
        #     pool.map(processFunction,self.FlightLines)
        
        for flightline in self.FlightLines:
            flightline.generateReflectanceH5(self.SpectrometerL1RadianceProcessingDir,self.SpectrometerL1ReflectanceProcessingDir,self.sbetTrajectoryFile,self.SpectrometerL1ReflectanceDir,self.metadataXml,self.spectrometerFlightLog,self.scriptsFile,isBrdf)

    def renameDirectionalReflectanceH5(self):
        
        for flightline in self.FlightLines:
            flightline.renameDirectionalReflectance(self.SpectrometerL1ReflectanceDir)

    def reverseRenameDirectionalReflectanceH5(self):
        
        for flightline in self.FlightLines:
            flightline.reverseRenameDirectionalReflectance(self.SpectrometerL1ReflectanceDir)

    def applyBrdfCorrection(self):
        outputCores = 10
        apply_brdf_correct(self.SpectrometerL1ReflectanceDir,outputCores)

    def generateVegetationIndices(self):

        os.makedirs(self.SpectrometerL2VegIndicesDir,exist_ok=True)
        with Pool(processes=30) as pool:
            processFunction = partial(processVegetationIndicesHelper,reflectanceDir=self.SpectrometerL1ReflectanceDir,outputDir=self.SpectrometerL2VegIndicesDir)
            pool.map(processFunction,self.FlightLines)

    def generateWaterIndices(self):

        os.makedirs(self.SpectrometerL2WaterIndicesDir,exist_ok=True)

        with Pool(processes=15) as pool:
            processFunction = partial(processWaterIndicesHelper,reflectanceDir=self.SpectrometerL1ReflectanceDir,outputDir=self.SpectrometerL2WaterIndicesDir)
            pool.map(processFunction,self.FlightLines)
        
        #for flightline in self.FlightLines:
        #    flightline.generateWaterIndices(self.SpectrometerL1ReflectanceDir,self.SpectrometerL2WaterIndicesDir)

    def zipVegetationIndices(self):

        with Pool(processes=15) as pool:
            processFunction = partial(zipVegetationIndicesHelper,inputDir=self.SpectrometerL2VegIndicesDir,outputDir=self.SpectrometerL2VegIndicesDir)
            pool.map(processFunction,self.FlightLines)

    def zipWaterIndices(self):

        with Pool(processes=15) as pool:
            processFunction = partial(zipWaterIndicesHelper,inputDir=self.SpectrometerL2WaterIndicesDir,outputDir=self.SpectrometerL2WaterIndicesDir)
            pool.map(processFunction,self.FlightLines)

    def generateFpar(self):

        os.makedirs(self.SpectrometerL2FparDir,exist_ok=True)

        with Pool(processes=30) as pool:
            processFunction = partial(processFparHelper,reflectanceDir=self.SpectrometerL1ReflectanceDir,outputDir=self.SpectrometerL2FparDir)
            pool.map(processFunction,self.FlightLines)

    def generateLai(self):

        os.makedirs(self.SpectrometerL2LaiDir,exist_ok=True)

        with Pool(processes=30) as pool:
            processFunction = partial(processLaiHelper,reflectanceDir=self.SpectrometerL1ReflectanceDir,outputDir=self.SpectrometerL2LaiDir)
            pool.map(processFunction,self.FlightLines)

    def generateAlbedo(self):

        os.makedirs(self.SpectrometerL2AlbedoDir,exist_ok=True)

        with Pool(processes=30) as pool:
            processFunction = partial(processAlbedoHelper,reflectanceDir=self.SpectrometerL1ReflectanceProcessingDir,outputDir=self.SpectrometerL2AlbedoDir)
            pool.map(processFunction,self.FlightLines)

    def generateRgbReflectanceTifs(self):

        os.makedirs(self.SpectrometerMetadataRgbTifsDir,exist_ok=True)

        with Pool(processes=30) as pool:
            processFunction = partial(processH5RgbTifHelper,H5Dir=self.SpectrometerL1BidirectionalReflectanceDir,outputDir=self.SpectrometerMetadataRgbTifsDir,raster='Reflectance')
            pool.map(processFunction,self.FlightLines)

    def generateRgbRadianceTifs(self):

        os.makedirs(self.SpectrometerMetadataRgbRadianceTifsDir,exist_ok=True)

        with Pool(processes=5) as pool:
            processFunction = partial(processH5RgbTifHelper,H5Dir=self.SpectrometerL1RadianceDir,outputDir=self.SpectrometerMetadataRgbRadianceTifsDir,raster='Radiance')
            pool.map(processFunction,self.FlightLines)
        
    def generateWaterMaskTifs(self):

        os.makedirs(self.SpectrometerL2QaReflectanceDir,exist_ok=True)
    
        with Pool(processes=30) as pool:
            processFunction = partial(generateWaterMaskFlightlineHelper,reflectanceH5Dir=self.SpectrometerL1BidirectionalReflectanceDir,outputDir=self.SpectrometerL2QaReflectanceDir)
            pool.map(processFunction,self.FlightLines)

    def generateReflectancePngs(self):

        os.makedirs(self.SpectrometerL2QaReflectanceDir,exist_ok=True)

        with Pool(processes=30) as pool:
            processFunction = partial(generateReflectancePngHelper,reflectanceH5Dir=self.SpectrometerL1BidirectionalReflectanceDir,outputDir=self.SpectrometerL2QaReflectanceDir)
            pool.map(processFunction,self.FlightLines)

    def generateSpectrometerProductQa(self,spectrometerProductQaList):

        os.makedirs(self.SpectrometerL2QaReflectanceProductDir,exist_ok=True)

        with Pool(processes=25) as pool:
            processFunction = partial(processReflectanceProductQa,siteFolder=os.path.join(self.productBaseDir,''),flightday=self.flightday,metadataXML=self.metadataXml,site=self.missionSite.site,productLevel=2,previousYearSiteVisit='')
            pool.map(processFunction,spectrometerProductQaList)

    def getReflectanceDifferenceFlightlineTifs(self):

        self.getProductFiles()
        os.makedirs(self.SpectrometerQaFlightlineDifferenceDir,exist_ok=True)
        filelist1,filelist2 = getOverlappingImageFiles(self.SpectrometerL1BidirectionalReflectanceFiles)
        getOverlapDifferenceTif(filelist1,filelist2,self.SpectrometerQaFlightlineDifferenceDir)

    def getReflectanceMeanDifferenceFlightlines(self):

        self.wavelengthSampleCount,self.meanDifferenceSpectra = getReflectanceMeanDifference(self.SpectrometerQaFlightlineDifferenceFiles)

    def getFlightlineReflectanceDifferenceSummaryStats(self):

        reflectanceArray, metadata, wavelengths = h5refl2array(self.SpectrometerL1BidirectionalReflectanceFiles[0], 'Reflectance',onlyMetadata = True)
        rmsArrays,maxIndicesArrays,maxDiffRaster,self.maxDifferenceWavelengths,self.sumWavelengthDifferenceSquared,self.sumWavelengthVariance = getFlightlineReflectanceDifferenceSummaryStats(self.SpectrometerQaFlightlineDifferenceFiles,self.meanDifferenceSpectra,wavelengths)
        return rmsArrays,maxIndicesArrays,maxDiffRaster

    def getFlightlineRmsAndMaxWavelgthErrorRasters(self,rmsArrays,maxIndicesArrays,maxDiffArrays):

        filebasename = []
                
        os.makedirs(self.SpectrometerL2QaFlightlineRmsDir,exist_ok=True)
        os.makedirs(self.SpectrometerL2QaFlightlineMaxIndicesDir,exist_ok=True)
        os.makedirs(self.SpectrometerL2QaFlightlineMaxDiffDir,exist_ok=True)
        
        for file in self.SpectrometerQaFlightlineDifferenceFiles:
            fileSplit = os.path.basename(file).split('_')
            filebasename.append(self.missionSite.site+'_'+self.flightday+'_'+fileSplit[1]+'_'+fileSplit[3])
        
        getRmsAndMaxWavelgthErrorRasters(rmsArrays,maxIndicesArrays,maxDiffArrays,self.SpectrometerQaFlightlineDifferenceFiles,self.SpectrometerL2QaFlightlineRmsDir,self.SpectrometerL2QaFlightlineMaxIndicesDir,self.SpectrometerL2QaFlightlineMaxDiffDir,filebasename)

        rmsArrays = []
        maxIndicesArrays = []

    def getWavelengthDifferenceSummaryPlots(self):

        os.makedirs(self.SpectrometerL2QaReflectanceProductDir,exist_ok=True)
        scaleReflectanceFactor = 100
        reflectanceArray, metadata, wavelengths = h5refl2array(self.SpectrometerL1BidirectionalReflectanceFiles[0], 'Reflectance',onlyMetadata = True)
        outputFileBaseName = self.missionSite.site+'_'+self.flightday
        generateReflectanceDifferenceLinePlots(wavelengths,self.meanDifferenceSpectra,self.wavelengthStandardDeviation,self.wavelengthTotalRms,self.maxDifferenceWavelengthsTotal,scaleReflectanceFactor,self.SpectrometerL2QaReflectanceProductDir,outputFileBaseName)

    def generateSpectrometerProductErrorL3QaMosaics(self):

        self.generateVegIndicesErrorQaMosaics()
        self.generateWaterIndicesErrorQaMosaics()
        self.generateFparErrorQaMosaic()
        self.generateLaiErrorQaMosaic()  

    def generateVegIndicesQaMosaics(self):

        os.makedirs(self.SpectrometerL2QaReflectanceProductDir,exist_ok=True)
        
        for vegIndex in self.vegIndices:
            vegIndexFiles = collectFilesInPath(self.SpectrometerL2VegIndicesDir,vegIndex+'.tif')
            generateMosaic(vegIndexFiles,os.path.join(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday+'_'+vegIndex+'_mosaic.tif'),mosaicType = 'last_in')

    def generateVegIndicesErrorQaMosaics(self):

        for vegIndex in self.vegIndices:
            vegIndexFiles = collectFilesInPath(self.SpectrometerL2VegIndicesDir,vegIndex+'_error.tif')
            generateMosaic(vegIndexFiles,os.path.join(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday+'_'+vegIndex+'_error_mosaic.tif'),mosaicType = 'last_in')

    def generateVegIndicesDifferenceQaMosaics(self):

        for vegIndex in self.vegIndices:
            vegIndexFiles = collectFilesInPath(self.SpectrometerL2VegIndicesDir,vegIndex+'.tif')
            generateMosaic(vegIndexFiles,os.path.join(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday+'_'+vegIndex+'_difference_mosaic.tif'),mosaicType = 'diff_overlap')

    def generateWaterIndicesQaMosaics(self):

        os.makedirs(self.SpectrometerL2QaReflectanceProductDir,exist_ok=True)
        
        for waterIndex in self.waterIndices:
            waterIndexFiles = collectFilesInPath(self.SpectrometerL2WaterIndicesDir,waterIndex+'.tif')
            generateMosaic(waterIndexFiles,os.path.join(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday+'_'+waterIndex+'_mosaic.tif'),mosaicType = 'last_in')

    def generateWaterIndicesErrorQaMosaics(self):

        os.makedirs(self.SpectrometerL2QaReflectanceProductDir,exist_ok=True)
        
        for waterIndex in self.waterIndices:
            waterIndexFiles = collectFilesInPath(self.SpectrometerL2WaterIndicesDir,waterIndex+'_error.tif')
            generateMosaic(waterIndexFiles,os.path.join(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday+'_'+waterIndex+'_error_mosaic.tif'),mosaicType = 'last_in')

    def generateWaterIndicesDifferenceQaMosaics(self):

        os.makedirs(self.SpectrometerL2QaReflectanceProductDir,exist_ok=True)
        
        for waterIndex in self.waterIndices:
            waterIndexFiles = collectFilesInPath(self.SpectrometerL2WaterIndicesDir,waterIndex+'.tif')
            generateMosaic(waterIndexFiles,os.path.join(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday+'_'+waterIndex+'_difference_mosaic.tif'),mosaicType = 'diff_overlap')

    def generateAlbedoQaMosaic(self):

        generateMosaic(self.SpectrometerL2AlbedoFiles,os.path.join(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday+'_'+'Albedo'+'_mosaic.tif'),mosaicType = 'last_in')

    def generateAlbedoDifferenceQaMosaic(self):

        generateMosaic(self.SpectrometerL2AlbedoFiles,os.path.join(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday+'_'+'Albedo'+'_difference_mosaic.tif'),mosaicType = 'diff_overlap')

    def generateFparQaMosaic(self):

        generateMosaic(self.SpectrometerL2FparFiles,os.path.join(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday+'_'+'Fpar'+'_mosaic.tif'),mosaicType = 'last_in')

    def generateFparErrorQaMosaic(self):

        generateMosaic(self.SpectrometerL2FparErrorFiles,os.path.join(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday+'_'+'Fpar'+'_error_mosaic.tif'),mosaicType = 'last_in')

    def generateFparDifferenceQaMosaic(self):

        generateMosaic(self.SpectrometerL2FparFiles,os.path.join(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday+'_'+'Fpar'+'_difference_mosaic.tif'),mosaicType = 'diff_overlap')

    def generateLaiQaMosaic(self):

        generateMosaic(self.SpectrometerL2LaiFiles,os.path.join(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday+'_'+'Lai'+'_mosaic.tif'),mosaicType = 'last_in')

    def generateLaiErrorQaMosaic(self):

        generateMosaic(self.SpectrometerL2LaiErrorFiles,os.path.join(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday+'_'+'Lai'+'_error_mosaic.tif'),mosaicType = 'last_in')

    def generateLaiDifferenceQaMosaic(self):

        generateMosaic(self.SpectrometerL2LaiFiles,os.path.join(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday+'_'+'Lai'+'_difference_mosaic.tif'),mosaicType = 'diff_overlap')

    def generateSpectrometerProductL2QaMosaics(self):
        
        self.generateVegIndicesQaMosaics()
        self.generateWaterIndicesQaMosaics()
        self.generateAlbedoQaMosaic()
        self.generateFparQaMosaic()
        self.generateLaiQaMosaic() 

    def generateSpectrometerProductErrorL2QaMosaics(self):
        
        self.generateVegIndicesErrorQaMosaics()
        self.generateWaterIndicesErrorQaMosaics()
        self.generateFparErrorQaMosaic()
        self.generateLaiErrorQaMosaic()
        
    def generateSpectrometerDifferenceQaMosaics(self):
       
        self.generateLaiDifferenceQaMosaic()
        self.generateFparDifferenceQaMosaic()
        self.generateAlbedoDifferenceQaMosaic()
        self.generateWaterIndicesDifferenceQaMosaics()
        self.generateVegIndicesDifferenceQaMosaics()
        
    def generateReflectanceDifferenceQaMosaics(self):
        
        self.generateReflectanceRmsQaMosaic()
        self.generateReflectanceMaxDiffWavelengthQaMosaic()
        self.generateReflectanceMaxDiffQaMosaic()
        
  
    def generateReflectanceRmsQaMosaic(self):
        
        self.getProductFiles()
        
        generateMosaic(self.SpectrometerL2QaRmsFiles,os.path.join(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday+'_'+'ReflectanceRMS'+'_mosaic.tif'),mosaicType = 'last_in')
        
    def generateReflectanceMaxDiffWavelengthQaMosaic(self):
        
        self.getProductFiles()
        
        generateMosaic(self.SpectrometerL2QaMaxWavelengthFiles,os.path.join(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday+'_'+'MaxDiffWavelength'+'_mosaic.tif'),mosaicType = 'last_in')
    
    def generateReflectanceMaxDiffQaMosaic(self):
        
        self.getProductFiles()
        
        generateMosaic(self.SpectrometerL2QaMaxDiffFiles,os.path.join(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday+'_'+'MaxDiff'+'_mosaic.tif'),mosaicType = 'last_in')

    def generateWaterMaskQaMosaic(self):
        
        self.getProductFiles()
        
        generateMosaic(self.SpectrometerL2QaWaterMaskFiles,os.path.join(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday+'_'+'water_mask'+'_mosaic.tif'),mosaicType = 'last_in')

    def generateReflectanceAncillaryRasterMosaics(self):
        
        self.getProductFiles()
    
        os.makedirs(self.SpectrometerQaTempAncillaryRastersDir,exist_ok=True)
        os.makedirs(self.SpectrometerL2QaReflectanceProductDir,exist_ok=True)
        
        generateAncillaryRastersFromH5s(self.spectrometerAncillaryProductQaList,self.SpectrometerL1BidirectionalReflectanceFiles,self.SpectrometerQaTempAncillaryRastersDir,self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday,'last_in')

    def generateReflectanceRgbRasterMosaics(self):
        
        self.getProductFiles()
        
        os.makedirs(self.SpectrometerQaTempAncillaryRastersDir,exist_ok=True)
        
        generateRgbAndNirTifMosaicFromH5s(self.SpectrometerL1BidirectionalReflectanceFiles,self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday,'last_in')    

    def generateRgbDistributionPlot(self):
 
        self.getProductFiles()
        
        generateRgbCumulativeDistribution(self.rgbMosaicFile,os.path.join(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday))

    def getSpectrometerQaReportMaps(self):

        self.getProductFiles()
        
        getSpectrometerQaReportMaps(self.SpectrometerL2HistsFiles,self.spectrometerMosaicColorMaps,'viridis','cividis','RdYlBu')

    def getRGBPngMap(self):
        
        self.getProductFiles()
        
        print(self.rgbMosaicFile)
        
        plot_multiband_geotiff(self.rgbMosaicFile, title='', stretch='linear5',save_path=self.rgbMosaicFile.replace('.tif','.png'), nodata_color='black',variable='')

    def getNIRGBPngMap(self):
        
        self.getProductFiles()
        
        plot_multiband_geotiff(self.nirgbMosaicFile, title='', stretch='linear5',save_path=self.nirgbMosaicFile.replace('.tif','.png'), nodata_color='black',variable='')

    def getWaterMaskPngMap(self):
        
        self.getProductFiles()
        
        waterMaskClassLabels = ['Non-water','Water']
        waterMaskClassColors=['Black','White']
        plot_classified_geotiff(self.waterMaskMosaicFile,waterMaskClassLabels,waterMaskClassColors,title='', save_path=self.waterMaskMosaicFile.replace('.tif','.png'), variable='Water Mask')

    def getDdvPngMap(self):
        
        self.getProductFiles()
        
        DdvClassLabels = ['NoData','Water','DDV', 'NonRef','Shadow']
        DdvClassColors=['White','Blue','Green','Grey','Black']
        plot_classified_geotiff(self.DdvMosaicFile,DdvClassLabels,DdvClassColors, title='', save_path=self.DdvMosaicFile.replace('.tif','.png'), variable='DDV')

    def getMosaicQaPngMaps(self):
        
        print('Getting PNG maps for product mosaics for '+self.missionSite.site+'_'+self.flightday)
        self.getSpectrometerQaReportMaps()
        
        print('Getting reflectance RGB map for '+self.missionSite.site+'_'+self.flightday)
        self.getRGBPngMap()
        
        print('Getting reflectance NIRGB map for '+self.missionSite.site+'_'+self.flightday)
        self.getNIRGBPngMap()
        
        print('Getting weqather mask map for '+self.missionSite.site+'_'+self.flightday)
        self.getWaterMaskPngMap()
        
        print('Getting DDV png map for '+self.missionSite.site+'_'+self.flightday)
        self.getDdvPngMap()

    def getSpectrometerQaReportHistograms(self):

        self.getProductFiles()
        
        for mosaicFile in self.SpectrometerL2HistsFiles:
            
            if 'Weather_Quality_Indicator' in mosaicFile or 'water_mask' in mosaicFile or '_RGB_' in mosaicFile:
                continue

            rasterVariable = os.path.basename(mosaicFile).split(self.missionSite.site+'_'+self.flightday)[-1]
            rasterVariable = rasterVariable.split('mosaic')[0]
            rasterVariable = rasterVariable.replace('_',' ')
            
            plotGtifHistogram(mosaicFile,mosaicFile.replace('.tif','_histogram.png'),rasterVariable)

    def getSpectrometerQaReportSummaryFiles(self):

        self.getProductFiles()
          
        getSpectrometerQaReportSummaryFiles(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday,self.spectrometerProductQaList)

    def getSpectrometerAncillaryQaReportSummaryFiles(self):
        
        getSpectrometerAncillaryQaReportSummaryFiles(self.SpectrometerL2QaReflectanceProductDir,self.missionSite.site+'_'+self.flightday,self.spectrometerAncillaryProductQaList)

    def getSpectrometerReflectanceDifferenceSummaryFiles(self):
        
        self.getProductFiles()
        
        getSpectrometerReflectanceDifferenceSummaryFiles(self.SpectrometerL2QaReflectanceProductDir,self.rmsMosaicFile,self.maxDiffMosaicFile,self.maxDiffWavelengthMosaicFile)

    def getSpectrometerRGBQaReportSummaryFiles(self):

        self.getProductFiles()
        
        getSpectrometerRGBQaReportSummaryFiles(self.SpectrometerL2QaReflectanceProductDir,self.redMosaicFile,self.greenMosaicFile,self.blueMosaicFile,self.nirMosaicFile)

    def getSpectrometerQaReports(self):

        print('Getting summary files for product mosaics for '+self.missionSite.site+'_'+self.flightday)
        self.getSpectrometerQaReportSummaryFiles()
        
        print('Getting ancillary raster summary files for product mosaics for '+self.missionSite.site+'_'+self.flightday)
        self.getSpectrometerAncillaryQaReportSummaryFiles()
        
        print('Getting reflectance difference raster summary files for product mosaics for '+self.missionSite.site+'_'+self.flightday)
        self.getSpectrometerReflectanceDifferenceSummaryFiles()
        
        print('Getting reflectance RGB raster summary files for product mosaics for '+self.missionSite.site+'_'+self.flightday)
        self.getSpectrometerRGBQaReportSummaryFiles()   

    def generateL2QaPdf(self):

        matlabEng = matlab.engine.start_matlab()
        matlabEng.cd(nisSpectrometerQaCodeDir, nargout=0)
        matlabEng.generateL2QAPdf(str(self.productBaseDir+os.path.sep),str(self.missionSite.site),str(self.flightday),str(self.payload.payloadId))
        matlabEng.quit()

    def generateL2QaHtml(self):
        # function to generate NIS L2 QA documents in markdown and html formats
        # requires the following folders / contents to be downloaded:
            # SpectrometerL1BidirectionalReflectanceDir (L1\Spectrometer\BidirectionalReflectanceH5\YYYYMMDDHH)
            # SpectrometerL2QaReflectanceProductDir (QA\Spectrometer\L2Hists)

        currentDir = os.getcwd()
        print(f'Changing directories to {nisSpectrometerQaCodeDir}')
        os.chdir(nisSpectrometerQaCodeDir)
        print('Running generateL2QAMarkdown.py with the following inputs:')
        print('site:', self.missionSite.site)
        print('flightday:',self.flightday)
        print('payloadCampaign:', self.payloadCampaign)
        print('productBaseDir:', self.productBaseDir)
        os.system(f"python generateL2QAMarkdown.py {self.missionSite.site} {self.flightday} {self.payloadCampaign} {self.productBaseDir}")
        os.chdir(currentDir)

    def processFlightlineQa(self):
        
        print('Running L2 RGBTifs for '+self.missionId)
        self.generateRgbReflectanceTifs()
    
        print('Getting data product mosaics for '+self.missionSite.site+'_'+self.flightday)
        self.generateSpectrometerProductL2QaMosaics()
        
        print('Getting data product error mosaics for '+self.missionSite.site+'_'+self.flightday)
        self.generateSpectrometerProductErrorL2QaMosaics()
       
        print('Getting data product difference mosaics for '+self.missionSite.site+'_'+self.flightday)
        self.generateSpectrometerDifferenceQaMosaics() 
       
        print('Running reflectance difference analysis for '+self.missionSite.site+'_'+self.flightday)
        self.generateFlightlineReflectanceProductQa()
        
        print('Getting reflectance difference mosaics for '+self.missionSite.site+'_'+self.flightday)
        self.generateReflectanceDifferenceQaMosaics()
        
        print('Getting reflectance water masks for '+self.missionSite.site+'_'+self.flightday) 
        self.generateWaterMaskTifs()

        print('Getting reflectance spectra for water and vegetation for '+self.missionSite.site+'_'+self.flightday) 
        self.generateReflectancePngs()

        print('Getting water mask mosaic for '+self.missionSite.site+'_'+self.flightday)
        self.generateWaterMaskQaMosaic()
        
        print('Getting reflectance ancillary data product mosaics for '+self.missionSite.site+'_'+self.flightday)
        self.generateReflectanceAncillaryRasterMosaics()
        
        print('Getting reflectance RGB data product mosaics for '+self.missionSite.site+'_'+self.flightday)
        self.generateReflectanceRgbRasterMosaics()
        
        print('Getting reflectance RGB data histogram plot for '+self.missionSite.site+'_'+self.flightday)
        self.generateRgbDistributionPlot()
        
        print('Getting PNG map plots for '+self.missionSite.site+'_'+self.flightday)
        self.getMosaicQaPngMaps()
        
        print('Getting PNG histograms for product mosaics for '+self.missionSite.site+'_'+self.flightday)
        self.getSpectrometerQaReportHistograms()
        
        print('Getting report summaries for '+self.missionSite.site+'_'+self.flightday)
        self.getSpectrometerQaReports()
        
        print('Generating Spectrometer L2 QA PDF Document for '+self.missionSite.site+'_'+self.flightday)
        self.generateL2QaPdf()

        print('Generating Spectrometer L2 QA Markdown and HTML Documents for '+self.missionSite.site+'_'+self.flightday)
        self.generateL2QaHtml()

        print('Finished '+self.missionSite.site+'_'+self.flightday)

    def generateFlightlineReflectanceProductQa(self):
        
        self.getReflectanceDifferenceFlightlineTifs()
        
        self.getProductFiles()
        
        self.getReflectanceMeanDifferenceFlightlines()
        
        rmsArrays,maxIndicesArrays,maxDiffArray = self.getFlightlineReflectanceDifferenceSummaryStats()
        
        self.getFlightlineRmsAndMaxWavelgthErrorRasters(rmsArrays,maxIndicesArrays,maxDiffArray)

        self.getProductFiles()

        self.wavelengthStandardDeviation = (np.sum(self.sumWavelengthVariance,axis=0)[:,None]/np.sum(self.wavelengthSampleCount,axis=0)[:,None])**0.5
        
        self.wavelengthTotalRms = (np.sum(self.sumWavelengthDifferenceSquared,axis=0)[:,None]/np.sum(self.wavelengthSampleCount,axis=0)[:,None])**0.5
        
        self.maxDifferenceWavelengthsTotal = np.max(self.maxDifferenceWavelengths,axis=0)[:,None]
        
        self.getWavelengthDifferenceSummaryPlots()

    def moveLidarFlightLog(self):

        os.makedirs(self.sbetQaGpsImuDir,exist_ok=True)

        
        for file in self.logFiles:
            if 'Lidar' in file or 'lidar' in file and file.endswith('.pdf'):
                self.lidarLogFile = file
                break                                                                                          

        print('\nLidar flight log: ' + self.lidarLogFile)

        lidarLogDestination = os.path.join(self.sbetQaGpsImuDir,os.path.basename(self.lidarLogFile))

        if os.path.exists(lidarLogDestination):
            print('\nLiDAR flight log already exists in QA folder: ' + str(lidarLogDestination))
        else:
            shutil.copy2(self.lidarLogFile, self.sbetQaGpsImuDir)
            print('\nLiDAR flight log copied to QA folder: ' + str(lidarLogDestination))

    def downloadL1DirectionalReflectanceH5s(self):

        print('Downloading spectrometer '+'L1'+' '+'ReflectanceH5'+' for '+self.missionId)
        
        self.dl.download_aop_object(self.SpectrometerL1DirectionalReflectanceDir)
        
        self.getProductFiles()

    def downloadL1BidirectionalReflectanceH5s(self):

        print('Downloading spectrometer '+'L1'+' '+'ReflectanceH5'+' for '+self.missionId)

        self.dl.download_aop_object(self.SpectrometerL1BidirectionalReflectanceDir)

        self.getProductFiles()

    def downloadL1RadianceH5s(self):

        print('Downloading spectrometer '+'L1'+' '+'RadianceH5'+' for '+self.missionId)

        self.dl.download_aop_object(self.SpectrometerL1RadianceDir)

        self.getProductFiles()

    def downloadAllL1SpectrometerProducts(self):

        print('Downloading spectrometer '+'L1 '+'products '+ 'for '+self.missionId)

        self.downloadL1ReflectanceH5s()
        self.downloadL1RadianceH5s()

    def downloadL2Albedo(self):

        print('Downloading '+'L2 '+'Albedo '+ 'for '+self.missionId)

        self.dl.download_aop_object(self.SpectrometerL2AlbedoDir)

        self.getProductFiles()

    def downloadL2Fpar(self):

        print('Downloading '+'L2 '+'FPAR '+ 'for '+self.missionSite.site+'_'+self.flightday)

        self.dl.download_aop_object(self.SpectrometerL2FparDir)

        self.getProductFiles()

    def downloadL2Lai(self):

        print('Downloading '+'L2 '+'LAI '+ 'for '+self.missionSite.site+'_'+self.flightday)

        self.dl.download_aop_object(self.SpectrometerL2FparDir)

        self.getProductFiles()

    def downloadL2VegetationIndices(self):

        print('Downloading '+'L2 '+'Vegetation Indices '+ 'for '+self.missionSite.site+'_'+self.flightday)

        self.dl.download_aop_object(self.SpectrometerL2VegIndicesDir)

        self.getProductFiles()
        self.unzipL2VegetationIndices()
        self.getProductFiles()
        for zipFile in self.SpectrometerL2VegIndicesZipFiles:
            os.remove(zipFile)
        self.getProductFiles()

    def downloadL2WaterIndices(self):

        print('Downloading '+'L2 '+'Water Indices '+ 'for '+self.missionSite.site+'_'+self.flightday)

        self.dl.download_aop_object(self.SpectrometerL2VegIndicesDir)

        self.getProductFiles()
        self.unzipL2WaterIndices()
        self.getProductFiles()
        for zipFile in self.SpectrometerL2WaterIndicesZipFiles:
            os.remove(zipFile)
        self.getProductFiles()

    def downloadAllL2SpectrometerProducts(self):

        print('Downloading all '+'L2 '+'spectrometer products '+ 'for '+self.missionSite.site+'_'+self.flightday)

        self.downloadL2Albedo()
        self.downloadL2Fpar()
        self.downloadL2Lai()
        self.downloadL2VegetationIndices()
        self.downloadL2WaterIndices()

    def downloadL1Camera(self):

        print('Downloading camera '+'L1'+' '+'camera'+' for '+self.missionSite.site+'_'+self.flightday)

        self.dl.download_aop_object(self.CameraL1ImagesDir)

        self.getProductFiles()

    def cleanupRadiance(self):

        self.getProductFiles()

        deleteExtensionList = ['_rdn_eph','_rdn','_rdn.hdr','_rdn_obs','_rdn_obs.hdr','_rdn_ort_igm5',
                              '_rdn_ort_igm5.hdr','_rdn_ort_igm',
                             '_rdn_ort_igm.hdr','_rdn_ort_navout_mout','_rdn_ort_tim','_rdn_ort_step1b_report.txt']

        deleteFiles(self.SpectrometerL1RadianceProcessingFiles,deleteExtensionList)

    def cleanupAtcor(self):

        self.getProductFiles()

        deleteExtensionListRadiance = ['_rdn_obs_sca','_rdn_obs_sca.bsq','_rdn_obs_sca.hdr',
                           '_rdn_ort_ele.bsq','_rdn_ort_ele.hdr','_rdn_ort_igm_ort',
                           '_rdn_ort_igm_ort.hdr']

        deleteFiles(self.SpectrometerL1RadianceProcessingFiles,deleteExtensionListRadiance)

        deleteExtensionListReflectance = ['_rdn_ort.bsq','_rdn_ort.hdr','_rdn_ort_original.bsq',
                                      '_rdn_ort_original.hdr','_rdn_ort_atm_qcl.bsq',
                                      '_rdn_ort_atm_qcl.hdr','_backup.cal']

        deleteFiles(self.SpectrometerL1ReflectanceProcessingFiles,deleteExtensionListReflectance)

    def cleanupFlxAlbedo(self):

        self.getProductFiles()

        deleteExtensionList = ['_flx.hdr','_flx.bsq']

        deleteFiles(self.SpectrometerL1ReflectanceProcessingFiles,deleteExtensionList)

    def cleanupSlopeAspect(self):

        self.getProductFiles()

        deleteExtensionList = ['_rdn_ort_sm_envislpasp.bsq','_rdn_ort_sm_envislpasp.hdr']

        deleteFiles(self.SpectrometerL1RadianceProcessingFiles,deleteExtensionList)

    def cleanupWaterVegetationTifs(self):

        self.getProductFiles()

        deleteExtensionList = ['.tif']

        deleteFiles(self.SpectrometerL2VegIndicesTifFiles,deleteExtensionList)
        deleteFiles(self.SpectrometerL2WaterIndicesTifFiles,deleteExtensionList)
        deleteFiles(self.SpectrometerL3VegIndicesTifFiles,deleteExtensionList)
        deleteFiles(self.SpectrometerL3WaterIndicesTifFiles,deleteExtensionList)

    def cleanupL1H5(self):

        self.getProductFiles()

        deleteExtensionListRadiance = ['_rdn_ort_sm_asp.bsq','_rdn_ort_sm_asp.hdr','_rdn_ort_sm_ele.bsq',
                               '_rdn_ort_sm_ele.hdr','_rdn_ort_sm_ele_shadow_report.log',
                               '_rdn_ort_sm_ele_skyview_report.log','_rdn_ort_sm_sky.bsq',
                               '_rdn_ort_sm_sky.hdr','_rdn_ort_sm_slp.bsq',
                               '_rdn_ort_sm_slp.hdr','_shd.bsq','_shd.hdr','_rdn_obs_ort',
                               '_rdn_obs_ort.hdr','_rdn_ort','_rdn_ort.hdr'
                               '_rdn_ort_glt','_rdn_ort_glt.hdr']

        deleteFiles(self.SpectrometerL1RadianceProcessingFiles,deleteExtensionListRadiance)

        deleteExtensionListReflectance = ['_rdn_ort.inn','_rdn_ort_atm.bsq','_rdn_ort_atm.bsq',
                                          '_rdn_ort_atm.hdr','_rdn_ort_atm.log','_rdn_ort_atm_aot.bsq',
                                          '_rdn_ort_atm_aot.hdr','_rdn_ort_atm_ddv.bsq','_rdn_ort_atm_ddv.hdr',
                                          '_rdn_ort_atm_nodata.bsq','_rdn_ort_atm_nodata.hdr','_rdn_ort_atm_visindex.bsq',
                                          '_rdn_ort_atm_visindex.hdr','_rdn_ort_atm_wv.bsq','_rdn_ort_atm_wv.hdr',
                                          '_rdn_ort_ilu.bsq','_rdn_ort_ilu.hdr','_rdn_ort_out_hcw.bsq','_rdn_ort_out_hcw.hdr']

        deleteFiles(self.SpectrometerL1ReflectanceProcessingFiles,deleteExtensionListReflectance)

    def organizeReflectanceFiles(self):
        
        self.getProductFiles()
        
        os.makedirs(self.SpectrometerL1DirectionalReflectanceDir,exist_ok=True)
        os.makedirs(self.SpectrometerL1BidirectionalReflectanceDir,exist_ok=True)
          
        
        for file in self.SpectrometerL1ReflectanceFiles:
            
            if 'bidirectional' in os.path.basename(file):
                shutil.move(file,os.path.join(self.SpectrometerL1BidirectionalReflectanceDir,os.path.basename(file)))
            else:
                shutil.move(file,os.path.join(self.SpectrometerL1DirectionalReflectanceDir,os.path.basename(file)))

    def runCameraOrtho(self):
        
        self.getProductFiles()

        IDL.run('cd, "' + self.productBaseDir + '"')
        IDL.run('.compile ' + '"' + 'run_aig_neon_dms_ortho.pro' + '"')

        IDL.orthoProcessingParamsFile = str(self.orthoProcessingParamsFile)

        IDL.run('run_aig_neon_dms_ortho(orthoProcessingParamsFile)',stdout=True)

    def cleanVegIndexTifs(self):
        
        for tifFile in self.SpectrometerL2VegIndicesTifFiles:
            os.remove(tifFile)

    def cleanWaterIndexTifs(self):
        
        for tifFile in self.SpectrometerL2WaterIndicesTifFiles:
            os.remove(tifFile)

    def cleanRadianceProcessingFolder(self):
        
        if os.path.exists(self.SpectrometerL1RadianceProcessingDir):
        
            shutil.rmtree(self.SpectrometerL1RadianceProcessingDir)

    def cleanReflectanceProcessingFolder(self):
        
        if os.path.exists(self.SpectrometerL1ReflectanceProcessingDir):
        
            shutil.rmtree(self.SpectrometerL1ReflectanceProcessingDir)

    def cleanL2SpectrometerProducts(self):
        
        self.cleanVegIndexTifs()

        self.cleanWaterIndexTifs()
        
        self.cleanRadianceProcessingFolder()
        
        self.cleanReflectanceProcessingFolder()

    def cleanCameraOrthoFolder(self):
        
        for file in self.orthoProcessingFiles:
            
            if file.endswith(self.flightday+'.txt') or file.endswith(self.flightday+'_report.txt'):
                continue
            else:
                os.remove(file)

    def processL1RadianceSpectrometer(self,geoidFile,demFile,sapFile,domain,visit):

        print('Running L1 DN to Radiance for '+self.missionId)
        self.processRadiance()
        print('Running L1 Radiance QA for '+self.missionId)
        self.processRadianceQa(domain,visit)
        print('Running L1 Radiance Ortho for '+self.missionId)
        self.processOrthoRadiance(geoidFile,demFile,sapFile)
        print('Running L1 Radiance Ortho QA for '+self.missionId)
        self.generateOrthoQa()
        print('Running L1 Radiance H5 for '+self.missionId)
        self.processH5RadianceWriter()
        print('Generating L1 Radiance H5 RGB tifs for '+self.missionId)
        self.generateRgbRadianceTifs()
        print('Cleaning up radiance files for '+self.missionId)
        self.cleanupRadiance()
        print('Cleaning up slope / aspect files for '+self.missionId)
        self.cleanupSlopeAspect()

    def processL1ReflectanceSpectrometer(self,aigdsm):

        print('Running Atcor Topo clips for '+self.missionId)
        self.clipTopoForAtcor(aigdsm)
        print('Running Smooth Elevation for '+self.missionId)
        self.processSmoothElevation()
        print('Running Slope Aspect for '+self.missionId)
        self.processSlopeAspect()
        print('Preparing ATCOR inputs for '+self.missionId)
        self.prepareAtcorInputs()
        print('Running ATCOR reflectance retrieval for '+self.missionId)
        self.processReflectance()
        print('Running post ATCOR updates for '+self.missionId)
        self.postAtcorUpdates()
        print('Running L1 reflectance H5 for '+self.missionId)
        self.postAtcorFileUpdates()
        self.processH5ReflectanceWriter()
        print('Cleaning up atcor files for '+self.missionId)
        self.cleanupAtcor()
        print('Running L1 BRDF correction for '+self.missionId)
        self.applyBrdfCorrection()
        print('Running L1 BRDF reflectance H5 for '+self.missionId)
        self.processH5ReflectanceWriter(isBrdf=True)
        
        self.renameDirectionalReflectanceH5()
        
        print('Cleaning up files used for H5 writer for '+self.missionId)
        self.cleanupL1H5()

    def processL2SpectrometerProducts(self):

        print('Running L2 Veg Indices for '+self.missionId)
        self.generateVegetationIndices()
        print('Zipping L2 Veg Indices fobr '+self.missionId)
        self.zipVegetationIndices()
        print('Running L2 Water Indices for '+self.missionId)
        self.generateWaterIndices()
        print('Zipping L2 Water Indices for '+self.missionId)
        self.zipWaterIndices()
        print('Running L2 FPAR for '+self.missionId)
        self.generateFpar()
        print('Running L2 LAI for '+self.missionId)
        self.generateLai()
        print('Running L2 Albedo for '+self.missionId)
        if self.reapplyBrdf:
            self.downloadL2Albedo()
            self.getProductFiles()
            if not self.SpectrometerL2AlbedoFiles:
                sDriveAlbedos = self.SpectrometerL2AlbedoFiles = collectFilesInPath(self.SpectrometerL2AlbedoDir.replace('D:','S:'),'albedo.tif')
                for file in sDriveAlbedos:
                    shutil.copy(file,os.path.join(self.SpectrometerL2AlbedoDir,os.path.basename(file)))
        else:
            self.generateAlbedo()
            
        print('Cleaning up flx files for '+self.missionId)
        self.cleanupFlxAlbedo()

    def spectrometerProcessWorkflow(self,geoidFile,aigdsm,sapFile,spectrometerProductQaList,domain,visit):
        print('Downloading Spectrometer Raw data for '+ self.missionId)
        self.downloadDataRawSpectrometerProcessing()
        print('Running Spectrometer flightline boundaries for '+ self.missionId)
        self.generateSpectrometerKmls()
        print('Running L1 Radiance Spectrometer Products for '+ self.missionId)
        self.processL1RadianceSpectrometer(geoidFile,aigdsm,sapFile,domain,visit)
        print('Running L1 Reflectance Spectrometer Products for '+ self.missionId)
        self.processL1ReflectanceSpectrometer(aigdsm)
        print('Running L2 Spectrometer Products for '+ self.missionId)
        self.processL2SpectrometerProducts()
        self.organizeReflectanceFiles()
        print('Running L2 Spectrometer Product QA for '+ self.missionId)
        self.processFlightlineQa()
        print('Cleaning L2 Spectrometer data for '+ self.missionId)
        self.cleanL2SpectrometerProducts()

    def processSbet(self):
        
        if os.path.exists(self.sbetTrajectoryFile):
            pass
        else:
            print('Downloading SBET Raw data for '+ self.missionId)
            self.downloadGpsImu()
            print('Moving lidar flight log into sbet QA for '+ self.missionId)
            self.moveLidarFlightLog()
            print('Getting non-CORS GPS stations for '+ self.missionId)
            self.getGpsBasestations()
            print('Downloading GPS basestations for '+ self.missionId)
            self.downloadGpsBasestations()
            print('Converting Trimble basestations to RINEX for '+ self.missionId)
            self.convertTrimble2Rinex()
            print('adding antenna information to RINEX for  '+ self.missionId)
            self.addAntennaInfoToRinex()
            print('Generating posbat file for '+ self.missionId)
            self.generatePosbatFile()
            print('Running POSPAC for '+ self.missionId)
            self.runPOSPac()
            print('Prepping POSPAC sbet QA figures for '+ self.missionId)
            self.prepQaFigures()
            
            self.moveSbetOutputs()

    def runWavexWorkflow(self):

        print('Downloading Riegl Raw Lidar data for '+ self.missionId)
        self.downloadRawLidar()
        print('Getting Rpp records data for '+ self.missionId)
        self.getRppRecords()
        print('Getting WavEx processing files for '+ self.missionId)
        self.getWavexProcessingFiles()
        print('Getting ATM files for WavEx for '+ self.missionId)
        self.createWavexAtmFiles()
        print('Running WavEx for '+ self.missionId)
        self.runWavex()
        print('Renaming WavEx files for '+ self.missionId)
        self.renameWavexOutfiles()

    def runOptechWorkflow(self):

        # print('Downloading Optech Raw Discrete Lidar data for '+ self.missionId)
        self.downloadRawLidar() # downloads the SBET files and .range file to the Daily and Raw folders, respectively

        # set up LMS project folder (eg. LMS/LMS_YYYYMMDDHH) and populate Data folder (YYYYMMDDHH) with the raw discrete data
        self.moveOptechMissionData()
        
        # print('Raw Discrete Lidar Files:')
        print(self.rawDiscreteLidarFiles)

        # determine and pull in the instrument files (from the lookup), grf files, and flight plan files
        self.getOptechProcessingFiles()

        # create atmospheric (.met) files
        self.makeOptechMetFile()

        # run Optech LMS CLI
        self.runOptechLMS()

        # move output files to Internal folder and use line-matching to re-name


    def reapplyBrdfCorrectionWorkflow(self):
        
        self.reapplyBrdf = True
        self.downloadProcessedSbet()
        self.downloadL1DirectionalReflectanceH5s()
        self.downloadL1RadianceH5s()
        os.makedirs(self.SpectrometerL1ReflectanceDir,exist_ok=True)
        for file in self.SpectrometerL1DirectionalReflectanceFiles:
            shutil.move(file,os.path.join(self.SpectrometerL1ReflectanceDir,os.path.basename(file)))
        self.reverseRenameDirectionalReflectanceH5()
        os.makedirs(self.SpectrometerL1ReflectanceProcessingDir,exist_ok=True)
        self.applyBrdfCorrection()
        self.prepareForBrdfH5Writing()
        self.rewriteAtcorLogs()
        self.processH5ReflectanceWriter(isBrdf=True)
        self.cleanupL1H5()
        self.processL2SpectrometerProducts()
        self.organizeReflectanceFiles()
        self.processFlightlineQa()

    def prepareForBrdfH5Writing(self):
        
        
        self.reconstructObs()
        #Only do one band for radiance to make processing faster. We need the header, not the data
        self.reconstructRadiance(bandIndexesToRead = [1,2]) 
        self.getReflectanceH5AncillaryAsEnvi()
        self.rewriteNamesForH5Writing()
        self.rewriteAtcorLogs()

    def convertH5sToTifs(self):
        
        self.getProductFiles()
        
        os.makedirs(self.brdfInputsTempDir,exist_ok=True)
        for h5File in self.SpectrometerL1DirectionalReflectanceFiles:
            getAllRastersFromH5(h5File,outDir = self.brdfInputsTempDir,skipRasters = ['Reflectance'])

    def convertH5TifsToEnvi(self):
         
        self.getProductFiles()
        
        for tifFile in self.brdfInputsTempFiles:
            outFile = os.path.splitext(tifFile)[0]
            convertGtiffToEnvi(tifFile,outFile,self.filePathToEnviProjCs)

    def reconstructObs(self,spatialIndexesToRead = None,bandIndexesToRead = None):
        print('Extracting ENVI Obs files from H5 for '+ self.missionId)
        self.getProductFiles()
        os.makedirs(self.SpectrometerL1RadianceProcessingDir,exist_ok=True)
        for flightline in self.FlightLines:
            flightline.convertRadianceH5ToEnviObs(self.SpectrometerL1RadianceDir,self.SpectrometerL1RadianceProcessingDir,self.filePathToEnviProjCs,spatialIndexesToRead = spatialIndexesToRead ,bandIndexesToRead = bandIndexesToRead )

    def reconstructIgm(self,spatialIndexesToRead = None,bandIndexesToRead = None):
        print('Extracting ENVI Igm files from H5 for '+ self.missionId)
        self.getProductFiles()
        os.makedirs(self.SpectrometerL1RadianceProcessingDir,exist_ok=True)
        for flightline in self.FlightLines:
            flightline.convertRadianceH5ToEnviIgm(self.SpectrometerL1RadianceDir,self.SpectrometerL1RadianceProcessingDir,self.filePathToEnviProjCs,spatialIndexesToRead = spatialIndexesToRead ,bandIndexesToRead = bandIndexesToRead )

    def reconstructGlt(self,spatialIndexesToRead = None,bandIndexesToRead = None):
        
        
        print('Extracting ENVI GLT files from H5 for '+ self.missionId)
        self.getProductFiles()
        os.makedirs(self.SpectrometerL1RadianceProcessingDir,exist_ok=True)
        for flightline in self.FlightLines:
            flightline.convertRadianceH5ToEnviGlt(self.SpectrometerL1RadianceDir,self.SpectrometerL1RadianceProcessingDir,self.filePathToEnviProjCs,spatialIndexesToRead = spatialIndexesToRead ,bandIndexesToRead = bandIndexesToRead )

    def reconstructRadiance(self,spatialIndexesToRead = None,bandIndexesToRead = None):
        
        print('Extracting ENVI Radiance files from H5 for '+ self.missionId)
        self.getProductFiles()
        os.makedirs(self.SpectrometerL1RadianceProcessingDir,exist_ok=True)
        # for flightline in self.FlightLines:
        #     flightline.convertH5ToEnviRadiance(self.SpectrometerL1RadianceDir,self.SpectrometerL1RadianceProcessingDir,self.filePathToEnviProjCs)              
        with Pool(processes=5) as pool:
            processFunction = partial(convertH5ToEnviRadianceHelper,radianceH5Folder=self.SpectrometerL1RadianceDir,outputRadianceDir=self.SpectrometerL1RadianceProcessingDir,filePathToEnviProjCs=self.filePathToEnviProjCs,spatialIndexesToRead = spatialIndexesToRead ,bandIndexesToRead = bandIndexesToRead )
            pool.map(processFunction,self.FlightLines)
        
    def getReflectanceH5AncillaryAsEnvi(self,spatialIndexesToRead = None,bandIndexesToRead = None):
        
        skipRasters = ['Reflectance']
        os.makedirs(self.brdfInputsTempDir,exist_ok=True)
        for flightline in self.FlightLines:
        
            flightline.convertReflectanceH5ToEnvi(self.SpectrometerL1ReflectanceDir,self.brdfInputsTempDir,self.filePathToEnviProjCs,skipRasters=skipRasters,spatialIndexesToRead = spatialIndexesToRead,bandIndexesToRead = bandIndexesToRead)

    def rewriteNamesForH5Writing(self):
        
        self.getProductFiles()
        
        os.makedirs(self.SpectrometerL1ReflectanceProcessingDir,exist_ok=True)
        
        for flightline in self.FlightLines:
            for file in self.brdfInputsTempFiles:
            
                if flightline.lineNumber in file:
                
                    if 'Aerosol_Optical_Thickness' in file:
                        
                        newFileName = file.replace(flightline.reflectanceSpectrometerH5FilePrefixLineNum,flightline.radianceOrtSpectrometerFile).replace('directional_Aerosol_Optical_Thickness','atm_aot')
                        newFileName = os.path.join(self.SpectrometerL1ReflectanceProcessingDir,os.path.basename(newFileName))
 
                    elif 'Cast_Shadow' in file:
                        
                        newFileName = file.replace(flightline.reflectanceSpectrometerH5FilePrefixLineNum,flightline.radianceOrtSpectrometerFile).replace('directional_Cast_Shadow','sm_zen00_azi000_shd')
                        newFileName = os.path.join(self.SpectrometerL1RadianceProcessingDir,os.path.basename(newFileName))
                        
                    elif 'Dark_Dense_Vegetation_Classification' in file:
                        
                        newFileName = file.replace(flightline.reflectanceSpectrometerH5FilePrefixLineNum,flightline.radianceOrtSpectrometerFile).replace('directional_Dark_Dense_Vegetation_Classification','atm_ddv')
                        newFileName = os.path.join(self.SpectrometerL1ReflectanceProcessingDir,os.path.basename(newFileName))
                        
                    elif 'Haze_Cloud_Water_Map' in file:
                        
                        newFileName = file.replace(flightline.reflectanceSpectrometerH5FilePrefixLineNum,flightline.radianceOrtSpectrometerFile).replace('directional_Haze_Cloud_Water_Map','out_hcw')
                        newFileName = os.path.join(self.SpectrometerL1ReflectanceProcessingDir,os.path.basename(newFileName))
                           
                    elif 'Illumination_Factor' in file:
                        
                        newFileName = file.replace(flightline.reflectanceSpectrometerH5FilePrefixLineNum,flightline.radianceOrtSpectrometerFile).replace('directional_Illumination_Factor','ilu')
                        newFileName = os.path.join(self.SpectrometerL1ReflectanceProcessingDir,os.path.basename(newFileName))
                        
                    elif 'Sky_View_Factor' in file:
                        
                        newFileName = file.replace(flightline.reflectanceSpectrometerH5FilePrefixLineNum,flightline.radianceOrtSpectrometerFile).replace('directional_Sky_View_Factor','sm_sky')
                        newFileName = os.path.join(self.SpectrometerL1RadianceProcessingDir,os.path.basename(newFileName))
                        
                    elif 'Slope' in file:
                        
                        newFileName = file.replace(flightline.reflectanceSpectrometerH5FilePrefixLineNum,flightline.radianceOrtSpectrometerFile).replace('directional_Slope','sm_slp')
                        newFileName = os.path.join(self.SpectrometerL1RadianceProcessingDir,os.path.basename(newFileName))
                        
                    elif 'Aspect' in file:
                        
                        newFileName = file.replace(flightline.reflectanceSpectrometerH5FilePrefixLineNum,flightline.radianceOrtSpectrometerFile).replace('directional_Aspect','sm_asp')
                        newFileName = os.path.join(self.SpectrometerL1RadianceProcessingDir,os.path.basename(newFileName))
                        
                    elif 'Smooth_Surface_Elevation' in file:
                        
                        newFileName = file.replace(flightline.reflectanceSpectrometerH5FilePrefixLineNum,flightline.radianceOrtSpectrometerFile).replace('directional_Smooth_Surface_Elevation','sm_ele')
                        newFileName = os.path.join(self.SpectrometerL1RadianceProcessingDir,os.path.basename(newFileName))
                        
                    elif 'Visibility_Index_Map' in file:
                        
                        newFileName = file.replace(flightline.reflectanceSpectrometerH5FilePrefixLineNum,flightline.radianceOrtSpectrometerFile).replace('directional_Visibility_Index_Map','atm_visindex')
                        newFileName = os.path.join(self.SpectrometerL1ReflectanceProcessingDir,os.path.basename(newFileName))
                        
                    elif 'Water_Vapor_Column' in file:
                        
                        newFileName = file.replace(flightline.reflectanceSpectrometerH5FilePrefixLineNum,flightline.radianceOrtSpectrometerFile).replace('directional_Water_Vapor_Column' ,'atm_wv')
                        newFileName = os.path.join(self.SpectrometerL1ReflectanceProcessingDir,os.path.basename(newFileName))
                    
                    else:
                        continue
                   
                    shutil.copy(file,newFileName)
                
        for flightline in self.FlightLines:
            for file in self.SpectrometerL1ReflectanceProcessingFiles:
                
                if flightline.lineNumber in file:
                
                    newFileName = file.replace(flightline.reflectanceSpectrometerH5FilePrefixLineNum,flightline.reflectanceSpectrometerH5FilePrefix)
                    newFileName = os.path.join(self.SpectrometerL1ReflectanceProcessingDir,os.path.basename(newFileName))
                    if os.path.exists(newFileName):
                        print(newFileName + ' already exists')
                    else:
                        os.rename(file,newFileName)
                    
    def rewriteAtcorLogs(self):
          
        for flightline in self.FlightLines:
        
            flightline.writeAtcorLogsFromReflectanceH5(self.SpectrometerL1ReflectanceDir,self.SpectrometerL1ReflectanceProcessingDir,self.SpectrometerL1RadianceProcessingDir)

    def generateLidarUncertainty(self):
     
        self.getProductFiles()
        os.makedirs(self.lidarProcessingLastoolsUncertaintyDir,exist_ok=True)
        self.getSbet()
        with Pool(processes=15) as pool:
            processFunction = partial(generateSimulatedLidarUncertainty,sbet=self.sbet,payload=self.payload,outputDir=self.lidarProcessingLastoolsUncertaintyDir)
            pool.map(processFunction,self.L1LidarLasFiles)
        
        # for file in self.L1LidarLasFiles:
        #     generateSimulatedLidarUncertainty(file,self.sbet,self.payload,self.metadataLidarUncertaintyDir)
        
    def resampleSpectrum(self, productType = 'bidirectional',outputType='ENVI'):
        
        os.makedirs(self.spectrallyResampledDir,exist_ok=True)
        
        if productType == 'bidirectional':
            inputH5Dir = self.SpectrometerL1BidirectionalReflectanceDir
        elif productType == 'directional':
            inputH5Dir = self.SpectrometerL1DirectionalReflectanceDir
        
        with Pool(processes=5) as pool:
            processFunction = partial(resampleH5SpectrumHelper,inputDir=inputH5Dir,outputDir=self.spectrallyResampledDir,outputType=outputType,filePathToEnviProjCs=self.filePathToEnviProjCs)
            pool.map(processFunction,self.FlightLines)
        

class payloadClass:
    def __init__(self,year,payloadId):

        self.payloadId = payloadId
        self.payload = payloadId[0:2]
        self.campaign = payloadId[2:4]

        currentFile = os.path.abspath(os.path.join(__file__))
        nisPayloadInfoLookupFile = os.path.join(currentFile.lower().split('gold_pipeline')[0], 'Gold_Pipeline', 'ProcessingPipelines', 'NIS', 'res', 'CameraModels', 'NIS_parameter_summary_for_pipeline.csv')
        self.nisCameraModelDir = os.path.join(currentFile.lower().split('gold_pipeline')[0], 'Gold_Pipeline', 'ProcessingPipelines', 'NIS', 'res', 'CameraModels')
        nisPayloadInfoDf = pd.read_csv(nisPayloadInfoLookupFile,dtype=str)
        nisPayloadRow = nisPayloadInfoDf[(nisPayloadInfoDf['Year'] == year) & (nisPayloadInfoDf['Payload'] == payloadId)].squeeze()
        self.nisId = nisPayloadRow['NIS']
        self.geoCalSite = nisPayloadRow['Site']
        self.nisLeverArm = [nisPayloadRow['Lever Arm X'],nisPayloadRow['Lever Arm Y'],nisPayloadRow['Lever Arm Z']]
        self.nisNominalTimingOffset = nisPayloadRow['Nominal Timing Offset']
        self.nisProcessingStartDate = nisPayloadRow['CM Start Date']
        self.nisProcessingEndDate = nisPayloadRow['CM End Date']
        self.nisCameraGeolocationModel = nisPayloadRow['Camera Model (CM)']
        self.nisAtcorSensorModel = nisPayloadRow['ATCOR Sensor']

        cameraInfoLookupFile = os.path.join(currentFile.lower().split('gold_pipeline')[0], 'Gold_Pipeline', 'ProcessingPipelines', 'Camera', 'data', 'camera', 'camera_parameters/CameraApplicability.txt')
        cameraPayloadInfoDf = pd.read_csv(cameraInfoLookupFile,dtype=str,delimiter="\t")
        cameraPayloadRow = cameraPayloadInfoDf[(cameraPayloadInfoDf['Year'] == year) & (cameraPayloadInfoDf['PnCm'] == payloadId)].squeeze()
        self.cameraMake = cameraPayloadRow['Camera_Body']
        self.cameraSerialNumber = cameraPayloadRow['SN']
        self.cameraGeolocationModel = cameraPayloadRow['Camera_Model']
        self.initialCameraGeolocationModel = cameraPayloadRow['Nominal_Camera_Model']
        self.cameraProcessingStartDate = cameraPayloadRow['From']
        self.cameraProcessingEndDate = cameraPayloadRow['To']
        self.cameraImagesExtension = cameraPayloadRow['Image_Ext']
        self.cameraSuffixExclude = cameraPayloadRow['Suffix_Exclude']
        self.cameraFilenameType = cameraPayloadRow['Filename_Type']
        self.cameraLogfileType = cameraPayloadRow['Logfile_Type']
        self.cameraLogfileRE = cameraPayloadRow['Logfile_RE']
        self.cameraMinDeltaT = cameraPayloadRow['Min_Delta_T']
        self.cameraMinDeltaTime = cameraPayloadRow['Min_Delta_Time']

        sbetLeverArmLookupFile = os.path.join(currentFile.lower().split('gold_pipeline')[0], 'Gold_Pipeline', 'ProcessingPipelines', 'SBET', 'res', 'lever_arm_lookup.csv')
        sbetLeverArmInfoDf = pd.read_csv(sbetLeverArmLookupFile,dtype=str)
        sbetLeverArmdRow = sbetLeverArmInfoDf[(sbetLeverArmInfoDf ['YYYY_PxCx'] == year+'_'+payloadId)].squeeze()
        self.sbetLeverArm = [sbetLeverArmdRow['x'],sbetLeverArmdRow['y'],sbetLeverArmdRow['z']]

        if 'P3' == self.payload:
            self.lidarSensorManufacturer = 'riegl'
            self.lidarSensorModel = 'q780'
            self.cameraTimingOffset = 18.0
            self.lidarScanAngleError = np.sqrt((0.003*np.pi/180)**2+(0.00025/2)**2)
            self.laserRangeError = 0.03
            
        else:
            self.cameraTimingOffset = 0.0
            if int(year) < 2021:
                self.lidarSensorManufacturer = 'optech'
                self.lidarSensorModel = 'gemini'
                self.lidarScanAngleError = np.sqrt((0.003*np.pi/180)**2+(0.00025/2)**2)
                self.laserRangeError = 0.08
            else:
                self.lidarSensorManufacturer = 'optech'
                self.lidarSensorModel = 'galaxy'
                self.lidarScanAngleError = np.sqrt((0.003*np.pi/180)**2+(0.0008/2)**2)
                self.laserRangeError = 0.03

class siteClass:
    def __init__(self,site):

        self.site = site

        currentFile = os.path.abspath(os.path.join(__file__))
        driveLetter = currentFile[0]
        self.site_lookup_db = driveLetter + ":/Gold_Pipeline/ProcessingPipelines/res/Lookups/neon_aop_lookup.db"
        conn = sqlite3.connect(self.site_lookup_db)
        epsgLookup = driveLetter + ':/Gold_Pipeline/ProcessingPipelines/res/Lookups/EPSG_lookup.txt'
        siteQuery = "SELECT * FROM site"
        siteLookupDf = pd.read_sql_query(siteQuery, conn)
        siteLookupRow = siteLookupDf[(siteLookupDf['site_code'] == self.site)].squeeze()
        self.domain = siteLookupRow['domain']
        self.fullSiteName = siteLookupRow['site_name']
        self.state = siteLookupRow['state']
        self.siteType = siteLookupRow['site_type']
        self.utmZone = str(int(siteLookupRow['utm_zone']))
        self.nad83wgs84offset = siteLookupRow['wgs84_nad83_offset']

        epsg_dict = {}
        with open(epsgLookup) as f:
            for line in f:
                utm=" ".join(line.split()[0:5]).replace('UTM Zone ','').replace('Northern Hemisphere','N').replace('Southern Hemisphere','S')
                epsg=line.split()[-1]
                epsg_dict[utm] = epsg
        self.epsg = epsg_dict[self.utmZone+' N']
        geoQuery = "SELECT * FROM flight_plan_geo_data"
        geoLookupDf = pd.read_sql_query(geoQuery, conn)
        geoLookupRow = siteLookupDf[(geoLookupDf['site_code'] == self.site)].squeeze()

    def get_flight_plans(self):
        conn = sqlite3.connect(self.site_lookup_db)
        db_sites = [site[0] for site in conn.cursor().execute("SELECT site_code FROM flight_plan_geo_data")]
        if self.site not in db_sites:
            print("ERROR: ",site," is not in the database. Check that site is entered correctly or run add_new_site_to_lookup_db.py to add site to lookup database.")
            sys.exit()
        else:
            try:
                fb_all = []
                fb_all.append(conn.cursor().execute("SELECT flightbox_p1 FROM flight_plan_geo_data WHERE end_date IS NULL AND site_code = ? ", (self.site,)).fetchone()[0])
                fb_all.append(conn.cursor().execute("SELECT flightbox_p2 FROM flight_plan_geo_data WHERE end_date IS NULL AND site_code = ? ", (self.site,)).fetchone()[0])
                fb_all.append(conn.cursor().execute("SELECT flightbox_p3 FROM flight_plan_geo_data WHERE end_date IS NULL AND site_code = ? ", (self.site,)).fetchone()[0])
                fb =  [x for x in fb_all if not x == 'N/A' and not x.strip() == '']
                if fb !=[]:
                    return fb #flightbox shapefiles
                else:
                    try:
                        fb_aquatic = []
                        fb_aquatic.append(conn.cursor().execute("SELECT aquatic_1 FROM flight_plan_geo_data WHERE end_date IS NULL AND site_code = ? ", (self.site,)).fetchone()[0])
                        fb_aquatic.append(conn.cursor().execute("SELECT aquatic_2 FROM flight_plan_geo_data WHERE end_date IS NULL AND site_code = ? ", (self.site,)).fetchone()[0])
                        fb_aquatic =  [x for x in fb_aquatic if not x == 'N/A' and not x.strip() == '']
                        if fb_aquatic ==[]:
                            print('WARNING could not find any terrestrial or aquatic flight plans for site',self.site)
                            return
                        else:
                            return fb_aquatic #aquatic flightbox shapefiles
                    except:
                        print('ERROR! check lookup_db flight_plan_geo_data table for site',self.site)
            except ValueError:
                print('ERROR!',sys.exc_info()[0])


class applanixSbetClass:
    def __init__(self,sbetFile,sbetErrorFile,piinkaruFile,iincalFile):
        
        self.sbetFile = sbetFile
        self.sbetErrorFile = sbetErrorFile
        self.piinkaruFile = piinkaruFile
        self.iincalFile = iincalFile
        self.sbetColumns = 17
        self.sbetErrorColumns = 10
        self.sbetPiinkarusColumns = 9
        self.sbetIincalColumns = 33
        self.sbetEpsg = 4326
        
    def getAllSbetData(self):
        
        self.getSbetData()
        self.getSbetErrorData()
        self.getSbetMetadata()
        self.getSbetCaldata()
        self.getGpsDayOfWeek()
        
    def getSbetData(self):
        
        sbetData = readFlatBinaryFile(self.sbetFile,np.double,numCols=self.sbetColumns)
        
        self.time = sbetData[:,0]               #GPS time of week
        self.latitudeRadians  = sbetData[:,1]            #latitude in degrees (-90.0 to +90.0)
        self.longitudeRadians  = sbetData[:,2]            #longitude in degrees (-180.0 to 180.0)
        self.elevation = sbetData[:,3]                #altitude
        self.xVelocity = sbetData[:,4]               #velocity in x direction
        self.yVelocity = sbetData[:,5]               #velocity in y direction
        self.zVelocity = sbetData[:,6]               #velocity in z direction
        self.roll = sbetData[:,7]               #roll angle
        self.pitch = sbetData[:,8]              #pitch angle
        self.heading = sbetData[:,9]            #heading angle
        self.wander  = sbetData[:,10]            #wander
        self.xForce = sbetData[:,11]             #force in x direction
        self.yForce = sbetData[:,12]             #force in y direction
        self.zForce = sbetData[:,13]             #force in z direction
        self.xAngularRate = sbetData[:,14]           #angular rate in x direction
        self.yAngularRate = sbetData[:,15]           #angular rate in y direction
        self.zAngularRate = sbetData[:,16]           #angular rate in z direction

    def transformSbetToProjection(self,outputEpsg):
        
        self.easting, self.northing = latlonToUtm(self.latitudeRadians* (180 / math.pi), self.longitudeRadians* (180 / math.pi), self.sbetEpsg,outputEpsg)

    def getSbetErrorData(self):
        
        sbetErrorData = readFlatBinaryFile(self.sbetErrorFile,np.double,numCols=self.sbetErrorColumns)
        
        self.errorTime = sbetErrorData[:,0]
        self.xError = sbetErrorData[:,1]
        self.yError = sbetErrorData[:,2]
        self.zError = sbetErrorData[:,3]
        
        self.rollError = sbetErrorData[:,7]
        self.pitchError = sbetErrorData[:,8]
        self.headingError = sbetErrorData[:,9]

    def getSbetMetadata(self):
        
        if os.path.exists(self.piinkaruFile):
            sbetMetadata = readFlatBinaryFile(self.piinkaruFile,np.double,numCols=self.sbetPiinkarusColumns)
            
            self.metadataTime = sbetMetadata[:,0]
            self.numberSatellites = sbetMetadata[:,1]
            self.processingMode = sbetMetadata[:,5]
            self.pdop = sbetMetadata[:,6]
            self.baselineLength = sbetMetadata[:,7]

    def getSbetCaldata(self):
        
        if os.path.exists(self.iincalFile):
            
            calMetadata = readFlatBinaryFile(self.iincalFile,np.double,numCols=self.sbetIincalColumns)
            
            self.metadataTime = calMetadata[:,0]
            self.xLeverArmCal = calMetadata[:,1]
            self.yLeverArmCal = calMetadata[:,2]
            self.zLeverArmCal = calMetadata[:,3]
    
    def getGpsDayOfWeek(self):
        
        self.gpsDayOfWeek = np.floor(self.time[0]/(24*60*60))


class LineClass:
    def __init__(self,nisLineMetadata,lidarCameraLineMetadata,payload,site,domain,rawNisDir):

        self.payload=payload
        self.site = site
        self.domain = domain
        self.nisFlightID = nisLineMetadata['FlightID']
        self.nisDomain = nisLineMetadata['Domain']
        self.nisSite = nisLineMetadata['Site']
        self.lineNumber = nisLineMetadata['LineNumberNew']
        self.nisFilename = nisLineMetadata['Filename']
        self.nisCloudCover = nisLineMetadata['CloudCover']
        self.nisCloudType = nisLineMetadata['CloudType']
        self.nisCrossStrip = nisLineMetadata['CrossStrip']
        self.nisRefly = nisLineMetadata['Refly']
        self.nisDNP = nisLineMetadata['DNP']

        if self.nisCrossStrip == 'True' or self.nisCrossStrip == 'TRUE':
            self.nisCrossStrip = True
        else:
            self.nisCrossStrip = False
        if self.nisRefly == 'True' or self.nisRefly == 'TRUE':
            self.nisRefly = True
        else:
            self.nisRefly = False
        if self.nisDNP == 'True' or self.nisDNP == 'TRUE':
            self.nisDNP = True
        else:
            self.nisDNP = False

        self.rgbBands = [18,34,54]
        self.rgbNirBands =  [18,34,54,85] 
        self.targetWavelengths = np.arange(380,2510,5)                                        
        self.rawSpectrometerFile =  self.nisFlightID[0:8]+'_'+self.nisFilename+'_hsi_raw_0000.hsi'
        self.rawKmlFile =  self.nisFlightID[0:8]+'_'+self.nisFilename+'_hsi_kml_0000.kml'
               
        self.rawSpectrometerDir = os.path.join(payload.nisId.replace('-','')+'_'+self.nisFlightID[0:8]+'_'+self.nisFilename)
        self.rawSpectrometerKmlDir = os.path.join(payload.nisId.replace('-','')+'_'+self.nisFlightID[0:8]+'_'+self.nisFilename,'hsi','kml')
        
        self.radianceSpectrometerFile = payload.nisId.replace('-','0')+'_'+self.nisFlightID[0:8]+'_'+self.nisFilename+'_rdn'
        self.radianceOrtSpectrometerFile = self.radianceSpectrometerFile+'_ort'
        self.obsOrtSpectrometerFile = self.radianceSpectrometerFile+'_obs_ort'
        self.igmOrtSpectrometerFile = self.radianceOrtSpectrometerFile+'_igm_ort'
        self.gltOrtSpectrometerFile = self.radianceOrtSpectrometerFile+'_glt'
        self.scaSpectrometerFile = self.radianceOrtSpectrometerFile+'_sca'

        self.elevationTifFileBoundingBox = self.radianceOrtSpectrometerFile+'_ele.tif'

        self.smEleOrtSpectrometerFile = self.radianceOrtSpectrometerFile+'_sm_ele.bsq'
        self.smSlopeOrtSpectrometerFile = self.radianceOrtSpectrometerFile+'_sm_slp.bsq'
        self.smAspectOrtSpectrometerFile = self.radianceOrtSpectrometerFile+'_sm_asp.bsq'

        self.reflectanceSpectrometerEnviFile = self.radianceOrtSpectrometerFile+'_atm_nodata.bsq'
        self.directionalSpectrometerFlxFile = self.radianceOrtSpectrometerFile+'_atm_flx.bsq'
        self.directionalSpectrometerDdvFile = self.radianceOrtSpectrometerFile+'_atm_ddv.bsq'

        self.reflectanceSpectrometerH5FilePrefix = 'NEON_'+self.domain+'_'+self.site+'_DP1_'+self.nisFlightID[0:8]+'_'+self.nisFilename

        self.reflectanceSpectrometerH5FilePrefixLineNum = 'NEON_'+self.domain+'_'+self.site+'_DP1_'+self.lineNumber+'_'+self.nisFlightID[0:8]
        self.spectrometerL2ProductPrefixLineNum = (self.reflectanceSpectrometerH5FilePrefixLineNum+ '_bidirectional').replace('_DP1_','_DP2_')
        
        self.radianceSpectrometerEnviBsqFile = self.radianceOrtSpectrometerFile+'.bsq'
        
        self.directionalReflectanceSpectrometerEnviFile = self.radianceOrtSpectrometerFile+'_atm.bsq'
        self.directionalReflectanceSpectrometerH5File = self.reflectanceSpectrometerH5FilePrefix + '_directional_reflectance.h5'
        self.directionalReflectanceSpectrometerH5FileLineNum = self.reflectanceSpectrometerH5FilePrefixLineNum + '_directional_reflectance.h5'

        self.bidirectionalReflectanceSpectrometerEnviFile = self.reflectanceSpectrometerH5FilePrefix + '_directional_reflectance_BRDF_topo_corrected'
        self.bidirectionalReflectanceSpectrometerH5File = self.reflectanceSpectrometerH5FilePrefixLineNum + '_bidirectional_reflectance.h5'
        self.radianceSpectrometerH5File = self.reflectanceSpectrometerH5FilePrefixLineNum + '_radiance.h5'
        
        self.directionalAlbedoSpectrometerTifFile =  self.reflectanceSpectrometerH5FilePrefixLineNum.replace('_DP1_','_DP2_') + '_albedo.tif'
        
        self.kmlOutFile = self.reflectanceSpectrometerH5FilePrefixLineNum.replace('_DP1_','_DPQA_')+'_boundary.kml'
        
        self.ArviTifFile = self.spectrometerL2ProductPrefixLineNum + '_ARVI.tif'
        self.NdviTifFile = self.spectrometerL2ProductPrefixLineNum + '_NDVI.tif'
        self.EviTifFile = self.spectrometerL2ProductPrefixLineNum + '_EVI.tif'
        self.PriTifFile = self.spectrometerL2ProductPrefixLineNum + '_PRI.tif'
        self.SaviTifFile = self.spectrometerL2ProductPrefixLineNum + '_SAVI.tif'
        
        self.ArviErrorTifFile = self.spectrometerL2ProductPrefixLineNum + '_ARVI_error.tif'
        self.NdviErrorTifFile = self.spectrometerL2ProductPrefixLineNum + '_NDVI_error.tif'
        self.EviErrorTifFile = self.spectrometerL2ProductPrefixLineNum + '_EVI_error.tif'
        self.PriErrorTifFile = self.spectrometerL2ProductPrefixLineNum + '_PRI_error.tif'
        self.SaviErrorTifFile = self.spectrometerL2ProductPrefixLineNum + '_SAVI_error.tif'
        
        self.vegIndicesZipFile = self.spectrometerL2ProductPrefixLineNum + '_VegIndices.zip'
        
        self.MsiTifFile = self.spectrometerL2ProductPrefixLineNum + '_MSI.tif'
        self.NdiiTifFile = self.spectrometerL2ProductPrefixLineNum + '_NDII.tif'
        self.NdwiTifFile = self.spectrometerL2ProductPrefixLineNum + '_NDWI.tif'
        self.NmdiTifFile = self.spectrometerL2ProductPrefixLineNum + '_NMDI.tif'
        self.WbiTifFile = self.spectrometerL2ProductPrefixLineNum + '_WBI.tif'
        
        self.MsiErrorTifFile = self.spectrometerL2ProductPrefixLineNum + '_MSI_error.tif'
        self.NdiiErrorTifFile = self.spectrometerL2ProductPrefixLineNum + '_NDII_error.tif'
        self.NdwiErrorTifFile = self.spectrometerL2ProductPrefixLineNum + '_NDWI_error.tif'
        self.NmdiErrorTifFile = self.spectrometerL2ProductPrefixLineNum + '_NMDI_error.tif'
        self.WbiErrorTifFile = self.spectrometerL2ProductPrefixLineNum + '_WBI_error.tif'
        
        self.waterIndicesZipFile = self.spectrometerL2ProductPrefixLineNum + '_WaterIndices.zip'
        
        self.waterMaskFile = self.spectrometerL2ProductPrefixLineNum + '_water_mask.tif'
        
        self.lidarFlightID = lidarCameraLineMetadata['FlightID']
        self.lidarDomain = lidarCameraLineMetadata['Domain']
        self.lidarSite = lidarCameraLineMetadata['Site']
        self.lidarLineNumber = lidarCameraLineMetadata['LineNumber']
        self.lidarAltitude = lidarCameraLineMetadata['Altitude']
        if 'P3' == self.payload.payload:
            self.lidarPRF = lidarCameraLineMetadata['PRR']
            self.lidarPowerLevel = lidarCameraLineMetadata['LaserPowerLevel']
        else:
            self.lidarPRF = lidarCameraLineMetadata['PRF']
            self.lidarScanFreq = lidarCameraLineMetadata['ScanFreq']

        self.cameraISO = lidarCameraLineMetadata['ISO']
        self.cameraAperture = lidarCameraLineMetadata['Aperture']
        self.cameraCameraOut = lidarCameraLineMetadata['CameraOut']
        self.lidarCrossStrip = lidarCameraLineMetadata['CrossStrip']
        self.lidarRefly = lidarCameraLineMetadata['Refly']
        self.lidarDNP = lidarCameraLineMetadata['DNP']

        # print(self.lidarFlightID + ' Line ' + self.lidarLineNumber + ' LIDAR REFLY: ',self.lidarRefly)
        if self.lidarRefly == 'True' or self.lidarRefly == 'TRUE':
            self.lidarRefly = True
        else:
            self.lidarRefly = False
        if self.lidarDNP == 'True' or self.lidarDNP == 'TRUE':
            self.lidarDNP = True
        else:
            self.lidarDNP = False
            
        self.nirBand = 850
        self.swirBand = 1600
        self.nirThreshold = 100
        self.swirThreshold = 50 

    def generateKmls(self,rawMissionSpectrometerDir,outputKmlDir):
        
        if int(self.nisCloudCover) == 1:
            color='Green'
        elif int(self.nisCloudCover) == 2:
            color='Yellow'
        elif int(self.nisCloudCover) == 3:
            color='Red'
        
        generateNisKml(str(os.path.join(rawMissionSpectrometerDir,self.rawSpectrometerKmlDir,self.rawKmlFile)),os.path.join(outputKmlDir,self.kmlOutFile),color)

    def processRadiance(self,rawSpectrometerDir,outputRadianceDir,outputQaDir,numCpus):
        
        if not os.path.exists(os.path.join(outputRadianceDir,self.radianceSpectrometerFile)):
            
            rawSpectrometerLineDir = os.path.join(rawSpectrometerDir,self.rawSpectrometerDir)
            
            print(rawSpectrometerLineDir)
            
            # K8M Fix: Replace missing rawNisLineClass with MATLAB implementation
            # The rawNisLineClass appears to be from a separate module not included
            # Using the MATLAB implementation that was commented out below
            
            if not os.path.exists(os.path.join(outputRadianceDir,self.radianceSpectrometerFile)):
                
                try:
                    print(f"Using MATLAB implementation for radiance processing")
                    
                    # Need to define dn2RadProcessingDir - this should be set somewhere
                    # For now, we'll try to find it or use a default
                    dn2RadProcessingDir = os.environ.get('DN2RAD_DIR', 'C:/NIS/Processing/DN2Radiance')
                    
                    if not os.path.exists(dn2RadProcessingDir):
                        # Try common locations
                        possible_dirs = [
                            'D:/Gold_Pipeline/ProcessingPipelines/NIS/DN2Radiance',
                            'C:/Gold_Pipeline/ProcessingPipelines/NIS/DN2Radiance',
                            'D:/NIS/Processing/DN2Radiance',
                            'C:/NIS/Processing/DN2Radiance'
                        ]
                        for pdir in possible_dirs:
                            if os.path.exists(pdir):
                                dn2RadProcessingDir = pdir
                                break
                    
                    print(f"DN2Rad directory: {dn2RadProcessingDir}")
                    
                    matlabEng = matlab.engine.start_matlab()
                    matlabEng.cd(dn2RadProcessingDir, nargout=0)
                    inputFile = str(os.path.join(rawSpectrometerLineDir, self.rawSpectrometerFile))
                    
                    print(f"Calling MATLAB nisCalCode_noGUI with:")
                    print(f"  Input: {inputFile}")
                    print(f"  Output: {outputRadianceDir}")
                    
                    matlabEng.nisCalCode_noGUI(inputFile, str(outputRadianceDir), matlab.double(2000), matlab.double(1))
                    
                    matlabEng.quit()
                    
                    print(f"MATLAB radiance processing completed")
                    
                except Exception as e:
                    print(f"Error in MATLAB radiance processing: {e}")
                    print(f"Please ensure:")
                    print(f"  1. MATLAB engine for Python is installed")
                    print(f"  2. DN2Radiance MATLAB code exists in: {dn2RadProcessingDir}")
                    print(f"  3. nisCalCode_noGUI.m is available")
                    raise
                    
            else:
                print(f"Radiance file already exists: {self.radianceSpectrometerFile}")
            
            # Generate QA for radiance
            # This part was calling rawNisLine.generateRadianceQaQc(outputQaDir)
            # We'll skip it for now as it requires the rawNisLineClass
            print(f"Note: Radiance QA generation skipped (requires rawNisLineClass)")
        
        #Old Matlab implementation
        
        # if not os.path.exists(os.path.join(outputRadianceDir,self.radianceSpectrometerFile)):
            
    
        #     matlabEng = matlab.engine.start_matlab()
        #     matlabEng.cd(dn2RadProcessingDir, nargout=0)
        #     inputFile = str(os.path.join(self.rawSpectrometerDir,self.rawSpectrometerFile))
    
        #     matlabEng.nisCalCode_noGUI(inputFile,str(outputRadianceDir),matlab.double(2000),matlab.double(1))
    
        #     matlabEng.quit()
        
    def processOrthoRadiance(self,rawMissionSpectrometerDir,radianceDir,payload,leapSeconds,sbetFile,geoidFile,demFile,sapFile):

        #from idlpy import IDL
        if not os.path.exists(os.path.join(radianceDir,self.radianceOrtSpectrometerFile)):
            
            IDL.run('cd, "' + orthoProcessingDir + '"')
            IDL.run('.compile ' + '"' + 'nisortho.pro' + '"')
    
            IDL.bfFlag = 1
            IDL.psFlag = 1.0
            IDL.lSec = leapSeconds
            IDL.sbetFile = str(sbetFile)
            IDL.demFile = str(demFile)
            IDL.saFile = str(sapFile)
            IDL.gps = float(payload.nisNominalTimingOffset)
            IDL.geoidFile = str(geoidFile)
            IDL.camFile = os.path.join(payload.nisCameraModelDir,payload.nisCameraGeolocationModel)
            IDL.leverArm = str(payload.nisLeverArm[0])+','+str(payload.nisLeverArm[1])+','+str(payload.nisLeverArm[2])
            IDL.bFileDelete = 1
            IDL.inMemory = 1
            IDL.rawFile = str(os.path.join(rawMissionSpectrometerDir,self.rawSpectrometerDir,'hsi','raw',self.rawSpectrometerFile))
            IDL.radFile = str(os.path.join(radianceDir,self.radianceSpectrometerFile))
    
            IDL.run('nisortho(bfFlag, psFlag, lSec, sbetFile, demFile, saFile, gps, geoidFile, camFile, leverArm, bFileDelete, inMemory, rawFile, radFile)')
            IDL.close()
            IDL.Heap_GC()
            #IDL.exit()

    def convertRdnOrtToTif(self,radianceDir,outputDir):
        
        radOrtFile = os.path.join(radianceDir,self.radianceOrtSpectrometerFile)
        
        outFile = os.path.join(outputDir,self.radianceOrtSpectrometerFile+'.tif')
        
        if not os.path.exists(outFile):
        
            convertEnvi2tif(radOrtFile,outFile=outFile,Bands=self.rgbNirBands)                                                       
    
    def processClippedTopoForAtcor(self,radianceDir,aigDsm):

        rasterClipFootprint = os.path.join(radianceDir,self.igmOrtSpectrometerFile)

        convertEnvi2tif(rasterClipFootprint,outFile=rasterClipFootprint+'.tif')

        clipRasterByRaster(aigDsm,rasterClipFootprint+'.tif',os.path.join(radianceDir,self.elevationTifFileBoundingBox))

    def processSmoothElevation(self,radianceDir):

        rasterData,metadata = readEnviRaster(os.path.join(radianceDir,self.igmOrtSpectrometerFile),justMetadata = True)

        ul_x, lr_y, lr_x, ul_y,nodata, data_layer = getCurrentFileExtentsAndData(os.path.join(radianceDir,self.elevationTifFileBoundingBox),Band=1)

        smoothedImage = smoothImage(data_layer, 41)
        metadata["num_bands"]= 1

        writeEnviRaster(os.path.join(radianceDir,self.smEleOrtSpectrometerFile),smoothedImage,metadata)

    def processSlopeAspect(self,radianceDir):

        inEnviFile = os.path.join(radianceDir,self.smEleOrtSpectrometerFile)
        outSlopeEnviFile = os.path.join(radianceDir,self.smSlopeOrtSpectrometerFile)
        outAspectEnviFile = os.path.join(radianceDir,self.smAspectOrtSpectrometerFile)

        calcSlopeAspectEnvi(inEnviFile,outSlopeEnviFile,outAspectEnviFile)

    def generateRadianceH5(self,radianceDir,sbetFile,outputDir,metadataXML,NISlog,ScriptsFile):

        radOrtFile = os.path.join(radianceDir,self.radianceOrtSpectrometerFile)
        obsOrtFile = os.path.join(radianceDir,self.obsOrtSpectrometerFile)
        igmOrtFile = os.path.join(radianceDir,self.igmOrtSpectrometerFile)

        H5WriterFunctionRadiance(radOrtFile, obsOrtFile, igmOrtFile, sbetFile, outputDir, metadataXML, NISlog, ScriptsFile,self.lineNumber)

    def processReflectance(self,radianceDir,reflectanceDir,sbetFile,payload):

        #from idlpy import IDL

        
        if os.path.exists(os.path.join(radianceDir,self.radianceOrtSpectrometerFile)) and not os.path.exists(os.path.join(reflectanceDir,self.directionalReflectanceSpectrometerEnviFile)):
                
            print('Processing: '+os.path.join(radianceDir,self.radianceOrtSpectrometerFile))
            
            IDL.run('cd, "' + atcorProcessingDir + '"')
            IDL.run('.compile ' + '"' + 'atmcorrection_update.pro' + '"')
    
            IDL.rdnOrtFile = str(os.path.join(radianceDir,self.radianceOrtSpectrometerFile))
            IDL.reflectanceDir = reflectanceDir
            IDL.sbetFile = str(sbetFile)
            IDL.atcorHomeDir = atcorHomeDir
            IDL.atcorSensorNew=payload.nisAtcorSensorModel
            IDL.avElevation=float(self.meanElevation)
            IDL.avAltitud=float(self.lineAverageAltitude)
            IDL.avHeadings=float(self.lineAverageHeading)
            IDL.avSunZenith=float(self.meanSunZenith)
            IDL.avSunAzimuth=float(self.meanSunAzimuth) 
                                                            
                                                                                                                                                                        
            IDL.run('atmcorrection_update(rdnOrtFile,reflectanceDir,sbetFile,atcorHomeDir,atcorSensorNew,avElevation,avAltitud,avHeadings,avSunZenith,avSunAzimuth)',stdout=True)
                                                                                                                                 
                                                                                                              
            IDL.close()
            IDL.Heap_GC()
            
    def generateReflectanceH5(self,radianceDir,reflectanceDir,sbetFile,outputDir,metadataXML,NISlog,ScriptsFile,isBrdf):

        radOrtFile = os.path.join(radianceDir,self.radianceOrtSpectrometerFile)
        if isBrdf:
            reflOrtFile = os.path.join(reflectanceDir,self.bidirectionalReflectanceSpectrometerEnviFile)
        else:
            reflOrtFile = os.path.join(reflectanceDir,self.directionalReflectanceSpectrometerEnviFile)
        smEleFile =  os.path.join(radianceDir,self.smEleOrtSpectrometerFile)

        for file in os.listdir(radianceDir):
            if 'shd.bsq' in file and self.nisFilename in file:
                self.reflectanceShdFile = file

        shdFile = os.path.join(radianceDir,self.reflectanceShdFile)

        H5WriterFunction(reflOrtFile,smEleFile,shdFile,radOrtFile,sbetFile, outputDir, metadataXML, NISlog, ScriptsFile,lineNum=self.lineNumber)

    def renameDirectionalReflectance(self,reflectanceDir):
        
        shutil.move(os.path.join(reflectanceDir,self.directionalReflectanceSpectrometerH5File),os.path.join(reflectanceDir,self.directionalReflectanceSpectrometerH5FileLineNum))
     
    def reverseRenameDirectionalReflectance(self,reflectanceDir):
        
        shutil.move(os.path.join(reflectanceDir,self.directionalReflectanceSpectrometerH5FileLineNum),os.path.join(reflectanceDir,self.directionalReflectanceSpectrometerH5File))
     
    def generateVegetationIndices(self,reflectanceH5Dir,outputDir):

        h5File = os.path.join(reflectanceH5Dir,self.bidirectionalReflectanceSpectrometerH5File)

        generateVegIndices(h5File,outputDir)

    def generateWaterIndices(self,reflectanceH5Dir,outputDir):

        h5File = os.path.join(reflectanceH5Dir,self.bidirectionalReflectanceSpectrometerH5File)

        generateWaterIndices(h5File,outputDir)

    def generateFpar(self,reflectanceH5Dir,outputDir):

        h5File = os.path.join(reflectanceH5Dir,self.bidirectionalReflectanceSpectrometerH5File)

        generateFpar(h5File,outputDir)

    def generateLai(self,reflectanceH5Dir,outputDir):

        h5File = os.path.join(reflectanceH5Dir,self.bidirectionalReflectanceSpectrometerH5File)

        generateLai(h5File,outputDir)

    def generateAlbedo(self,reflectanceDir,outputDir):

        flxFile = os.path.join(reflectanceDir,self.directionalSpectrometerFlxFile)
        outputFile = os.path.join(outputDir,self.directionalAlbedoSpectrometerTifFile)

        convertEnvi2tif(flxFile,outputFile,Bands=[3],ScaleFactor=10000,NoDataOverRide=-9999,CorrectNoData=0)

    def generateTifFromH5(self,h5Dir,outputDir,Raster):

        if Raster == 'Reflectance':        

            h5File = os.path.join(h5Dir,self.bidirectionalReflectanceSpectrometerH5File)
            outTifFile = os.path.join(outputDir,self.bidirectionalReflectanceSpectrometerH5File.replace('reflectance.h5','RGB.tif'))
      
        elif Raster == 'Radiance':
        
            h5File = os.path.join(h5Dir,self.radianceSpectrometerH5File)
            outTifFile = os.path.join(outputDir,self.radianceSpectrometerH5File.replace('radiance.h5','radiance_RGB.tif'))

        generateTifs(h5File,Raster,outTifFile,makeCog=True,bands=self.rgbBands)

    def zipL2VegIndices(self,inputDir,outputDir):
        
        vegIndexFiles = [] 
        
        vegIndexFiles.append(self.ArviTifFile) 
        vegIndexFiles.append(self.NdviTifFile)
        vegIndexFiles.append(self.EviTifFile)
        vegIndexFiles.append(self.PriTifFile) 
        vegIndexFiles.append(self.SaviTifFile) 
        
        vegIndexFiles.append(self.ArviErrorTifFile)
        vegIndexFiles.append(self.NdviErrorTifFile)
        vegIndexFiles.append(self.EviErrorTifFile) 
        vegIndexFiles.append(self.PriErrorTifFile) 
        vegIndexFiles.append(self.SaviErrorTifFile)
        
        vegIndexFiles = [os.path.join(inputDir,file) for file in vegIndexFiles]   
              
        zipFiles(vegIndexFiles,os.path.join(outputDir,self.vegIndicesZipFile))

    def zipL2WaterIndices(self,inputDir,outputDir):
        
        waterIndexFiles = [] 
        
        waterIndexFiles.append(self.MsiTifFile) 
        waterIndexFiles.append(self.NdiiTifFile)
        waterIndexFiles.append(self.NdwiTifFile)
        waterIndexFiles.append(self.NmdiTifFile)
        waterIndexFiles.append(self.WbiTifFile) 
        
        waterIndexFiles.append(self.MsiErrorTifFile)
        waterIndexFiles.append(self.NdiiErrorTifFile)
        waterIndexFiles.append(self.NdwiErrorTifFile)
        waterIndexFiles.append(self.NmdiErrorTifFile)
        waterIndexFiles.append(self.WbiErrorTifFile) 
      
        waterIndexFiles = [os.path.join(inputDir,file) for file in waterIndexFiles]   
      
        zipFiles(waterIndexFiles,os.path.join(outputDir,self.waterIndicesZipFile))

    def generateWaterMask(self,inputDir,outputWaterMaskDir):
        
        outTif = os.path.join(outputWaterMaskDir,self.waterMaskFile)
        
        h5File = os.path.join(inputDir,self.bidirectionalReflectanceSpectrometerH5File)
        
        generateWaterMask(h5File,outTif,self.nirBand,self.swirBand,self.nirThreshold,self.swirThreshold)

    def generateReflectancePlots(self,inputh5Dir,outPngDir):
        
        outFile = os.path.join(outPngDir,self.spectrometerL2ProductPrefixLineNum)
                             
        h5File = os.path.join(inputh5Dir,self.bidirectionalReflectanceSpectrometerH5File)
                
        generateReflectancePlots(h5File,os.path.join(outPngDir,self.spectrometerL2ProductPrefixLineNum),self.nirBand,self.swirBand,self.nirThreshold,self.swirThreshold)

    def convertRdnOrtForAtcor(self,radianceDir,reflectanceDir):
        
        print('Working on '+self.radianceOrtSpectrometerFile)
        
        radOrtFile = os.path.join(radianceDir,self.radianceOrtSpectrometerFile)
        
        outRadOrtBsqFile =  os.path.join(reflectanceDir,self.radianceSpectrometerEnviBsqFile)
        
        if not os.path.exists(outRadOrtBsqFile):
        
            quickBiltoBsq(radOrtFile,outRadOrtBsqFile,updateNoData=np.nan)
        
    def generateScaFileForAtcor(self,radianceDir):
    
        inObsFile = os.path.join(radianceDir,self.obsOrtSpectrometerFile)
        
        outScaFile = os.path.join(radianceDir,self.scaSpectrometerFile)
    
        convertObsToSca(inObsFile,outScaFile)
        
    # def fixNansInAtcorRaster(self,reflectanceDir):
        
    #     outAtcorReflFile = os.path.join(reflectanceDir,self.directionalReflectanceSpectrometerEnviFile)
        
    #     preAtcorRadianceFile = os.path.join(reflectanceDir,self.radianceSpectrometerEnviBsqFile)
        
    #     preAtcorRaster,preAtcorMetadata = readEnviRaster(preAtcorRadianceFile,bands = [85])
        
    #     postAtcorRaster,postAtcorMetadata = readEnviRaster(outAtcorReflFile)
        
    #     outAtcorReflFileHeader = outAtcorReflFile.replace('.bsq','.hdr')
        
    #     outAtcorReflFileHeaderTemp = outAtcorReflFileHeader.replace('.hdr','_temp.hdr')
        
    #     shutil.copy(outAtcorReflFileHeader,outAtcorReflFileHeaderTemp)
        
    #     for band in np.arange(postAtcorRaster.shape[0]):
    #         postAtcorRaster[band,np.isnan(preAtcorRaster[0,:,:])] = float(preAtcorMetadata['data_ignore_value'])
            
    #     writeEnviRaster(outAtcorReflFile,np.moveaxis(postAtcorRaster,[0,1,2],[2,0,1]),postAtcorMetadata)
        
    #     updateNoDataEnviHeader(outAtcorReflFileHeaderTemp,outEnviHdrFile = outAtcorReflFileHeader,noDataUpdate=preAtcorMetadata['data_ignore_value'])
        
    #     os.remove(outAtcorReflFileHeaderTemp)
        
    def fixNansInAtcorRaster(self,reflectanceDir):
        outAtcorReflFile = os.path.join(reflectanceDir,self.directionalReflectanceSpectrometerEnviFile)
        preAtcorRadianceFile = os.path.join(reflectanceDir,self.radianceSpectrometerEnviBsqFile)
        preAtcorRaster,preAtcorMetadata = readEnviRaster(preAtcorRadianceFile,bands = [85])
        postAtcorRaster,postAtcorMetadata = readEnviRaster(outAtcorReflFile)
        outAtcorReflFileHeader = outAtcorReflFile.replace('.bsq','.hdr')
        outAtcorReflFileHeaderTemp = outAtcorReflFileHeader.replace('.hdr','_temp.hdr')
        shutil.copy(outAtcorReflFileHeader,outAtcorReflFileHeaderTemp)
        for band in np.arange(postAtcorRaster.shape[0]):
            postAtcorRaster[band,np.isnan(preAtcorRaster[0,:,:])] = float(preAtcorMetadata['data_ignore_value'])
        def write_envi():
            writeEnviRaster(outAtcorReflFile,np.moveaxis(postAtcorRaster,[0,1,2],[2,0,1]),postAtcorMetadata)
            updateNoDataEnviHeader(outAtcorReflFileHeaderTemp,outEnviHdrFile = outAtcorReflFileHeader,noDataUpdate=preAtcorMetadata['data_ignore_value'])
            os.remove(outAtcorReflFileHeaderTemp)
        qa_check_and_retry(lambda: write_envi(), lambda: qa_check_envi_file(outAtcorReflFile), max_retries=3)

        
    
    def createMissingDDV(self,reflectanceDir):
        
        outAtcorDdvFile = os.path.join(reflectanceDir,self.directionalSpectrometerDdvFile)
        outAtcorReflFile = os.path.join(reflectanceDir,self.directionalReflectanceSpectrometerEnviFile)
              
        if not os.path.exists(outAtcorDdvFile):
            ddvRaster,metadata = readEnviRaster(outAtcorReflFile,bands = [1])
            metadata['Classes'] = 5
            metadata['Class_Names'] = ['0: geocoded backg'.encode('utf-8'), '1: water'.encode('utf-8'), '2: DDV reference'.encode('utf-8'), '3: non-reference'.encode('utf-8'), '4: topogr.shadow'.encode('utf-8')]
            metadata['Class_Lookup'] = ['0,0,0','0,0,250','0,250,0','150,150,150','60,60,60']
            ddvRaster[ddvRaster!=metadata['data_ignore_value']] = 3
            ddvRaster[ddvRaster==metadata['data_ignore_value']] = -1
            ddvRaster = ddvRaster.astype(np.uint8)
            metadata['num_bands'] = 1
            metadata['band_names'] = ['DDV reference map']
            metadata['data_ignore_value'] = -1
 
              
            writeEnviRaster(outAtcorDdvFile,np.moveaxis(ddvRaster,[0,1,2],[2,0,1]),metadata)   

    def convertRadianceH5ToEnviObs(self,inputh5Dir,outObsDir,filePathToEnviProjCs,spatialIndexesToRead = None,bandIndexesToRead = None):
        
        print('Working on '+self.obsOrtSpectrometerFile)
        
        h5File = os.path.join(inputh5Dir,self.radianceSpectrometerH5File)
        
        convertH5RasterToEnvi(h5File,'OBS_Data',os.path.join(outObsDir,self.obsOrtSpectrometerFile),filePathToEnviProjCs,spatialIndexesToRead = spatialIndexesToRead,bandIndexesToRead = bandIndexesToRead)

    def convertRadianceH5ToEnviIgm(self,inputh5Dir,outObsDir,filePathToEnviProjCs,spatialIndexesToRead = None,bandIndexesToRead = None):
        
        print('Working on '+self.igmOrtSpectrometerFile)
        
        h5File = os.path.join(inputh5Dir,self.radianceSpectrometerH5File)
        
        convertH5RasterToEnvi(h5File,'IGM_Data',os.path.join(outObsDir,self.igmOrtSpectrometerFile),filePathToEnviProjCs,spatialIndexesToRead = spatialIndexesToRead,bandIndexesToRead = bandIndexesToRead)

    def convertRadianceH5ToEnviGlt(self,inputh5Dir,outObsDir,filePathToEnviProjCs,spatialIndexesToRead = None,bandIndexesToRead = None):
        
        print('Working on '+self.gltOrtSpectrometerFile)
        
        h5File = os.path.join(inputh5Dir,self.radianceSpectrometerH5File)
        
        convertH5RasterToEnvi(h5File,'GLT_Data',os.path.join(outObsDir,self.gltOrtSpectrometerFile),filePathToEnviProjCs,spatialIndexesToRead = spatialIndexesToRead,bandIndexesToRead = bandIndexesToRead)

    def convertRadianceH5ToEnviRadiance(self,inputh5Dir,outObsDir,filePathToEnviProjCs,spatialIndexesToRead = None,bandIndexesToRead = None):
        
        print('Working on '+self.radianceOrtSpectrometerFile)
        
        h5File = os.path.join(inputh5Dir,self.radianceSpectrometerH5File)
        
        convertH5RasterToEnvi(h5File,'Radiance',os.path.join(outObsDir,self.radianceOrtSpectrometerFile),filePathToEnviProjCs,spatialIndexesToRead = spatialIndexesToRead,bandIndexesToRead = bandIndexesToRead)

    def convertReflectanceH5ToEnvi(self,inputh5Dir,outputDir,filePathToEnviProjCs,skipRasters = [],keepRasters = [],spatialIndexesToRead = None,bandIndexesToRead = None):
        
        print('Working on '+self.directionalReflectanceSpectrometerH5File)

        h5File = os.path.join(inputh5Dir,self.directionalReflectanceSpectrometerH5FileLineNum)
        
        #outFile = os.path.join(outputDir,os.path.basename(h5File))
        
        getAllRastersFromH5(h5File,'ENVI',outDir = outputDir,skipRasters = skipRasters,keepRasters = keepRasters,filePathToEnviProjCs=filePathToEnviProjCs,spatialIndexesToRead =spatialIndexesToRead,bandIndexesToRead = bandIndexesToRead)
        
      #  (h5_filename,ouputFormat,outDir = None,skipRasters = [],filePathToEnviProjCs=None,spatialIndexesToRead = None,bandIndexesToRead = None)

    def convertReflectanceH5ToGtif(inputh5Dir,outputDir,skipRasters = [],keepRasters = [],spatialIndexesToRead = None,bandIndexesToRead = None):
        
        print('Working on '+self.directionalReflectanceSpectrometerH5File)
             
        h5File = os.path.join(inputh5Dir,self.directionalReflectanceSpectrometerH5FileLineNum)
        
        getAllRastersFromH5(h5File,'Gtif',outDir = outputDir,skipRasters = skipRasters,keepRasters =keepRasters,filePathToEnviProjCs=None,spatialIndexesToRead = spatialIndexesToRead,bandIndexesToRead = bandIndexesToRead)

    def writeAtcorLogsFromReflectanceH5(self,inputh5Dir,outputReflDir,outputRadDir):
        
        print('Working on '+self.directionalReflectanceSpectrometerH5File)
             
        h5File = os.path.join(inputh5Dir,self.directionalReflectanceSpectrometerH5FileLineNum)

        ATCOR_log,ATCOR_inn,ATCOR_shadow_processing_log,ATCOR_skyview_processing_log,BRDF_coeffs,BRDF_config,avSolarAzimuthAngles,avSolarZenithAngles =  getH5ReflectanceLogs(h5File)
        
        atcorLogFileName =  os.path.join(outputReflDir,self.radianceOrtSpectrometerFile+'_atm.log')
        with open(atcorLogFileName, "w") as file:
            file.write(ATCOR_log[()].decode('utf-8'))
            
        atcorInnFileName =  os.path.join(outputReflDir,self.radianceOrtSpectrometerFile+'.inn')
        with open(atcorInnFileName, "w") as file:
            file.write(ATCOR_inn[()].decode('utf-8'))
            
        shadowProcessingLogFileName =  os.path.join(outputRadDir,self.radianceOrtSpectrometerFile+'_sm_ele_shadow_report.log')
        with open(shadowProcessingLogFileName, "w") as file:
            file.write(ATCOR_shadow_processing_log[()].decode('utf-8'))
            
        skyviewProcessingLogFileName =  os.path.join(outputRadDir,self.radianceOrtSpectrometerFile+'_sm_ele_skyview_report.log')
        with open(skyviewProcessingLogFileName, "w") as file:
            file.write(ATCOR_skyview_processing_log[()].decode('utf-8'))
            
    def getBeginningEndFlightTime(self,radianceDir,gpsDayOfWeek):
        
        inObsFile = os.path.join(radianceDir,self.obsOrtSpectrometerFile)
        obsData,metadata = readEnviRaster(inObsFile)
        obsData[obsData == float(metadata["data_ignore_value"])] = np.nan
        gpsTime = obsData[9,:,:]
        secondsInDay = 24*60*60
        self.spectrometerMinimumTime = np.nanmin(gpsTime)*3600+gpsDayOfWeek*secondsInDay
        self.spectrometerMaximumTime = np.nanmax(gpsTime)*3600+gpsDayOfWeek*secondsInDay
         
        #;if (t0 LT start_time) then begin
 #;  t0 = t0+seconds_in_day
 #;  tf = tf+seconds_in_day
 #;endif
 
#; if (abs(t0-tf) GT 23.5) then begin
# ;  indexes_next_day = where(GPSTimeData < 12.0,num_next_day_count)
# ;  GPSTimeData[indexes_next_day] = GPSTimeData[indexes_next_day]+24.0
# ;  t0 = min(GPSTimeData)*3600.0+num_days*seconds_in_day
# ;  tf = max(GPSTimeData)*3600.0+num_days*seconds_in_day 
#; endif 
        
    def getLineAverageAltitude(self,sbet):
        
        self.lineAverageAltitude = np.nanmean(sbet.elevation[(sbet.time>self.spectrometerMinimumTime) & (sbet.time<self.spectrometerMaximumTime)])
        
        #print(self.lineAverageAltitude)
        
    def getLineAverageHeading(self,sbet):
        
        self.lineHeadings = sbet.heading[(sbet.time>self.spectrometerMinimumTime) & (sbet.time<self.spectrometerMaximumTime)]
        self.lineWanders = sbet.wander[(sbet.time>self.spectrometerMinimumTime) & (sbet.time<self.spectrometerMaximumTime)]
        self.lineHeadings = self.lineHeadings-self.lineWanders
        cosHeadings = np.cos(self.lineHeadings)
        sinHeadings = np.sin(self.lineHeadings)
        self.lineAverageHeading = np.degrees(np.arctan2(np.mean(sinHeadings),np.mean(cosHeadings)))
        self.lineAverageHeading = (self.lineAverageHeading+360)%360
        #print(self.nisFilename)    
        #print(self.lineAverageHeading)
      
    def getAverageSunZenith(self,radianceDir):
        
        inObsFile = os.path.join(radianceDir,self.obsOrtSpectrometerFile)
        obsData,metadata = readEnviRaster(inObsFile)
        obsData[obsData == float(metadata["data_ignore_value"])] = np.nan
        sunZenith = obsData[4,:,:]
        self.meanSunZenith = np.nanmean(sunZenith)
        #print(self.meanSunZenith)
          
    def getAverageSunAzimuth(self,radianceDir):
        
        inObsFile = os.path.join(radianceDir,self.obsOrtSpectrometerFile)
        obsData,metadata = readEnviRaster(inObsFile)
        obsData[obsData == float(metadata["data_ignore_value"])] = np.nan
        sunAzimuth = obsData[3,:,:]
        self.meanSunAzimuth = np.nanmean(sunAzimuth)
        #print(self.meanSunAzimuth)
        
    def getAverageElevation(self,radianceDir):
        
        inElevationFile = os.path.join(radianceDir,self.smEleOrtSpectrometerFile)
        elevationData,metadata = readEnviRaster(inElevationFile)
        elevationData[elevationData == float(metadata["data_ignore_value"])] = np.nan
        self.meanElevation = np.nanmean(elevationData)
        #print(self.meanElevation)
        
    def createAtcorSensorAndAtmLib(self,payload):
        
        atcorSensorDir = os.path.join(atcorHomeDir,'sensor',payload.nisAtcorSensorModel)
        payloadFlightFilename = payload.nisAtcorSensorModel+'_'+self.nisFlightID[0:8]+'_'+self.nisFilename
        self.newSensorLocation = os.path.join(atcorHomeDir,'sensor',payloadFlightFilename)
        
        if os.path.exists(self.newSensorLocation):
            shutil.rmtree(self.newSensorLocation)
        
        shutil.copytree(atcorSensorDir,self.newSensorLocation)
        
        shutil.move(os.path.join(self.newSensorLocation,'sensor_'+payload.nisAtcorSensorModel+'.cal'),os.path.join(self.newSensorLocation,'sensor_'+payloadFlightFilename+'.cal'))
        shutil.move(os.path.join(self.newSensorLocation,'sensor_'+payload.nisAtcorSensorModel+'.dat'),os.path.join(self.newSensorLocation,'sensor_'+payloadFlightFilename+'.dat'))
        shutil.move(os.path.join(self.newSensorLocation,'e0_solar_'+payload.nisAtcorSensorModel+'.spc'),os.path.join(self.newSensorLocation,'e0_solar_'+payloadFlightFilename+'.spc'))

        currentAtmLibLocation = os.path.join(atcorHomeDir,'atm_lib',payload.nisAtcorSensorModel)
        newAtmLibTempLocation = os.path.join(atcorHomeDir,'atm_lib','temp')
        self.newAtmLibLocation = os.path.join(atcorHomeDir,'atm_lib','temp',payloadFlightFilename)    
        
        if os.path.exists(self.newAtmLibLocation):
            shutil.rmtree(self.newAtmLibLocation)

        os.makedirs(newAtmLibTempLocation,exist_ok=True)
        shutil.copytree(currentAtmLibLocation,self.newAtmLibLocation)
        
    def cleanupAtcorSensorAndAtmLib(self):
        
        shutil.rmtree(self.newAtmLibLocation)
        shutil.rmtree(self.newSensorLocation)
        
    def resampleSpectrum(self,inputDir,outputDir,filePathToEnviProjCs,outputType='ENVI'):
        
        h5File = os.path.join(inputDir,self.bidirectionalReflectanceSpectrometerH5File)
        
        generateResampleH5Spectrum(h5File,self.targetWavelengths,outputDir,outputType,filePathToEnviProjCs)
        
        