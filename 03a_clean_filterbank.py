# import libraries

import logging
import os
import tempfile
import statistics
import sys
import argparse
import math

import pylab as plt
import numpy as np
import blimpy as bl

from pathlib import Path
from tqdm import tqdm
from sigpyproc.readers import FilReader
from scipy.signal import correlate
from scipy.linalg import toeplitz
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from astropy import units as u
from astropy.coordinates import SkyCoord
from urllib.request import urlretrieve
from your.formats.filwriter import make_sigproc_object
from your import Your, Writer
#from sigpyproc.Filterbank import Filterbank
#from sigpyproc.Readers import FilReader

def generate_filterbank():
    logging.info("01c_generate_filterbank: filterbank generator launched")
    data_written = np.random.randint(low=0, high=255, size=(8192,1024), dtype=np.uint)
    gen_path = os.path.join('./data/', 'foo.fil')   
    sigproc_object = make_sigproc_object(
    rawdatafile  = gen_path,
    source_name = "bar",
    nchans  = 1024, 
    foff = -0.1,                # MHz
    fch1 = 2000,                # MHz
    tsamp = 256e-6,             # seconds
    tstart = 59246,             # MJD
    src_raj=112233.44,          # HHMMSS.SS
    src_dej=112233.44,          # DDMMSS.SS
    machine_id=0,
    nbeams=0,
    ibeam=0,
    nbits=8,
    nifs=1,
    barycentric=0,
    pulsarcentric=0,
    telescope_id=6,
    data_type=0,
    az_start=-1,
    za_start=-1,
    )
    logging.info("01c_generate_filterbank: matrix generated: " + str(data_written.shape))
    sigproc_object.write_header(gen_path)
    logging.info("01c_generate_filterbank: header written to: " + str(gen_path))
    sigproc_object.append_spectra(data_written, gen_path)
    logging.info("01c_generate_filterbank: spectra appended to: " + str(gen_path))
    return (gen_path)

def eigenbasis(matrix):

    """
    Compute the eigenvalues and the eigenvectors of a square matrix and return the eigenspectrum (eigenvalues sorted in decreasing order) and the sorted eigenvectors respect
    to the eigenspectrum for the KLT analysis
    """

    eigenvalues,eigenvectors = np.linalg.eigh(matrix)

    if eigenvalues[0] < eigenvalues[-1]:
        eigenvalues = np.flipud(eigenvalues)
        eigenvectors = np.fliplr(eigenvectors)
    eigenspectrum = eigenvalues
    return eigenspectrum,eigenvectors

def count_elements_for_threshold(arr, threshold):
    sorted_arr = np.sort(arr)[::-1]  # Sort array in descending order
    total_sum = np.sum(sorted_arr)
    cumulative_sum = np.cumsum(sorted_arr)
    num_elements = np.searchsorted(cumulative_sum, threshold * total_sum, side='right') + 1
    return num_elements

def klt(signals, threshold):
    logging.info("KLT function called with matrix element, threshold: " + str(signals[10,10]) + " | " + str(threshold))
    R = np.cov((signals-np.mean(signals,axis=0)),rowvar=False)

    eigenspectrum,eigenvectors = eigenbasis(R)

    neig = count_elements_for_threshold(eigenspectrum, threshold)

    coeff = np.matmul((signals[:,:]-np.mean(signals,axis=0)),np.conjugate((eigenvectors[:,:])))
    recsignals = np.matmul(coeff[:,0:int(neig)],np.transpose(eigenvectors[:,0:int(neig)])) + np.mean(signals,axis=0)

    return neig,eigenspectrum,eigenvectors,recsignals

def process_data(filename, output_dir, output_name,
    sk_flag, sk_sig, klt_clean, var_frac, klt_window, verbose, writer
    ):
    
    changed = False
    
    logging.info("01_process_data: called with parameters: ")
    logging.info(str((locals().keys())) + str((locals().values())))
    
    # load input file
    if (filename == "sample_your"):
        logging.info("01a_process_data: sample from Y-O-U-R library chosen")
        download_path = os.path.join('./data', "FRB180417.fil")
        if not (os.path.isfile(download_path)):
            logging.info("01a_process_data: no sample found, will download")
            url = "https://zenodo.org/record/3905426/files/FRB180417.fil"
            urlretrieve(
                url, download_path,
            )
        filename = download_path
        logging.info("01a_process_data: your sample loaded at: " + str(filename))
    elif (filename == "sample_matteo"):
        logging.info("01b_process_data: sample of burst was chosen: ")
        upload_path = "./data/burst1.fil"
        filename = upload_path
        logging.info("01b_process_data: burst sample loaded at: " + str(filename))
    elif (filename == "sample_generate"):
        logging.info("01c_process_data: filterbank generator chosen " + str(filename))
        filename = generate_filterbank()
        logging.info("01c_process_data: filterbank generated at: " + str(filename))

    if(writer.lower() == "your"):
        input_data = Your(filename)
        header = input_data.your_header
        nchans = header.native_nchans
        nbits = header.nbits
        matrix = input_data.get_data(nstart=0, nsamp=header.nspectra)
        logging.info("02a_process_data: your sample loaded via library YOUR at: " + str(filename))
    if(writer.lower() == "blimpy"):
        input_data = bl.Waterfall(filename)
        header = input_data.header
        nchans = header['nchans']
        nbits = header['nbits']
        matrix = np.squeeze(input_data.data)
        logging.info("02b_process_data: your sample loaded via library blimpy at: " + str(filename))
    if(writer.lower() == "sigproc"):
        input_data = FilReader(filename)
        header  = input_data.header
        nsamp   = input_data.header.nsamples
        nchans   = input_data.header.nchans
        nbits   = input_data.header.nbits
        df      = input_data.header.foff
        dt      = input_data.header.tsamp
        matrix  = input_data.read_block(0, nsamp).T
    

    logging.info(str(type(header)) + str(header))
    logging.info(str(type(matrix)) + str(matrix.shape) + str(matrix[0][15]))
    
    if klt_clean is True:
        logging.info("03a_KLT method was chosen to clean with: " + str(klt_clean) + str(klt_window))
        nchunks = math.ceil(nchans / klt_window)
        logging.info("03a_values of variables nchunks, nchans " + str(nchans) + str(nchunks))
        rfitemplate_full = np.zeros(matrix.T.shape)
        for ii in tqdm(range(nchunks)):
            if((ii +1) * klt_window <= nchans):
                datagrabbed = matrix[:,ii * klt_window : (ii + 1) * klt_window].astype(float)
                neig, ev, evecs, rfitemplate = klt(datagrabbed.T, var_frac)
                rfitemplate_full[ii * klt_window : (ii + 1) * klt_window,:] = rfitemplate
            else:
                datagrabbed = matrix[:,ii * klt_window : nchans].astype(float)
                neig, ev, evecs, rfitemplate = klt(datagrabbed.T, var_frac)
                rfitemplate_full[ii * klt_window : nchans,:] = rfitemplate
            #logging.info("Shape of grabbed data: " + str(datagrabbed.shape))
            #logging.info("Shape of resulting KLT values: " + str(rfitemplate.shape))
        #avg_value = np.mean(matrix)
        #med_value = np.median(matrix)
        #max_value = np.max(matrix)
        #max_rfi   = np.max(np.abs(rfitemplate_full))
        output_matrix = (matrix - np.abs(rfitemplate_full.T))
        logging.info("CHECKLOG: input_matrix-last element: " + str(matrix[matrix.shape[0]-1][matrix.shape[1]-1]))
        logging.info("CHECKLOG: rfitemplate-last element: " + str(rfitemplate.T[rfitemplate.T.shape[0]-1][rfitemplate.T.shape[1]-1]))
        logging.info("CHECKLOG: output_matrix-last element: " + str(output_matrix[matrix.shape[0]-1][matrix.shape[1]-1]))
        #logging.info("CHECKLOG: average values input / output: " + str(avg_value) + " / " + str(np.mean(output_matrix)))
        #logging.info("CHECKLOG: median values input / output: " + str(med_value) + " / " + str(np.median(output_matrix)))
        #logging.info("CHECKLOG: max values input / output: " + str(max_value) + " / " + str(np.max(output_matrix)))
        #logging.info("CHECKLOG: max rfitemplate: " + str(max_rfi) + " / " + str(np.max(output_matrix)))
        #logging.info("CHECKLOG: output_matrix after correction -last element: " + str(output_matrix[matrix.shape[0]-1][matrix.shape[1]-1]))
        changed = True

    if changed is False:
        output_matrix = matrix
        logging.info("03d_No cleaning method chosen, will store untouched matrix" + str(output_matrix.shape))

    # produce output
    if output_name is None:
        filedir, name = os.path.split(filename)
        output_name = name.replace(".fil","") + "_cleaned" + ".fil"
    if output_dir is None:
        output_dir = './data/'
    
    output_path = os.path.join(output_dir, output_name)
    
    logging.info("04_process_data: output path created: filename, output_dir, output_name, output_path")
    logging.info(str(filename) + ", " + str(output_dir) + ", " + str(output_name) + ", " + str(output_path))

    if int(nbits) == int(8):
        output_matrix = output_matrix.astype("uint8")
    if int(nbits) == int(16):
        output_matrix = output_matrix.astype("uint16")
    if int(nbits) == int(32):
        output_matrix = output_matrix.astype("uint32")

    if (writer.lower() == "your"):
        c = SkyCoord(ra=header.ra_deg*u.degree, dec=header.dec_deg*u.degree)
        dec_dms = c.dec.dms
        ra_hms = c.ra.hms
        
        output_data = make_sigproc_object(
            rawdatafile   = 'foo.fil',
            source_name   = header.source_name,
            nchans        = header.nchans,
            foff          = header.foff,            # MHz
            fch1          = header.fch1,            # MHz
            tsamp         = header.tsamp,           # seconds
            tstart        = header.tstart,          # MJD
            nbits         = header.nbits,
            src_raj       = float(str(str(int(ra_hms.h)) + str(int(abs(ra_hms.m))) + str(abs(ra_hms.s)))),
            src_dej       = float(str(str(int(dec_dms.d)) + str(int(abs(dec_dms.m))) + str(abs(dec_dms.s))))
        )
        output_data.write_header('foo.fil')
        logging.info("05a_process_data: header of output file written")
        output_data.append_spectra(output_matrix, 'foo.fil')
        logging.info("05a_process_data: data spectrum written to the output")

        os.rename('foo.fil', os.path.join(output_dir,output_name))
        logging.info("05a_process_data: output file renamed / moved if specified")
    
    if (writer.lower() == "blimpy"):
        input_data.data = np.copy(np.expand_dims(output_matrix, axis=1))
        bl.Waterfall.write_to_fil(input_data, os.path.join(output_dir, output_name))
        logging.info('05b_blimpy library succesfully written new file')
    
    if(writer.lower() == "sigproc"):
        output_data = input_data.header.prep_outfile(os.path.join(output_dir,output_name), back_compatible = True, nbits = nbits)
        output_data.cwrite(output_matrix.ravel())
        output_data.close()

def _get_parser():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description = "Clean a SIGPROC filterbank file from RFI and produces a cleaned filterbank" + "\n"
                      "It performs an RFI excision in frequency via spectral kurtosis " + "\n"
                      "It performs an RFI excision in time via a Gaussian thresholding" + "\n"
                      "It also makes an RFI template, computed via a PCA (KLT), which will be subtracted to the data" + "\n"
                      "It works only with > 8-bits filterbanks...")
    parser.add_argument('-f',
                        '--fil_file',
                        action = "store" ,
                        help = "SIGPROC .fil file to be processed (REQUIRED)",
                        required = True)
    parser.add_argument('-o',
                        '--output_dir',
                        action = "store" ,
                        help = "Output directory (Default: your current path)",
                        default = None
                        )
    parser.add_argument('-n',
                        '--output_name',
                        action = "store" ,
                        help = "Output File Name (Default: filename_cleaned.fil)",
                        default = None
                        )
    parser.add_argument('-sk',
                        '--spectral_kurtosis',
                        help = "Find the bad channels via a spectral kurtosis (Bad channels will be set to zero). Default = False.",
                        action = 'store_true',
                        )
    parser.add_argument('-sksig',
                        '--spectral_kurtosis_sigma',
                        type = int,
                        default = 3,
                        action = "store" ,
                        help = "Sigma for the Spectral Kurtosis (Default: 3)"
                        )
    parser.add_argument('-klt',
                        '--karhunen_loeve_cleaning',
                        help = "Evaluate an RFI template via a KLT and remove it from the data. Default = False.",
                        action = 'store_true',
                        )
    parser.add_argument('-var_frac',
                        '--variance_fraction',
                        type = float,
                        default = 0.3,
                        action = "store" ,
                        help = "The fraction of the total variance of the signal to consider (between 0 and 1). The number of associated eigenvalues will be computed from this. (Default: 0.3)"
                        )
    parser.add_argument('-klt_win',
                        '--klt_window',
                        type = int,
                        default = 1024,
                        action = "store" ,
                        help = "Number of time bins to consider in each read to make the KLT. (Default: 1024)"
                        )
    parser.add_argument('-v',
                        '--verbose',
                        action = "store_true" ,
                        help = "prints out debugging logs"
                        )
    parser.add_argument('-w',
                        '--writer',
                        action="store",
                        help="Choose writing library from YOUR, BLIMPY, SIGPROC. Default is YOUR",
                        default="your"
    )
    return parser.parse_args()

if __name__ == '__main__':

    args = _get_parser()

    # Set up logging
    log_level = logging.INFO if args.verbose else logging.WARN
    logging.basicConfig(filename="clean_filterbank.log", filemode='a', level=log_level, format="%(levelname)s: %(message)s")

    process_data(args.fil_file,
                args.output_dir,
                args.output_name,
                args.spectral_kurtosis,
                args.spectral_kurtosis_sigma,
                args.karhunen_loeve_cleaning,
                args.variance_fraction,
                args.klt_window,
                args.verbose,
                args.writer
    )