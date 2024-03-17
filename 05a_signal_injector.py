import time
import os
import logging
import argparse

import blimpy as bl
import setigen as stg
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord
from your.formats.filwriter import make_sigproc_object
from your import Your, Writer

def setigen_injection(frame, snr, df, header, f_start, t_start, t_length):

    if(df == 'nudot'):
        nchans  = header['nchans']
        tsamp   = header['tsamp']
        fch0    = header['fch1']
        foff    = header['foff']
        nsamp   = frame.shape[0]
        bw      = nchans * foff
        fc      = fch0 + bw / 2
        obslen  = nsamp * tsamp
        df      = ((fc - fch0) / obslen) * u.MHz / u.s
    else:
        df = float(df) * u.Hz / u.s

    signal = frame.add_signal(path=stg.constant_path(f_start=frame.get_frequency(index=int(f_start)), drift_rate= df),
                          t_profile=stg.constant_t_profile(level=frame.get_intensity(snr=float(snr))), 
                          f_profile=stg.gaussian_f_profile(width=3000000*u.Hz),
                          bp_profile=None
                          )
    
    logging.debug('03a_setigen injection succesfully finished')
    return frame.data

def gauss(x,a,x0,sigma):

    """
    Simple Gaussian Function. I use this to fit the data to get FRB width.
    """

    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def get_stdoff(x, chan, wchan):

    mask = np.ones(x.shape[0], dtype = bool)
    mask[chan - wchan : chan + wchan ] = 0

    std = np.std(x[mask])

    return std

class FakeTelemetry:
    def __init__(
        self,
        fc,
        bw,
        driftrate,
        tstart,
        cstart,
        cstop,
        nchans,
        nbins,
        sigmaf,
        duration,
        snr
        ):
        self.fc        = fc # central frequency of the observation
        self.bw        = bw # observational bandwidth
        self.driftrate = driftrate # drift rate of the SETI signal
        self.cstart    = cstart # starting frequency channel of the SETI signal
        self.cstop     = cstop  # stopping frequency channel of the SETI signal
        self.nchans    = nchans # number of frequency channels of the observation
        self.nbins     = nbins # number of time bins of the observation
        self.snr       = snr # integrated (in time) S/N of the SETI signal
        self.tstart    = tstart # starting time of the SETI signal
        self.sigmaf    = sigmaf # spectral width of the SETI signal
        self.duration  = duration # duration of the observation

    def simulate(self):

        fc = self.fc.to(u.MHz)
        bw = self.bw.to(u.MHz)
        duration = self.duration.to(u.s)
        sigmaf = self.sigmaf.to(u.MHz)
        driftrate = self.driftrate.to(u.MHz / u.s)
        tstart = self.tstart.to(u.s)


        data = np.random.normal(0,1, size = (self.nbins, self.nchans))
        freqs = np.linspace((fc + bw / 2).value , (fc - bw / 2).value, self.nchans)
        times = np.linspace(0, duration.value , self.nbins)
        df = np.abs(freqs[1] - freqs[0])
        dt = np.abs(times[1] - times[0])
        nstart = np.rint(tstart.value / dt).astype(int)

        fstart = freqs[self.cstart]
        fstop  = freqs[self.cstop]

        DT = (fstop - fstart) / driftrate.value
        NT = np.rint(DT / dt).astype(int)

        for binidx in range(self.nbins):
            snr_bin = np.sqrt(1 / self.nbins) * int(self.snr)
            spectrum = data[binidx, :]
            wchan = np.rint(sigmaf.value / df).astype(int)

            chan = np.rint( (fstart + driftrate.value * (times[binidx] - tstart.value ) )  / df).astype(int)
            std_off = get_stdoff(spectrum, chan, wchan)

            # Generate Gaussian signal and add it to the data
            signal = gauss(freqs, snr_bin * std_off, fstart + driftrate.value * (times[binidx] - tstart.value), sigmaf.value)
            data[binidx, :] += signal

        return data

    def inject(self, data):

        data = np.array(data, dtype = "float64")

        if self.nchans != data.shape[1]:
            raise ValueError("The number of frequency channels of the injected telemetry should be the same with the data")
        if self.nbins != data.shape[0]:
            raise ValueError("The number of time bins of the injected telemetry should be the same with the data")


        fc = self.fc.to(u.MHz)
        bw = self.bw.to(u.MHz)
        duration = self.duration.to(u.s)
        sigmaf = self.sigmaf.to(u.MHz)
        driftrate = self.driftrate.to(u.MHz / u.s)
        tstart = self.tstart.to(u.s)



        freqs = np.linspace((fc + bw / 2).value , (fc - bw / 2).value, self.nchans)
        times = np.linspace(0, duration.value , self.nbins)
        df = np.abs(freqs[1] - freqs[0])
        dt = np.abs(times[1] - times[0])
        nstart = np.rint(tstart.value / dt).astype(int)

        fstart = freqs[self.cstart]
        fstop  = freqs[self.cstop]

        DT = (fstop - fstart) / driftrate.value
        NT = np.rint(DT / dt).astype(int)

        for binidx in range(self.nbins):
            snr_bin = np.sqrt(1 / self.nbins) * int(self.snr)
            spectrum = data[binidx, :]
            wchan = np.rint(sigmaf.value / df).astype(int)

            chan = np.rint( (fstart + driftrate.value * (times[binidx] - tstart.value ) )  / df).astype(int)
            std_off = get_stdoff(spectrum, chan, wchan)

            # Generate Gaussian signal and add it to the data
            signal = gauss(freqs, snr_bin * std_off, fstart + driftrate.value * (times[binidx] - tstart.value), sigmaf.value)
            data[binidx, :] += signal

        return data


def telemetry_simulation(data, header, snr, df, f_start, t_start, t_length):
    nchans  = header['nchans']
    tsamp   = header['tsamp']
    nsamp   = data.shape[0]
    foff    = header['foff']
    fch0    = header['fch1']
    tstart  = header['tstart']
    bw      = nchans * foff
    fc      = fch0 + bw / 2
    obslen  = nsamp * tsamp

    freqs = np.arange(fc - bw / 2, fc + bw / 2, foff)
    times = np.linspace(0,obslen,int(nsamp))

    if(df == 'nudot'):
        df = (freqs[0] - freqs[-1]) / 2 / obslen * u.MHz / u.s
    else:
        df = float(df) * u.Hz / u.s

    sf = bw * u.MHz / nchans
    
    FT = FakeTelemetry(fc = fc * u.MHz, bw = bw * u.MHz, driftrate = df, nchans = int(nchans), nbins = int(nsamp), duration = obslen * u.s, sigmaf = sf, snr = snr, tstart = 0 * u.s, cstart = 168, cstop = nchans-1)
    
    injected_data = FT.inject(data)
    
    return injected_data


def write_with_your(matrix, header, output_dir, output_name):
    c = SkyCoord(ra=header['src_raj'], dec=header['src_dej'])
    dec_dms = c.dec.dms
    ra_hms = c.ra.hms

    output_data = make_sigproc_object(
        rawdatafile   = 'foo.fil',
        source_name   = header['source_name'],
        nchans        = header['nchans'],
        foff          = header['foff'],            # MHz
        fch1          = header['fch1'],            # MHz
        tsamp         = header['tsamp'],           # seconds
        tstart        = header['tstart'],          # MJD
        nbits         = header['nbits'],
        src_raj       = float(str(str(int(ra_hms.h)) + str(abs(int(ra_hms.m))) + str(abs(ra_hms.s)))),
        src_dej       = float(str(str(int(dec_dms.d)) + str(abs(int(dec_dms.m))) + str(abs(dec_dms.s))))
    )

    output_data.write_header('foo.fil')
    output_data.append_spectra(matrix, 'foo.fil')
    os.rename('foo.fil', os.path.join(output_dir, output_name))
    logging.debug('04a_your library succesfully written new file')

def write_with_blimpy(matrix, bl_object, output_dir, output_name):
    bl_object.data = []
    bl_object.data = np.expand_dims(matrix, axis=1)
    bl.Waterfall.write_to_fil(bl_object, os.path.join(output_dir, output_name))
    logging.debug('04b_blimpy library succesfully written new file')

def plot_data(data, name, output_dir, timestamp):
    plt.clf()
    plt.imshow(np.squeeze(data), aspect="auto")
    plt.ylabel("Time (bins)")
    plt.xlabel("Frequency (channels)")
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, timestamp + "/" + name + ".png"))
    logging.debug('Plot saved: ' + str(output_dir) + "/" + timestamp + "/" + name)

def process_data(file, output_dir, output_name, setigen_inj, telsim_inj, writer, snr, df, f_start, t_start, sig_length):
    timestamp = str(time.time())
    os.mkdir(os.path.join(output_dir, timestamp))

    wf = bl.Waterfall(file)
    header = wf.header
    frame = stg.Frame(wf)
    
    logging.debug('01_script initialized and ready to process data')
    logging.debug(str((locals().keys())) + str((locals().values())))

    plot_data(wf.data, "00_input_matrix", output_dir, timestamp)

    logging.debug('02_input data plot saved')
    
    if(setigen_inj == True):
        output_matrix = setigen_injection(frame, snr, df, header, f_start, t_start, sig_length)
    if(telsim_inj == True):
        output_matrix = telemetry_simulation(np.squeeze(wf.data), header, snr, df, f_start, t_start, sig_length)

    plot_data(output_matrix, "01_output_matrix", output_dir, timestamp)
    
    if output_name is None:
        filedir, name = os.path.split(file)
        output_name = name.replace(".fil","") + "_injected" + ".fil"

    if(writer == 'your'):
        write_with_your(output_matrix, header, output_dir, output_name)
    if(writer == 'blimpy'):
        write_with_blimpy(output_matrix, wf, output_dir, output_name)

    wf2 = bl.Waterfall(os.path.join(output_dir, output_name))
    plot_data(wf2.data, "03_saved&loaded", output_dir, timestamp)

    logging.debug(str(np.array_equal(output_matrix, np.squeeze(wf2.data))))

def _get_parser():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description = "Load .fil, inject signal, save as new .fil" + "\n"
                      "setigen or telsim available" + "\n")
    parser.add_argument('-f',
                        '--fil_file',
                        action = "store" ,
                        help = "filterbank to be processed (REQUIRED)",
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
                        help = "Output File Name (Default: filename_injected.fil)",
                        default = None
                        )
    parser.add_argument('-stg',
                        '--setigen',
                        help = "Using setigen library to inject a signal",
                        action = 'store_true',
                        )
    parser.add_argument('-tls',
                        '--telsim',
                        help = "Using telemetry ",
                        action = 'store_true',
                        )
    parser.add_argument('-v',
                        '--verbose',
                        action = "store_true" ,
                        help = "prints out debugging logs"
                        )
    parser.add_argument('-w',
                        '--writer',
                        action = "store" ,
                        help = "choose from your or blimpy (REQUIRED)",
                        required = True
                        )
    parser.add_argument('-snr',
                        '--signal_to_noise_ratio',
                        action="store",
                        help = "provide with integer - the higher number, the stronger signal to background generated",
                        default=400
                        )
    parser.add_argument('-df',
                        '--drift_rate',
                        action="store",
                        help = "provide with integer (Hz/s) or 'nudot' for calculated. Default is zero",
                        default=0
                        )
    parser.add_argument('-fs',
                        '--f_start',
                        action="store",
                        help = "provide integer - frequency channel. Default is zero",
                        default=0
                        )
    parser.add_argument('-ts',
                        '--t_start',
                        action="store",
                        help="Provide time when signals starts as integer (bins). Default is zero",
                        default=0
                        )
    parser.add_argument('-sl',
                        "--sig_length",
                        action="store",
                        help="Provide with length of signal as integer (bins). Default is full time",
                        default="all"
                        )
    return parser.parse_args()

# init script

if __name__ == '__main__':

    args = _get_parser()

    verbose = args.verbose

    # Set up logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    process_data(args.fil_file,
                args.output_dir,
                args.output_name,
                args.setigen,
                args.telsim,
                args.writer,
                args.signal_to_noise_ratio,
                args.drift_rate,
                args.f_start,
                args.t_start,
                args.sig_length
    )