import sys
import os
import matplotlib

import setigen as stg
import numpy as np

from scipy.signal import correlate
from scipy.linalg import toeplitz
from astropy import units as u
from astropy.coordinates import SkyCoord
from your.formats.filwriter import make_sigproc_object
from your import Your
from blimpy import Waterfall




data_written = np.random.randint(low=0, high=255, size=(8192,1024), dtype=np.uint)
def create_sigproc_object():
    data_written = np.random.randint(low=0, high=255, size=(8192,1024), dtype=np.uint)
    sigproc_object = make_sigproc_object(
    rawdatafile  = "foo.fil",
    source_name = "bar",
    nchans  = 1024, 
    foff = -0.1, #MHz
    fch1 = 2000, # MHz
    tsamp = 256e-6, # seconds
    tstart = 59246, #MJD
    src_raj=112233.44, # HHMMSS.SS
    src_dej=112233.44, # DDMMSS.SS
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
    sigproc_object.write_header("foo.fil")
    sigproc_object.append_spectra(data_written, "foo.fil")
    return Your("foo.fil")

def generate_signal():
    os.environ['SETIGEN_ENABLE_GPU'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    frame_1 = stg.Frame(fchans=1024*u.pixel,
                    tchans=32*u.pixel,
                    df=2.7939677238464355*u.Hz,
                    dt=18.253611008*u.s,
                    fch1=6095.214842353016*u.MHz)

    noise = frame_1.add_noise(x_mean=10, noise_type='chi2')
    signal = frame_1.add_signal(stg.constant_path(f_start=frame_1.get_frequency(index=200),
                                                drift_rate=2*u.Hz/u.s),
                            stg.constant_t_profile(level=frame_1.get_intensity(snr=30)),
                            stg.gaussian_f_profile(width=40*u.Hz),
                            stg.constant_bp_profile(level=1))
    return frame_1.fch1


def test_python():
    assert(sys.version.split(' ')[0].strip() == "3.11.4")
    assert(os.system('conda info | grep ": klt"') == 0)

def test_libraries():
    assert(np.__version__ == "1.24.4")
    assert(matplotlib.__version__ == "3.7.1")
    assert(correlate(np.array([[1, 2], [3, 4]], np.int32),np.array([[1, 2], [3, 4]], np.int32))[0][0] == 4)
    assert(toeplitz(np.array([[1, 2], [3, 4]], np.int32)[1:4])[0][0] == 3)
    assert(str(SkyCoord(ra=10.68458*u.deg, dec=41.26917*u.deg, frame='icrs').ra) == "10d41m04.488s")
    assert(create_sigproc_object().your_header.fch1 == 2000)
    assert(Waterfall('foo.fil').header['fch1'] == 2000)
    assert(generate_signal() == 6095214842.353016)