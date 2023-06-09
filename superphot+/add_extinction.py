from dustmaps.sfd import SFDQuery
from astropy.coordinates import SkyCoord
import extinction
import numpy as np

sfd = SFDQuery()

def get_band_extinctions(ra, dec):
    """
    Get green and red band extinctions in magnitudes for
    a single supernova LC based on RA and DEC.
    """
    #First look up the amount of mw dust at this location
    coords = SkyCoord(ra,dec, frame='icrs', unit="deg")
    Av_sfd = 2.742 * sfd(coords) # from https://dustmaps.readthedocs.io/en/latest/examples.html
    
    # for gr, the was are:
    band_wvs = 1./ (0.0001 * np.asarray([4741.64, 6173.23])) # in inverse microns
    
    #Now figure out how much the magnitude is affected by this dust
    ext_list = extinction.fm07(band_wvs, Av_sfd, unit='invum') # in magnitudes
    
    return ext_list

def get_sn_ra_dec(ztf_name, filtered_csv):
    """
    Get the RA and DEC of a supernova from
    filtered summary info.
    """
    with open(filtered_csv, "r") as cf:
        for row in cf:
            if row.split(",")[0] == ztf_name:
                ra = float(row.split(",")[2])
                dec = float(row.split(",")[3])
                return ra, dec