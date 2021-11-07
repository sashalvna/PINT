#!/usr/bin/env python
import sys

import astropy.io.fits as pyfits
import astropy.units as u
import numpy as np
from astropy import log

import pint.models
import pint.residuals
import pint.toa as toa
from pint.event_toas import load_event_TOAs, load_fits_TOAs
from pint.eventstats import h2sig, hm
from pint.fits_utils import read_fits_event_mjds
from pint.observatory.satellite_obs import get_satellite_observatory
from pint.plot_utils import phaseogram_binned
from pint.pulsar_mjd import Time

import fitsio
import polycos2 as polycos
import os.path
import os

try:
    from erfa import DAYSEC as SECS_PER_DAY
except ImportError:
    from astropy._erfa import DAYSEC as SECS_PER_DAY


__all__ = ["main"]


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Use PINT to compute event phases and make plots of photon event files."
    )
    parser.add_argument(
        "eventfile",
        help="Photon event FITS file name (e.g. from NICER, RXTE, XMM, Chandra).",
    )
    parser.add_argument("parfile", help="par file to construct model from")
    parser.add_argument("--orbfile", help="Name of orbit file", default=None)
    parser.add_argument(
        "--maxMJD", help="Maximum MJD to include in analysis", default=None
    )
    parser.add_argument(
        "--minMJD", help="Minimum MJD to include in analysis", default=None
    )
    parser.add_argument(
        "--plotfile", help="Output figure file name (default=None)", default=None
    )
    parser.add_argument(
        "--addphase",
        help="Write FITS file with added phase column",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--addorbphase",
        help="Write FITS file with added orbital phase column",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--absphase",
        help="Write FITS file with integral portion of pulse phase (ABS_PHASE)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--barytime",
        help="Write FITS file with a column containing the barycentric time as double precision MJD.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--outfile",
        help="Output FITS file name (default=same as eventfile)",
        default=None,
    )
    parser.add_argument(
        "--ephem", help="Planetary ephemeris to use (default=DE421)", default="DE421"
    )
    parser.add_argument(
        "--tdbmethod",
        help="Method for computing TT to TDB (default=astropy)",
        default="default",
    )
    parser.add_argument(
        "--plot", help="Show phaseogram plot.", action="store_true", default=False
    )
    parser.add_argument(
        "--use_gps",
        default=False,
        action="store_true",
        help="Apply GPS to UTC clock corrections",
    )
    parser.add_argument(
        "--use_bipm",
        default=False,
        action="store_true",
        help="Use TT(BIPM) instead of TT(TAI)",
    )
    #    parser.add_argument("--fix",help="Apply 1.0 second offset for NICER", action='store_true', default=False)

    parser.add_argument(
        "--polycos",
        #default=False,
        action="store_true",
        help="Use polycos to calculate phases; use this when working with very large event files. Creates multiple, smaller event files"
    )

    parser.add_argument(
        "--counts",
        default=1000000,
        help="Specify the number of counts per event file; only use with the polycos argument. Default=1000000"
    )

    parser.add_argument(
        "--overwrite",
        #default=False,
        action="store_true",
        help="Overwrites existing event files; only use with the polycos argument"
    )

    args = parser.parse_args(argv)

    # If outfile is specified, that implies addphase
    if args.outfile is not None:
        args.addphase = True

    # If plotfile is specified, that implies plot
    if args.plotfile is not None:
        args.plot = True

    # set MJD ranges
    maxmjd = np.inf if (args.maxMJD is None) else float(args.maxMJD)
    minmjd = 0.0 if (args.minMJD is None) else float(args.minMJD)

    # Read event file header to figure out what instrument is is from
    hdr = pyfits.getheader(args.eventfile, ext=1)

    log.info(
        "Event file TELESCOPE = {0}, INSTRUMENT = {1}".format(
            hdr["TELESCOP"], hdr["INSTRUME"]
        )
    )

    telescope = hdr["TELESCOP"].lower()

    # Instantiate observatory once so it gets added to the observatory registry
    if args.orbfile is not None:
        log.info(f"Setting up {telescope.upper()} observatory")
        try:
            get_satellite_observatory(telescope, args.orbfile)
        except Exception:
            log.error(
                "The orbit file is not recognized. It is likely that this mission is not supported. "
                "Please barycenter the event file using the official mission tools before processing with PINT"
            )

    # Read in model
    modelin = pint.models.get_model(args.parfile)
    use_planets = False
    if "PLANET_SHAPIRO" in modelin.params:
        if modelin.PLANET_SHAPIRO.value:
            use_planets = True
    if "AbsPhase" not in modelin.components:
        log.error(
            "TimingModel does not include AbsPhase component, which is required "
            "for computing phases. Make sure you have TZR* parameters in your par file!"
        )
        raise ValueError("Model missing AbsPhase component.")

    if args.addorbphase and (not hasattr(modelin, "binary_model_name")):
        log.error(
            "TimingModel does not include a binary model, which is required for "
            "computing orbital phases. Make sure you have BINARY and associated "
            "model parameters in your par file!"
        )
        raise ValueError("Model missing BINARY compo:wnent.")

    if args.polycos:
        log.info("Using polycos to get pulse phases")
        segLength = 120 #in minutes
        ncoeff = 10
        obsfreq = 0
        event_hdr = fitsio.read_header(args.eventfile, ext=1)
        print(event_hdr)
        n_counts= event_hdr["NAXIS2"]
        minmjd = fitsio.FITS(args.eventfile)[1][0]['TIME']
        maxmjd = fitsio.FITS(args.eventfile)[1][n_counts-1]['TIME']
        if "TIMEZERO" not in event_hdr and "TIMEZERI" not in event_hdr:
            TIMEZERO = 0
        else:
            try:
                TIMEZERO = np.longdouble(event_hdr["TIMEZERO"])
            except KeyError:
                TIMEZERO = np.longdouble(event_hdr["TIMEZERI"]) + np.longdouble(event_hdr["TIMEZERF"])
        try:
            MJDREF = np.longdouble(event_hdr["MJDREF"])
        except KeyError:
            if isinstance(event_hdr["MJDREFF"], (str, bytes)):
                MJDREF = np.longdouble(event_hdr["MJDREFI"]) + fortran_float(event_hdr["MJDREFF"])
            else:
                MJDREF = np.longdouble(event_hdr["MJDREFI"]) + np.longdouble(event_hdr["MJDREFF"])
        minmjd = MJDREF + ((minmjd + TIMEZERO)/SECS_PER_DAY)
        maxmjd = MJDREF + ((maxmjd + TIMEZERO)/SECS_PER_DAY)
        print(minmjd, maxmjd)
        telescope_n = '@'
        p = polycos.Polycos()
        ptable = p.generate_polycos(
            modelin, minmjd, maxmjd, telescope_n,
            segLength, ncoeff, obsfreq)

        #read in header to build sections
        h = fitsio.read_header(args.eventfile, ext=1) 
        n_entries = h["NAXIS2"] 

        #get sections
        n = float(args.counts)
        sections = n_entries/n
        sectionlist = []
        val = 0
        while val <= n_entries:
            sectionlist.append(val)
            val = val + n
        low = []
        high = []
        for i in range(len(sectionlist)-1):
            m = sectionlist[i]
            n = sectionlist[i+1]
            low.append(int(m))
            high.append(int(n))
        low.append(int(n))
        high.append(n_entries)

        #column names (omit columns which have issues reading in, for now)
        #also omit pulse phase column, if it exists
        columns = []
        for i in range(1, h['TFIELDS']+1):
            if (h["TTYPE%s" %i] != 'DEADTIME') & (h["TTYPE%s" %i] != 'MPU_A_TEMP') & (h["TTYPE%s" %i] != 'PULSE_PHASE'):
                columns.append(h["TTYPE%s" %i])

        #get GTI values for the second HDU
        gti = fitsio.read(args.eventfile, ext=2)
        gti = tuple(list(gti))

        log.info('Creating new directory for event files inside %s_pipe'% h['OBS_ID'])
        os.chdir("%s_pipe"% h['OBS_ID'])
        try:
            os.mkdir('evt')
        except FileExistsError:
            log.info('evt directory already exists, putting event files into it')
        
        #check whether event files exist
        if args.overwrite:
            log.info('Overwriting event files')
            os.chdir('evt')
            for i in range(len(os.listdir('.'))):
                os.remove('cleanfilt_%s.evt'%i)
            os.chdir('..')
            os.chdir('..')
            sections = range(len(low))
            total = len(low)
        else:
            os.chdir('evt')
            sections = []
            for m in range(len(low)):
                if os.path.isfile('cleanfilt_%s.evt'% m) == True:
                    log.info('cleanfilt_%s.evt already exists, skipping'% m)
                else:
                    sections.append(m)
            total = len(sections)
            os.chdir("..")
            os.chdir("..")
        
        #create files for each section
        for m in sections:
            log.info("Calculating phases for section %s of %s"% (m, len(low)-1))
            newlow = []
            newhigh = []
            val = low[m]
            if (val+1000) <= high[m]:
                while (val+1000) <= high[m]:
                    newlow.append(val)
                    val += 1000
                    newhigh.append(val)
            else:
                while (val+1) <= high[m]:
                    newlow.append(val)
                    val += 1
                    newhigh.append(val)
            times = []
            for i in range(len(newlow)):  #get phase values from times
                time = fitsio.read(args.eventfile, rows=[np.arange(newlow[i], newhigh[i])], columns='TIME', ext=1)
                if "TIMEZERO" not in event_hdr and "TIMEZERI" not in event_hdr:
                    TIMEZERO = 0
                else:
                    try:
                        TIMEZERO = np.longdouble(event_hdr["TIMEZERO"])
                    except KeyError:
                        TIMEZERO = np.longdouble(event_hdr["TIMEZERI"]) + np.longdouble(event_hdr["TIMEZERF"])
                try:
                    MJDREF = np.longdouble(event_hdr["MJDREF"])
                except KeyError:
                    if isinstance(event_hdr["MJDREFF"], (str, bytes)):
                        MJDREF = np.longdouble(event_hdr["MJDREFI"]) + fortran_float(event_hdr["MJDREFF"])
                    else:
                        MJDREF = np.longdouble(event_hdr["MJDREFI"]) + np.longdouble(event_hdr["MJDREFF"])
                for b in time:
                    mjd = MJDREF + ((b + TIMEZERO)/SECS_PER_DAY)
                    times.append(mjd)

            times = np.array(times)
            phases = p.eval_phase(times)
            print(phases[0:10])

            log.info("Writing new event file for section %s of %s"% (m, len(low)-1))
            length = len(phases)
            counter=0
            while counter<len(phases):
                for i in range(len(newlow)): #create new fits file with same columns and header + new column
                    colvals = fitsio.read(args.eventfile, rows=[np.arange(newlow[i], newhigh[i])], columns=columns, ext=1) #get other column values
                    rows = []
                    for r in range(len(colvals)):
                        values = []
                        for n in range(len(colvals[r])):
                            try:
                                values.append(float(colvals[r][n]))
                            except TypeError:
                                values.append(colvals[r][n])
                        phaseval = float(phases[counter+r])
                        if phaseval <0:
                            phaseval += 1
                        row = list(np.append(values, phaseval))
                        rows.append(row)

                    dtype = np.dtype([('TIME', '>f8'), ('RAWX', 'u1'), ('RAWY', 'u1'), ('PHA', '>i2'), ('PHA_FAST', '>i2'), ('DET_ID', 'u1'), ('EVENT_FLAGS', '?', (8,)), ('TICK', '>i8'), ('MPU_UNDER_COUNT', '>i4'), ('PI_FAST', '>i2'), ('PI', '>i2'), ('PI_RATIO', '>f4'), ('PULSE_PHASE', '>f8')])
                    rows = np.array([tuple(k) for k in rows], dtype=dtype)
                    os.chdir("%s_pipe/evt"% h['OBS_ID'])
                    newfilename = 'cleanfilt_%s.evt'% m
                    if os.path.isfile(newfilename):
                        fits = fitsio.FITS(newfilename, 'rw')
                        fits[1].append(rows) #write to existing (new) fits file
                        fits.close()
                    else:
                        fitsio.write(newfilename, rows, header=h) #create new fits file

                    counter += newhigh[0]-newlow[0]
                    os.chdir('..')
                    os.chdir('..')
            
            os.chdir("%s_pipe/evt"% h['OBS_ID'])
            fitsio.write(newfilename, np.array([gti[0]], dtype=[('START', '>f8'), ('STOP', '>f8')])) 
            for j in range(1, len(gti)):
                fits = fitsio.FITS(newfilename, 'rw')
                fits[2].write(np.array([gti[j]], dtype=[('START', '>f8'), ('STOP', '>f8')]), firstrow=j)
                fits.close()
            os.chdir('..')
            os.chdir('..')

    else:
        # Read event file and return list of TOA objects
        try:
            tl = load_event_TOAs(args.eventfile, telescope, minmjd=minmjd, maxmjd=maxmjd)
        except KeyError:
            log.error(
                "Observatory not recognized. This probably means you need to provide an orbit file or barycenter the event file."
            )
            sys.exit(1)


        # Now convert to TOAs object and compute TDBs and posvels
        if len(tl) == 0:
            log.error("No TOAs, exiting!")
            sys.exit(0)

    
        ts = toa.get_TOAs_list(
            tl,
            ephem=args.ephem,
            include_bipm=args.use_bipm,
            include_gps=args.use_gps,
            planets=use_planets,
            tdb_method=args.tdbmethod,
        )   
        ts.filename = args.eventfile
        #    if args.fix:
        #        ts.adjust_TOAs(TimeDelta(np.ones(len(ts.table))*-1.0*u.s,scale='tt'))

        print(ts.get_summary())
        mjds = ts.get_mjds()
        print(mjds.min(), mjds.max())
    
        # Compute model phase for each TOA
        iphss, phss = modelin.phase(ts, abs_phase=True)
        phases = phss.value % 1
        h = float(hm(phases))
        print("Htest : {0:.2f} ({1:.2f} sigma)".format(h, h2sig(h)))
        if args.plot:
            phaseogram_binned(mjds, phases, bins=100, plotfile=args.plotfile)

        # Compute orbital phases for each photon TOA
        if args.addorbphase:
            delay = modelin.delay(ts)
            orbits = modelin.binary_instance.orbits()
            # These lines are already in orbits.orbit_phase() in binary_orbits.py.
            # What is the correct syntax is to call this function here?
            norbits = np.array(np.floor(orbits), dtype=int)
            orbphases = orbits - norbits  # fractional phase
        """ """
        if args.addphase or args.addorbphase:
            # Read input FITS file (again).
            # If overwriting, open in 'update' mode
            if args.outfile is None:
                hdulist = pyfits.open(args.eventfile, mode="update")
            else:
                hdulist = pyfits.open(args.eventfile)
        
            datacol = []
            data_to_add = {}

            # Handle case where minMJD/maxMJD do not exceed length of events
            mjds_float = read_fits_event_mjds(hdulist[1])
            time_mask = np.logical_and((mjds_float > minmjd), (mjds_float < maxmjd))

            if args.addphase:
                if time_mask.sum() != len(phases):
                    raise RuntimeError(
                        "Mismatch between data selection {0} and length of phase array ({1})!".format(
                            time_mask.sum(), len(phases)
                        )
                    )
                data_to_add["PULSE_PHASE"] = [phases, "D"]

            if args.absphase:
                data_to_add["ABS_PHASE"] = [iphss, "K"]

            if args.barytime:
                bats = modelin.get_barycentric_toas(ts)
                data_to_add["BARY_TIME"] = [bats, "D"]

            if args.addorbphase:
                if time_mask.sum() != len(orbphases):
                    raise RuntimeError(
                        "Mismatch between data selection ({0}) and length of orbital phase array ({1})!".format(
                            time_mask.sum(), len(orbphases)
                        )
                    )
                data_to_add["ORBIT_PHASE"] = [orbphases, "D"]
            # End if args.addorbphase

            for key in data_to_add.keys():
                if key in hdulist[1].columns.names:
                    log.info("Found existing %s column, overwriting..." % key)
                    # Overwrite values in existing Column
                    hdulist[1].data[key][time_mask] = data_to_add[key][0]
                else:
                    # Construct and append new column, preserving HDU header and name
                    log.info("Adding new %s column." % key)
                    new_dat = np.full(time_mask.shape, -1, dtype=data_to_add[key][0].dtype)
                    new_dat[time_mask] = data_to_add[key][0]
                    datacol.append(
                        pyfits.ColDefs(
                            [
                                pyfits.Column(
                                    name=key, format=data_to_add[key][1], array=new_dat
                                )
                            ]
                        )
                    )

            if len(datacol) > 0:
                cols = hdulist[1].columns
                for c in datacol:
                    cols = cols + c
                bt = pyfits.BinTableHDU.from_columns(
                    cols, header=hdulist[1].header, name=hdulist[1].name
                )
                hdulist[1] = bt

            if args.outfile is None:
                # Overwrite the existing file
                log.info("Overwriting existing FITS file " + args.eventfile)
                hdulist.flush(verbose=True, output_verify="warn")
            else:
                # Write to new output file
                log.info("Writing output FITS file " + args.outfile)
                hdulist.writeto(
                    args.outfile, overwrite=True, checksum=True, output_verify="warn"
                )

    

if __name__ == '__main__':
    main()
                              


