from abc import ABC, abstractmethod
import os
import re

import pandas as pd
from datetime import timedelta

import mya
import utils


class BaseModel(ABC):
    """An abstract class defining the model API and providing several useful concrete functions.

    Inheriting from this class requires that both the describe and analzye methods be defined which form the basis of
    the model API.  This class provides a number of concrete methods that can be used to perform data processing and
    validation.  Child classes may use these directly or as a guide for writing their own.

    Attributes:
        event_dir (str): The path to the event directory being analyzed by this model object.
        event_df  (pandas.dataframe): A Pandas dataframe containing the parsed waveform data from event_dir.

    """
    file_regex = re.compile(r"R.*harv\..*\.txt")

    def __init__(self, event_dir):
        """Instantiates a BaseModel object that contains a reference to the filesystem location of the event data.

            Args:
                event_dir (str): Path to the fault event directory.
        """
        self.event_dir = event_dir
        self.event_df = None

    @abstractmethod
    def analyze(self):
        """A method that performs some analysis and classifies the fault event by cavity number and fault type.

            This method is the most important part of any model package.  Here is where the desired analytical
            approaches should be implemented.  In addition to the classification label output, this method should
            include information about the confidence of those classifications and list which fault event is being
            analyzed.  Confidence numbers should be given on the range [0,1] with lower numbers implying more
            uncertainty and higher numbers implying greater certainty.


            Returns:
                dict: A dictionary containing the results of the analysis.  Detailed key/value information given in the table below.

            +---------------------+------------+-----------------------------------------------------------------+
            | Key                 | Value Type | Value Descriptions                                              |
            +=====================+============+=================================================================+
            | "location"          | str        | Zone of the fault event. (e.g. "1L22")                          |
            +---------------------+------------+-----------------------------------------------------------------+
            | "timestamp"         | str        | Timestamp of the fault event, (e.g. "2019-12-25 01:23:45.6")    |
            +---------------------+------------+-----------------------------------------------------------------+
            | "cavity-label"      | str        | Label of the cavity that faulted (e.g., "cav1")                 |
            +---------------------+------------+-----------------------------------------------------------------+
            | "cavity-confidence" | float      | Number between [0,1] representing cavity label confidence       |
            +---------------------+------------+-----------------------------------------------------------------+
            | "fault-label"       | str        | Label of the identified fault type (e.g., quench)               |
            +---------------------+------------+-----------------------------------------------------------------+
            | "fault-confidence"  | float      | Number between [0,1] representing fault type label confidence   |
            +---------------------+------------+-----------------------------------------------------------------+
        """
        pass

    def is_capture_file(self, filename):
        """Validates if filename appears to be a valid capture file.

            Args:
                filename (str): The name of the file that is to be validated

            Returns:
                bool: True if the filename appears to be a valid capture file.  Otherwise False.
        """
        return BaseModel.file_regex.match(filename)

    # Parse the capture file specified by name (not full path) and return a pandas dataframe
    def parse_capture_file(self, filename):
        """Parses an individual capture file into a Pandas dataframe object.

            Args:
                filename: The name of the file, , relative to event_dir, to be parsed

            Returns:
                dataframe: A pandas dataframe containing the data from the specified capture file
        """
        return pd.read_csv(os.path.join(self.event_dir, filename), sep="\t", comment='#', skip_blank_lines=True)

    def parse_event_dir(self):
        """Parses the  capture files in the BaseModel's event_dir and sets event_df to the appropriate pandas dataframe.

        The waveform names are converted from <EPICS_NAME><Waveform> (e.g., R123WFSGMES), to <Cavity_Number>_<Waveform>
        (e.g., 3_GMES).  This allows analysis code to more easily handle waveforms from different zones.

            Returns:
                None

            Raises:
                 ValueError: if a column name is discovered with an unexpected format
        """
        zone_df = None

        for filename in sorted(os.listdir(self.event_dir)):
            # Only try to process files that look like capture files
            if not self.is_capture_file(filename):
                continue
            if zone_df is None:
                zone_df = self.parse_capture_file(filename)
            else:
                # Join the existing zone data with the new capture file by using the "Time" column as an index to
                # match rows
                zone_df = zone_df.join(self.parse_capture_file(filename).set_index("Time"), on="Time")

        # Now format the column names to remove the zone information but keep a cavity and signal identifiers
        pattern = re.compile(r'R\d\w\dWF[TS]')
        new_columns = []
        for column in zone_df.columns:
            if column != "Time":
                # This only works for PV/waveform names of the proper format.  That's all we should be working with.
                if not pattern.match(column):
                    raise ValueError("Found unexpected waveform data - " + column)
                column = column[3] + "_" + column[7:]
            new_columns.append(column)
        zone_df.columns = new_columns

        self.event_df = zone_df

    def get_event_df(self):
        """Accessor method for class attribute event_df."""
        return self.event_df

    def validate_capture_file_counts(self):
        """This method checks that we have exactly one capture file per cavity/IOC.

        The harvester grouping logic coupled with unreliable IOC behavior seems to produce fault event directories where
        either an IOC has multiple capture files or are missing.  We want to make sure we have exactly eight capture
        files - one per IOC.  Raises an exception in the case that something is amiss.

            Returns:
                None

            Raises:
                ValueError: if either missing or "duplicate" capture files are found.
        """

        # Count capture files per cavity
        capture_file_counts = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0}
        for filename in os.listdir(self.event_dir):
            cavity = filename[3]
            if cavity not in capture_file_counts.keys():
                raise ValueError("Found capture file for an unsupported cavity - " + cavity)

            capture_file_counts[cavity] += 1

        for cavity in capture_file_counts.keys():
            if capture_file_counts[cavity] > 1:
                raise ValueError("Duplicate capture files exist for zone '" + cavity + "'")
            if capture_file_counts[cavity] == 0:
                raise ValueError("Missing capture file for zone '" + cavity + "'")

    def validate_capture_file_waveforms(self):
        """Checks that all of the required waveforms are present exactly one time across all capture files.

            Returns:
                None

            Raises:
                ValueError: if any required waveform is repeated
        """

        # Get a structure for counting matches of waveforms
        req_signals = ["IMES", "QMES", "GMES", "PMES", "IASK", "QASK", "GASK", "PASK", "CRFP", "CRFPP",
                       "CRRP", "CRRPP", "GLDE", "PLDE", "DETA2", "CFQE2", "DFQES"]

        # Will contain regexs that are used to check for required waveforms, and the count of how many matches
        req_waveforms = {re.compile("Time"): 0}
        for sig in req_signals:
            for cav in [1, 2, 3, 4, 5, 6, 7, 8]:
                wf = r"R\d\w" + str(cav) + "WF[ST]" + sig + "$"
                req_waveforms[re.compile(wf)] = 0

        # Metadata lines are lines at the top of the file that start with a #.  Probably no spaces, but just to be safe
        metadata_regex = re.compile(r"^\s*#")

        # Go through each capture file and make sure that the required waveforms are present
        for filename in os.listdir(self.event_dir):
            if not self.is_capture_file(filename):
                continue
            file = open(os.path.join(self.event_dir, filename), "r")

            # Get the header line, which should be either the first line just after the metadata or the very first line
            line = "#"
            while metadata_regex.match(line):
                line = file.readline().rstrip("\n")

            # Check each column header for a match
            for col_name in line.split("\t"):
                for pattern in req_waveforms.keys():
                    if pattern.match(col_name):
                        req_waveforms[pattern] += 1

            # Supposedly garbage collector would take care of this, but this seems cleaner
            file.close()

        # Validate that each of the patterns had exactly one match
        for pattern in req_waveforms.keys():
            if pattern.pattern == "Time":
                if req_waveforms[pattern] != 8:
                    raise ValueError("Model found " + str(req_waveforms[pattern]) + " Time columns.  Expected eight.")
            else:
                if req_waveforms[pattern] > 1:
                    raise ValueError("Model found multiple waveforms that matched pattern '" + pattern.pattern + "'")
                if req_waveforms[pattern] < 1:
                    raise ValueError(
                        "Model could not identify require waveform matching pattern '" + pattern.pattern + "'")

    def validate_waveform_times(self, time_limits=(-1600, 1600), delta_max=0.025):
        """Verify the Time column of all capture files are identical and have a valid range and sample interval.

            Args:
                time_limits (tuple): A two-valued tuple (min_t, max_t) which gives the minimum and maximum time values
                    that are allowed for a valid waveform.  Values in milliSeconds.
                delta_max (float): The maximum difference between the smallest and largest time steps in milliseconds.
                    (default is 0.025)

            Returns:
                None

            Raises:
                ValueError: if either Time columns mismatch or Time columns are beyond expected thresholds

        """

        # Check that all of the file have the same time series
        first_filename = ""
        time = None
        for filename in os.listdir(self.event_dir):
            if self.is_capture_file(filename=filename):
                if time is None:
                    first_filename = filename
                    time = self.parse_capture_file(filename)['Time']
                else:
                    if not time.equals(self.parse_capture_file(filename)['Time']):
                        raise ValueError(
                            "Found Time series mismatch between '{}' and '{}'".format(first_filename, filename))

        # Check that the time range is somewhere in the [-1.6s, 1.6s] range
        min_t = min(time)
        max_t = max(time)
        if min_t < time_limits[0] or max_t > time_limits[1]:
            raise ValueError(
                "Invalid time range of [{},{}] found outside of normal [{}, {}] bounds".format(min_t, max_t,
                                                                                               time_limits[0],
                                                                                               time_limits[1]))

        # Check that the time sample interval is approximately the same.  Since this is floating point, there may be
        # slight differences
        lag = time - time.shift(1)
        lag = lag[1:len(lag)]
        delta = max(lag) - min(lag)
        if delta > delta_max:
            raise ValueError("Found discrepancies among sample intervals.  Range of intervals is {}".format(delta))

    def validate_cavity_modes(self, mode=4, offset=-1.0):
        """Checks that each cavity was in the appropriate control mode.

        A request is made to the internal CEBAF myaweb myquery HTTP service at the specified offset from the event
        timestamp.  Currently the proper mode is GDR (I/Q).

        According to the RF low-level software developer (lahti), the proper PV for C100 IOCs is
        R<Linac><Zone><Cavity>CNTL2MODE which is a float treated like a bitword.  At the time of writing, the most
        common modes are:

        * 2 == SEL
        * 4 == GDR (I/Q)

        A single cavity may be bypassed by operations to alleviate performance problems.  In the situation the rest of
        the zone is working normally and is considered to produce valid data for modeling purposes.  Only the control
        modes of the non-bypassed cavities will be considered for invalidating the data.

            Args:
                mode (int):  The mode number associated with the proper control mode.
                offset (float): The number of seconds before the fault event the mode setting should be checked.

            Returns:
                None

            Raises:
                ValueError: if any cavity mode does not match the value specified by the mode parameter.
        """

        # The R???CNTL2MODE PV is a float, treated like a bitword.  GDR (I/Q) mode corresponds to a value of 4.
        mode_template = '{}CNTL2MODE'

        # "Newer" C100 bypass control.  It's a bitword that represents the bypass status of all cavities
        bypassed_template = '{}XMOUT'

        # Still need to check if GSET == 0 since this is how many operators "bypass" a cavity, at least historically.
        gset_template = '{}GSET'

        # Check these PVs just before the fault, since they may have changed in response to the fault
        datetime = utils.path_to_datetime(self.event_dir) + timedelta(seconds=offset)

        # We need the zone to check the bypass bitword that has bit 0-7 corresponding to cavity 1-8
        for filename in os.listdir(self.event_dir):
            if not self.is_capture_file(filename):
                continue
            zone = filename[0:3]
            break

        # Get the bypassed bitword.  Check each cavity's status in the loop below.
        bypassed = None
        try:
            bypassed = mya.get_pv_value(PV=bypassed_template.format(zone), datetime=datetime, deployment='ops')
        except ValueError:
            # Do nothing here as this bypassed flag was not always archived.  Faults prior to Fall 2019 may predate
            # archival of the R...MOUT PVs
            pass

        # Switch to binary string.  "08b" means include leading zeros ("0"), have eight bits ("8"), and format string as
        # binary number ("b").  The [::-1] is an extended slice that says to step along the characters in reverse.
        # The reversal puts the bits in cavity order - bit_0 -> cav_1, bit_1 -> cav_2, ...
        bypassed_bits = format(0, "08b")
        if bypassed is not None:
            bypassed_bits = format(bypassed, "08b")[::-1]

        for filename in os.listdir(self.event_dir):
            if self.is_capture_file(filename):
                cav = filename[0:4]

                # Check if the cavity was gset == 0.  Ops meant to bypass this if so, and we don't care about it's
                # control mode
                gset = mya.get_pv_value(PV=gset_template.format(cav), datetime=datetime, deployment='ops')
                if gset == 0:
                    continue

                # Check if the cavity was formally bypassed.  bypassed_bits is zero indexed, while cavities are one
                # indexed.  1 is bypassed, 0 is not
                if bypassed_bits[int(cav[3]) - 1] == 0:
                    continue

                val = mya.get_pv_value(PV=mode_template.format(cav), datetime=datetime, deployment='ops')
                if val != mode:
                    raise ValueError("Cavity '" + cav + "' not in GDR mode.  Mode = " + str(val))
