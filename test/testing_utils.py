import datetime

import requests
import urllib.parse
import pandas as pd
import numpy as np
import os
import shutil

from io import StringIO
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.ssl_ import create_urllib3_context


# requests on Windows won't use default trust store unless you manually provide the SSLContextAdapter
class SSLContextAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        kwargs['ssl_context'] = context
        context.load_default_certs()  # this loads the OS defaults on Windows
        return super(SSLContextAdapter, self).init_poolmanager(*args, **kwargs)


class EventData:

    def __init__(self, zone, timestamp):
        """"Construct an object representing and event and it's data on disk
            Args:
                zone (str): The CED zone name
                timestamp (str): The timestamp of the event.  Ex. 2019/02/28 21:47:24
        """
        self.zone = zone
        # Supplied timestamps are in weird format, and don't have a decimal.  Add '.0' to keep consistent with real data
        self.timestamp = timestamp.replace("/", "-").replace(":", "") + ".0"
        self.test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "test-data", "tmp"))
        self.event_dir_base = os.path.join(self.test_dir, zone)
        self.event_dir = os.path.join(self.event_dir_base, self.timestamp.split(" ")[0].replace("-", "_"),
                                      self.timestamp.split(" ")[1].replace(":", ""))

    def get_event_data(self):
        """Downloads the data from accweb for the specified zone and timestamp."""

        # Setup to download the data
        base = 'https://accweb.acc.jlab.org/wfbrowser/ajax/event?'
        z = urllib.parse.quote_plus(self.zone)
        in_fmt = '%Y-%m-%d %H%M%S.%f'
        out_fmt = '%Y-%m-%d %H:%M:%S'
        begin = datetime.datetime.strptime(self.timestamp, in_fmt)
        end = begin + datetime.timedelta(seconds=1)
        b = urllib.parse.quote_plus(begin.strftime(out_fmt))
        e = urllib.parse.quote_plus(end.strftime(out_fmt))
        url = base + 'out=csv&includeData=true&location=' + z + '&begin=' + b + '&end=' + e

        # Download the data - supply the session/SSLContextAdapter to use Windows trust store
        s = requests.Session()
        adapter = SSLContextAdapter()
        s.mount(url, adapter)
        r = s.get(url)

        # Read the data in from the response stream
        data = StringIO(r.text.replace('time_offset', 'Time'))
        df = pd.read_csv(data)

        # Create the event directory tree
        try:
            os.makedirs(self.event_dir)
        except:
            print("Failed to make directory - " + self.event_dir)
            return

        # Write out the data into per-cavity capture files that the model expects to find
        date = self.timestamp.split(" ")[0].replace("-", "_")
        ctime = self.timestamp.split(" ")[1].replace(":", "")
        base = df.columns.values[3][:3]
        for i in range(1, 9):
            cav = base + str(i)
            cav_columns = ['Time'] + [col for col in df.columns.values if cav in col]
            out_file = os.path.join(self.event_dir, "{}WFSharv.{}_{}.txt".format(cav, date, ctime))
            df[cav_columns].to_csv(out_file, index=False, sep='\t')

    def delete_event_data(self):
        shutil.rmtree(self.event_dir_base)

    def get_event_path(self):
        return os.path.join(self.event_dir)


class TestSet:
    """This class is used to represent the set of events in a single test set represented by a TSV file.

    These files should be of the following formatted represented by this example:

    zone	cavity	cav#	fault	time	cav_pred	cav_conf	fault_pred	fault_conf
    1L22	5	13	Quench_3ms	2019/01/19 09:14:37	5	78.0	Quench_3ms	60.57
    1L22	0	9	Heat Riser Choke	2019/01/22 08:06:01	0	85.67	Multi Cav turn off	85.67
    1L22	4	12	Multi Cav turn off	2019/03/15 10:46:22	4	87.0	Multi Cav turn off	91.23
    ...
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.test_set_df = pd.read_csv(file_path, sep='\t', header=0, index_col=False)

        # Confidences in file are given as percentages
        self.test_set_df.loc[:, "cav_conf"] = np.round(self.test_set_df.loc[:, "cav_conf"]/100, 4)
        self.test_set_df.loc[:, "fault_conf"] = np.round(self.test_set_df.loc[:, "fault_conf"]/100, 4)

    def get_events(self):
        out = []
        for i in range(0, len(self.test_set_df)):
            out.append({
                'zone': self.test_set_df.loc[i, 'zone'],
                'timestamp': self.test_set_df.loc[i, 'time'],
                'expected': {
                    'location': self.test_set_df.loc[i, 'zone'],
                    'timestamp': self.test_set_df.loc[i, 'time'].replace("/", "-") + '.0',
                    'cavity-label': str(self.test_set_df.loc[i, 'cav_pred']),
                    'cavity-confidence': self.test_set_df.loc[i, 'cav_conf'],
                    'fault-label': self.test_set_df.loc[i, 'fault_pred'],
                    'fault-confidence': self.test_set_df.loc[i, 'fault_conf']
                }
            })
        return out
