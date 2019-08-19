import re
import os
from datetime import datetime


def path_to_datetime(path):
    """Returns the datetime object associated with an event path.

        Args:
            path (str): The path on the filesystem matching a fault event directory.  Ending in .../<date>/<time> where
                        <date> is of the format YYYY_MM_DD and <time> is formatted hhmmss.S

        Returns:
            datetime: A datetime object corresponding to the event timestamp embedded in the supplied path

        Raises:
            ValueError: if the path is not of the expected format
    """

    time_pattern = re.compile(r'\d\d\d\d\d\d\.\d')
    date_pattern = re.compile(r'\d\d\d\d_\d\d_\d\d')
    path = os.path.abspath(path).split(os.path.sep)

    time = str(path[-1])
    date = str(path[-2])
    if not time_pattern.match(time):
        raise ValueError("Path includes invalid time format - " + time)

    if not date_pattern.match(date):
        raise ValueError("Path includes invalid date format - " + date)

    return datetime(year=int(date[0:4]), month=int(date[5:7]), day=int(date[8:10]), hour=int(time[0:2]),
                    minute=int(time[2:4]), second=int(time[4:6]), microsecond=int(time[7:8]) * 100000)


def path_to_zone_and_timestamp(path, fmt="%Y-%m-%d %H:%M:%S.%f"):
    """Returns a tuple containing the event zone and timestamp.

        Args:
            path (str): The path on the filesystem matching a fault event directory.  Ending in .../<date>/<time> where
                        <date> is of the format YYYY_MM_DD and <time> is formatted hhmmss.S
            fmt (str): The format string used by datetime.strftime.
        Returns:
            tuple: A tuple object containing strings for the event zone and timestamp

        Raises:
            ValueError: if the path is not of the expected format
    """

    time_pattern = re.compile(r'\d\d\d\d\d\d\.\d')
    date_pattern = re.compile(r'\d\d\d\d_\d\d_\d\d')
    path = os.path.abspath(path).split(os.path.sep)

    time = str(path[-1])
    date = str(path[-2])
    zone = str(path[-3])
    if not time_pattern.match(time):
        raise ValueError("Path includes invalid time format - " + time)

    if not date_pattern.match(date):
        raise ValueError("Path includes invalid date format - " + date)

    dt = datetime(year=int(date[0:4]), month=int(date[5:7]), day=int(date[8:10]), hour=int(time[0:2]),
                  minute=int(time[2:4]), second=int(time[4:6]), microsecond=int(time[7:8]) * 100000)

    return zone, dt.strftime(fmt)[:-5]
