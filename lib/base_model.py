from abc import ABC, abstractmethod
import os


class BaseModel(ABC):
    """An abstract class defining the model API.

    Inheriting from this class requires that both the describe and analyze methods be defined which form the basis of
    the model API.  Helpful functionality of this class has been rolled into the rfwtools package.

    Attributes:
        event_dir (str): The path to the event directory being analyzed by this model object.
        zone_name (str): The zone where the fault occurred.

    """

    def __init__(self, event_dir):
        """Instantiates a BaseModel object that contains a reference to the filesystem location of the event data.

            Args:
                event_dir (str): Path to the fault event directory.
        """
        self.event_dir = event_dir
        self.zone_name = os.path.split(os.path.split(os.path.split(self.event_dir)[0])[0])[-1]

    @abstractmethod
    def analyze(self):
        """A method that performs some analysis and classifies the fault event by cavity number and fault type.

            This method is the most important part of any model package.  Here is where the desired analytical
            approaches should be implemented.  In addition to the classification label output, this method should
            include information about the confidence of those classifications and list which fault event is being
            analyzed.  Confidence numbers should be given on the range [0,1] with lower numbers implying more
            uncertainty and higher numbers implying greater certainty.


            Returns:
                dict: A dictionary containing the results of the analysis.  Detailed key/value information given in the
                table below.

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
