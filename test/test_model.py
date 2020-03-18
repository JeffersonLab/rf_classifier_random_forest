import datetime
import unittest
import warnings

from unittest import TestCase
import os
import sys

# Put the lib dir at the front of the search path.  Makes the sys.path correct regardless of the context this test is
# run.
import testing_utils

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "lib")))
from model import Model


class TestRandomForest(TestCase):

    def test_analyze(self):
        # Context manager resets warnings filters to default after code section exits
        # I've spent tons of time trying to figure out why the analyze method and specifically it call of
        # sklearn.externals.joblib.load(...) produces RuntimeWarnings for numpy.ufunc binary incompatibility _ONLY_ when
        # running from unittest.  I give up.  Just suppress these warnings .
        with warnings.catch_warnings():
            # warnings.filterwarnings("ignore", "can't resolve package from __spec__ or __package__, "
            #                                   "falling back on __name__ and __path__")
            # warnings.filterwarnings("ignore", "numpy.ufunc size changed, may indicate binary incompatibility. "
            #                                   "Expected 192 from C header, got 216 from PyObject")
            # warnings.filterwarnings("ignore", "numpy.ufunc size changed, may indicate binary incompatibility. "
            #                                   "Expected 216, got 192")
            failed = 0
            test_set = testing_utils.TestSet('test_set.txt')
            for test_event in test_set.get_events():
                print("##### Testing: Event {} - {} #####".format(test_event['zone'], test_event['timestamp']))
                event = testing_utils.EventData(zone=test_event['zone'], timestamp=test_event['timestamp'])

                try:
                    event.get_event_data()
                except:
                    failed += 1
                    print("Failed to get data for test.")
                    continue

                expect = test_event['expected']
                if expect['cavity-label'] == '0':
                    expect['cavity-label'] = 'multiple'
                path = event.event_dir

                mod = Model(path)
                try:
                    # The history archiver has everything, except recent data.  ops archiver has recent data, but not
                    # anything more than maybe two years old.
                    deployment = 'history'
                    if datetime.datetime.now() - event.timestamp < datetime.timedelta(days=31):
                        deployment = 'ops'
                    result = mod.analyze(deployment=deployment)
                except Exception as e:
                    failed += 1
                    print("Error analyzing data")
                    print(e)
                    continue

                # The test set file only has four decimal places.
                result['cavity-confidence'] = round(result['cavity-confidence'], 4)
                result['fault-confidence'] = round(result['fault-confidence'], 4)

                try:
                    self.assertDictEqual(expect, result)
                except Exception as e:
                    failed += 1
                    print(e)

                try:
                    event.delete_event_data()
                except Exception as e:
                    print("Error deleting data")
                    print(e)

            if failed > 0:
                self.fail("Failed {} example tests".format(failed))


if __name__ == '__main__':
    unittest.main()
