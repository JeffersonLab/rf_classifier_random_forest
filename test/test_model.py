import unittest
import warnings

from unittest import TestCase
import os
import sys


# Put the lib dir at the front of the search path.  Makes the sys.path correct regardless of the context this test is
# run.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "lib")))
from model import Model


exp = {
    '0L04': [
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '0L04', '2018_04_26', '070734.9'),
            'result': {
                'cavity-label': '2',
                'fault-label': 'Microphonics',
                'cavity-confidence': 0.90,
                'fault-confidence': 0.7825079365079365,
                'location': '0L04',
                'timestamp': '2018-04-26 07:07:34.9'
            }
        }
    ],
    '1L22': [
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L22', '2018_05_02', '122116.5'),
            'result': {
                'cavity-label': '5',
                'fault-label': 'Quench',
                'cavity-confidence': 0.8833333333333333,
                'fault-confidence': 0.4421483331483332,
                'location': '1L22',
                'timestamp': '2018-05-02 12:21:16.5'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L22', '2018_05_02', '132811.9'),
            'result': {
                'cavity-label': '5',
                'fault-label': 'Quench',
                'cavity-confidence': 0.7796296296296297,
                'fault-confidence': 0.6147303437303439,
                'location': '1L22',
                'timestamp': '2018-05-02 13:28:11.9'
            }
        }
    ],
    '1L23': [
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L23', '2018_05_02', '152747.4'),
            'result': {
                'cavity-label': '2',
                'fault-label': 'Single Cav Turn off',
                'cavity-confidence': 0.9833333333333333,
                'fault-confidence': 0.8466666666666667,
                'location': '1L23',
                'timestamp': '2018-05-02 15:27:47.4'
            }
        }
    ],
    '1L24': [
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L24', '2018_05_03', '074836.5'),
            'result': {
                'cavity-label': '6',
                'fault-label': 'Quench',
                'cavity-confidence': 1.00,
                'fault-confidence': 0.88,
                'location': '1L24',
                'timestamp': '2018-05-03 07:48:36.5'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L24', '2018_05_04', '044822.5'),
            'result': {
                'cavity-label': '6',
                'fault-label': 'Quench',
                'cavity-confidence': 1.00,
                'fault-confidence': 0.88,
                'location': '1L24',
                'timestamp': '2018-05-04 04:48:22.5'
            }
        }
    ],
    '1L25': [
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L25', '2018_04_26', '161322.6'),
            'result': {
                'cavity-label': '4',
                'fault-label': 'Microphonics',
                'cavity-confidence': 0.9142515980751273,
                'fault-confidence': 0.9053333333333334,
                'location': '1L25',
                'timestamp': '2018-04-26 16:13:22.6'
            }
        }
    ],
    '1L26': [
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L26', '2018_04_27', '143918.4'),
            'result': {
                'cavity-label': '1',
                'fault-label': 'E_Quench',
                'cavity-confidence': 0.9333333333333333,
                'fault-confidence': 1.00,
                'location': '1L26',
                'timestamp': '2018-04-27 14:39:18.4'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L26', '2018_04_29', '193409.3'),
            'result': {
                'cavity-label': '8',
                'fault-label': 'Single Cav Turn off',
                'cavity-confidence': 0.9833333333333333,
                'fault-confidence': 0.6602247382247383,
                'location': '1L26',
                'timestamp': '2018-04-29 19:34:09.3'
            }
        },
        # Chris said this should produced a div by zero error during feature extraction - but he was using
        # a development version of tsfresh to get around a different bug.  Seems OK here, but I don't have
        # his numbers to compare against.  Just use the values output by this model and check for future
        # deviations.
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L26', '2018_05_03', '144259.6'),
            'result': {
                'cavity-label': '8',
                'fault-label': 'E_Quench',
                'cavity-confidence': 0.9333333333333333,
                'fault-confidence': 1.0,
                'location': '1L26',
                'timestamp': '2018-05-03 14:42:59.6'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L26', '2018_05_04', '085459.2'),
            'result': {
                'cavity-label': '6',
                'fault-label': 'Microphonics',
                'cavity-confidence': 0.8375,
                'fault-confidence': 0.6704021164021164,
                'location': '1L26',
                'timestamp': '2018-05-04 08:54:59.2'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L26', '2018_05_05', '181545.5'),
            'result': {
                'cavity-label': '1',
                'fault-label': 'E_Quench',
                'cavity-confidence': 0.8333333333333334,
                'fault-confidence': 0.9733333333333334,
                'location': '1L26',
                'timestamp': '2018-05-05 18:15:45.5'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L26', '2018_05_06', '002724.6'),
            'result': {
                'cavity-label': '1',
                'fault-label': 'Quench',  # Model predicts the wrong fault.  This is that "wrong" prediction.
                'cavity-confidence': 0.9875,
                'fault-confidence': 0.7817777777777777,
                'location': '1L26',
                'timestamp': '2018-05-06 00:27:24.6'
            }
        },
    ],
    '2L22': [
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '2L22', '2018_05_03', '081145.9'),
            'result': {
                'cavity-label': '7',  # Model get's it wrong.  This is the expected prediction.
                'fault-label': 'Single Cav Turn off',  # Model get's it wrong.  This is the expected "wrong" prediction.
                'cavity-confidence': 0.48333333333333334,
                'fault-confidence': 0.46882450882450893,
                'location': '2L22',
                'timestamp': '2018-05-03 08:11:45.9'
            }
        }
    ],
    '2L23': [
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '2L23', '2018_05_05', '090120.3'),
            'result': {
                'cavity-label': '1',
                'fault-label': 'E_Quench',
                'cavity-confidence': 0.9128205128205128,
                'fault-confidence': 0.9361904761904764,
                'location': '2L23',
                'timestamp': '2018-05-05 09:01:20.3'
            }
        }
    ],
    '2L24': [
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '2L24', '2018_04_25', '130853.9'),
            'result': {
                'cavity-label': '3',
                'fault-label': 'Single Cav Turn off',
                'cavity-confidence': 0.8531746031746033,
                'fault-confidence': 0.86,
                'location': '2L24',
                'timestamp': '2018-04-25 13:08:53.9'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '2L24', '2018_04_26', '142849.9'),
            'result': {
                'cavity-label': '5',
                'fault-label': 'Microphonics',
                'cavity-confidence': 0.6553885158051824,
                'fault-confidence': 0.5162380952380953,
                'location': '2L24',
                'timestamp': '2018-04-26 14:28:49.9'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '2L24', '2018_04_27', '021043.7'),
            'result': {
                'cavity-label': '4',
                'fault-label': 'Single Cav Turn off',
                'cavity-confidence': 0.48253968253968255,
                'fault-confidence': 0.86,
                'location': '2L24',
                'timestamp': '2018-04-27 02:10:43.7'
            }
        }
    ]
}


class TestRandomForest(TestCase):

    def test_analyze(self):
        # Context manager resets warnings filters to default after code section exits
        # I've spent tons of time trying to figure out why the analyze method and specifically it call of
        # sklearn.externals.joblib.load(...) produces RuntimeWarnings for numpy.ufunc binary incompatibility _ONLY_ when
        # running from unittest.  I give up.  Just suppress these warnings .
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "can't resolve package from __spec__ or __package__, "
                                              "falling back on __name__ and __path__")
            warnings.filterwarnings("ignore", "numpy.ufunc size changed, may indicate binary incompatibility. "
                                              "Expected 192 from C header, got 216 from PyObject")
            warnings.filterwarnings("ignore", "numpy.ufunc size changed, may indicate binary incompatibility. "
                                              "Expected 216, got 192")
            for zone in exp:
                for example in exp[zone]:
                    expect = example['result']
                    path = example['path']
                    print(path)
                    mod = Model(path)
                    result = mod.analyze(deployment='history')
                    self.assertDictEqual(expect, result)


if __name__ == '__main__':
    unittest.main()
