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
                'cavity-confidence': 0.69,
                'fault-confidence': 0.79,
                'location': '0L04',
                'timestamp': '2018-04-26 07:07:34.9'
            }
        }
    ],
    '1L22': [
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L22', '2018_05_02', '122116.5'),
            'result': {
                'cavity-label': '7',  # Model gets this wrong
                'fault-label': 'Quench', # Model gets this wrong
                'cavity-confidence': 0.29,
                'fault-confidence': 0.74,
                'location': '1L22',
                'timestamp': '2018-05-02 12:21:16.5'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L22', '2018_05_02', '132811.9'),
            'result': {
                'cavity-label': '7', # Model gets this wrong
                'fault-label': 'Quench',
                'cavity-confidence': 0.27,
                'fault-confidence': 0.71,
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
                'cavity-confidence': 0.88,
                'fault-confidence': 1.00,
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
                'fault-confidence': 1.00,
                'location': '1L24',
                'timestamp': '2018-05-03 07:48:36.5'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L24', '2018_05_04', '044822.5'),
            'result': {
                'cavity-label': '6',
                'fault-label': 'Quench',
                'cavity-confidence': 0.99,
                'fault-confidence': 0.99,
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
                'cavity-confidence': 0.81,
                'fault-confidence': 0.87,
                'location': '1L25',
                'timestamp': '2018-04-26 16:13:22.6'
            }
        },
        # Skip this one since it fails the 'all cavities either bypassed or in GDR mode' test
        # {
        #     'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L25', '2018_04_29', '221542.6'),
        #     'result': {
        #         'cavity-label': '1',
        #         'fault-label': 'Quench',
        #         'cavity-confidence': 0.75,
        #         'fault-confidence': 0.83,
        #         'location': '1L25',
        #         'timestamp': '2018-04-29 22:15:42.6'
        #     }
        # }
    ],
    '1L26': [
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L26', '2018_04_27', '143918.4'),
            'result': {
                'cavity-label': '1',
                'fault-label': 'E_Quench',
                'cavity-confidence': 0.89,
                'fault-confidence': 0.92,
                'location': '1L26',
                'timestamp': '2018-04-27 14:39:18.4'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L26', '2018_04_29', '193409.3'),
            'result': {
                'cavity-label': '8',
                'fault-label': 'Single Cav Turn off',
                'cavity-confidence': 0.69,
                'fault-confidence': 1.00,
                'location': '1L26',
                'timestamp': '2018-04-29 19:34:09.3'
            }
        },
        # Skip this one since it fails the 'all cavities either bypassed or in GDR mode' test
        # {
        #     'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L26', '2018_05_02', '111441.1'),
        #     'result': {
        #         'cavity-label': '8',
        #         'fault-label': 'E_Quench',
        #         'cavity-confidence': 0.91,
        #         'fault-confidence': 0.89,
        #         'location': '1L26',
        #         'timestamp': '2018-05-02 11:14:41.1'
        #     }
        # },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L26', '2018_05_03', '144259.6'),
            'result': {
                'cavity-label': '8',
                'fault-label': 'E_Quench',
                'cavity-confidence': 0.85,
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
                'cavity-confidence': 0.92,
                'fault-confidence': 0.50,
                'location': '1L26',
                'timestamp': '2018-05-04 08:54:59.2'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L26', '2018_05_05', '181545.5'),
            'result': {
                'cavity-label': '1',
                'fault-label': 'E_Quench',
                'cavity-confidence': 0.85,
                'fault-confidence': 0.71,
                'location': '1L26',
                'timestamp': '2018-05-05 18:15:45.5'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L26', '2018_05_06', '002724.6'),
            'result': {
                'cavity-label': '1',
                'fault-label': 'Quench',  # Model predicts the wrong fault.  This is that "wrong" prediction.
                'cavity-confidence': 0.85,
                'fault-confidence': 0.57,
                'location': '1L26',
                'timestamp': '2018-05-06 00:27:24.6'
            }
        },
    ],
    '2L22': [
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '2L22', '2018_05_03', '081145.9'),
            'result': {
                'cavity-label': '8',
                'fault-label': 'Quench',
                'cavity-confidence': 0.60,
                'fault-confidence': 0.64,
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
                'cavity-confidence': 0.87,
                'fault-confidence': 0.80,
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
                'cavity-confidence': 0.75,
                'fault-confidence': 1.00,
                'location': '2L24',
                'timestamp': '2018-04-25 13:08:53.9'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '2L24', '2018_04_26', '142849.9'),
            'result': {
                'cavity-label': 'multiple', # Model gets this wrong
                'fault-label': 'Multi Cav Turn off', # Hence, model also gets this wrong
                'cavity-confidence': 0.24,
                'fault-confidence': 0.24,
                'location': '2L24',
                'timestamp': '2018-04-26 14:28:49.9'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '2L24', '2018_04_27', '021043.7'),
            'result': {
                'cavity-label': '4',
                'fault-label': 'Single Cav Turn off',
                'cavity-confidence': 0.50,
                'fault-confidence': 0.98,
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
            failed = 0
            for zone in exp:
                for example in exp[zone]:
                    expect = example['result']
                    path = example['path']
                    print(path)
                    mod = Model(path)
                    result = mod.analyze(deployment='history')
                    result['cavity-confidence'] = round(result['cavity-confidence'], 2)
                    result['fault-confidence'] = round(result['fault-confidence'], 2)
                    try:
                        self.assertDictEqual(expect, result)
                    except Exception as e:
                        failed = failed + 1
                        print(e)
            if failed > 0:
                self.fail("Failed {} example tests".format(failed))


if __name__ == '__main__':
    unittest.main()
