from unittest import TestCase
import os
import sys

# Put these here so IDE can find the packages from this project as they would when called from rf_classifier
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "rf_classifier")))
sys.path.insert(0, os.path.abspath(

    os.path.join(os.path.dirname(__file__), "..", "..", "rf_classifier", "venv", "Lib", "site-packages")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "venv", "Lib", "site-packages")))

# Add the local module to the sys.path to guarantee that external tester scripts can access the model module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from model import Model

# sys.path.pop(0)

exp = {
    '0L04': [
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '0L04', '2018_04_26', '070734.9'),
            'result': {
                'cavity-label': '2',
                'fault-label': 'Microphonics',
                'cavity-confidence': 0.92,
                'fault-confidence': 0.856,
                'location': '0L04',
                'timestamp': '2018-04-26 07:07:34.9'
            }
        }
    ],
    # '1L22': [
    #     {
    #         'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L22', '2018_05_02', '122116.5'),
    #         'result': {
    #             'cavity-label': '5',
    #             'fault-label': 'Microphonics',
    #             'cavity-confidence': 0.94,
    #             'fault-confidence': 0.65,
    #             'location': '1L22',
    #             'timestamp': '2018-05-02 12:21:16.5'
    #         }
    #     },
    #     {
    #         'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L22', '2018_05_02', '132811.9'),
    #         'result': {
    #             'cavity-label': '5',
    #             'fault-label': 'Quench',  # Should be something else
    #             'cavity-confidence': 0.74,
    #             'fault-confidence': 0.56,
    #             'location': '1L22',
    #             'timestamp': '2018-05-02 13:28:11.9'
    #         }
    #     }
    # ],
    # '1L23': [
    #     {
    #         'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L23', '2018_05_02', '152747.4'),
    #         'result': {
    #             'cavity-label': '2',
    #             'fault-label': 'Single Cav Turn off',
    #             'cavity-confidence': 0.96,
    #             'fault-confidence': 0.60,
    #             'location': '1L23',
    #             'timestamp': '2018-05-02 15:27:47.4'
    #         }
    #     }
    # ],
    # '1L24': [
    #     {
    #         'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L24', '2018_05_03', '074836.5'),
    #         'result': {
    #             'cavity-label': '6',
    #             'fault-label': 'Quench',
    #             'cavity-confidence': 0.98,
    #             'fault-confidence': 0.57,
    #             'location': '1L24',
    #             'timestamp': '2018-05-03 07:48:36.5'
    #         }
    #     },
    #     {
    #         'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L24', '2018_05_04', '044822.5'),
    #         'result': {
    #             'cavity-label': '6',
    #             'fault-label': 'Quench',
    #             'cavity-confidence': 0.97,
    #             'fault-confidence': 0.57,
    #             'location': '1L24',
    #             'timestamp': '2018-05-04 04:47:22.5'
    #         }
    #     }
    # ],
    '1L25': [
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L25', '2018_04_26', '161322.6'),
            'result': {
                'cavity-label': '4',
                'fault-label': 'Microphonics',
                'cavity-confidence': 0.896,
                'fault-confidence': 0.884,
                'location': '1L25',
                'timestamp': '2018-04-26 16:13:22.6'
            }
        },
        # {
        #     'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L25', '2018_04_29', '221542.6'),
        #     'result': {
        #         'cavity-label': '1',
        #         'fault-label': 'Quench',  # WRONG FAULT
        #         'cavity-confidence': 0.86,
        #         'fault-confidence': 0.31,
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
                'cavity-confidence': 0.908,
                'fault-confidence': 0.924,
                'location': '1L26',
                'timestamp': '2018-04-27 14:39:18.4'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L26', '2018_04_29', '193409.3'),
            'result': {
                'cavity-label': '8',
                'fault-label': 'Single Cav Turn off',
                'cavity-confidence': 0.988,
                'fault-confidence': 0.788,
                'location': '1L26',
                'timestamp': '2018-04-29 19:34:09.3'
            }
        },
        # {
        #     'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L26', '2018_05_02', '111441.1'),
        #     'result': {
        #         'cavity-label': '8',
        #         'fault-label': 'E_Quench',
        #         'cavity-confidence': 0.89,
        #         'fault-confidence': 0.80,
        #         'location': '1L26',
        #         'timestamp': '2018-05-02 11:14:41.1'
        #     }
        # },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L26', '2018_05_03', '144259.6'),
            'result': {
                'cavity-label': '8',
                'fault-label': 'E_Quench',
                'cavity-confidence': 0.944,
                # 'fault-confidence': 0.68,
                'fault-confidence': 0.756,
                'location': '1L26',
                'timestamp': '2018-05-03 14:42:59.6'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L26', '2018_05_04', '085459.2'),
            'result': {
                'cavity-label': '6',
                'fault-label': 'Microphonics',
                'cavity-confidence': 0.8097619047619047,
                'fault-confidence': 0.7974,
                'location': '1L26',
                'timestamp': '2018-05-04 08:54:59.2'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L26', '2018_05_05', '181545.5'),
            'result': {
                'cavity-label': '1',
                'fault-label': 'E_Quench',
                'cavity-confidence': 0.884,
                'fault-confidence': 0.824,
                'location': '1L26',
                'timestamp': '2018-05-05 18:15:45.5'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '1L26', '2018_05_06', '002724.6'),
            'result': {
                'cavity-label': '1',
                'fault-label': 'Microphonics',  # Model predicts the wrong fault.  This is that "wrong" prediction.
                'cavity-confidence': 0.968,
                'fault-confidence': 0.4187142857142857,
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
                'fault-label': 'Microphonics',  # Model get's it wrong.  This is the expected prediction.
                #'cavity-confidence': 0.48,
                'cavity-confidence': 0.4970666666666666,
                'fault-confidence': 0.46,
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
                'cavity-confidence': 0.884,
                'fault-confidence': 0.6833333333333332,
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
                'cavity-confidence': 0.8176,
                'fault-confidence': 0.708,
                'location': '2L24',
                'timestamp': '2018-04-25 13:08:53.9'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '2L24', '2018_04_26', '142849.9'),
            'result': {
                'cavity-label': '5',
                'fault-label': 'Microphonics',
                'cavity-confidence': 0.70,
                'fault-confidence': 0.737,
                'location': '2L24',
                'timestamp': '2018-04-26 14:28:49.9'
            }
        },
        {
            'path': os.path.join(os.path.dirname(__file__), 'test-data', '2L24', '2018_04_27', '021043.7'),
            'result': {
                'cavity-label': '4',
                'fault-label': 'Single Cav Turn off',
                'cavity-confidence': 0.424,
                'fault-confidence': 0.792,
                'location': '2L24',
                'timestamp': '2018-04-27 02:10:43.7'
            }
        }
    ]
}


class TestRandomForest(TestCase):

    def test_analyze(self):
        for zone in exp:
            for example in exp[zone]:
                expect = example['result']
                path = example['path']
                print(path)
                mod = Model(path)
                result = mod.analyze()
                self.assertDictEqual(expect, result)

        # mod = Model(e1)
        #
        # exp = {
        #     'cavity-label': '4',
        #     'fault-label': 'Microphonics',
        #     'cavity-confidence': None,
        #     'fault-confidence': None,
        #     'location': '0L04',
        #     'timestamp': '2019-11-12 13:36:15.8'
        # }
        #
        # result = mod.analyze()
        # self.assertDictEqual(exp, result)
