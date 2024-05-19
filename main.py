from nilmtk.api import API
import warnings
warnings.filterwarnings("ignore")
from nilmtk.disaggregate import Seq2Point
from nilmtk.disaggregate import TimesNet
from nilmtk.disaggregate import FFSTT

e = {
    # Specify power type, sample rate and disaggregated appliance
    'power': {
        'mains': ['apparent'],
        'appliance': ['active']
    },
    'sample_rate': 5,
    'appliances': ['fridge'],  #washing machine fridge dish washer microwave
    # Universally no pre-training
    'pre_trained': False,
    'fine_tuning': True,
    'transfer': False,

    'hyper_parameters': {
        'sequence_length': 699,
        'n_epochs': 200,
        'batch_size': 128,
        'appliance_params': {},
        'mains_mean': None,
        'mains_std': None,
    },
    # Specify algorithm
    'method': FFSTT,

    # Specify train and test data
    'train': {
        'datasets': {
            'ukdale': {
                'path': './mnt/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': '2013-07-01',
                        'end_time': '2013-07-07'
                        # 'start_time': '2011-04-18',
                        # 'end_time': '2011-04-28'
                        # 'start_time': 2013-07-01,
                        # 'end_time': 2013-08-01
                    }
                }
            },
        }
    },
    'test': {
        'datasets': {
            'ukdale': {
                'path': './mnt/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': '2013-07-08',
                        'end_time': '2013-07-14'
                        # 'start_time': '2011-05-18',
                        # 'end_time': '2011-05-18'
                    }
                }
            },

        },
        # Specify evaluation metrics
        'metrics': ['mae', 'f1score', 'recall', 'precision', 'nep', 'omae', 'MCC']
    }
}

API(e)