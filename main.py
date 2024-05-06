from nilmtk.api import API
import warnings
warnings.filterwarnings("ignore")
from nilmtk.disaggregate import BiLSTM
from nilmtk.disaggregate import EnerGAN
from nilmtk.disaggregate import TimesNet
from nilmtk.disaggregate import Seq2Point
from nilmtk.disaggregate import FFSTT

e = {
    # Specify power type, sample rate and disaggregated appliance
    'power': {
        'mains': ['active'],
        'appliance': ['active']
    },
    'sample_rate': 5,
    'appliances': ['washing machine'],  #kettle fridge washing machine microwave dish washer
    # Universally no pre-training
    'pre_trained': False,
    'fine_tuning': False,
    'transfer': False,

    'hyper_parameters': {
        'sequence_length': 799,
        'n_epochs': 30,
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
                    2: {
                        'start_time': '2013-07-01',
                        'end_time': '2013-07-07'
                        # 'start_time': '2011-04-18',
                        # 'end_time': '2011-05-18'
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
                    2: {
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