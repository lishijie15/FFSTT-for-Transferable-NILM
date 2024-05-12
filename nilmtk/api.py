from nilmtk.dataset import DataSet
import pandas as pd
from nilmtk.losses import *
import numpy as np
import matplotlib.pyplot as plt
import datetime
import torch
import os

all_type_power = ['active', 'apparent']


class API():
    # The API is designed for rapid experimentation with NILM Algorithms. 
    def __init__(self, params):
        # Initializes the API with default parameters
        self.power = {}
        self.sample_period = 1
        self.appliances = []
        self.method = None
        self.chunk_size = None
        self.pre_trained = False
        self.fine_tuning = False
        self.metrics = []
        self.transfer = False
        self.train_datasets_dict = {}
        self.test_datasets_dict = {}
        self.artificial_aggregate = False
        self.train_submeters = []
        self.train_mains = pd.DataFrame()
        self.test_submeters = []
        self.test_mains = pd.DataFrame()
        self.gt_overall = {}
        self.pred_overall = {}
        self.classifiers = []
        self.DROP_ALL_NANS = True
        self.mae = pd.DataFrame()
        self.rmse = pd.DataFrame()
        self.errors = []
        self.predictions = []
        self.errors_keys = []
        self.predictions_keys = []
        self.params = params
        for elems in params['power']:
            self.power = params['power']
        self.sample_period = params['sample_rate']
        for elems in params['appliances']:
            self.appliances.append(elems)

        self.pre_trained = params['pre_trained']
        self.fine_tuning = params['fine_tuning']
        self.transfer = params['transfer']
        self.train_datasets_dict = params['train']['datasets']
        self.test_datasets_dict = params['test']['datasets']
        self.metrics = params['test']['metrics']

        self.hyper_parameters = params['hyper_parameters']
        self.method = params['method']

        self.artificial_aggregate = params.get('artificial_aggregate', self.artificial_aggregate)
        self.chunk_size = params.get('chunk_size', self.chunk_size)
        self.experiment(params)

    def experiment(self, params):
        # Calls the Experiments with the specified parameters

        self.store_classifier_instances()

        if self.transfer==False:
            d = self.train_datasets_dict
            for model_name, clf in self.classifiers:
                # If the model is a neural net, it has an attribute n_epochs, Ex: DAE, Seq2Point
                print("Started training for ", clf.MODEL_NAME)
                # If the model has the filename specified for loading the pretrained model, then we don't need to load training data
                if hasattr(clf, 'load_model_path'):
                    if clf.load_model_path:
                        print(clf.MODEL_NAME, " is loading the pretrained model")
                        continue

                print("Joint training for ", clf.MODEL_NAME)
                self.train_jointly(clf, d)
                print("Finished training for ", clf.MODEL_NAME)

            d = self.test_datasets_dict

            print("Joint Testing for all algorithms")
            self.test_jointly(d)
        else:
            for model_name, clf in self.classifiers:
                print("Transfer for ", clf.MODEL_NAME)
                d = self.test_datasets_dict

                print("Joint Testing for all algorithms")
                self.transfer_jointly(clf, d)

    def train_jointly(self, clf, d):
        # This function has a few issues, which should be addressed soon
        print("............... Loading Data for training ...................")
        # Store the train_main readings for all buildings
        self.train_mains = pd.DataFrame()
        self.train_submeters = [pd.DataFrame() for i in range(len(self.appliances))]
        for dataset in d:
            print("Loading data for ", dataset, " dataset")
            train = DataSet(d[dataset]['path'])
            for building in d[dataset]['buildings']:
                print("Loading building ... ", building)

                train.set_window(start=d[dataset]['buildings'][building]['start_time'],
                                 end=d[dataset]['buildings'][building]['end_time'])
                # train.set_window(start = None,end = None)

                train_df = next(
                    train.buildings[building].elec.mains().load(physical_quantity='power', ac_type=all_type_power,
                                                                sample_period=self.sample_period))

                # print columns in train_df['power']
                # print(train_df['power'].columns)

                train_df['active'] = train_df['power']['active']
                train_df['apparent'] = train_df['power']['apparent']
                train_df['reactive'] = np.sqrt(train_df['power']['apparent'] ** 2 - train_df['power']['active'] ** 2)
                train_df.drop(['power'], axis=1, inplace=True)
                train_df = train_df[self.power['mains']]
                # print(train_df.columns)
                # print(train_df.shape)
                # print(train_df)
                # train_df = train_df[[list(train_df.columns)[0]]]
                # print(train_df.shape)

                appliance_readings = []
                for appliance_name in self.appliances:
                    appliance_df = next(train.buildings[building].elec[appliance_name].load(physical_quantity='power',
                                                                                            ac_type=self.power[
                                                                                                'appliance'],
                                                                                            sample_period=self.sample_period))
                    # appliance_df = appliance_df[[list(appliance_df.columns)[0]]]
                    appliance_readings.append(appliance_df)
                if self.DROP_ALL_NANS:
                    train_df, appliance_readings = self.dropna(train_df, appliance_readings)
                if self.artificial_aggregate:
                    print("Creating an Artificial Aggregate")
                    train_df = pd.DataFrame(np.zeros(appliance_readings[0].shape), index=appliance_readings[0].index,
                                            columns=appliance_readings[0].columns)
                    for app_reading in appliance_readings:
                        train_df += app_reading

                print("Train Jointly")
                self.train_mains = self.train_mains.append(train_df)
                for i, appliance_name in enumerate(self.appliances):
                    self.train_submeters[i] = self.train_submeters[i].append(appliance_readings[i])

        appliance_readings = []
        for i, appliance_name in enumerate(self.appliances):
            appliance_readings.append((appliance_name, [self.train_submeters[i]]))

        self.train_mains = [self.train_mains]
        self.train_submeters = appliance_readings
        clf.partial_fit(self.train_mains, self.train_submeters, self.pre_trained, self.fine_tuning, True)

    def test_jointly(self, d):
        # Store the test_main readings for all buildings
        for dataset in d:
            print("Loading data for ", dataset, " dataset")
            test = DataSet(d[dataset]['path'])
            for building in d[dataset]['buildings']:

                test.set_window(start=d[dataset]['buildings'][building]['start_time'],
                                end=d[dataset]['buildings'][building]['end_time'])
                # test.set_window(start=None, end=None)

                test_mains = next(
                    test.buildings[building].elec.mains().load(physical_quantity='power', ac_type=all_type_power,
                                                               sample_period=self.sample_period))
                test_mains['active'] = test_mains['power']['active']
                test_mains['apparent'] = test_mains['power']['apparent']
                test_mains['reactive'] = np.sqrt(test_mains['power']['apparent'] ** 2 - test_mains['power']['active'] ** 2)
                test_mains.drop(['power'], axis=1, inplace=True)
                test_mains = test_mains[self.power['mains']]
                '''
                train_df = next(train.buildings[building].elec.mains().load(physical_quantity='power', ac_type= all_type_power,sample_period = self.sample_period))
                train_df['active'] = train_df['power']['active']
                train_df['apparent'] = train_df['power']['apparent']
                train_df['reactive'] = np.sqrt(train_df['power']['apparent']**2 - train_df['power']['active']**2)
                train_df.drop(['power'], axis = 1,inplace = True)
                train_df = train_df[self.power['mains']]
                '''

                appliance_readings = []
                for appliance in self.appliances:
                    test_df = next(test.buildings[building].elec[appliance].load(physical_quantity='power',
                                                                                  ac_type=self.power['appliance'],
                                                                                  sample_period=self.sample_period))
                    appliance_readings.append(test_df)

                if self.DROP_ALL_NANS:
                    test_mains, appliance_readings = self.dropna(test_mains, appliance_readings)

                if self.artificial_aggregate:
                    print("Creating an Artificial Aggregate")
                    test_mains = pd.DataFrame(np.zeros(appliance_readings[0].shape), index=appliance_readings[0].index,
                                              columns=appliance_readings[0].columns)
                    for app_reading in appliance_readings:
                        test_mains += app_reading

                self.test_mains = [test_mains]
                for i, appliance_name in enumerate(self.appliances):
                    self.test_submeters.append((appliance_name, [appliance_readings[i]]))

                self.storing_key = str(dataset) + "_" + str(building)
                self.call_predict(self.classifiers)

    def transfer_jointly(self, clf, d):
        # Store the test_main readings for all buildings
        for dataset in d:
            print("Loading data for ", dataset, " dataset")
            test = DataSet(d[dataset]['path'])
            for building in d[dataset]['buildings']:

                test.set_window(start=d[dataset]['buildings'][building]['start_time'],
                                end=d[dataset]['buildings'][building]['end_time'])
                # test.set_window(start=None, end=None)

                test_mains = next(
                    test.buildings[building].elec.mains().load(physical_quantity='power', ac_type=all_type_power,
                                                               sample_period=self.sample_period))
                test_mains['active'] = test_mains['power']['active']
                test_mains['apparent'] = test_mains['power']['apparent']
                test_mains['reactive'] = np.sqrt(test_mains['power']['apparent'] ** 2 - test_mains['power']['active'] ** 2)
                test_mains.drop(['power'], axis=1, inplace=True)
                test_mains = test_mains[self.power['mains']]
                '''
                train_df = next(train.buildings[building].elec.mains().load(physical_quantity='power', ac_type= all_type_power,sample_period = self.sample_period))
                train_df['active'] = train_df['power']['active']
                train_df['apparent'] = train_df['power']['apparent']
                train_df['reactive'] = np.sqrt(train_df['power']['apparent']**2 - train_df['power']['active']**2)
                train_df.drop(['power'], axis = 1,inplace = True)
                train_df = train_df[self.power['mains']]
                '''

                appliance_readings = []
                for appliance in self.appliances:
                    test_df = next(test.buildings[building].elec[appliance].load(physical_quantity='power',
                                                                                  ac_type=self.power['appliance'],
                                                                                  sample_period=self.sample_period))
                    appliance_readings.append(test_df)

                if self.DROP_ALL_NANS:
                    test_mains, appliance_readings = self.dropna(test_mains, appliance_readings)

                if self.artificial_aggregate:
                    print("Creating an Artificial Aggregate")
                    test_mains = pd.DataFrame(np.zeros(appliance_readings[0].shape), index=appliance_readings[0].index,
                                              columns=appliance_readings[0].columns)
                    for app_reading in appliance_readings:
                        test_mains += app_reading

                self.test_mains = [test_mains]
                for i, appliance_name in enumerate(self.appliances):
                    self.test_submeters.append((appliance_name, [appliance_readings[i]]))

                self.storing_key = str(dataset) + "_" + str(building)
                self.transfer_predict(self.classifiers)


            # # Create optimizer, loss function, and dataloader
            # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
            # loss_fn = torch.nn.MSELoss(reduction='mean')
            #
            # train_dataset = TensorDataset(torch.from_numpy(train_mains).float().permute(0, 2, 1),
            #                               torch.from_numpy(train_appliance).float())
            # train_loader = tud.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
            #                               drop_last=True)
            #
            # # valid_dataset = TensorDataset(torch.from_numpy(valid_mains).float().permute(0, 2, 1), torch.from_numpy(valid_appliance).float())
            # # valid_loader = tud.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
            #
            # # Freeze the layers if required
            # # Assume that we are only training the last layers named 'projection'
            # for param in model.parameters():
            #     param.requires_grad = False
            # for param in model.projection.parameters():
            #     param.requires_grad = True

    def dropna(self, mains_df, appliance_dfs):
        # Drop missing value
        print("Dropping missing values")

        # The below steps are for making sure that data is consistent by doing intersection across appliances
        mains_df = mains_df.dropna()
        for i in range(len(appliance_dfs)):
            appliance_dfs[i] = appliance_dfs[i].dropna()
        ix = mains_df.index
        for app_df in appliance_dfs:
            ix = ix.intersection(app_df.index)
        mains_df = mains_df.loc[ix]
        new_appliances_list = []
        for app_df in appliance_dfs:
            new_appliances_list.append(app_df.loc[ix])
        return mains_df, new_appliances_list

    def store_classifier_instances(self):
        # This function is reponsible for initializing the models with the specified model parameters
        if self.method is None:
            print("Please specify the method to use for disaggregation")
            raise ValueError("Method not specified")

        try:
            clf = self.method(self.hyper_parameters)
            model_name = clf.__class__.__name__
            print(model_name)
            self.classifiers.append((model_name, clf))

            print("Initialized", model_name)

        except Exception as e:
            print("\n\nThe method specied does not exist. \n\n")
            print(e)

    def call_predict(self, classifiers):
        # This functions computers the predictions on the self.test_mains using all the trained models and then compares different learn't models using the metrics specified
        pred_overall = {}
        gt_overall = {}
        for name, clf in classifiers:
            gt_overall, pred_overall[name] = self.predict(clf, self.test_mains, self.test_submeters, self.sample_period,
                                                          'Europe/London')
            path = "predict" + clf.MODEL_NAME + ".csv"
            pred_overall[name].to_csv(path)
        self.gt_overall = gt_overall
        self.pred_overall = pred_overall
        self.gt_overall.to_csv("truth.csv")
        if gt_overall.size == 0:
            print("No samples found in ground truth")
            return None

        for i in gt_overall.columns:

            plt.figure(figsize=(28, 10))
            plt.plot(self.test_mains[0], label='Mains reading')
            plt.plot(gt_overall[i], label='Truth')
            for clf in pred_overall:
                plt.plot(pred_overall[clf][i], label=clf)
            plt.title(i)
            plt.legend()

        for metric in self.metrics:
            try:
                loss_function = globals()[metric]
            except:
                print("Loss function ", metric, " is not supported currently!")
                continue

            computed_metric = {}
            for clf_name, clf in classifiers:
                computed_metric[clf_name] = self.compute_loss(gt_overall, pred_overall[clf_name], loss_function)
            computed_metric = pd.DataFrame(computed_metric)
            print("............ ", metric, " ..............")
            print(computed_metric)
            self.errors.append(computed_metric)
            self.errors_keys.append(self.storing_key + "_" + metric)

    def transfer_predict(self, classifiers):
        # This functions computers the predictions on the self.test_mains using all the trained models and then compares different learn't models using the metrics specified
        pred_overall = {}
        gt_overall = {}
        for name, clf in classifiers:

            gt_overall, pred_overall[name] = self.predict(clf, self.test_mains, self.test_submeters, self.sample_period,
                                                          'Europe/London')
            path = "predict" + clf.MODEL_NAME + ".csv"
            pred_overall[name].to_csv(path)
        self.gt_overall = gt_overall
        self.pred_overall = pred_overall
        self.gt_overall.to_csv("truth.csv")
        if gt_overall.size == 0:
            print("No samples found in ground truth")
            return None

        for i in gt_overall.columns:

            plt.figure(figsize=(28, 10))
            plt.plot(self.test_mains[0], label='Mains reading')
            plt.plot(gt_overall[i], label='Truth')
            for clf in pred_overall:
                plt.plot(pred_overall[clf][i], label=clf)
            plt.title(i)
            plt.legend()

        for metric in self.metrics:
            try:
                loss_function = globals()[metric]
            except:
                print("Loss function ", metric, " is not supported currently!")
                continue

            computed_metric = {}
            for clf_name, clf in classifiers:
                computed_metric[clf_name] = self.compute_loss(gt_overall, pred_overall[clf_name], loss_function)
            computed_metric = pd.DataFrame(computed_metric)
            print("............ ", metric, " ..............")
            print(computed_metric)
            self.errors.append(computed_metric)
            self.errors_keys.append(self.storing_key + "_" + metric)

    def predict(self, clf, test_elec, test_submeters, sample_period, timezone):
        # Generates predictions on the test dataset using the specified classifier.
        print("Generating predictions for :", clf.MODEL_NAME)
        # "ac_type" varies according to the dataset used.
        # Make sure to use the correct ac_type before using the default parameters in this code.

        pred_list = clf.disaggregate_chunk(test_elec)
        concat_pred_df = pd.concat(pred_list, axis=0)

        gt = {}
        for meter, data in test_submeters:
            concatenated_df_app = pd.concat(data, axis=1)
            index = concatenated_df_app.index
            gt[meter] = pd.Series(concatenated_df_app.values.flatten(), index=index)

        gt_overall = pd.DataFrame(gt, dtype='float32')
        pred = {}
        for app_name in concat_pred_df.columns:
            app_series_values = concat_pred_df[app_name].values.flatten()
            # Neural nets do extra padding sometimes, to fit, so get rid of extra predictions
            app_series_values = app_series_values[:len(gt_overall[app_name])]
            pred[app_name] = pd.Series(app_series_values, index=gt_overall.index)
        pred_overall = pd.DataFrame(pred, dtype='float32')
        return gt_overall, pred_overall

    # metrics
    def compute_loss(self, gt, clf_pred, loss_function):
        error = {}
        for app_name in gt.columns:
            error[app_name] = loss_function(app_name, gt[app_name], clf_pred[app_name])
        return pd.Series(error)        
