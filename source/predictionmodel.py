# import modules
from sklearn.ensemble import GradientBoostingRegressor as GBR
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from joblib import dump, load
from matplotlib import pyplot as plt

class GBR_model():

    def __init__(self, n_estimators = 2000, period = 12, savepath = 'Results', modeltype = None):

        """
        parameters
        """
        self.n_estimators = n_estimators
        self.timegap = period * 5
        self.savepath=savepath
        self.modeltype= modeltype

        # model save path
        self.weightspath = self.savepath + '/{}_GBR_model_{}_estimators.joblib'.format(self.modeltype, self.n_estimators)

        # containers for predictions
        self.train_pred = None
        self.test_pred = None

        # Design the model
        self.model = GBR(loss='ls', learning_rate=0.1,
                         validation_fraction=0.1, n_iter_no_change=300,
                         n_estimators=self.n_estimators)

    def trainmodel(self, X_train, X_test, y_train, y_test, savemodel = True):

        file = open(self.savepath + '/' + str(self.timegap) + ' min {} Results.txt'.format(self.modeltype), 'w')
        file.close()

        self.model.fit(X_train, y_train)
        self.logError(X_train, y_train, X_test, y_test)
        self.generatePlots(y_train, y_test)

        if savemodel:
            dump(self.model, self.weightspath)

    def loadweights(self, weightspath=None):
        if weightspath is None:
            self.model = load(self.weightspath)
        else:
            self.model = load(weightspath)

    def logError(self, X_train, y_train, X_test, y_test):

        # storing results for plotting
        self.train_pred = self.model.predict(X_train)
        self.test_pred = self.model.predict(X_test)

        # Calculate train error
        rmse = sqrt(mean_squared_error(self.train_pred, y_train))
        cvrmse = 100 * (rmse / np.mean(y_train))
        mae = mean_absolute_error(self.train_pred, y_train)
        file = open(self.savepath + '/' + str(self.timegap) + ' min {} Results.txt'.format(self.modeltype), 'a')
        file.write('Train RMSE={} |Train CVRMSE={} |Train MAE={} \n'.format(rmse, cvrmse, mae))
        file.close()

        # Calculate test error
        rmse = sqrt(mean_squared_error(self.test_pred, y_test))
        cvrmse = 100 * (rmse / np.mean(y_test))
        mae = mean_absolute_error(self.test_pred, y_test)
        file = open(self.savepath + '/' + str(self.timegap) + ' min {} Results.txt'.format(self.modeltype), 'a')
        file.write('Test RMSE={} |Test CVRMSE={} |Test MAE={} \n'.format(rmse, cvrmse, mae))
        file.close()

    def generatePlots(self, y_train, y_test):

        # plt.rc('font', family='serif', serif='Times')
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)
        plt.rc('axes', labelsize=8)

        # width as measured in inkscape
        width = 25
        height = 5  # width / 1.618
        plt.rcParams["figure.figsize"] = (width, height)

        # Plotting the prediction versus target curve:train
        fig, axs = plt.subplots(1)
        # plt.xlim(0, len(y_train))
        plt.xlim(0, 2000)
        # plot target
        axs.plot(y_train, 'g--', label='Actual {}'.format(self.modeltype))
        # plot predicted
        axs.plot(self.train_pred, 'r--', label='Predicted {}'.format(self.modeltype))
        # Plot Properties
        axs.set_title('Predicted vs Actual {}'.format(self.modeltype))
        axs.set_xlabel('Time points at {} minutes'.format(self.timegap))
        axs.set_ylabel('Temperature in Fahrenheit')
        axs.grid(which='both', alpha=100)
        axs.legend()
        fig.savefig(self.savepath + '/' + str(self.timegap) +
                    ' minutes Train Temp Comparison {}.pdf'.format(self.modeltype),
                    bbox_inches='tight')
        # fig.savefig(self.savepath + '/' + str(self.timegap) +
        #             ' minutes Train Temp Comparison {}.png'.format(self.modeltype),
        #             bbox_inches='tight')
        plt.close(fig)

        # Plotting the prediction versus target curve":test
        fig, axs = plt.subplots(1)
        # plt.xlim(0, len(y_test))
        plt.xlim(0, 2000)
        # plot target
        axs.plot(y_test, 'g--', label='Actual {}'.format(self.modeltype))
        # plot predicted
        axs.plot(self.test_pred, 'r--', label='Predicted {}'.format(self.modeltype))
        # Plot Properties
        axs.set_title('Predicted vs Actual {}'.format(self.modeltype))
        axs.set_xlabel('Time points at {} minutes'.format(self.timegap))
        axs.set_ylabel('Temperature in Fahrenheit')
        axs.grid(which='both', alpha=100)
        axs.legend()
        fig.savefig(self.savepath + '/' + str(self.timegap) +
                    ' minutes Test Temp Comparison {}.pdf'.format(self.modeltype),
                    bbox_inches='tight')
        # fig.savefig(self.savepath + '/' + str(self.timegap) +
        #             ' minutes Test Temp Comparison {}.png'.format(self.modeltype),
        #             bbox_inches='tight')
        plt.close(fig)
