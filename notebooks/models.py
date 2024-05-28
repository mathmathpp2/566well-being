import enum
from typing import Union, List
from datetime import datetime
import humanize
import time
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from pathlib import Path
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

import matplotlib.pyplot as plt

NJOBS = 8

num_prior_days = 10
date_covid = datetime(2020, 3, 1)
# rough date
date_vaccine = datetime(2021, 4, 1)

class WBModelType(enum.Enum):
    LINEAR = 1

class SplitMethod(enum.Enum):
    MEDIAN = 1

class COVIDStatus(enum.Enum):
    PRE_COVID = 0
    POST_COVID = 1

    def __str__(self):
        return f"{self.name}"

ema_dictionary = {
    "Y1": "pam",
    "Y2": "phq2_score",
    "Y3": "phq4_score",
    "Y4": "gad2_score",
    "Y5": "social_level",
    "Y6": "sse_score",
    "Y7": "stress",
}
reverse_ema_dictionary = {v: k for k, v in ema_dictionary.items()}

physical_dictionary = {
    "P1": "excercise",
    "P2": "studying",
    "P3": "in house",
    "P4": "sports",
}
social_dictionary = {
    "S1": "traveling",
    "S2": "distance traveled",
    "S3": "time in social location",
    "S4": "visits",
    "S5": "duration unlocked phone in social locations",
    "S6": "frequency of unlocked phone in social locations",
    "S7": "motion at social locations",
}

sleep_dictionary = {
    "Z1": "sleep_duration",
    "Z2": "sleep start time",
    "Z3": "sleep end time",
}


demographic_dictionary = {
    "D1": "gender",
    "D2": "race",
    "D3": "os",
    "D4": "cohort year",

}


full_dictionary = (
    physical_dictionary | social_dictionary | sleep_dictionary | ema_dictionary | {'C': COVIDStatus} | demographic_dictionary
)

ema = [f"Y{i}" for i in range(1, 8, 1)]
physical = [f"P{i}" for i in range(1, 5, 1)]
social = [f"S{i}" for i in range(1, 8, 1)]
sleep = [f"Z{i}" for i in range(1, 4, 1)]

datafile = "../data/features_v3.csv"
# _longest is actually shorter and only has the Y-s ?
sets_file = "../2.causal discovery/pc_ici_longest.parquet"
#sets_file = "../2.causal discovery/pc_ici.parquet"



class RandomForestModelBuilder:
    def __init__(self, data, outcome, covariates):
        self.data = data
        self.outcome = outcome
        self.covariates = covariates

    def fit_random_forest(self, test_size=0.2, random_state=None, n_estimators=100, max_depth=None, ccp_alpha=0):
        # Extract the X (covariates) and y (outcome) from the data
        X = self.data[self.covariates]
        y = self.data[self.outcome]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=NJOBS, ccp_alpha=1e-4))
        ])
        model.fit(X_train, y_train)

        # Predict the outcome on the training and testing data
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        r2_train = r2_score(y_train, y_pred_train)
        mean_absolute_error(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        # Return the fitted model and the R^2 scores for training and testing sets
        return model, (r2_train, r2_test)

class LinearModelBuilder:
    def __init__(self, data, outcome, covariates):
        self.data = data
        self.outcome = outcome
        self.covariates = covariates

    def fit_linear_model(self, test_size=0.2, random_state=None):
        # Extract the X (covariates) and y (outcome) from the data
        X = self.data[self.covariates]
        y = self.data[self.outcome]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Create and fit the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict the outcome on the testing data
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        # Return the fitted model and the R^2 scores for training and testing sets
        return model, (r2_train, r2_test)

    def fit_polynomial_model(self, degree, test_size=0.2, random_state=None):
        # Extract the X (covariates) and y (outcome) from the data
        X = self.data[self.covariates]
        y = self.data[self.outcome]

        # Generate polynomial and interaction features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=test_size, random_state=random_state)

        # Create and fit the linear regression model with polynomial features
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict the outcome on the testing data
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        # Return the fitted model and the R^2 scores for training and testing sets
        return model, (r2_train, r2_test)



class KernelModelBuilder:
    def __init__(self, data, outcome, covariates):
        self.data = data
        self.outcome = outcome
        self.covariates = covariates

    def fit_linear_model(self, test_size=0.2, random_state=None, alpha=1.0):
        # Extract the X (covariates) and y (outcome) from the data
        X = self.data[self.covariates]
        y = self.data[self.outcome]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Create and fit the linear regression model
        model = KernelRidge(kernel='linear', alpha=alpha)
        model.fit(X_train, y_train)

        # Predict the outcome on the training and testing data
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        # Return the fitted model and the R^2 scores for training and testing sets
        self.model = model

        return model, (r2_train, r2_test)

    def fit_gaussian_kernel_model(self, test_size=0.2, random_state=None, alpha=1.0, gamma=None):
        # Extract the X (covariates) and y (outcome) from the data
        X = self.data[self.covariates]
        y = self.data[self.outcome]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma))
        ])
        model.fit(X_train, y_train)
        # Create and fit the Gaussian kernel regression model
        model.fit(X_train, y_train)

        # Predict the outcome on the training and testing data
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        # Return the fitted model and the R^2 scores for training and testing sets
        self.model = model
        return model, (r2_train, r2_test)

    def predict(self, X_new):
        return self.model.predict(X_new)



class NeuralNetModelBuilder:
    def __init__(self, data, outcome, covariates):
        self.data = data
        self.outcome = outcome
        self.covariates = covariates

    def fit_neural_net(self, hidden_layer_sizes=(100, ), activation='relu', solver='adam', test_size=0.2, random_state=None, max_iter=500, alpha=1e-4):
        # Extract the X (covariates) and y (outcome) from the data
        X = self.data[self.covariates]
        y = self.data[self.outcome]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, max_iter=max_iter, random_state=random_state, alpha=alpha))
        ])
        model.fit(X_train, y_train)


        # Predict the outcome on the training and testing data
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        # Return the fitted model and the R^2 scores for training and testing sets
        self.model = model
        return model, (r2_train, r2_test)

    def predict(self, X_new):
        return self.model.predict(X_new)


class WBModel:
    """ if binarization_threshold is None, leave the treatment column as is
    """
    def __init__(self,
                 data: pd.DataFrame,
                 treatment: str,
                 outcome: str,
                 separating_set: List[str],
                 binarization_threshold: Union[float, None]=None,
                 name: str=""):
        self.start_time = time.time()
        self.name = name
        self.data = data.copy()
        if isinstance(treatment, str) and treatment in self.data.columns:
            self.treatment = treatment
        else:
            raise ValueError(
                "treatment not in data or treatment not a string "
                "(with the column name of the treatment)")
        if isinstance(outcome, str) and outcome in self.data.columns:
            self.outcome = outcome
        else:
            raise ValueError("outcome not in data or outcome not a string"
                             "(with the column name of the outcome)")

        # check that all covariates are in data
        if not all(covariate in self.data.columns
                   for covariate in separating_set):
            raise ValueError("some covariate not in data")
        else:
            self.separating_set = separating_set

        self.binarization_threshold = binarization_threshold

        if self.binarization_threshold is not None:
            self.data[self.treatment] = (
                self.data[treatment] > binarization_threshold)
        else:
            if set(self.data[self.treatment].unique()) != {0,1}:
                print(
                    "Binarizing using median value "
                    f"({full_dictionary[self.treatment]} "
                    f"median={self.data[self.treatment].median():.2e})")
                self.data[self.treatment] = (
                    self.data[self.treatment] > self.data[self.treatment].median())

        self.covariates_dictionary = {
            key: value for key,value in full_dictionary.items()
            if key in self.data.columns}

        self.model_covariates=[self.treatment] + self.separating_set

        now = datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S') + f'_{now.microsecond:06d}'
        #timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Step 2: Create a Path object with the folder name
        self.folder_path = Path(timestamp)

        # Step 3: Check if the folder exists, if not, create it
        if not self.folder_path.exists():
            self.folder_path.mkdir()
            print(f"Directory {self.folder_path} created.")
        else:
            print(f"Directory {self.folder_path} already exists.")

        self.X = self.data[self.model_covariates]
        self.y = self.data[self.outcome]

    def plot_data_and_fit(self, X, y_actual, y_pred, title="Model Fit", xlabel="X", ylabel="Y"):
        """
        Plots the actual data points and the model fit.

        Parameters:
        - X: array-like, shape (n_samples,)
            The input data.
        - y_actual: array-like, shape (n_samples,)
            The actual target values.
        - y_pred: array-like, shape (n_samples,)
            The predicted target values from the model.
        - title: str, default="Model Fit"
            The title of the plot.
        - xlabel: str, default="X"
            The label for the x-axis.
        - ylabel: str, default="Y"
            The label for the y-axis.
        """

        fig, ax = plt.subplots(figsize=(10, 6))

        y_pred = self.model.predict(self.X)

        ax.scatter(self.X, y_actual, color='blue', label='Actual Data')
        ax.plot(self.X, y_pred, color='red', label='Model Fit', linewidth=2)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
        fig.savefig(Path(self.folder_path, f"{title}.png"), dpi=400)
        fig.savefig(Path(self.folder_path, f"{title}.svg"))
        plt.show()


    def document(self):
        end_time = time.time()

        # Calculate the time taken to fit the model
        self.fit_time = end_time - self.start_time


        results_json = {
            "pre_r_squared train": f"{self.pre_r_squared[0]:.2e}",
            "pre_r_squared test": f"{self.pre_r_squared[1]:.2e}",
            "post_r_squared train": f"{self.post_r_squared[0]:.2e}",
            "post_r_squared test": f"{self.post_r_squared[1]:.2e}",
            "time": humanize.precisedelta(self.fit_time),
            "path": str(self.folder_path),
            "name": self.name,
            "outcome": self.outcome,
            "treatment": self.treatment,
            "covariates": f"{self.model_covariates}",
            "description": (
                "Modeling treatment:"
                f"{full_dictionary[self.treatment]} "
                f"on outcome:{full_dictionary[self.outcome]}")
        }
        with open(Path(self.folder_path, "results.json"), 'w') as f:
            json.dump(results_json | self.model_type_specific, f, indent=4)


    # @property
    # def post_ace(self):
    #     return self.post_coefficients[self.treatment]

    # @property
    # def pre_ace(self):
    #     return self.pre_coefficients[self.treatment]

    @property
    def summary(self):
        print(f"pre-covid ACE: {self.pre_ace}, post_covid:{self.post_ace}"
              f"pre_r_squared: {self.pre_r_squared}, post_r_squared: {self.post_r_squared}"
              #f"pre_coefficients: {self.pre_coefficients}"
              #f"post_coefficients:{self.post_coefficients}")

        )

class WBNeuralNetModel(WBModel):
    def __init__(self, **kwargs):
        self.alpha = kwargs.pop('alpha', 1e-4)
        self.hidden_layer_sizes = kwargs.pop('hidden_layer_sizes', (32,))
        self.max_iter = kwargs.pop('max_iter', 400)
        self.activation = kwargs.pop('activation', 'relu')
        super().__init__(**kwargs)

        self.pre_model, self.pre_r_squared = NeuralNetModelBuilder(
            self.data[self.data['C'] == COVIDStatus.PRE_COVID],
            self.outcome, covariates=self.model_covariates
            ).fit_neural_net(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter)

        self.post_model, self.post_r_squared = NeuralNetModelBuilder(
            self.data[self.data['C'] == COVIDStatus.POST_COVID],
            self.outcome, covariates=self.model_covariates
            ).fit_neural_net(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter)



        self.model_type_specific = {"model": "NN",
                                    "alpha": self.alpha,
                               "max_iter": self.max_iter,
                               "activation": self.activation,
                               "hidden_layers": list(self.hidden_layer_sizes)}
        self.document()

class WBRandomForestModel(WBModel):
    def __init__(self, **kwargs):
        self.n_estimators = kwargs.pop('n_estimators', 100)
        self.ccp_alpha = kwargs.pop('ccp_alpha', 1e-5)
        super().__init__(**kwargs)

        self.pre_model, self.pre_r_squared = RandomForestModelBuilder(
            self.data[self.data['C'] == COVIDStatus.PRE_COVID],
            self.outcome, covariates=self.model_covariates
            ).fit_random_forest(n_estimators=self.n_estimators, ccp_alpha=self.ccp_alpha)

        self.post_model, self.post_r_squared = RandomForestModelBuilder(
            self.data[self.data['C'] == COVIDStatus.POST_COVID],
            self.outcome, covariates=self.model_covariates
            ).fit_random_forest(n_estimators=self.n_estimators, ccp_alpha=self.ccp_alpha)

        self.model_type_specific = {"type": "random forest",
                                    "n_estimators": self.n_estimators,
                                    "ccp_alpha": self.ccp_alpha}
        self.document()

class WBLinearModel(WBModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pre_model, self.pre_r_squared = LinearModelBuilder(
            self.data[self.data['C'] == COVIDStatus.PRE_COVID],
            self.outcome, covariates=self.model_covariates
            ).fit_linear_model()

        self.post_model, self.post_r_squared = LinearModelBuilder(
            self.data[self.data['C'] == COVIDStatus.POST_COVID],
            self.outcome, covariates=self.model_covariates
            ).fit_linear_model()

        self.model_type_specific = {"type": "linear"}
        self.document()



class WBKernelModel(WBModel):
    def __init__(self, **kwargs):
        self.alpha = kwargs.pop('alpha', 1)
        self.gamma = kwargs.pop('gamma', None)
        super().__init__(**kwargs)

        self.pre_model, self.pre_r_squared = KernelModelBuilder(
            self.data[self.data['C'] == COVIDStatus.PRE_COVID],
            self.outcome, covariates=self.model_covariates
            ).fit_gaussian_kernel_model(alpha=self.alpha, gamma=self.gamma)

        self.post_model, self.post_r_squared = KernelModelBuilder(
            self.data[self.data['C'] == COVIDStatus.POST_COVID],
            self.outcome, covariates=self.model_covariates
            ).fit_gaussian_kernel_model(alpha=self.alpha, gamma=self.gamma)

        self.model_type_specific = {
            "type": "Gaussian kernel", "alpha": self.alpha, "gamma": self.gamma}
        self.document()

class WBLinearPolyModel(WBModel):
    def __init__(self, **kwargs):
        self.degree = kwargs.pop('degree', 2)

        super().__init__(**kwargs)

        self.pre_model, self.pre_r_squared = LinearModelBuilder(
            self.data[self.data['C'] == COVIDStatus.PRE_COVID],
            self.outcome, covariates=self.model_covariates
            ).fit_polynomial_model(degree=self.degree)

        self.post_model, self.post_r_squared = LinearModelBuilder(
            self.data[self.data['C'] == COVIDStatus.POST_COVID],
            self.outcome, covariates=self.model_covariates
            ).fit_polynomial_model(degree=self.degree)

        self.model_type_specific = {
            "type": "linear with poly features", "degree": self.degree}
        self.document()



def aggregate_results_from_subfolders(base_folder):
    # Convert base_folder to a Path object
    base_path = Path(base_folder)

    # Initialize an empty list to store data
    all_data = []

    # Iterate over all subfolders in the base folder
    for subfolder in base_path.iterdir():
        # Check if the subfolder matches the format YYYYMMDD_HHMMSS
        if subfolder.is_dir() and subfolder.name.count('_') == 2:
            # Define the path to the results.json file
            results_file = subfolder / 'results.json'

            # Check if the results.json file exists
            if results_file.exists():
                # Read and parse the JSON file
                with results_file.open('r') as f:
                    data = json.load(f)
                    all_data.append(data)

    # Create a DataFrame from the aggregated data
    df = pd.DataFrame(all_data)
    df['pre_r_squared test'] = pd.to_numeric(df['pre_r_squared test'], errors='coerce')
    df['pre_r_squared train'] = pd.to_numeric(df['pre_r_squared train'], errors='coerce')
    df.sort_values(by='pre_r_squared test', ascending=False, inplace=True)
    return df
