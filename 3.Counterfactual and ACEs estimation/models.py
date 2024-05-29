import enum
from multiprocessing.dummy import Value
from typing import Union, Set
from datetime import datetime
import humanize
import time
import json
import logging

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression, Ridge  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler, PolynomialFeatures  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.neural_network import MLPRegressor  # type: ignore
from pathlib import Path  # type: ignore
from sklearn.kernel_ridge import KernelRidge  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.metrics import r2_score, mean_absolute_error

from utils import get_git_root
from pretty_logger import get_logger
from names import *

project_root = get_git_root()

logger = get_logger(
    name="modelslog",
    full_path=Path(project_root, "log/models.log"),
    add_console_hander=True,
    level=logging.INFO,
)

NJOBS = 8

num_prior_days = 10
date_covid = datetime(2020, 3, 1)
# rough date
date_vaccine = datetime(2021, 4, 1)

datafile = "../data/features_v3.csv"
# _longest is actually shorter and only has the Y-s ?
sets_file = "../2.causal discovery/pc_ici_longest.parquet"
# sets_file = "../2.causal discovery/pc_ici.parquet"


class LassoModelBuilder:
    def __init__(self, data, outcome, covariates):
        self.data = data
        self.outcome = outcome
        self.covariates = covariates

    def fit_lasso(
        self,
        test_size=0.2,
        random_state=None,
        alpha=1.0,
    ):
        # Extract the X (covariates) and y (outcome) from the data
        X = self.data[list(self.covariates)]
        y = self.data[self.outcome]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = Lasso(alpha=alpha, random_state=random_state)

        model.fit(X_train, y_train)

        # Predict the outcome on the training and testing data
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)

        self.model = model
        return model, (r2_train, r2_test, mae)


class RandomForestModelBuilder:
    def __init__(self, data, outcome, covariates):
        self.data = data
        self.outcome = outcome
        self.covariates = covariates

    def fit_random_forest(
        self,
        test_size=0.2,
        random_state=None,
        n_estimators=100,
        max_depth=None,
        ccp_alpha=0,
    ):
        # Extract the X (covariates) and y (outcome) from the data
        X = self.data[list(self.covariates)]
        y = self.data[self.outcome]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=NJOBS,
            ccp_alpha=1e-4,
        )

        model.fit(X_train, y_train)

        # Predict the outcome on the training and testing data
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        self.model = model
        return model, (r2_train, r2_test, mae)


class LinearModelBuilder:
    def __init__(self, data, outcome, covariates):
        self.data = data
        self.outcome = outcome
        self.covariates = covariates

    def fit_linear_model(
        self, test_size=0.2, random_state=None, alpha: int = 0
    ):
        # Extract the X (covariates) and y (outcome) from the data
        X = self.data[list(self.covariates)]
        y = self.data[self.outcome]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Create and fit the linear regression model
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        # Predict the outcome on the testing data
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        test_residuals = y_test - y_pred_test
        train_residuals = y_train - y_pred_train

        # Compute the mean error
        # (not mean absolute, not mean square)
        train_mean_average_error = np.mean(train_residuals) / len(y_train)
        test_mean_average_error = np.mean(test_residuals) / len(y_test)

        self.model = model
        # Return the fitted model and the R^2 scores
        # for training and testing sets
        return model, (r2_train, r2_test, test_mean_average_error)

    def fit_polynomial_model(self, degree, test_size=0.2, random_state=None):
        # Extract the X (covariates) and y (outcome) from the data
        X = self.data[list(self.covariates)]
        y = self.data[self.outcome]

        # Generate polynomial and interaction features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_poly, y, test_size=test_size, random_state=random_state
        )

        # Create and fit the linear regression model with polynomial features
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict the outcome on the testing data
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        mae = mean_absolute_error(y_test, y_pred_test)

        # Return the fitted model and the R^2 scores
        # for training and testing sets
        return model, (r2_train, r2_test, mae)


class KernelModelBuilder:
    def __init__(self, data, outcome, covariates):
        self.data = data
        self.outcome = outcome
        self.covariates = covariates

    def fit_gaussian_kernel_model(
        self, test_size=0.2, random_state=None, alpha=1.0, gamma=None
    ):
        # Extract the X (covariates) and y (outcome) from the data
        X = self.data[list(self.covariates)]
        y = self.data[self.outcome]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        model = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)
        model.fit(X_train, y_train)
        # Create and fit the Gaussian kernel regression model
        model.fit(X_train, y_train)

        # Predict the outcome on the training and testing data
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        # Return the fitted model and the R^2 scores
        # for training and testing sets
        mae = mean_absolute_error(y_test, y_pred_test)
        self.model = model
        # Return the fitted model and the R^2 scores
        # for training and testing sets
        return model, (r2_train, r2_test, mae)

    def predict(self, X_new):
        return self.model.predict(X_new)


class NeuralNetModelBuilder:
    def __init__(self, data, outcome, covariates):
        self.data = data
        self.outcome = outcome
        self.covariates = covariates

    def fit_neural_net(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        test_size=0.2,
        random_state=None,
        max_iter=500,
        alpha=1e-4,
    ):
        # Extract the X (covariates) and y (outcome) from the data
        X = self.data[self.covariates]
        y = self.data[self.outcome]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    MLPRegressor(
                        hidden_layer_sizes=hidden_layer_sizes,
                        activation=activation,
                        solver=solver,
                        max_iter=max_iter,
                        random_state=random_state,
                        alpha=alpha,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)

        # Predict the outcome on the training and testing data
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        # Return the fitted model and the R^2 scores
        # for training and testing sets
        mae = mean_absolute_error(y_test, y_pred_test)
        self.model = model
        # Return the fitted model and the R^2 scores
        # for training and testing sets
        return model, (r2_train, r2_test, mae)

    def predict(self, X_new):
        return self.model.predict(X_new)


class WBModel:
    """if binarization_threshold is None, leave the treatment column as is"""

    def __init__(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        separating_set: Set[str],
        binarization_threshold: Union[float, None] = None,
        name: str = "",
    ):
        self.start_time = time.time()
        self.name = name
        self.data = data.copy()
        if isinstance(treatment, str) and treatment in self.data.columns:
            self.treatment = treatment
        else:
            raise ValueError(
                "treatment not in data or treatment not a string "
                "(with the column name of the treatment)"
            )
        if isinstance(outcome, str) and outcome in self.data.columns:
            self.outcome = outcome
        else:
            raise ValueError(
                "outcome not in data or outcome not a string"
                "(with the column name of the outcome)"
            )

        # check that all covariates are in data
        if not all(
            covariate in self.data.columns for covariate in separating_set
        ):
            raise ValueError("some covariate not in data")
        else:
            self.separating_set = separating_set

        self.binarization_threshold = binarization_threshold

        if self.binarization_threshold is not None:
            self.data[self.treatment] = (
                self.data[treatment] > binarization_threshold
            )
        else:
            if set(self.data[self.treatment].unique()) != {0, 1}:
                q = np.percentile(self.data[self.treatment], 75)
                logger.debug(
                    "Binarizing using value "
                    f"({full_dictionary[self.treatment]} "
                    f"q={q:.2e})"
                )
                self.data[self.treatment] = self.data[self.treatment] > q

        self.covariates_dictionary = {
            key: value
            for key, value in full_dictionary.items()
            if key in self.data.columns
        }
        # set([self.treatment]), set(self.treatment) will split the string
        self.covariates = set([self.treatment]) | self.separating_set

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond:06d}"
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Step 2: Create a Path object with the folder name
        self.folder_path = Path(timestamp)

        # Step 3: Check if the folder exists, if not, create it
        if not self.folder_path.exists():
            self.folder_path.mkdir()
            logger.debug(f"Directory {self.folder_path} created.")
        else:
            logger.debug(f"Directory {self.folder_path} already exists.")

        self.X = self.data[list(self.covariates)]
        self.y = self.data[self.outcome]

        self.pre_model = None
        self.post_model = None

    def document(self):
        end_time = time.time()

        # Calculate the time taken to fit the model
        self.fit_time = end_time - self.start_time
        postcovid_ace = self.ace(COVIDStatus.POST_COVID)
        precovid_ace = self.ace(COVIDStatus.PRE_COVID)
        difference_of_ace = self.ace_post_minus_pre()
        relative_difference_of_ace = (
            100 * difference_of_ace / np.abs(precovid_ace)
        )

        results_json = {
            "pre_r_squared test": f"{self.pre_r_squared[1]:.2e}",
            "pre_r_squared train": f"{self.pre_r_squared[0]:.2e}",
            "post_r_squared test": f"{self.post_r_squared[1]:.2e}",
            "post_r_squared train": f"{self.post_r_squared[0]:.2e}",
            "postcovid ACE": f"{postcovid_ace:.2e}",
            "precovid ACE": f"{precovid_ace:.2e}",
            "difference of ACE": f"{difference_of_ace:.2e}",
            "Relative difference of ACE": f"{relative_difference_of_ace:.2e}",
            "time": humanize.precisedelta(self.fit_time),
            "path": str(self.folder_path),
            "name": self.name,
            "outcome": self.outcome,
            "treatment": self.treatment,
            "covariates": f"{self.covariates}",
            "description": (
                "Modeling treatment:"
                f"{full_dictionary[self.treatment]} "
                f"on outcome:{full_dictionary[self.outcome]}"
            ),
        }
        with open(Path(self.folder_path, "results.json"), "w") as f:
            json.dump(results_json | self.model_type_specific, f, indent=4)

    def y_hat(self, era: COVIDStatus, actual: bool):
        # returns a vector of values obtained when all
        # the units received the treatement 'actual', by using the model that
        # was fit on units that actually received the treatment 'actual'
        # select only era data (either post or pre covid)
        era_data = self.data[self.data["C"] == era]
        # data for which the treatment was 'actual'
        era_data_factual = era_data[era_data[self.treatment] == actual].copy()

        # select data for which the treatment was not 'actual'
        # and flip the value of treatment to actual.
        # There datapoints are now counterfactuals
        # We will use our models to estimate the counterfactual outcome values
        era_data_counterfactual = era_data[
            era_data[self.treatment] != actual
        ].copy()
        era_data_counterfactual[self.treatment] = actual

        if era == COVIDStatus.PRE_COVID:
            mu_counterfactual = self.pre_model.predict(
                era_data_counterfactual[list(self.covariates)]
            )

        elif era == COVIDStatus.POST_COVID:
            mu_counterfactual = self.post_model.predict(
                era_data_counterfactual[list(self.covariates)]
            )
        else:
            raise ValueError("There should be only two eras")

        y_hat = np.concatenate(
            [era_data_factual[self.outcome].values, mu_counterfactual]
        )
        return y_hat

    def ace(self, era: COVIDStatus):
        # ACE for self.treatment, either in the post or in the pre-covid era.
        return self.y_hat(era, True).mean() - self.y_hat(era, False).mean()

    def ace_post_minus_pre(self):
        # ACE in the post-pandemic era minus ACE in the pre-pandemic era.
        return self.ace(COVIDStatus.POST_COVID) - self.ace(
            COVIDStatus.PRE_COVID
        )


class WBNeuralNetModel(WBModel):
    def __init__(self, **kwargs):
        self.alpha = kwargs.pop("alpha", 1e-4)
        self.hidden_layer_sizes = kwargs.pop("hidden_layer_sizes", (32,))
        self.max_iter = kwargs.pop("max_iter", 400)
        self.activation = kwargs.pop("activation", "relu")
        super().__init__(**kwargs)

        self.pre_model, self.pre_r_squared = NeuralNetModelBuilder(
            self.data[self.data["C"] == COVIDStatus.PRE_COVID],
            self.outcome,
            covariates=self.covariates,
        ).fit_neural_net(
            hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter
        )

        self.post_model, self.post_r_squared = NeuralNetModelBuilder(
            self.data[self.data["C"] == COVIDStatus.POST_COVID],
            self.outcome,
            covariates=self.covariates,
        ).fit_neural_net(
            hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter
        )

        self.model_type_specific = {
            "type": "NN",
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "activation": self.activation,
            "hidden_layers": list(self.hidden_layer_sizes),
        }
        self.document()


class WBLassoModel(WBModel):
    def __init__(self, **kwargs):
        self.alpha = kwargs.pop("alpha", 1e-5)
        super().__init__(**kwargs)

        # fit a model only on the precovid data
        self.pre_model, self.pre_r_squared = LassoModelBuilder(
            self.data[self.data["C"] == COVIDStatus.PRE_COVID],
            self.outcome,
            covariates=self.covariates,
        ).fit_lasso(alpha=self.alpha)

        # fit a model only on the postcovid data
        self.post_model, self.post_r_squared = LassoModelBuilder(
            self.data[self.data["C"] == COVIDStatus.POST_COVID],
            self.outcome,
            covariates=self.covariates,
        ).fit_lasso(alpha=self.alpha)

        self.model_type_specific = {
            "type": "lasso",
            "alpha": self.alpha,
        }
        self.document()


class WBRandomForestModel(WBModel):
    def __init__(self, **kwargs):
        self.n_estimators = kwargs.pop("n_estimators", 100)
        self.ccp_alpha = kwargs.pop("ccp_alpha", 1e-5)
        super().__init__(**kwargs)

        # fit a model only on the precovid data
        self.pre_model, self.pre_r_squared = RandomForestModelBuilder(
            self.data[self.data["C"] == COVIDStatus.PRE_COVID],
            self.outcome,
            covariates=self.covariates,
        ).fit_random_forest(
            n_estimators=self.n_estimators, ccp_alpha=self.ccp_alpha
        )

        # fit a model only on the postcovid data
        self.post_model, self.post_r_squared = RandomForestModelBuilder(
            self.data[self.data["C"] == COVIDStatus.POST_COVID],
            self.outcome,
            covariates=self.covariates,
        ).fit_random_forest(
            n_estimators=self.n_estimators, ccp_alpha=self.ccp_alpha
        )

        self.model_type_specific = {
            "type": "random forest",
            "n_estimators": self.n_estimators,
            "ccp_alpha": self.ccp_alpha,
        }
        self.document()


class WBLinearModel(WBModel):
    def __init__(self, **kwargs):
        self.alpha = kwargs.pop("alpha", 0)
        super().__init__(**kwargs)

        self.pre_model, self.pre_r_squared = LinearModelBuilder(
            self.data[self.data["C"] == COVIDStatus.PRE_COVID],
            self.outcome,
            covariates=self.covariates,
        ).fit_linear_model(alpha=self.alpha)

        self.post_model, self.post_r_squared = LinearModelBuilder(
            self.data[self.data["C"] == COVIDStatus.POST_COVID],
            self.outcome,
            covariates=self.covariates,
        ).fit_linear_model(alpha=self.alpha)

        self.model_type_specific = {
            "type": "linear",
            "alpha": self.alpha,
            "mean_average_error": self.pre_r_squared[2],
        }
        self.document()


class WBKernelModel(WBModel):
    def __init__(self, **kwargs):
        self.alpha = kwargs.pop("alpha", 1)
        self.gamma = kwargs.pop("gamma", None)
        super().__init__(**kwargs)

        self.pre_model, self.pre_r_squared = KernelModelBuilder(
            self.data[self.data["C"] == COVIDStatus.PRE_COVID],
            self.outcome,
            covariates=self.covariates,
        ).fit_gaussian_kernel_model(alpha=self.alpha, gamma=self.gamma)

        self.post_model, self.post_r_squared = KernelModelBuilder(
            self.data[self.data["C"] == COVIDStatus.POST_COVID],
            self.outcome,
            covariates=self.covariates,
        ).fit_gaussian_kernel_model(alpha=self.alpha, gamma=self.gamma)

        self.model_type_specific = {
            "type": "Gaussian kernel",
            "alpha": self.alpha,
            "gamma": self.gamma,
        }
        self.document()


class WBLinearPolyModel(WBModel):
    def __init__(self, **kwargs):
        self.degree = kwargs.pop("degree", 2)

        super().__init__(**kwargs)

        self.pre_model, self.pre_r_squared = LinearModelBuilder(
            self.data[self.data["C"] == COVIDStatus.PRE_COVID],
            self.outcome,
            covariates=self.covariates,
        ).fit_polynomial_model(degree=self.degree)

        self.post_model, self.post_r_squared = LinearModelBuilder(
            self.data[self.data["C"] == COVIDStatus.POST_COVID],
            self.outcome,
            covariates=self.covariates,
        ).fit_polynomial_model(degree=self.degree)

        self.model_type_specific = {
            "type": "linear with poly features",
            "degree": self.degree,
        }
        self.document()


def aggregate_results_from_subfolders(base_folder):
    # Convert base_folder to a Path object
    base_path = Path(base_folder)

    # Initialize an empty list to store data
    all_data = []

    # Iterate over all subfolders in the base folder
    for subfolder in base_path.iterdir():
        # Check if the subfolder matches the format YYYYMMDD_HHMMSS
        if subfolder.is_dir() and subfolder.name.count("_") == 2:
            # Define the path to the results.json file
            results_file = subfolder / "results.json"

            # Check if the results.json file exists
            if results_file.exists():
                # Read and parse the JSON file
                with results_file.open("r") as f:
                    data = json.load(f)
                    all_data.append(data)

    # Create a DataFrame from the aggregated data
    df = pd.DataFrame(all_data)

    df["difference of ACE"] = pd.to_numeric(
        df["difference of ACE"], errors="coerce"
    )

    return df


class CovariateSet:
    @staticmethod
    def model_row_series_valid(
        row: pd.Series,
        data: pd.DataFrame,
    ):
        adjustment_set = row["sets"].tolist()
        outcome = row["outcome"]
        treatment = row["treatment"]

        if treatment not in data.columns or outcome not in data.columns:
            return logger.debug(
                f"treatment or outcome not in columns: "
                f"{treatment} or {outcome} (columns:{data.columns})"
            )
        else:
            return set(data.columns).intersection(set(adjustment_set))

    @staticmethod
    def row_string(outcome, treatment, adjustment_set):
        return (
            f"treatment: {treatment}:{full_dictionary[treatment]}, "
            f"outcome: {outcome}:{full_dictionary[outcome]}, "
            f"adjustment set={adjustment_set}"
        )

    def __init__(
        self,
        row: pd.Series,
        data: pd.DataFrame,
        treatments_to_skip=set([]),
        outcomes_to_skip=set([]),
    ) -> None:
        # passed adjustment set
        self.outcomes_to_skip = outcomes_to_skip
        self.treatments_to_skip = treatments_to_skip
        self.original_adjustment_set = row["sets"].tolist()
        # adjustment set restricted to valid data (columns in the dataframe)
        self.restricted_adjustment_set = CovariateSet.model_row_series_valid(
            row=row, data=data
        )
        self.treatment = row["treatment"]
        self.outcome = row["outcome"]

    @property
    def set_to_fit(self) -> Union[tuple, None]:
        if self.valid_set:
            return (
                self.outcome,
                set([self.treatment] + list(self.restricted_adjustment_set)),
            )
        else:
            return None

    @property
    def valid_set(self):
        return (
            (self.restricted_adjustment_set is not None)
            and (len(self.restricted_adjustment_set) != 0)
            # exclude demographics
            # and (not self.treatment.startswith("D"))
            and (self.outcome not in self.outcomes_to_skip)
            and (self.treatment not in self.treatments_to_skip)
        )

    def __repr__(self):
        return CovariateSet.row_string(
            self.outcome, self.treatment, self.restricted_adjustment_set
        )

    def __str__(self):
        return f"{self.set_to_fit}"
