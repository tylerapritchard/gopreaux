import os
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split

from caat.utils import WLE

from .DataCube import DataCube
from .Kernels import Kernel
from .Plot import Plot

# from .SN import SN
from .SNCollection import SNCollection, SNType

warnings.filterwarnings("ignore")


class Fitter(ABC):
    """
    An AbstractBaseClass representing any Fitter object.

    This is inherited by the GP and GP3D classes in this package.
    More generally, a user is free to design a custom Fitter
    class to support their specific use case that inherits the 
    `Fitter` ABC.

    Required Methods:
    -------------
    _prepare_data:
        Initializes the data set to be used as input in the fitting routine
    _process_dataset:
        Processes the initialized data into the form used in the fitting routine
    predict:
        Runs the fitting routine and produces a prediction using the inputted data
    """

    @abstractmethod
    def _prepare_data():
        raise NotImplementedError(
            "Subclasses must implement this method."
        )  # pragma: no cover

    @abstractmethod
    def _process_dataset():
        raise NotImplementedError(
            "Subclasses must implement this method."
        )  # pragma: no cover

    @abstractmethod
    def predict():
        raise NotImplementedError(
            "Subclasses must implement this method."
        )  # pragma: no cover


class GP(Fitter):
    """
    Generate a GP fit for a single band.

    Unlike `GP3D`, the `GP` class provides functionality to
    perform Gaussian Process Regression on photometry from a
    single filter. This class inherits the `Fitter` ABC and
    prepares all data for the input sample, processes the data
    to be used in the Gaussian process regression routine, and
    performs the fitting.
    """

    wle = WLE

    def __init__(
        self,
        sne_collection: SNCollection | SNType,
        kernel: Kernel,
        filtlist: list,
        phasemin: int,
        phasemax: int,
        log_transform: float,
    ):
        """
        Initialize a GP object.

        Provides methods to perform Gaussian Process Regression on 
        photometry from a single filter for a collection of SN objects.
        NOTE: This method is not used frequently within the base
        GOPREAUX package. Instead, it is used as a building block for the
        `GP3D` class, and to provides backwards support for more traditional
        Gaussian Process fitting routines.

        Args:
            sne_collection (SNCollection | SNType): A collection of SN objects to fit.
            kernel (Kernel): A Kernel object used in the Gaussian Process Regression.
            filtlist (list): The filter of the photometry to fit.
            phasemin (int): The minimum phase, relative to peak brightness, to be fit.
            phasemax (int): The maximum phase, relative to peak brightness, to be fit.
            log_transform (float): The offset in the log transform. Must be larger than `phasemin`. 
                Effectively controls the light curve "stretch" in log space.
        """
        self.collection = sne_collection
        self.kernel = kernel
        self.filtlist = filtlist
        self.phasemin = phasemin
        self.phasemax = phasemax
        self.log_transform = log_transform

    def _prepare_data(self):
        """
        Use the flags set in __init__ to filter the pandas dataframes for each SN
        in the science and control samples
        """
        for sn in self.collection.sne:
            data_cube_filename = os.path.join(
                sn.base_path,
                sn.classification,
                sn.subtype,
                sn.name,
                sn.name + "_datacube.csv",
            )
            if os.path.exists(data_cube_filename):
                cube = pd.read_csv(data_cube_filename)
            else:
                # For now, we'll just construct it
                datacube = DataCube(sn=sn)
                datacube.construct_cube()
                cube = datacube.cube

            # Drop rows that are out of the phase range
            inds_to_drop_phase = cube.loc[
                (cube["Phase"] < self.phasemin) | (cube["Phase"] > self.phasemax)
            ].index
            cube = cube.drop(inds_to_drop_phase).reset_index(drop=True)

            # Drop nondetections farther than 2 days away from first/last detection
            try:
                min_phase = min(cube.loc[cube["Nondetection"] == False]["Phase"].values)
                inds_to_drop_nondets_before_first_det = cube.loc[
                    (cube["Nondetection"] == True) & (cube["Phase"] < (min_phase - 2.0))
                ].index
                cube = cube.drop(inds_to_drop_nondets_before_first_det).reset_index(
                    drop=True
                )
            except ValueError:  # no values
                pass

            try:
                max_phase = max(cube.loc[cube["Nondetection"] == False]["Phase"].values)
                inds_to_drop_nondets_after_last_det = cube.loc[
                    (cube["Nondetection"] == True) & (cube["Phase"] > (max_phase + 2.0))
                ].index
                cube = cube.drop(inds_to_drop_nondets_after_last_det).reset_index(
                    drop=True
                )
            except ValueError:  # no values
                pass

            # Drop rows corresponding to filters not in the user-provided filter list
            inds_to_drop_filts = cube.loc[~cube["Filter"].isin(self.filtlist)].index
            cube = cube.drop(inds_to_drop_filts).reset_index(drop=True)

            # Log transform the data (as a separate column)
            cube["LogPhase"] = np.log(
                cube["Phase"].values.astype(float) + self.log_transform
            )
            cube["LogWavelength"] = np.log10(cube["Wavelength"].values.astype(float))
            cube["LogShiftedWavelength"] = np.log10(
                cube["ShiftedWavelength"].values.astype(float)
            )

            # Drop nondetections that are within the phase range and less constraining
            # than the first or last detection in each filter
            try:
                min_flux_before_peak = min(
                    cube.loc[(cube["Nondetection"] == False) & (cube["Phase"] < 0)][
                        "Flux"
                    ].values
                )
                inds_to_drop_nondets_before_peak = cube.loc[
                    (cube["Nondetection"] == True)
                    & (cube["Phase"] < 0)
                    & (cube["Flux"] > min_flux_before_peak)
                ].index
                cube = cube.drop(inds_to_drop_nondets_before_peak).reset_index(
                    drop=True
                )
            except ValueError:  # No values pre-peak, so pass
                pass

            try:
                min_flux_after_peak = min(
                    cube.loc[(cube["Nondetection"] == False) & (cube["Phase"] > 0)][
                        "Flux"
                    ].values
                )
                inds_to_drop_nondets_after_peak = cube.loc[
                    (cube["Nondetection"] == True)
                    & (cube["Phase"] > 0)
                    & (cube["Flux"] > min_flux_after_peak)
                ].index
                cube = cube.drop(inds_to_drop_nondets_after_peak).reset_index(drop=True)
            except ValueError:  # No values post-peak, so pass
                pass

            # Drop nondetections between the first and last detection
            cube_only_dets = cube.loc[cube["Nondetection"] == False]
            if len(cube_only_dets["Phase"].values) > 0:
                first_detection = min(cube_only_dets["Phase"].values)
                last_detection = max(cube_only_dets["Phase"].values)
                inds_to_drop_nondets_between_dets = cube.loc[
                    (cube["Phase"] > first_detection)
                    & (cube["Phase"] < last_detection)
                    & (cube["Nondetection"] == True)
                ].index
                cube = cube.drop(inds_to_drop_nondets_between_dets).reset_index(
                    drop=True
                )

            # Construct anchor points
            # TODO: Implement

            sn.cube = cube

    def _process_dataset(
        self,
        filt,
        sn_set=None,
    ):
        """
        Loads all the data, shifts the data to peak,
        and concatenates the data for the object's SN collection
        or a provided SN set
        """

        phases, mags, errs, wls = (
            np.asarray([]),
            np.asarray([]),
            np.asarray([]),
            np.asarray([]),
        )

        if sn_set is None:
            sn_set = self.collection

        for sn in sn_set.sne:
            if hasattr(sn, "cube"):
                current_phases = sn.cube.loc[sn.cube["ShiftedFilter"] == filt][
                    "LogPhase"
                ].values
                current_mags = sn.cube.loc[sn.cube["ShiftedFilter"] == filt][
                    "ShiftedFlux"
                ]
                current_errs = sn.cube.loc[sn.cube["ShiftedFilter"] == filt][
                    "ShiftedFluxerr"
                ]
                current_wls = sn.cube.loc[sn.cube["ShiftedFilter"] == filt][
                    "LogShiftedWavelength"
                ].values

                phases = np.concatenate((phases, current_phases))
                mags = np.concatenate((mags, current_mags))
                errs = np.concatenate((errs, current_errs))
                wls = np.concatenate((wls, current_wls))

        return (
            phases.reshape(-1, 1),
            mags.reshape(-1, 1),
            errs.reshape(-1, 1),
            wls.reshape(-1, 1),
        )

    def run(self, filt: str, test_size: float):
        """
        Run the Gaussian Process Regression.

        This method runs the `_prepare_data` and 
        `_process_dataset` methods to prepare the data for fitting.

        A random sample of the input photometry is chosen via the 
        `sklearn` `train_test_split` function. The GaussianProcessRegressor
        object is returned along with the input photometry.

        Args:
            filt (str): The filter of the photometry to fit.
            test_size (float): The fraction of the input dataset to fit
                with the Gaussian process. Must be between 0 and 1.

        Returns:
            GaussianProcessRegressor: The trained Gaussian process fit
            np.ndarray: The input phases in the fit
            np.ndarray: The input magnitudes in the fit
            np.ndarray: The uncertainties on the input magnitudes.
        """
        self._prepare_data()

        phases, mags, errs, _ = self._process_dataset(filt, sn_set=self.collection)
        X_train, _, Y_train, _, Z_train, _ = train_test_split(
            phases, mags, errs, test_size=test_size
        )

        ### Get array of errors at each timestep
        min_phase, max_phase = sorted(X_train)[0], sorted(X_train)[-1]
        phase_grid = np.linspace(min_phase, max_phase, len(X_train))
        phase_grid_space = (max_phase - min_phase) / len(X_train)

        err_grid = np.ones(len(phase_grid))
        for mjd in phase_grid:
            ind = np.where(
                (X_train < mjd + phase_grid_space / 2)
                & (X_train > mjd - phase_grid_space / 2)
            )[0]
            mags_at_this_phase = Y_train[ind]
            if len(mags_at_this_phase) > 1:
                std_mag = max(np.std(mags_at_this_phase), 0.01)
            elif len(mags_at_this_phase) == 1:
                std_mag = Z_train[ind]
            else:
                std_mag = 0.1
            err_grid[ind] = std_mag

        ### Run the GP
        gaussian_process = GaussianProcessRegressor(
            kernel=self.kernel, alpha=err_grid, n_restarts_optimizer=9
        )
        gaussian_process.fit(X_train, Y_train)

        self.gaussian_process = gaussian_process

        return gaussian_process, phases, mags, errs

    def predict(self, filt: str, test_size: float, plot=False):
        """
        Run the Gaussian Process Regression and use it to predict
        a light curve for the input filter.

        Calls the `run` method and uses the trained, outputted
        `GaussianProcessRegressor` object to produce a predicted light curve.

        Args:
            filt (str): The filter of the photometry to fit.
            test_size (float): The fraction of the input dataset to fit
                with the Gaussian process. Must be between 0 and 1.
            plot (bool, optional): Plot the predicted light curve. Defaults to False.
        """
        gaussian_process, phases, mags, errs = self.run(filt, test_size)

        mean_prediction, std_prediction = gaussian_process.predict(
            sorted(phases), return_std=True
        )

        if plot:
            Plot().plot_gp_predict_gp(
                phases=phases,
                mean_prediction=mean_prediction,
                std_prediction=std_prediction,
                mags=mags,
                errs=errs,
                filt=filt,
            )
