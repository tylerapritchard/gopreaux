import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

from caat.utils import ROOT_DIR, WLE, colors, convert_shifted_fluxes_to_shifted_mags

from .DataCube import DataCube
from .Plot import Plot
from .SN import SN
from .SNCollection import SNCollection, SNType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SurfaceArray:
    """
    Class that stores final SED surface models as a numpy array,
    rather than a GaussianProcessRegressor object.

    Contains a `predict` method that is called with the same inputs as the
    `GaussianProcessRegressor.predict` method, so that both can objects
    can be used interchangeably.
    """

    def __init__(
        self,
        surface: np.ndarray,
        phase_grid: np.ndarray,
        wl_grid: np.ndarray,
        kernel: RBF | Matern | WhiteKernel | None = None,
    ):
        """
        Initialize a `SurfaceArray` object containing a surface, phase grid,
        wavelength grid, and kernel.

        Args:
            surface (np.ndarray): The final SED surface model,
                represented as a numpy array.
            phase_grid (np.ndarray): The grid of phases used to construct
                the surface model.
            wl_grid (np.ndarray): The grid of wavelengths used to construct
                the surface model.
            kernel (RBF | Matern | WhiteKernel | None, optional): The kernel
                used to construct the surface model. Defaults to None.
        """
        self.surface = surface[0]
        self.iqr = surface[1]

        self.phase_grid = phase_grid
        self.wl_grid = wl_grid
        self.kernel = kernel

    def predict(self, input: np.ndarray, return_std: bool = False):
        """
        Predict a light curve or SED from the SurfaceArray.
        Analogous to the `GaussianProcessRegressor.predict` method

        Args:
            input (np.ndarray): An mxn array of phase and wavelength values
                at which to predict relative magnitudes.
            return_std (bool, optional): Return the uncertainty in the
                predicted values. Defaults to False.

        Returns:
            tuple[np.ndarray | None]: The returned predicted magnitudes
                and (optional) uncertainties.
        """
        # Interpolate over the surface
        interp = RegularGridInterpolator(
            (self.phase_grid, self.wl_grid), self.surface.T
        )
        result = interp(input)

        if not return_std:
            return result, None

        interp_std = RegularGridInterpolator(
            (self.phase_grid, self.wl_grid), self.iqr.T
        )
        std = interp_std(input)

        return result, std


class SNModel:
    """
    A Supernova Model object. Saves the outputted 3D SED surface from the
    Gaussian process regression routine for a SN object or a collection of SNe.
    Provides routines for saving and loading the final GP fits, as well as
    for extracting 2D light curves or SEDs from the interpolated surfaces.
    """

    def __init__(
        self,
        surface: GaussianProcessRegressor | str | SurfaceArray | None = None,
        template_mags: np.ndarray | None = None,
        phase_grid: np.ndarray | None = None,
        phase_bounds: tuple | None = None,
        wl_grid: np.ndarray | None = None,
        filters_fit: list[str] | None = None,
        sn: SN | None = None,
        sncollection: SNType | SNCollection | None = None,
        norm_set: SNType | SNCollection | None = None,
        log_transform: int | float | bool = False,
    ):
        """
        Initialize an SNModel object from a number of arguments.

        Args:
            surface (GaussianProcessRegressor | str | SurfaceArray | None, optional): The
                final SED surface model. Can be either an initialized and fit
                `GaussianProcessRegressor` object, a `SurfaceArray` object, or filepath
                to a saved model. Defaults to None.
            template_mags (np.ndarray | None, optional): The magnitudes used to
                normalize the input photometry prior to fitting. Defaults to None.
            phase_grid (np.ndarray | None, optional): The phase grid used to
                produce the final SED surface model. Defaults to None.
            phase_bounds (tuple | None, optional): The minimum and maximum
                phase values provided to the fitting. Defaults to None.
            wl_grid (np.ndarray | None, optional): The wavelength grid used to
                produce the final SED surface model. Defaults to None.
            filters_fit (list[str] | None, optional): The specified filters of the photometry
                used in the fitting. Defaults to None.
            sn (SN | None, optional): The SN object to which this model belongs.
                Defaults to None.
            sncollection (SNType | SNCollection | None, optional): The collection of SN
                objects to which this model belongs. Defaults to None.
            norm_set (SNType | SNCollection | None, optional): The set of SN objects
                used in the normalization process during the fitting. Defaults to None.
            log_transform (int | float | bool, optional): The value of `log_transform` used
                in the fitting, if one was used. Defaults to False.

        Raises:
            ValueError: Must initialize an SNModel object by passing in either
                an SN or SNCollection, to which this model belongs.
        """
        self.base_path = os.path.join(ROOT_DIR, "data/final_models/")

        if type(surface) == str:
            ### This will load everything from the fits file
            self.load_from_fits(surface)

        else:
            if not sn and not sncollection:
                raise ValueError(
                    "Need to specify either a SN or SNCollection for this model!"
                )

            if sn:
                self.sn = sn
                self._initialize_surface_fit()
            if sncollection:
                self.collection = sncollection
            if norm_set:
                self.norm_set = norm_set

            self.surface = surface
            self.kernel = surface.kernel
            self.template = template_mags

            if (
                phase_grid is None
                and phase_bounds is not None
                and self.template is not None
            ):
                self.phase_grid = np.linspace(
                    phase_bounds[0], phase_bounds[1], self.template.shape[0]
                )
                self.min_phase, self.max_phase = phase_bounds
            else:
                self.phase_grid = phase_grid
                self.min_phase, self.max_phase = (
                    int(min(phase_grid)),
                    int(max(phase_grid)),
                )

            if (
                wl_grid is None
                and filters_fit is not None
                and self.template is not None
            ):
                self.wl_grid = np.linspace(
                    min(WLE[f] for f in filters_fit) - 500,
                    max(WLE[f] for f in filters_fit) + 500,
                    self.template.shape[1],
                )
            else:
                self.wl_grid = wl_grid
            self.min_wl, self.max_wl = int(min(self.wl_grid)), int(max(self.wl_grid))
            self.filters = filters_fit

            self.log_transform = log_transform

    def _initialize_surface_fit(self):
        """
        Initialize the Surface model by fitting the photometry that
        was used to create the model.
        This is necessary to produce sensible predictions for models
        that fit individual SN ojects.
        """
        data_cube_filename = os.path.join(
            self.sn.base_path,
            self.sn.classification,
            self.sn.subtype,
            self.sn.name,
            self.sn.name + "_datacube_mangled.csv",
        )

        if os.path.exists(data_cube_filename):
            cube = pd.read_csv(data_cube_filename)

            filtered_df = cube[
                (
                    cube["Phase"] > self.min_phase
                ) & (
                    cube["Phase"] < self.max_phase
                ) & (
                    cube["ShiftedWavelength"] > self.min_wl
                ) & (
                    cube["ShiftedWavelength"] < self.max_wl
                )
            ]

            phases_to_fit = filtered_df["Phase"].values
            wls_to_fit = filtered_df["ShiftedWavelength"].values
            shifted_mags = filtered_df["ShiftedFlux"].values
            residuals = []

            for phase, wl, mag in zip(phases_to_fit, wls_to_fit, shifted_mags):
                phase_ind = np.argmin(abs(phase - self.phase_grid))
                wl_ind = np.argmin(abs(wl - self.wl_grid))

                residuals.append(mag - self.template[phase_ind, wl_ind])

        x = np.vstack((np.log(phases_to_fit + self.log_transform), np.log10(wls_to_fit))).T
        y = residuals
        # TODO: Use actual errors here:
        self.surface.alpha = np.asarray(np.ones((len(residuals),))*0.1)
        self.surface.fit(x, y)

    def save_fits(self, filename: str = None, force: bool = False):
        """
        Save the SNModel as a .fits file. The .fits file will store
        the pickled SED surface model, the associated uncertainties,
        and all metadata passed in through the initialization. This
        format allows it to be easily loaded and used without redoing
        the fitting process.

        Args:
            filename (str, optional): The name of the file to
                be saved. Defaults to None.
            force (bool, optional): Overwrite an existing file, if one
                already exists with the same `filename`. Defaults to False.
        """
        if not self.surface:
            logger.warning(
                "Need to instantiate the SNModel class with a GP model to save it!"
            )
            return

        if not filename:
            try:
                filename = f"{self.sn.name}_GP_model.fits"
            except:
                filename = ""
                if hasattr(self.collection, "type"):
                    filename += self.collection.type
                    if hasattr(self.collection, "subtype"):
                        filename += f"_{self.collection.subtype}"
                else:
                    filename += f"{', '.join(sn.name for sn in self.collection.sne)}"

                filename += "_GP_model.fits"

        # Save the GP model, template grid, phase grid, and wavelength grid
        # as fits HDUs
        model_bytes = pickle.dumps(self.surface)
        model_array = np.frombuffer(model_bytes, dtype=np.uint8)
        model_array = model_array.reshape((1, -1))
        model_hdu = fits.PrimaryHDU(model_array)

        model_hdu.header["FILTERS_FIT"] = ",".join(self.filters)
        model_hdu.header["LOG_TRANSFORM"] = self.log_transform
        if hasattr(self, "sn"):
            model_hdu.header["OBJECTS"] = self.sn.name
        else:
            model_hdu.header["OBJECTS"] = ",".join(
                [sn.name for sn in self.collection.sne]
            )
        model_hdu.header["NORM_SET"] = ",".join([sn.name for sn in self.norm_set.sne])
        if isinstance(self.surface, GaussianProcessRegressor):
            model_hdu.header["TYPE"] = "Gaussian Process Regressor"
        elif isinstance(self.surface, SurfaceArray):
            model_hdu.header["TYPE"] = "Numpy Array"

        template_hdu = fits.ImageHDU(self.template, name="TEMPLATE")

        phase_array = self.phase_grid
        phase_hdu = fits.ImageHDU(phase_array, name="PHASE ARRAY")

        wavelength_array = self.wl_grid
        wavelength_hdu = fits.ImageHDU(wavelength_array, name="WAVELENGTH ARRAY")

        hdul = fits.HDUList([model_hdu, template_hdu, phase_hdu, wavelength_hdu])
        hdul.writeto(os.path.join(self.base_path, filename), overwrite=force)
        hdul.close()

    def load_from_fits(self, filename: str):
        """
        Load a specified .fits file as an `SNModel` object.
        This is used during the initialization of the object,
        if a string is passed as the `surface` argument.
        Loading the saved .fits file will initialize the
        underlying SED surface model and load all
        associated metadata.

        Args:
            filename (str): The path to and name of the .fits file.

        """
        with fits.open(os.path.join(self.base_path, filename)) as hdul:
            surface = pickle.loads(hdul[0].data)
            log_transform = hdul[0].header["LOG_TRANSFORM"]
            filters_fit = hdul[0].header["FILTERS_FIT"]
            object_names = hdul[0].header["OBJECTS"]
            norm_set_names = hdul[0].header["NORM_SET"]

            try:
                self.kernel = surface.kernel
            except:
                print("No Kernel, need to implement")

            template = hdul[1].data
            phase_grid = hdul[2].data
            wl_grid = hdul[3].data

        self.surface = surface
        self.template = template
        self.phase_grid = phase_grid
        self.min_phase, self.max_phase = min(phase_grid), max(phase_grid)
        self.wl_grid = wl_grid
        self.min_wl, self.max_wl = min(wl_grid), max(wl_grid)

        self.log_transform = log_transform
        self.filters = filters_fit
        if "," not in object_names:
            # Only one object in our sample, so load it as a SN object
            self.sn = SN(name=object_names)
            self.collection = None
            self._initialize_surface_fit()
        else:
            self.collection = SNCollection(names=object_names.split(","))
            self.sn = None

        self.norm_set = SNCollection(names=norm_set_names.split(","))

    def predict_lightcurve(
        self,
        phase_min: float | int,
        phase_max: float | int,
        wavelength: float | int,
        show=True,
    ):
        """
        Predict a light curve using the stored surface model.
        Utilizes the `predict()` method on `self.surface` to predict
        relative magnitudes given phase bounds and wavelength.
        Works for both stored and initialized `GaussianProcessRegressor`
        objects as well as `SurfaceArray` objects.

        Args:
            phase_min (float | int): The minimum phase to predict,
                i.e. the first point of the light curve.
            phase_max (float | int): The maximum phase to predict,
                i.e. the last point of the light curve.
            wavelength (float | int): The wavelength at which to
                predict the light curve.
            show (bool, optional): Plot the predicted light curve.
                Defaults to True.

        Raises:
            ValueError: If the input phase_min or phase_max are
                outside the fitted bounds of the Gaussian process.
            ValueError: If the input wavelength is outside the
                fitted bounds of the Gaussian Process.
        """
        if phase_max > self.max_phase or phase_min < self.min_phase:
            raise ValueError("Phases need to be within the bounds of the GP")
        if wavelength < self.min_wl or wavelength > self.max_wl:
            raise ValueError("Wavelength needs to be within the bounds of the GP")

        linear_phases = np.linspace(
            phase_min,
            phase_max,
            int(max(min(len(self.phase_grid) / 2, len(self.wl_grid) / 2), 40)),
        )
        phases = np.log(linear_phases + self.log_transform)
        waves = np.ones(len(phases)) * np.log10(wavelength)

        prediction, dev = self.surface.predict(
            np.vstack((phases, waves)).T, return_std=True
        )

        # Add back on template mag for the correct phase and wavelength inds
        if self.template is not None:
            template_lc = []
            wl_ind = np.argmin((abs(self.wl_grid - wavelength)))
            for i in range(len(phases)):
                phase_ind = np.argmin((abs(self.phase_grid - linear_phases[i])))
                template_lc.append(self.template[phase_ind, wl_ind])
            template_lc = np.asarray(template_lc)

        else:
            template_lc = np.zeros(len(prediction))

        plt.plot(linear_phases, prediction + template_lc)
        plt.plot(
            linear_phases,
            prediction + template_lc - 1.96 * dev,
            alpha=0.2,
            color="blue",
        )
        plt.plot(
            linear_phases,
            prediction + template_lc + 1.96 * dev,
            alpha=0.2,
            color="blue",
        )
        plt.xlabel("Phase (days)")
        plt.ylabel("Log10(Flux) Relative to Peak")
        plt.title(f"Light curve at {wavelength} Angstroms")
        if show:
            plt.show()

    def predict_sed(self, wavelength_min, wavelength_max, phase, show=True):
        """
        Predict a spectral energy distribution (SED) using the stored surface model.
        Utilizes the `predict()` method on `self.surface` to predict
        relative magnitudes given wavelength bounds and phase.
        Works for both stored and initialized `GaussianProcessRegressor`
        objects as well as `SurfaceArray` objects.

        Args:
            wavelength_min (float | int): The minimum wavelength to predict,
                i.e. the first point of the SED.
            phase_max (float | int): The maximum wavelength to predict,
                i.e. the last point of the SED.
            phase (float | int): The phase at which to
                predict the SED.
            show (bool, optional): Plot the predicted light curve.
                Defaults to True.

        Raises:
            ValueError: If the input wavelength_min or wavelength_max are
                outside the fitted bounds of the Gaussian process.
            ValueError: If the input phase is outside the
                fitted bounds of the Gaussian Process.
        """
        if wavelength_max > self.max_wl or wavelength_min < self.min_wl:
            raise ValueError("Wavelengths need to be within the bounds of the GP")
        if phase < self.min_phase or phase > self.max_phase:
            raise ValueError("Phase needs to be within the bounds of the GP")

        linear_waves = np.linspace(
            wavelength_min,
            wavelength_max,
            int(min(len(self.phase_grid) / 2, len(self.wl_grid) / 2)),
        )
        waves = np.log10(linear_waves)
        phases = np.ones(len(waves)) * np.log(phase + self.log_transform)

        prediction, dev = self.surface.predict(
            np.vstack((phases, waves)).T, return_std=True
        )

        # Add back on template mag for the correct phase and wavelength inds
        if self.template is not None:
            template_lc = []
            phase_ind = np.argmin((abs(self.phase_grid - phase)))
            for i in range(len(waves)):
                wl_ind = np.argmin((abs(self.wl_grid - linear_waves[i])))
                template_lc.append(self.template[phase_ind, wl_ind])
            template_lc = np.asarray(template_lc)
        else:
            template_lc = np.zeros(len(prediction))

        plt.plot(linear_waves, prediction + template_lc)
        plt.plot(
            linear_waves, prediction + template_lc - 1.96 * dev, alpha=0.2, color="blue"
        )
        plt.plot(
            linear_waves, prediction + template_lc + 1.96 * dev, alpha=0.2, color="blue"
        )
        plt.xlabel("Phase (days)")
        plt.xlabel("Wavelength (Angstrom)")
        plt.ylabel("Log10(Flux) Relative to Peak")
        plt.title(f"SED at {phase} days")
        if show:
            plt.show()

    def predict_photometry_points(
        self, wavelengths: np.ndarray, phases: np.ndarray, show: bool = True, **kwargs
    ):
        """
        Predict a series of photometry points given arrays of wavelength and phase.
        Utilizes the `predict()` method on `self.surface` to predict
        relative magnitudes at discrete points of wavelength and phase.
        Works for both stored and initialized `GaussianProcessRegressor`
        objects as well as `SurfaceArray` objects.

        Example:
            To predict points in two filters at each of three phases,
            `wavelengths` and `phases` must be length 6 arrays, e.g.
            `wavelengths = [3500, 3500, 3500, 5000, 5000, 5000]`
            `phases = [-10, 0, 10, -10, 0, 10]`


        Args:
            wavelengths (np.ndarray): array of wavelengths,
                in Angstroms.
            phases (np.ndarray): array of phases, in days.
            show (bool, optional): Plot the predicted
                photometry points. Defaults to True.

        Raises:
            ValueError: If any input wavelength values are outside
                the bounds of the Gaussian process fit.
            ValueError: If any input phase values are outside
                the bounds of the Gaussian process fit.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The predicted phases,
                magnitudes, and uncertainties of the photometry points.
        """
        if any(wavelengths > max(self.wl_grid)) or any(wavelengths < min(self.wl_grid)):
            raise ValueError("Wavelengths need to be within the bounds of the GP")
        if any(phases < self.min_phase) or any(phases > self.max_phase):
            raise ValueError("Phase needs to be within the bounds of the GP")

        log_phases = np.log(phases + self.log_transform)
        log_waves = np.log10(wavelengths)

        predicted_lc, dev = self.surface.predict(
            np.vstack((log_phases, log_waves)).T, return_std=True
        )

        # For each predicted point, define a Gaussian distribution with uncertainty=dev and sample from it
        prediction = []
        for i in range(len(predicted_lc)):
            prediction.append(np.random.normal(predicted_lc[i], dev[i], 1)[0])
        prediction = np.asarray(prediction)

        # Add back on template mag for the correct phase and wavelength inds
        if self.template is not None:
            template_lc = []
            for i in range(len(phases)):
                phase_ind = np.argmin((abs(self.phase_grid - phases[i])))
                wl_ind = np.argmin((abs(self.wl_grid - wavelengths[i])))
                template_lc.append(self.template[phase_ind, wl_ind])
            template_lc = np.asarray(template_lc)

        else:
            template_lc = np.zeros(len(prediction))

        plt.errorbar(
            phases,
            prediction + template_lc,
            yerr=dev,
            fmt="o",
            color=kwargs.get("color", "k"),
        )
        plt.xlabel("Phase (days)")
        plt.ylabel("Log10(Flux) Relative to Peak")
        plt.title("Predicted Photometry Points")
        if show:
            plt.show()

        return phases, prediction + template_lc, dev

    def compare_lightcurve_with_photometry(self, sn: SN, filt: str, show: bool = True):
        """
        Compare a predicted light curve with observed photometry.
        Useful when validating the accuracy of a model's predicted light curve.

        Args:
            sn (SN): A SN object with observed photometry.
            filt (str): The filter of the light curve to compare.
            show (bool, optional): Plot the resulting comparison.
                Defaults to True.
        """
        datacube = DataCube(sn=sn)
        datacube.construct_cube()
        cube = datacube.cube

        filtered_cube = cube[
            (cube["ShiftedFilter"] == filt)
            & (cube["Phase"] > self.min_phase)
            & (cube["Phase"] < self.max_phase)
        ]

        observed_phases = filtered_cube["Phase"].values
        observed_fluxes = filtered_cube["ShiftedMag"].values
        observed_flux_errs = filtered_cube["ShiftedFluxerr"].values

        plt.errorbar(
            observed_phases,
            observed_fluxes,
            yerr=observed_flux_errs,
            color="lime",
            fmt="o",
            mec="k",
            label=sn.name,
        )
        plt.legend()
        if show:
            plt.show()

    def fit_photometry(
        self,
        photometry: dict | pd.DataFrame,
        phase_min: float | None = None,
        phase_max: float | None = None,
        show: bool = False,
        nsamples: int = 1,
    ):
        """
        Fit input photometry using the GaussianProcessRegressor model.
        If a phase min or phase max is specified, extrapolates the fit to those bounds.

        Args:
            photometry (dict | pd.DataFrame): The input photometry
                to fit. Must be a dict or DataFrame that contains
                these columns: Filter, Phase, Mag, and MagErr
            phase_min (float, optional): The minimum phase to constrain our
                GP prediction. Defaults to None.
            phase_max (float, optional): The maximum phase to constrain
                our GP prediction. Defaults to None.
            show (bool, optional): Plot the resulting fit. Defaults to False.
            nsamples (int, optional): Number of samples to draw from the GP
                for the fit. If 1, plots the usual GP prediction with error bars
                If >1, plots nsamples of randomly drawn GP fits. Defaults to 1.
        """
        if self.sn is not None:
            sn = self.sn
        else:
            # We just need a SN object to convert to fluxes, any will do
            sn = self.collection.sne[0]

        if isinstance(photometry, dict):
            try:
                photometry = pd.DataFrame(photometry)
            except Exception as e:
                raise ValueError(
                    "Either provide photometry as a DataFrame or in a valid dictionary",
                    e,
                )

        if nsamples < 1:
            raise ValueError("Number of samples must be >= 1")

        ### Get residuals for the photometry from the saved template grid
        residuals = []
        for filt in list(set(photometry["Filter"].values)):
            mags = photometry.loc[photometry["Filter"] == filt]["Mag"].values
            errs = photometry.loc[photometry["Filter"] == filt]["MagErr"].values
            phases = photometry.loc[photometry["Filter"] == filt]["Phase"].values
            current_wls = np.ones(len(phases)) * WLE[filt]

            if len(phases) > 0:
                for i, phase in enumerate(phases):
                    ### Get index of current phase in phase grid
                    ### The phase corresponding to phase_ind is no more than the phase grid spacing away from the true phase being measured
                    phase_ind = np.argmin(abs(self.phase_grid - phase))

                    wl_ind = np.argmin(abs(self.wl_grid - current_wls[i]))

                    if not np.isnan(self.template[phase_ind, wl_ind]) and not np.isinf(
                        mags[i] - self.template[phase_ind, wl_ind]
                    ):
                        residuals.append(
                            {
                                "Filter": filt,
                                "Phase": phase,
                                "MagErr": errs[i],
                                "Mag": mags[i],
                            }
                        )
        residuals = pd.DataFrame(residuals)
        if len(residuals) == 0:
            raise ValueError("Photometry not within bounds of this GP")

        ### Fit the photometry with the GP model
        err = residuals["MagErr"].values

        gp = self.surface
        gp.alpha = err

        ### Predict lightcurves given the GP fit
        if not phase_min:
            phase_min = min(residuals["Phase"].values)
        if not phase_max:
            phase_max = max(residuals["Phase"].values)

        _, ax = plt.subplots()
        for filt in list(set(residuals["Filter"].values)):
            test_times_linear = np.arange(phase_min, phase_max, 1.0 / 24)
            test_times = np.log(test_times_linear + self.log_transform)
            test_waves = np.ones(len(test_times)) * np.log10(WLE[filt])

            wl_ind = np.argmin(abs(self.wl_grid - WLE[filt]))
            template_mags = []
            for i in range(len(test_times_linear)):
                j = np.argmin(abs(self.phase_grid - test_times_linear[i]))
                template_mags.append(self.template[j, wl_ind])
            template_mags = np.asarray(template_mags)

            if nsamples == 1:
                test_prediction, std_prediction = gp.predict(
                    np.vstack((test_times, test_waves)).T, return_std=True
                )
            elif nsamples > 1:
                # TODO: Create a SurfaceArray.sample_y method
                samples = gp.sample_y(
                    np.vstack((test_times, test_waves)).T, n_samples=nsamples
                )

            test_times = np.exp(test_times) - self.log_transform
            residuals_for_filt = residuals[residuals["Filter"] == filt]

            if nsamples == 1:
                residuals_for_filt["Phase"] = np.log(
                    residuals_for_filt["Phase"].values + self.log_transform
                )

                Plot().plot_run_gp_overlay(
                    ax=ax,
                    test_times=test_times,
                    test_prediction=test_prediction,
                    std_prediction=std_prediction,
                    template_mags=template_mags,
                    residuals=residuals_for_filt,
                    log_transform=self.log_transform,
                    filt=filt,
                    sn=sn,
                )
            else:
                for sample in samples.T:
                    log_fluxes = sample + template_mags
                    shifted_mags = convert_shifted_fluxes_to_shifted_mags(
                        log_fluxes, sn, sn.zps[filt]
                    )

                    ax.plot(
                        test_times, shifted_mags, color=colors.get(filt, "k"), alpha=0.2
                    )
                    ax.errorbar(
                        residuals_for_filt["Phase"].values,
                        residuals_for_filt["Mag"].values,
                        yerr=residuals_for_filt["MagErr"].values,
                        fmt="o",
                        color=colors.get(filt, "k"),
                        mec="k",
                    )

        if show:
            plt.show()
