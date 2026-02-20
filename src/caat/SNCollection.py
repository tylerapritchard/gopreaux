import logging
import warnings

from .CAAT import CAAT
from .Plot import Plot
from .SN import SN

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore")


class SNCollection:
    """
    A SNCollection object, which holds an arbitrary number of SNe.
    
    The SNCollection object is the primary way of storing and passing
    a number of SN objects to the fitting routines in this package.
    SNCollection objects can be initialized with either a list of names,
    each corresponding to a SN recorded in the CAAT file, a list of
    SN objects, or a type and subtype. In the latter case, the corresponding
    SN objects are automatically found and instantiated.
    """

    base_path = "../data/"

    def __init__(
        self,
        names: str | list | None = None,
        sntype: str | None = None,
        snsubtype: str | None = None,
        SNe: list[SN] | None = None,
        **kwargs,
    ):
        """
        Initialize a SNCollection object to store a collection of SNe. 
        Can be initalized with either a list of names or SNe objects,
        or with a type and subtype.

        Args:
            names (str | list | None, optional): A list of names, each
                corresponding to a SN object described in the CAAT file.
                Defaults to None.
            sntype (str | None, optional): The classification of SNe to instantiate.
                Defaults to None.
            snsubtype (str | None, optional): The subtype of SNe to instantiate. Must be
                a valid subtype of `sntype`. Defaults to None.
            SNe (list[SN] | None, optional): The SN or SNe to store. Defaults to None.
        """
        self.subtypes = list(kwargs.keys())

        if isinstance(SNe, SN):
            self.sne = [SNe]
        else:
            if isinstance(SNe, list):
                self.sne = [sn for sn in SNe]
            elif isinstance(names, list):
                self.sne = [SN(name) for name in names]
            else:
                if type(sntype) is not None:
                    logger.info(f"Loading SN Type: {sntype}, Subtype: {snsubtype}")
                    caat = CAAT()
                    type_list = caat.get_sne_by_type(sntype, snsubtype)
                    logger.info(type_list)
                    self.sne = [SN(name) for name in type_list]
                    self.type = sntype
                    self.subtype = snsubtype

    def __repr__(self):
        """
        Return the SNe that make up this object.

        Returns:
            list[SN]: The SNe in this object.
        """
        return self.sne

    def plot_all_lcs(
        self, filts=["all"], log_transform=False, plot_fluxes=False, ax=None, show=True
    ):
        """
        Plot all light curves for the SNe in this SNCollection object.
        Can choose a subset of filters to plot, or to plot all filter data.
        Can plot log-transformed or natural light curves, fluxes or magnitudes.

        Args:
            filts (list, optional): The filter(s) to plot. Defaults to ["all"].
            log_transform (bool, optional): Log transform the time axis. Defaults to False.
            plot_fluxes (bool, optional): Plot photometry in flux space,
                instead of magnitude space. Defaults to False.
            ax (_type_, optional): Provide an existing `matplotlib.axes` object.
                Used to customize plotting. Defaults to None.
            show (bool, optional): Show the light curves. Defaults to True.
        """
        Plot().plot_all_lcs(
            sn_class=self,
            filts=filts,
            log_transform=log_transform,
            plot_fluxes=plot_fluxes,
            ax=ax,
            show=show,
        )


class SNType(SNCollection):
    """
    A SNType object, building a collection of all SNe of a given type (classification).

    SNType inherits SNCollection and provides more tailored methods for all objects
    of a given classification.
    """

    subtypes = []
    sne = []

    def __init__(self, type):
        """
        Initialize a SNType object.

        Args:
            type (str): The classification of the objects to load.
                Will automatically retrieve all subtypes of the given type
                and load the correct SN objects.
        """
        self.type = type

        self.get_subtypes()
        self.build_object_list()

    def get_subtypes(self):
        """
        Get all subtypes of the given type from the CAAT file.
        """
        caat = CAAT()
        subtype_list = caat.caat["Type"] == self.type
        self.subtypes = list(set(caat.caat[subtype_list].Subtype.values))

    def build_object_list(self):
        """
        Load all objects of the given type from the CAAT file.
        First finds all subtypes, then loads all SN objects of those
        type/subtypes. 
        """
        caat = CAAT()
        type_list = caat.get_sne_by_type(self.type)
        self.sne = [SN(name) for name in type_list]
