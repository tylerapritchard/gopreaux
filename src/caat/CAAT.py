import logging
import os
import warnings

import numpy as np
import pandas as pd

from caat.utils import ROOT_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore")


class CAAT:
    """
    The CAAT class stores and retrieves metadata about the transients
    in this repository.

    This class provides functionality to retrieve the types, redshifts,
    coordinates, and peak information for the transients. It loads the
    saved metadata from the appropriate .csv file. This class also
    contains helper functions to create new .csv files, or merge
    existing files into one.
    """
    def __init__(self, filename: str = "caat.csv"):
        """
        Initialize a CAAT object to handle an existing caat database file.
        The database file should live within the `data/` directory in this
        repository and be a csv file with the necessary metadata. This method
        loads the existing file as a pandas DataFrame, or raises a Warning
        if none is found.

        Args:
            filename (str, optional): The name of the caat .csv file. 
            Defaults to "caat.csv".

        Raises:
            Warning: No CAAT database file could be found.
        """
        base_path = os.path.join(ROOT_DIR, "data/")
        db_loc = base_path + filename

        if os.path.isfile(db_loc):
            # Chech to see if db file exists
            self.caat = pd.read_csv(db_loc)
        else:
            raise Warning("No database file found")

    def get_sne_by_type(self, sntype: str, snsubtype: str | None = None):
        """
        Return a list of all the transients of a given type and subtype.

        Args:
            sntype (str): The type of the objects to load.
            snsubtype (str, optional): The optional subtype of the objects
                to load. Should be a valid subtype of `sntype`. Defaults to None.

        Returns:
            list: A list of the names of the filtered transients.
        """
        if snsubtype is not None:
            sne_list = (self.caat["Type"] == sntype) & (
                self.caat["Subtype"] == snsubtype
            )
        else:
            sne_list = self.caat["Type"] == sntype
        return self.caat[sne_list].Name.values

    @staticmethod
    def save_db_file(db_loc: str, sndb: pd.DataFrame, force=False):
        """
        Save the CAAT database file to disk.

        Args:
            db_loc (str): The filepath to save the new CAAT database file.
            sndb (pd.DataFrame): The CAAT database file, as a pandas DataFrame.
            force (bool, optional): Overwrite an existing file. Defaults to False.
        """
        if not force and os.path.exists(db_loc):
            logger.warning(
                "WARNING: CAAT file with this name already exists. To overwrite, use force=True"
            )

        else:
            sndb.to_csv(db_loc, index=False)

    @staticmethod
    def read_info_from_tns_file(tns_file: str, sn_names: list, col_names: list):
        """
        Retrieve transient metadata from a TNS .csv file.
        The TNS file is available on the TNS website daily.
        This method retrieves the necessary metadata from the file,
        specified by the `col_names` parameter, for an input list of
        transient names. 

        Args:
            tns_file (str): The file path to a TNS .csv file.
            sn_names (list): The list of names of transients to search for in
                the TNS file.
            col_names (list): The names of the columns to extract from the TNS file.

        Returns:
            dict: A dictionary of the retrieved column values.
        """
        # First, sanitize the names if not in TNS format (no SN or AT)
        sn_names = [sn_name.replace("SN", "").replace("AT", "") for sn_name in sn_names]

        ### This is really gross but works for now
        ### In the future, want to make this more pandas-esque and
        ### be able to handle multiple col_names at once
        tns_df = pd.read_csv(tns_file)
        tns_values = {}

        for col_name in col_names:
            tns_values[col_name] = []
            for name in sn_names:
                row = tns_df[tns_df["name"] == name]
                if len(row.values) == 0:
                    tns_values[col_name].append(np.nan)
                else:
                    tns_value = row[col_name].values[0]
                    if not tns_value:
                        tns_values[col_name].append(np.nan)
                    else:
                        tns_values[col_name].append(tns_value)

        return tns_values

    @classmethod
    def create_db_file(
        cls, type_list: list | None = None, base_db_name="caat.csv", tns_file="", force=False
    ):
        """
        Create a CAAT database file given a TNS csv file.
        Optionally provide a list of transient types to limit the
        database file to.

        Args:
            type_list (list, optional): An optional list of transient types to load.
                If none are provided, a preset set of types will be loaded. Defaults to None.
            base_db_name (str, optional): The name of the file to save. Defaults to "caat.csv"
            tns_file (str, optional): The name of the TNS file to parse for transient metadata,
                if one exists. Defaults to "".
            force (bool, optional): Overwrite an existing CAAT database file, if one exists by
                the same name. Defaults to False.
        """      
        base_path = os.path.join(ROOT_DIR, "data/")
        db_loc = base_path + base_db_name

        # Create A List Of Folders To Parse
        if type_list is None:
            type_list = ["SESNe", "SLSN-I", "SLSN-II", "SNII", "SNIIn", "FBOT", "Other"]

        sndb_name = []
        sndb_type = []
        sndb_subtype = []
        sndb_z = []
        sndb_tmax = []
        sndb_mmax = []
        sndb_filtmax = []
        sndb_ra = []
        sndb_dec = []

        # etc
        for sntype in type_list:
            """For each folder:
            get a list of subfolders
            get a list of objects in each folder
            assign SN, subtypes to list
            """
            subtypes = os.listdir(base_path + sntype + "/")
            for snsubtype in subtypes:
                sn_names = os.listdir(base_path + sntype + "/" + snsubtype + "/")

                sndb_name.extend(sn_names)
                sndb_type.extend([sntype] * len(sn_names))
                sndb_subtype.extend([snsubtype] * len(sn_names))

                if tns_file:
                    tns_info = cls.read_info_from_tns_file(
                        tns_file, sn_names, ["redshift", "ra", "declination"]
                    )
                    tns_z = tns_info["redshift"]
                    tns_ra = tns_info["ra"]
                    tns_dec = tns_info["declination"]
                else:
                    tns_z = [np.nan] * len(sn_names)
                    tns_ra = [np.nan] * len(sn_names)
                    tns_dec = [np.nan] * len(sn_names)

                sndb_z.extend(tns_z)
                sndb_ra.extend(tns_ra)
                sndb_dec.extend(tns_dec)

                sndb_tmax.extend([np.nan] * len(sn_names))
                sndb_mmax.extend([np.nan] * len(sn_names))
                sndb_filtmax.extend([""] * len(sn_names))

        sndb = pd.DataFrame(
            {
                "Name": sndb_name,
                "Type": sndb_type,
                "Subtype": sndb_subtype,
                "Redshift": sndb_z,
                "RA": sndb_ra,
                "Dec": sndb_dec,
                "Tmax": sndb_tmax,
                "Magmax": sndb_mmax,
                "Filtmax": sndb_filtmax,
            }
        )

        cls.save_db_file(db_loc, sndb, force=force)

    @classmethod
    def combine_db_files(cls, file1: str, file2: str, outfile: str):
        """
        Combine multiple CAAT database files into one.
        Assumes that the second file passed has the same columns
        as the first and in the same order.

        Args:
            file1 (str): The filepath of the first database file.
            file2 (str): The filepath of the second database file.
            outfile (str): The filepath of the output database file.
        """
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        merged = df1.combine_first(df2)

        ### Want to reorder columns in nicer way
        ### Here we are assuming df2 has all the columns in the merged df
        ### and in the correct order
        merged = merged[df2.columns.tolist()]
        cls.save_db_file(outfile, merged)

    @property
    def db(self):
        return self.caat
