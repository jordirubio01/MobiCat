import os
import importlib
import sys
import subprocess
import pandas as pd
from typing import Literal

def install_if_missing(package):
    try:
        importlib.import_module(package)
        print(f"Package {package} already installed.")
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Installing package {package}...")


def install_missing_packages(packages):
    for package in packages:
        install_if_missing(package)


def to_float(s):
    return float(s.replace(",", "."))


def ine_to_idescat(codimuni: pd.Series) -> pd.Series:
    return codimuni.apply(lambda s: s[0:5])


def group_by_municipality_type(df: pd.DataFrame, _type: Literal["origen", "destino"]) -> pd.DataFrame:
    """
    Reduces the mobility dataframe to origin or destination municipalities.

    Parameters:
        df (DataFrame): A dataframe of type #3 (Movilidad Municipios).
        _type (Literal): A string that must be "origen" or "destino".

    Returns:
        DataFrame: If ``_type`` is "origen", it returns a dataframe with columns 
        "municipio_origen_name", "municipio_origen", "viajes", where "viajes" is 
        the sum of the out-trips from the municipality. Otherwise, it returns a
        dataframe with columns "municipio_destino", "municipio_destino_name", "viajes", 
        where "viajes" is the sum of the in-trips to the municipality.
    """
    return pd.DataFrame(df
        .groupby([f"municipio_{_type}", f"municipio_{_type}_name"])["viajes"]
        .sum()
        .reset_index()
    )


def group_by_municipality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduces the mobility dataframe to municipalities.

    Parameters:
        df (DataFrame): A dataframe of type #3 (Movilidad Municipios).

    Returns:
        DataFrame: A dataframe grouped by municipalities with columns
        ``municipio``, ``municipio_name``, ``viajes``, where "viajes"
        the sum of the out- and in- trips.
    """
    origin_df = (group_by_municipality_type(df, "origen")
        .rename(columns={
            "municipio_origen": "municipio", 
            "municipio_origen_name": "municipio_name"
        })
    )

    destination_df = (group_by_municipality_type(df, "destino")
        .rename(columns={
            "municipio_destino": "municipio", 
            "municipio_destino_name": "municipio_name",
        })
    )

    return pd.DataFrame(pd.concat([origin_df, destination_df])
        .groupby(["municipio", "municipio_name"])["viajes"]
        .sum()
        .reset_index()
    )


def group_by_origin_destination_directed(mobility_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduces the mobility dataframe to ordered pairs of municipalities.

    Parameters:
        df (DataFrame): A dataframe of type #3 (Movilidad Municipios).

    Returns:
        DataFrame: A dataframe with columns ``"municipio_origen"``, ``"municipio_destino"``,
        ``"viajes"``, where ``"viajes"`` is the total number of trips for this oreded pair.
    """
    return pd.DataFrame(mobility_df
        .groupby(["municipio_origen", "municipio_destino"])["viajes"]
        .sum()
        .reset_index()
    )

def group_by_origin_destination_undirected(mobility_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduces the mobility dataframe to unordered pairs of municipalities.

    Parameters:
        df (DataFrame): A dataframe of type #3 (Movilidad Municipios).

    Returns:
        DataFrame: A dataframe with columns ``"municipio_origen"``, ``"municipio_destino"``,
        ``"viajes"``, where ``"viajes"`` is the total number of trips for this unordered pair.
    """

    directed_edges_df = group_by_origin_destination_directed(mobility_df)

    undirected_edges = []
    for _, node in directed_edges_df.iterrows():
        municipality_1, municipality_2 = tuple(sorted([
            node["municipio_origen"], 
            node["municipio_destino"]
        ]))
        undirected_edges.append({
            "municipio_1": municipality_1,
            "municipio_2": municipality_2,
            "viajes": node["viajes"]
        })

    return (pd.DataFrame(undirected_edges)
        .groupby(["municipio_1", "municipio_2"])["viajes"]
        .sum()
        .reset_index()
    )
