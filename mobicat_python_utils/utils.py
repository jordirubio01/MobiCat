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


WEEK_COLOR: list[tuple] = [
    ('Lunes',     '#08306b'), 
    ('Martes',    '#2171b5'),  
    ('Miércoles', '#4292c6'),  
    ('Jueves',    '#6baed6'), 
    ('Viernes',   '#9ecae1'),  
    ('Sábado',    '#006400'), 
    ('Domingo',   '#32CD32')
]

MONTH_COLOR: list[tuple] = [
    ("01", "#182F5D"),
    ("02", "#29509E"),
    ("03", "#4784FF"),
    ("04", "#165623"),
    ("05", "#27993D"),
    ("06", "#3FFF65"),
    ("07", "#514816"),
    ("08", "#928229"),
    ("09", "#D9C13D"),
    ("10", "#792929"),
    ("11", "#B53C3C"),
    ("12", "#FF7676"),
]

YEAR_COLOR: list[tuple] = [
    ("2023", "#29509E"),
    ("2024", "#928229"),
    ("2025", "#B53C8F")
]

def get_week_color(week_color: tuple[str, str] = WEEK_COLOR) -> pd.DataFrame:
    return pd.DataFrame(data=week_color, columns=["day_of_week", "day_of_week_color"])

def get_datasets_names(all_datasets_directory: str) -> pd.DataFrame:

    if not os.path.exists(all_datasets_directory):
        raise Exception("Path %s does not exists" % all_datasets_directory)

    datasets_names = []
    for dirpath, dirnames, filenames in os.walk(all_datasets_directory):
        if len(dirnames) > 0:
            continue
        year_month = str.split(dirpath, "\\")[-1]
        year, month = tuple(str.split(year_month, "-"))
        barrios, mun_barrios, municipios = tuple(sorted(filenames))
        datasets_names.append({
            "year": year,
            "month": month,
            "barrios": os.path.join(all_datasets_directory, year_month, barrios),
            "mun_barrios": os.path.join(all_datasets_directory, year_month, mun_barrios),
            "municipios": os.path.join(all_datasets_directory, year_month, municipios)
        })

    return (pd.DataFrame(datasets_names)
        .sort_values(by=["year", "month"])
        .reset_index()
        .drop(columns="index")
    )


def group_by_day(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(df
        .groupby(by=['day', 'day_of_week', "month"])["viajes"]
        .sum()
        .reset_index()
    )


def group_by_day_full_datasets(full_datasets_directory: str) -> pd.DataFrame:
    """
    Creates a dataframe with all datasets of type #3 of Telefonica 
    "grouped" by day using the sum of "viajes".
    """
    datasets_names_df = get_datasets_names(full_datasets_directory)

    concatenated_df = pd.DataFrame(columns=['day', 'day_of_week', "month", "viajes"])
    for municipios in datasets_names_df["municipios"]:
        municipios_df = pd.read_csv(municipios)
        day_municipios_df = group_by_day(municipios_df)
        concatenated_df = pd.concat([concatenated_df, day_municipios_df])
    return concatenated_df.reset_index()


def filter_by_day(
        day_df: pd.DataFrame, 
        start: str = "2023-01-01", 
        end: str = "2023-01-31"
    ) -> pd.DataFrame:
    return day_df[(day_df["day"] >= pd.Timestamp(start)) 
                  & (day_df["day"] <= pd.Timestamp(end))]     

def filter_day_by_year_month(
        day_df: pd.DataFrame, 
        year: int = 2023, 
        month: int = 1
    ) -> pd.DataFrame:
    return day_df[(day_df["day"].dt.year == year) 
                  & (day_df["day"].dt.month == month)]
