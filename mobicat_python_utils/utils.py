import os
import importlib
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal

WEEK_COLOR: list[tuple] = [
    ('Lunes',     '#08306b'), 
    ('Martes',    '#2171b5'),  
    ('Miércoles', '#4292c6'),  
    ('Jueves',    '#6baed6'), 
    ('Viernes',   '#9ecae1'),  
    ('Sábado',    '#006400'), 
    ('Domingo',   '#22CD32')
]

MONTH_COLOR: list[tuple] = [
    ("01", "#182F5D"),
    ("02", "#29509E"),
    ("03", "#4784FF"),
    ("04", "#165623"),
    ("05", "#27993D"),
    ("06", "#2FFF65"),
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
        df (DataFrame): A dataframe of type #2 (Movilidad Municipios).
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
        df (DataFrame): A dataframe of type #2 (Movilidad Municipios).

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
        df (DataFrame): A dataframe of type #2 (Movilidad Municipios).

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
        df (DataFrame): A dataframe of type #2 (Movilidad Municipios).

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

def get_week_color(week_color: tuple[str, str] = WEEK_COLOR) -> pd.DataFrame:
    return pd.DataFrame(data=week_color, columns=["day_of_week", "day_of_week_color"])

def get_datasets_names(all_datasets_directory: str) -> list:
    """
    """
    if not os.path.exists(all_datasets_directory):
        raise Exception("Path %s does not exists" % all_datasets_directory)

    file_types = ["barrios", "mun_barrios", "municipios"]
    datasets_names = []
    for dirpath, dirnames, filenames in os.walk(all_datasets_directory):
        if len(dirnames) > 0:
            continue
        year_month = str.split(dirpath, "\\")[-1]
        year, month = tuple(str.split(year_month, "-"))
        dataset_dir = os.path.join(all_datasets_directory, year_month)
        file_paths = [os.path.join(dataset_dir, f) for f in sorted(filenames)]
        file_sizes = [os.path.getsize(f) for f in file_paths]
        for i in range(len(file_types)):
            datasets_names.append({
                "year": year,
                "month": month,
                "type": file_types[i],
                "path": file_paths[i],
                "size_in_bytes": file_sizes[i]
            })
    return datasets_names

def get_datasets_names_df(all_datasets_directory: str) -> pd.DataFrame:
    """
    """
    datasets_names = get_datasets_names(all_datasets_directory)
    return (pd.DataFrame(datasets_names)
        .sort_values(by=["year", "month"])
        .reset_index()
        .drop(columns="index")
        .astype({"year": "int16", "month": "int8"})
    )

def full_datasets_groupby(
        datasets_names_df: pd.DataFrame, 
        by: list[str], 
        verbose: bool = False
    ) -> pd.DataFrame:
    """
    """
    day_full_df = pd.DataFrame(columns=by + ["viajes"])
    for dataset_name in datasets_names_df["path"]:
        print("Processing file %s" % (dataset_name)) if verbose else None
        df = pd.read_csv(dataset_name, dtype=str)
        df = df.astype({"viajes": "int"})
        day_df = df.groupby(by)["viajes"].sum().reset_index()
        day_full_df = pd.concat([day_full_df, day_df])
    return day_full_df.reset_index().drop(columns="index")


def full_datasets_filter_zeros(
        datasets_paths_df: pd.DataFrame, 
        verbose: bool = False
    ) -> pd.DataFrame:
    """
    """
    for dataset_path in datasets_paths_df["path"]:
        print("Processing file %s" % (dataset_path)) if verbose else None
        df = pd.read_csv(dataset_path, dtype=str)
        df = df.astype({"viajes": "int"})
        filtered_df = df[df["viajes"] > 0]
        filtered_df = filtered_df.reset_index()
        dirname = os.path.dirname(dataset_path)
        basename = os.path.basename(dataset_path).split(".")
        name, extension = "".join(basename[0:-1]), basename[-1]
        filepath = os.path.join(r".\test-data", name + "_filtered." + extension)
        filtered_df.drop(columns=["month", "index"]).to_csv(filepath, index=False)
        print("Original size %d, Filtered size %d, Stored in %s" %
             (df.shape[0], filtered_df.shape[0], filepath))


def filter_by_day(
        day_df: pd.DataFrame, 
        start: str = "2023-01-01", 
        end: str = "2023-01-31"
    ) -> pd.DataFrame:
    return day_df[(day_df["day"] >= pd.Timestamp(start)) 
                  & (day_df["day"] <= pd.Timestamp(end))]     

def filter_day_by_year(
        day_df: pd.DataFrame, 
        year: str = "2023"
    ) -> pd.DataFrame:
    return day_df[day_df["day"].dt.year == year]     

def filter_day_by_year_month(
        day_df: pd.DataFrame, 
        year: int = 2023, 
        month: int = 1
    ) -> pd.DataFrame:
    return day_df[(day_df["day"].dt.year == year) 
                  & (day_df["day"].dt.month == month)]

def plot_histogram_and_boxplot(
        array, 
        figsize = (9, 6), 
        height_ratios = [5, 2], 
        title = "", 
        xlabel = "", 
        ylabel = "", 
        ticks=None, 
        labels=None,
        histogram_bins = "auto", 
        histogram_color = None, 
        boxplot_width = 0.7, 
        boxplot_color=None
    ) -> None:

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, 
        ncols=1, 
        figsize=figsize, 
        height_ratios=height_ratios
    )
    
    ax1.hist(x=array, bins=histogram_bins, color=histogram_color)
    ax1.set_ylabel(ylabel)

    
    sns.boxplot(x=array, ax=ax2, width=boxplot_width, color=boxplot_color)
    ax2.set_xlabel(xlabel) 
    ax2.set_ylabel("") 

    fig.suptitle(title)
    if ticks is not None and labels is not None:
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(labels)
        ax2.set_xticks(ticks) 
        ax2.set_xticklabels(labels)
    plt.show()

def plot_histogram_with_density(
        array,
        figsize = (9,5),
        histogram_color = None,
        histogram_bins = "auto",
        density_color = None,
        title = "",
        xlabel = "",
        ylabel = "",
        ticks = None,
        labels = None
    ) -> None:
    plt.figure(figsize=figsize)
    plt.hist(x=array, density=True, bins=histogram_bins, color=histogram_color)
    sns.kdeplot(x=array, color=density_color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ticks is not None and labels is not None:
        plt.xticks(ticks, labels)
    plt.show()