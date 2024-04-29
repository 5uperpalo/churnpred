from typing import Optional

import numpy as np
import pandas as pd
import pycountry
import openapi_client
from geopy.geocoders import Nominatim
from openapi_client.apis.tags import personal_api

from churn_pred.config import (
    BIG_MAC_INDEX,
    GDP_PER_CAPITA,
    GDP_PER_CAPITA_INCOMEGROUP,
)


def surname_origin(df: pd.DataFrame, surname_col: str, api_key: str) -> pd.DataFrame:
    """
    Surname classification that uses Namsor API to get additional information
    about the surname, i.e.['countryOrigin', 'regionOrigin', 'subRegionOrigin',
    'probabilityCalibrated']. Based on API documentation:
    https://github.com/namsor/namsor-python-sdk2/blob/master/docs/apis/tags/PersonalApi.md#origin

    Issues:
    * uses 10 units per name to Infer the likely country of origin of a personal name
      * only 5k units are available in trial version and 2930 unique surnames are in
      the toy dataset; additional 10k units cost 19$/month and 100k units 130$/month

    Args:
        df (pd.DataFrame): input dataset
        surname_col (str): column name with surnames
        personal_api (str): namsor API key
    """
    dfc = df.copy()

    configuration = openapi_client.Configuration(
        host="https://v2.namsor.com/NamSorAPIv2"
    )

    configuration.api_key["api_key"] = api_key

    surnames = list(dfc[surname_col].unique())
    surname_origin_country = []
    surname_origin_region = []
    surname_origin_subregion = []
    surname_origin_namsor_probability = []

    with openapi_client.ApiClient(configuration) as api_client:
        api_instance = personal_api.PersonalApi(api_client)
        for surname in surnames:
            path_params = {
                "firstName": "unset",
                "lastName": surname,
            }
            try:
                api_response = api_instance.origin(
                    path_params=path_params,
                )
                surname_origin_country.append(api_response.body["countryOrigin"])
                surname_origin_region.append(api_response.body["regionOrigin"])
                surname_origin_subregion.append(api_response.body["subRegionOrigin"])
                surname_origin_namsor_probability.append(
                    float(api_response.body["probabilityCalibrated"])
                )
            except openapi_client.ApiException:
                surname_origin_country.append("NA")
                surname_origin_region.append("NA")
                surname_origin_subregion.append("NA")
                surname_origin_namsor_probability.append(np.nan)
    name_origin_df = pd.DataFrame(
        {
            "Surname": surnames,
            "surname_origin_country": surname_origin_country,
            "surname_origin_region": surname_origin_region,
            "surname_origin_subregion": surname_origin_subregion,
            "surname_origin_namsor_probability": surname_origin_namsor_probability,
        }
    )
    dfc = dfc.merge(name_origin_df, on="Surname", how="left")
    return dfc


def hemisphere(
    df: pd.DataFrame,
    cc_col: str,
    loc_db_df: Optional[pd.DataFrame] = None,
    city_col: Optional[str] = None,
) -> pd.Series:
    """Returns string pd.Series identifing hemisphere: "northern" or "southern".

    Args:
        df (pd.DataFrame): input dataframe
        city_col (str): city name column
        cc_col (str): country code column
        loc_db_df (pd.DataFrame): pandas dataframe with previously saved locations
    """
    if city_col:
        uq_locs = (df[city_col] + "," + df[cc_col]).unique()
    else:
        uq_locs = (" ," + df[cc_col]).unique()
    dfc = df.copy()
    geolocator = Nominatim(user_agent="hemisphere_identification")
    dfc["hemisphere"] = "northern"

    for uq_loc in uq_locs:
        if (loc_db_df is not None) and (uq_loc in loc_db_df["location"].values):
            location = loc_db_df[loc_db_df["location"] == uq_loc][
                ["latitude", "longitude"]
            ]
            latitude = float(location.latitude)
        else:
            latitude = geolocator.geocode(uq_loc).latitude

        if latitude < 0:
            city, country = uq_loc.split(",")
            dfc.loc[
                (dfc[city_col] == city) & (dfc[cc_col] == country), "hemisphere"
            ] = "southern"

    return dfc


def age_categories(df: pd.DataFrame, age_col: str) -> pd.DataFrame:
    """Returns string pd.DataFrame identifing different age classification
    categories according to the age:
    * working_class (https://ourworldindata.org/age-structure)
      * children_and_adolescents: <0, 15)
      * working_age: <15, 65)
      * elderly: <65, inf)
    * stage_of_life (https://integrishealth.org/resources/on-your-health/2015/october/stages-of-life-health-for-every-age)
      * infant: <0, 2)
      * toddler: <2, 5)
      * child: <5, 13)
      * teen: <13, 20)
      * adult: <20, 40)
      * middle_age_adult: <40, 60)
      * senior_adult: <60, inf)
    * generation (https://www.beresfordresearch.com/age-range-by-generation/)
      * gen_z: <12, 28)
      * millennials: <28, 44)
      * gen_x: <44, 60)
      * boomers_2: <60, 70)
      * boomers_1: <70, 79)
      * post_war: <79, 97)
      * ww2: <97, 102)
      * vampire: <102, inf)

    Args:
        df (pd.DataFrame): input dataset
        age_col (str): age column
    """
    dfc = df.copy()
    dfc["working_class"] = pd.cut(
        dfc[age_col],
        bins=[0, 15, 65, np.inf],
        labels=["children_and_adolescents", "working_age", "elderly"],
        include_lowest=True,
    )
    dfc["stage_of_life"] = pd.cut(
        dfc[age_col],
        bins=[0, 2, 5, 13, 20, 40, 60, np.inf],
        labels=[
            "infant",
            "toddler",
            "child",
            "teen",
            "adult",
            "middle_age_adult",
            "senior_adult",
        ],
        include_lowest=True,
    )
    dfc["generation"] = pd.cut(
        dfc[age_col],
        bins=[0, 28, 44, 60, 70, 79, 97, 102, np.inf],
        labels=[
            "gen_z",
            "millennials",
            "gen_x",
            "boomers_2",
            "boomers_1",
            "post_war",
            "ww2",
            "vampire",
        ],
        include_lowest=True,
    )
    return dfc


def big_mac_index(df: pd.DataFrame, country_name_col: str) -> pd.DataFrame:
    """
    Returns dataframe with Big Max Index corresponding to country codes.
    Downloaded on 26/4/2024.

    Note:
        Unfortunatelly Spain, France, Germany (unique companies in the dataset) are not included, other index to explore:
        https://en.wikipedia.org/wiki/Big_Mac_Index

    Based on:
        https://www.economist.com/big-mac-index

    Downloaded from:
        https://github.com/TheEconomist/big-mac-data
        https://raw.githubusercontent.com/TheEconomist/big-mac-data/master/output-data/big-mac-full-index.csv

    Args:
        df (pd.DataFrame): input dataset
        country_name_col: column name with country names
    """
    big_mac_index = pd.read_csv(BIG_MAC_INDEX)
    # filter the last available price
    big_mac_index["date"] = pd.to_datetime(big_mac_index["date"])
    big_mac_index = (
        big_mac_index.sort_values("date").groupby("iso_a3").tail(1).reset_index()
    )
    big_mac_index = big_mac_index[["iso_a3", "dollar_price"]]
    big_mac_index.rename(
        columns={"dollar_price": "big_mac_index_dollar_price"}, inplace=True
    )

    big_mac_index = big_mac_index.dropna()

    dfc = df.copy()
    dfc["iso_a3"] = get_iso_a3(df=dfc, country_name_col=country_name_col)
    dfc = dfc.merge(big_mac_index, on="iso_a3", how="left")
    dfc.drop(columns=["iso_a3"], inplace=True)
    return dfc


def gdppc(df: pd.DataFrame, country_name_col: str) -> pd.DataFrame:
    """
    Returns dataframe with Gross Domestic Product Per Capita corresponding to country
    codes. Uses information from worldbank API downloaded on 26/4/2024.
    Last file update: 3/28/2024.
    Posprocessed file has initial 3 lines removed for easier processing using pandas df.

    Downloaded from:
        * https://data.worldbank.org/indicator/NY.GDP.PCAP.PP.CD
        * https://api.worldbank.org/v2/en/indicator/NY.GDP.PCAP.PP.CD?downloadformat=csv
    Main file before preprocessing:
        'data/gdpp/API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_213153.csv'
    Other included files:
        'data/gdpp/Metadata_Country_API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_213153.csv'
        'data/gdpp/Metadata_Indicator_API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_213153.csv'

    Args:
        df (pd.DataFrame): input dataset
        country_name_col: column name with country names
    """
    gdp_per_capita = pd.read_csv(GDP_PER_CAPITA)
    gdp_per_capita_ic = pd.read_csv(GDP_PER_CAPITA_INCOMEGROUP)
    # add last available year
    gdp_per_capita = gdp_per_capita[["Country Code", "2022"]]
    gdp_per_capita.rename(columns={"2022": "gdp_per_capita_2022"}, inplace=True)
    gdp_per_capita.rename(columns={"Country Code": "iso_a3"}, inplace=True)

    gdp_per_capita_ic = gdp_per_capita_ic[["Country Code", "IncomeGroup"]]
    gdp_per_capita_ic.rename(columns={"Country Code": "iso_a3"}, inplace=True)

    gdp_per_capita = gdp_per_capita.dropna()
    gdp_per_capita_ic = gdp_per_capita_ic.dropna()

    dfc = df.copy()
    dfc["iso_a3"] = get_iso_a3(df=dfc, country_name_col=country_name_col)

    dfc = dfc.merge(gdp_per_capita, on="iso_a3", how="left")
    dfc = dfc.merge(gdp_per_capita_ic, on="iso_a3", how="left")
    dfc.drop(columns=["iso_a3"], inplace=True)

    return dfc


def get_iso_a3(df: pd.DataFrame, country_name_col: str):
    """Returns pd.Series with mapped country names to iso_a3.

    Args:
        df (pd.Series): input country name pandas series
    """
    dfc = df.copy()
    return dfc[country_name_col].apply(_get_iso_a3_map)


def _get_iso_a3_map(country_name: str) -> str:
    """Helper function for get_iso_a3"""
    result = pycountry.countries.search_fuzzy(country_name)
    if result:
        iso_a3_str = result[0].alpha_3
    else:
        iso_a3_str = "NA"
    return iso_a3_str
