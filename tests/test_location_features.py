import random
import datetime

import numpy as np
import pandas as pd
from data_preparation.location_features import (
    gdppc,
    city_size,
    hemisphere,
    big_mac_index,
    iso_a3_to_a2_map,
)


def randomtimes(start, end, n):
    frmt = "%d-%m-%Y %H:%M:%S"
    stime = datetime.datetime.strptime(start, frmt)
    etime = datetime.datetime.strptime(end, frmt)
    td = etime - stime
    random.seed(0)
    return [random.random() * td + stime for _ in range(n)]


df = pd.DataFrame(
    {
        "TS": randomtimes("01-01-2023 00:00:00", "31-12-2023 00:00:00", 10),
        "country_code_isoa2": [
            "SK",
            "SK",
            "SK",
            "SK",
            "SK",
            "IN",
            "IN",
            "BO",
            "BO",
            "BO",
        ],
        "country_code": [
            "SVK",
            "SVK",
            "SVK",
            "SVK",
            "SVK",
            "IND",
            "IND",
            "BOL",
            "BOL",
            "BOL",
        ],
        "city": [
            "Komarno",
            "Komarno",
            "Kosice",
            "Kosice",
            "Bratislava",
            "Panipat",
            "Kolkata",
            "La Paz",
            "Sucre",
            "",
        ],
    }
)

# day before holiday
df.loc[0, "TS"] = "2023-11-16 00:00:00"
# add prolonged weekend with holliday
df.loc[1, "TS"] = "2023-11-17 00:00:00"
# weekend
df.loc[2, "TS"] = "2023-11-19 00:00:00"
# day after prolonged weekend
df.loc[3, "TS"] = "2023-11-20 00:00:00"

ts_col = "TS"
cc_isoa2_col = "country_code_isoa2"
cc_isoa3_col = "country_code"
city_col = "city"

loc_db_df = pd.DataFrame(
    {
        "location": [
            "Komarno,SK",
            "Kosice,SK",
            "Bratislava,SK",
            "Panipat,IN",
            "Kolkata,IN",
            "La Paz,BO",
            "Sucre,BO",
            ",BO",
        ],
        "latitude": [
            47.7574079,
            48.7172272,
            48.1435149,
            29.3912753,
            22.5414185,
            -16.4955455,
            -19.0477251,
            -17.0568696,
        ],
        "longitude": [
            18.1298249,
            21.2496774,
            17.108279,
            76.9771675,
            88.35769124388872,
            -68.1336229,
            -65.2594306,
            -64.9912286,
        ],
    }
)

df[[cc_isoa2_col, cc_isoa3_col, city_col]] = df[
    [cc_isoa2_col, cc_isoa3_col, city_col]
].astype("string")

city_size_db_df = pd.DataFrame(
    {
        "location": [
            "Komarno,SK",
            "Kosice,SK",
            "Bratislava,SK",
            "Panipat,IN",
            "Kolkata,IN",
            "La Paz,BO",
            "Sucre,BO",
        ],
        "city_size": [
            33927,
            228249,
            423737,
            292808,
            4631392,
            2004652,
            224838,
        ],
    }
)


def test_hemisphere_w_db():
    ground_truth = np.array(
        [
            "northern",
            "northern",
            "northern",
            "northern",
            "northern",
            "northern",
            "northern",
            "southern",
            "southern",
            "southern",
        ]
    )
    hemispheres_np = hemisphere(
        df=df, city_col=city_col, cc_col=cc_isoa2_col, loc_db_df=loc_db_df
    ).values
    assert np.equal(hemispheres_np, ground_truth).sum() == len(ground_truth)


def test_big_mac_index():
    ground_truth = np.array(
        [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            2.38895428,
            2.38895428,
            np.nan,
            np.nan,
            np.nan,
        ]
    )
    bmi = big_mac_index(df=df, iso_a2_col=cc_isoa2_col)[
        "big_mac_index_dollar_price"
    ].values
    assert np.isclose(bmi, ground_truth, equal_nan=True).sum() == len(ground_truth)


def test_gdppc():
    ground_truth = np.array(
        [
            33010.29074847,
            33010.29074847,
            33010.29074847,
            33010.29074847,
            33010.29074847,
            7333.50561184,
            7333.50561184,
            9030.38768129,
            9030.38768129,
            9030.38768129,
        ]
    )
    gdppc_np = gdppc(df=df, iso_a2_col=cc_isoa2_col)["gdp_per_capita_2021"].values
    assert np.isclose(gdppc_np, ground_truth).sum() == len(ground_truth)


def test_iso_a3_to_a2_map():
    dfs_mapped = iso_a3_to_a2_map(df=df, iso_a3_col=cc_isoa3_col)
    assert df[cc_isoa2_col].eq(dfs_mapped).all()


def test_city_size():
    ground_truth = np.array(
        [
            33927,
            33927,
            228249,
            228249,
            423737,
            292808,
            4631392,
            2004652,
            224838,
            0,
        ]
    )
    result = city_size(
        df=df,
        city_col=city_col,
        cc_col=cc_isoa2_col,
        city_size_db_df=city_size_db_df,
        approximate=False,
    ).values
    assert np.equal(result, ground_truth).sum() == len(ground_truth)


def test_city_size_approx():
    ground_truth = np.array(
        [
            "semi_dense_urban_area",
            "semi_dense_urban_area",
            "medium_urban_area",
            "medium_urban_area",
            "medium_urban_area",
            "medium_urban_area",
            "large_metropolitan_area",
            "large_metropolitan_area",
            "medium_urban_area",
            "low_dense_urban_area",
        ]
    )
    result = city_size(
        df=df,
        city_col=city_col,
        cc_col=cc_isoa2_col,
        city_size_db_df=city_size_db_df,
        approximate=True,
    ).values
    assert np.equal(result, ground_truth).sum() == len(ground_truth)
