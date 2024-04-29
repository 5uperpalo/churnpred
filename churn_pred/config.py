import logging

LOCAL_DATA_DIRNAME = "data"

GDP_PER_CAPITA = (
    LOCAL_DATA_DIRNAME + "/gdpp/" + "API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_213153.csv"
)
GDP_PER_CAPITA_INCOMEGROUP = (
    LOCAL_DATA_DIRNAME
    + "/gdpp/"
    + "Metadata_Country_API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_213153.csv"
)

BIG_MAC_INDEX = LOCAL_DATA_DIRNAME + "/big_mac_index/" + "big-mac-full-index.csv"
COUNTRY_CODES = (
    LOCAL_DATA_DIRNAME + "/country_codes/" + "countries_codes_and_coordinates.csv"
)

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "f": {"format": "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"}
    },
    "handlers": {
        "stream": {
            "class": "logging.StreamHandler",
            "formatter": "f",
            "level": logging.INFO,
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "/logs/classifier.log",
            "mode": "a",
            "maxBytes": 100000,
            "backupCount": 5,
            "formatter": "f",
            "level": logging.INFO,
        },
    },
    "loggers": {
        "": {"handlers": ["stream", "file"], "level": logging.INFO, "propagate": False},
        "root": {
            "handlers": ["stream", "file"],
            "level": logging.INFO,
            "propagate": False,
        },
    },
}
