import logging
from logging.config import dictConfig

from churn_pred.config import logging_config

dictConfig(logging_config)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    print("done")
