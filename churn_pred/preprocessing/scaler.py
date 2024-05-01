from typing import List, Optional

from sklearn_pandas import DataFrameMapper, gen_features


def scaler_mapper(
    cont_cols: List[str],
    cat_cols: List[str],
    id_cols: List[str],
    scaler_mapper_def: Optional[dict] = None,
) -> DataFrameMapper:
    """Function that maps scaler functions to appropriate columns.

    By default does not assign any scaler to continuous, categorical or
    identifier columns. The scalers must be set in scaler_mapper_def. Use sklearn scalers.
    Only columns defined in mapper object will be present in the transformed dataset.

    Args:
        cont_cols (list): list of continuous feature columns in the dataset
        cat_cols (list): list of categorical feature columns in the dataset
        id_cols (list): identifier columns
        scaler_mapper_def (dict): optional dictionary that contains keys
            ['cont_cols', 'cat_cols', 'id_cols'] with their corresponding
            scalers (defined by names, not instantiated) from sklearn library
    Returns:
        scaler (DataFrameMapper): scaler object mapping sklearn scalers to columns in
            pandas dataframe
    """
    if scaler_mapper_def:
        cont_cols_def = gen_features(
            columns=list(map(lambda x: [x], cont_cols)),
            classes=[scaler_mapper_def["cont_cols"]],
        )

        cat_cols_def = gen_features(
            columns=list(map(lambda x: [x], cat_cols)),
            classes=[scaler_mapper_def["cat_cols"]],
        )

        id_cols_def = gen_features(
            columns=list(map(lambda x: [x], id_cols)),
            classes=[scaler_mapper_def["id_cols"]],
        )

    else:
        cont_cols_def = gen_features(
            columns=list(map(lambda x: [x], cont_cols)), classes=[None]
        )

        cat_cols_def = gen_features(
            columns=list(map(lambda x: [x], cat_cols)), classes=[None]
        )

        id_cols_def = gen_features(
            columns=list(map(lambda x: [x], id_cols)), classes=[None]
        )

    scaler = DataFrameMapper(cont_cols_def + cat_cols_def + id_cols_def, df_out=True)
    return scaler
