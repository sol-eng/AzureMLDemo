# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

input_sample = pd.DataFrame({"Account Length": pd.Series([0], dtype="int64"), "Area Code": pd.Series(["example_value"], dtype="object"), "VMail Message": pd.Series(["example_value"], dtype="object"), "Day Mins": pd.Series([0.0], dtype="float64"), "Day Calls": pd.Series(["example_value"], dtype="object"), "Eve Mins": pd.Series([0.0], dtype="float64"), "Eve Calls": pd.Series(["example_value"], dtype="object"), "Night Mins": pd.Series([0.0], dtype="float64"), "Night Calls": pd.Series(["example_value"], dtype="object"), "Intl Mins": pd.Series([0.0], dtype="float64"), "Intl Calls": pd.Series(["example_value"], dtype="object"), "CustServ Calls": pd.Series(["example_value"], dtype="object"), "intlplan": pd.Series(["example_value"], dtype="object"), "VMail plan": pd.Series(["example_value"], dtype="object"), "AK": pd.Series(["example_value"], dtype="object"), "AL": pd.Series(["example_value"], dtype="object"), "AR": pd.Series(["example_value"], dtype="object"), "AZ": pd.Series(["example_value"], dtype="object"), "CA": pd.Series(["example_value"], dtype="object"), "CO": pd.Series(["example_value"], dtype="object"), "CT": pd.Series(["example_value"], dtype="object"), "DC": pd.Series(["example_value"], dtype="object"), "DE": pd.Series(["example_value"], dtype="object"), "FL": pd.Series(["example_value"], dtype="object"), "GA": pd.Series(["example_value"], dtype="object"), "HI": pd.Series(["example_value"], dtype="object"), "IA": pd.Series(["example_value"], dtype="object"), "ID": pd.Series(["example_value"], dtype="object"), "IL": pd.Series(["example_value"], dtype="object"), "IN": pd.Series(["example_value"], dtype="object"), "KS": pd.Series(["example_value"], dtype="object"), "KY": pd.Series(["example_value"], dtype="object"), "LA": pd.Series(["example_value"], dtype="object"), "MA": pd.Series(["example_value"], dtype="object"), "MD": pd.Series(["example_value"], dtype="object"), "ME": pd.Series(["example_value"], dtype="object"), "MI": pd.Series(["example_value"], dtype="object"), "MN": pd.Series(["example_value"], dtype="object"), "MO": pd.Series(["example_value"], dtype="object"), "MS": pd.Series(["example_value"], dtype="object"), "MT": pd.Series(["example_value"], dtype="object"), "NC": pd.Series(["example_value"], dtype="object"), "ND": pd.Series(["example_value"], dtype="object"), "NE": pd.Series(["example_value"], dtype="object"), "NH": pd.Series(["example_value"], dtype="object"), "NJ": pd.Series(["example_value"], dtype="object"), "NM": pd.Series(["example_value"], dtype="object"), "NV": pd.Series(["example_value"], dtype="object"), "NY": pd.Series(["example_value"], dtype="object"), "OH": pd.Series(["example_value"], dtype="object"), "OK": pd.Series(["example_value"], dtype="object"), "OR": pd.Series(["example_value"], dtype="object"), "PA": pd.Series(["example_value"], dtype="object"), "RI": pd.Series(["example_value"], dtype="object"), "SC": pd.Series(["example_value"], dtype="object"), "SD": pd.Series(["example_value"], dtype="object"), "TN": pd.Series(["example_value"], dtype="object"), "TX": pd.Series(["example_value"], dtype="object"), "UT": pd.Series(["example_value"], dtype="object"), "VA": pd.Series(["example_value"], dtype="object"), "VT": pd.Series(["example_value"], dtype="object"), "WA": pd.Series(["example_value"], dtype="object"), "WI": pd.Series(["example_value"], dtype="object"), "WV": pd.Series(["example_value"], dtype="object"), "WY": pd.Series(["example_value"], dtype="object")})
output_sample = np.array([0])
method_sample = StandardPythonParameterType("predict")

try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise

@input_schema('method', method_sample, convert_to_provided_type=False)
@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data, method="predict"):
    try:
        if method == "predict_proba":
            result = model.predict_proba(data)
        elif method == "predict":
            result = model.predict(data)
        else:
            raise Exception(f"Invalid predict method argument received ({method})")
        if isinstance(result, pd.DataFrame):
            result = result.values
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
