import pdb

import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
#%%
# use default data
# NOTE: need to download data from remote: python scripts/get_data.py qlib_data_cn --target_dir ~/.qlib/qlib_data/cn_data
provider_uri = r"C:\Users\linsh\Downloads\ai-dataset\qlib_data_cn_1d_latest"  # target_dir
if not exists_qlib_data(provider_uri):
    print(f"Qlib data is not found in {provider_uri}")
    # sys.path.append(str(scripts_dir))
    # from get_data import GetData
    #
    # GetData().qlib_data(target_dir=provider_uri, region=REG_CN)
qlib.init(provider_uri=provider_uri, region=REG_CN)


market = "csi300"
benchmark = "SH000300"


###################################
# train model
###################################
data_handler_config = {
        "start_time": "2015-01-01",
        "end_time": "2020-08-01",
        "fit_start_time": "2015-01-01",
        "fit_end_time": "2018-12-31",
        "instruments": market,
        "infer_processors": {
            "RobustZScoreNorm": {
                "kwargs": {
                    "fields_group": "feature",
                    "clip_outlier": "true"
                }
            },
            "Fillna": {
                "kwargs": {
                    "fields_group": "feature"
                }
            }
        },
        "learn_processors": {
            "DropnaLabel":{},
            "CSRankNorm": {
                "kwargs": {
                    "field_group": "label"
                }
            }
        }
    }

task = {
    "model": {
        "class": "LinearModel",
        "module_path": "qlib.contrib.model.linear",
        "kwargs": {
            "estimator": "ols",
        }
    },
   "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": ["2015-01-01", "2018-12-31"],
                "valid": ["2019-01-01", "2019-12-31"],
                "test": ["2020-01-01", "2020-08-01"],
            },
        },
    },
}


# pdb.set_trace()

# if __name__ == '__main__':



###################################
# prediction, backtest & analysis
###################################


if __name__ == '__main__':
    # model initiaiton
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    # start exp to train model
    with R.start(experiment_name="train_model"):
        R.log_params(**flatten_dict(task))

        # moddel.fit -> qlib.contrib.model.linear.fit() -> sklearn.linear_model._base.fit()[618] -> scipy.linalg._basic.py.lstlq()
        model.fit(dataset)

        R.save_objects(trained_model=model)
        rid = R.get_recorder().id

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "model": model,
                "dataset": dataset,
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }


    # backtest and analysis
    with R.start(experiment_name="backtest_analysis"):
        recorder = R.get_recorder(recorder_id=rid, experiment_name="train_model")
        model = recorder.load_object("trained_model")

        # prediction
        recorder = R.get_recorder()
        ba_rid = recorder.id
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # backtest & analysis
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()