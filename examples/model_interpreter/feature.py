#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.


import qlib_bak
from qlib_bak.constant import REG_CN

from qlib_bak.utils import init_instance_by_config
from qlib_bak.tests.data import GetData
from qlib_bak.tests.config import CSI300_GBDT_TASK


if __name__ == "__main__":
    # use default data
    provider_uri = "~/.qlib_bak/qlib_data/cn_data"  # target_dir
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)

    qlib_bak.init(provider_uri=provider_uri, region=REG_CN)

    ###################################
    # train model
    ###################################
    # model initialization
    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])
    model.fit(dataset)

    # get model feature importance
    feature_importance = model.get_feature_importance()
    print("feature importance:")
    print(feature_importance)
