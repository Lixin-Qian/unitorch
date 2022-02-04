# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


class GeneralWriter(object):
    def _init___(self):
        pass

    def __call__(self, outputs: Dict):
        return pd.DataFrame(outputs).to_json(
            orient="records",
            lines=True,
        )


class GeneralCSVWriter(object):
    def __init__(
        self,
        headers: Optional[List[str]] = None,
        sep: str = "\t",
    ):
        self.headers = headers
        self.sep = sep

    def __call__(self, outputs: Dict):
        outputs = dict(
            {
                k: [vv if isinstance(vv, str) else json.dumps(vv) for vv in v]
                for k, v in outputs.items()
            }
        )

        outputs = pd.DataFrame(outputs)

        if self.headers is None:
            self.headers = outputs.columns

        assert len(set(self.headers) & set(outputs.columns)) == len(set(self.headers))
        outputs = outputs[self.headers]

        return outputs.to_csv(
            index=False,
            sep=self.sep,
            quoting=3,
            header=False,
        )
