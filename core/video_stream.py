from typing import Generator

import numpy as np


class IVideoStreamSource:
    def start(self):
        raise NotImplementedError

    def frames(self) -> Generator[np.ndarray, None, None]:
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError





