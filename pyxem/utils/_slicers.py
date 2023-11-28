import numpy as np


def slice_signal(arr, col_slice, row_slice):
    return arr[row_slice, col_slice]


# Note this functionality should be upstreamed to hyperspy and replaced by equivalent functionality
# added to the ``isig`` attribute of hyperspy signals.
class Slicer:
    def __init__(self, signal):
        self.signal = signal

    def __getitem__(self, item):
        if isinstance(item, tuple):  # multiple dimensions
            if len(item) == 0 or len(item) > 2:
                raise ValueError(
                    "Only column and row slicing 2-D arrays is currenlty supported"
                )
            col_slice = self.str2slice(item[0])
            if len(item) == 2:
                row_slice = item[1]
            else:
                row_slice = slice(None)
        else:
            col_slice = self.str2slice(item)
            row_slice = slice(None)
        if isinstance(col_slice, int):
            col_slice = [
                col_slice,
            ]
        slic = self.signal.map(
            slice_signal,
            col_slice=col_slice,
            row_slice=row_slice,
            inplace=False,
            ragged=self.signal._is_object_dtype,
        )
        if (
            not self.signal._is_object_dtype and len(col_slice) == 1
        ):  # potential bug upstream
            slic.data = slic.data[..., np.newaxis]
        if self.signal.scales is not None:
            slic.scales = np.array(self.signal.scales)[col_slice]
        if self.signal.offsets is not None:
            slic.offsets = np.array(self.signal.offsets)[col_slice]
        if self.signal.column_names is not None:
            slic.column_names = np.array(self.signal.column_names)[col_slice]
        if self.signal.units is not None:
            slic.units = np.array(self.signal.units)[col_slice]
        return slic

    def str2slice(self, item):
        if isinstance(item, str):
            item = self.signal.column_names.index(item)
        elif isinstance(item, (np.ndarray, list)):
            item = np.array([self.str2slice(i) for i in item])
        elif isinstance(item, (slice, int)):
            pass
        else:
            raise ValueError(
                "item must be a string or an int or an array of strings or ints"
            )
        return item
