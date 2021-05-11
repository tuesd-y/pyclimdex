import xarray as xr
import numpy as np
import climdex.utils as utils
from typing import Union

def indices(time_dim='time', convert_units_fn=lambda x: x):
    return PrecipitationIndices(time_dim=time_dim, convert_units_fn=convert_units_fn)

class PrecipitationIndices:
    def __init__(self, time_dim='time', convert_units_fn=lambda x: x):
        self.time_dim = time_dim
        self.convert_units_fn = convert_units_fn
        
    def rx1day(self, X: Union[xr.DataArray, xr.Dataset], period='1M', varname='PRCP'):
        """
        Maximum 1-day precipitation over 'period' (default: monthly) (1M or 1y)
        """
        X_arr = utils.data_array_or_dataset_var(X, var=varname)
        X_arr = utils.resample_daily(X_arr, lambda x: x.sum(), time_dim=self.time_dim)
        return X_arr.resample({self.time_dim: period}).max()
        
    def rx5day(self, X: Union[xr.DataArray, xr.Dataset], period='1M', varname='PRCP'):
        """
        Maximum 5-day precipitation over 'period' (default: monthly) (1M or 1y)
        """
        X_arr = utils.data_array_or_dataset_var(X, var=varname)
        X_arr = utils.resample_daily(X_arr, lambda x: x.sum(), time_dim=self.time_dim)
        return X_arr.rolling({self.time_dim: 5}, min_periods=1, center=True).sum().resample({self.time_dim: period}).max()
    
    def annual_rnmm(self, X: Union[xr.DataArray, xr.Dataset], nmm, varname='PRCP'):
        """
        Annual count of days when precipitation exceeds n mm.
        """
        def _count_rnmm(x, axis):
            return np.sum(x >= self.convert_units_fn(nmm), axis=axis)
        X_arr = utils.data_array_or_dataset_var(X, var=varname)
        X_arr = utils.resample_daily(X_arr, lambda x: x.sum(), time_dim=self.time_dim)
        return X_arr.groupby(f'{self.time_dim}.year').reduce(_count_rnmm)
    
    def annual_r10mm(self, X: Union[xr.DataArray, xr.Dataset], varname='PRCP'):
        """
        Annual count of days when precipitation exceeds 10mm.
        """
        return self.annual_rnmm(X, self.convert_units_fn(10.0), varname=varname)

    def annual_r20mm(self, X: Union[xr.DataArray, xr.Dataset], varname='PRCP'):
        """
        Annual count of days when precipitation exceeds 20 mm.
        """
        return self.annual_rnmm(X, self.convert_units_fn(20.0), varname=varname)
    
    def prcptot(self, X: Union[xr.DataArray, xr.Dataset], period='1y', wet_day_threshold=0, varname='PRCP'):
        """
        Total precipitation over 'period' (default: annual)
        wet_day_threshold: amount in mm for the day to be considered (default: 0, recommended: 1)
        """
        X_arr = utils.data_array_or_dataset_var(X, var=varname)
        X_arr = utils.resample_daily(X_arr, lambda x: x.sum(), time_dim=self.time_dim)
        X_arr = X_arr.where(X_arr >= wet_day_threshold)
        return X_arr.resample({self.time_dim: period}).sum(skipna=True)
    
    def sdii(self, X: Union[xr.DataArray, xr.Dataset], period='1M', varname='PRCP'):
        """
        Simple precipitation intensity index. Ratio of total precipitation of period to the number of wet days.
        """
        def _sdii(x, axis):
            # count wet days
            has_precip = x >= self.convert_units_fn(1.0)
            num_wet_days = np.sum(has_precip.astype(np.float32), axis=axis)
            return np.sum(np.where(has_precip, x, 0.0), axis=axis) / np.where(num_wet_days > 0, num_wet_days, 1.0)
        X_arr = utils.data_array_or_dataset_var(X, var=varname)
        X_arr = utils.resample_daily(X_arr, lambda x: x.sum(), time_dim=self.time_dim)
        return X_arr.resample({self.time_dim: period}).reduce(_sdii, dim=self.time_dim)
    
    def cdd(self, X: Union[xr.DataArray, xr.Dataset], period='1M', varname='PRCP'):
        """
        Number of consecutive dry days in 'period' (default: monthly)
        """
        def _cdd(x, axis):
            has_no_precip = x <= self.convert_units_fn(1.0)
            return utils.max_consecutive_count(has_no_precip)
        X_arr = utils.data_array_or_dataset_var(X, var=varname)
        X_arr = utils.resample_daily(X_arr, lambda x: x.sum(), time_dim=self.time_dim)
        return X_arr.resample({self.time_dim: period}).reduce(_cdd, dim=self.time_dim)
    
    def cwd(self, X: Union[xr.DataArray, xr.Dataset], period='1M', varname='PRCP'):
        """
        Number of consecutive wet days in 'period' (default: monthly)
        """
        def _cwd(x, axis):
            has_precip = x >= self.convert_units_fn(1.0)
            return utils.max_consecutive_count(has_precip)
        X_arr = utils.data_array_or_dataset_var(X, var=varname)
        X_arr = utils.resample_daily(X_arr, lambda x: x.sum(), time_dim=self.time_dim)
        return X_arr.resample({self.time_dim: period}).reduce(_cwd, dim=self.time_dim)
    
