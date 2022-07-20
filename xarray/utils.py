import xarray as xr
import numpy as np
from sklearn.decomposition import PCA, SparsePCA
from sklearn.cluster import KMeans, DBSCAN
from typing import Union

def _assemble(computed_data: np.array, 
              original_stack: xr.DataArray, 
              z_values: list, 
              asDataArray: bool=True) -> xr.DataArray:
    """Assemble numpy array as xarray DataArray
    
    Utility function that allows the transformation of a numpy array into a 
    xarray DataArray. Usually the numpy array is the result of a computation 
    originated from a xarray DataArray.
    
    Args:
        computed_data: A 3D numpy array with X, Y and Z dimensions
        original_stack: An original 3D xarray DataArray from which computed_data
            is derived. It has the same X, Y coordinates and can be bigger or 
            the same size in the Z coordinate.
        z_values: A list with the new Z coordinate values
        asDataArray: boolean. If True, returns a xarray.DataArray. Default is 
            True.

    Returns:
        An xarray.DataArray with the same extension (X and Y) that the template,
        but different data and (possible) Z coordinates.
    """
    coords_ = z_values
    coords_['y'] = original_stack['y']
    coords_['x'] = original_stack['x']
    
    if asDataArray:
        out = xr.DataArray(computed_data, 
                           coords=coords_, 
                           dims=original_stack.dims)
    else:
        pass
    
    return out


def applyTransCluster(data: xr.DataArray, 
                    decomposition: Union[PCA, SparsePCA], 
                    cluster: Union[KMeans, DBSCAN],
                    output_zdim: list = [1]) -> xr.DataArray:
    """Apply sklearn decomposition and then a cluster prediction
    
    Given a 3D xarray DataArray, apply a decomposition and a cluster prediction
    over the Z coordinate or axis.
    
    Args:
        data: a 3D xarray.DataAray. Must be in the order Z, Y, X or Z, X, Y, 
            where Z is the coordinate that would be traverse.
        decomposition: a sklearn.preprocessing fitted object that has a 
            transform method.
        cluster: a sklearn.decomposition fitted object (using the previous
            decomposition) that as a transform method.
        output_zdim: A list with the coordinates values of the new Z dim. Only 
            tested with [1] as value.
    
    Returns:
        An xarray.DataArray with the same extension (X and Y) that the original,
        but only one Z coordinate.
    """
    shape_ = data.shape
    dt = data.values.reshape(shape_[0], -1).T  # first coord is the one to be preserved!!
    pcat = decomposition.transform(dt)
    clustert = cluster.predict(pcat)
    ans = clustert.T.reshape(len(output_zdim), shape_[1], shape_[2])
    # TODO: test flexible Z coordinate output and not just 1.
    return _assemble(ans, data, {'band': output_zdim}, True)    


def createTemplate(data: xr.DataArray,
                   new_coords: list,
                   chunk: Union[dict, tuple] = None) -> xr.DataArray:
    """Create a template from an xarray.DataArray filled with 0s
    
    This function is intented to be used as part of the dask.array.map_blocks
    when a template is required (the output of map_blocks is of different size).
    This case only covers the change of Z coordinate, while X and Y remain the 
    same.
    
    Args:
        data: a 3D xarray.DataAray. Must be in the order Z, Y, X, where Z is the 
            coordinate that would be transformed.
        new_coords: a list with the value of the new coords.
        chunk: an optional dict with the chunksize of the output.
    
    Returns:
        An xarray.DataArray filled with 0s, with the same extension (X and Y) 
        that the original, but the Z coordinates coudl be different.
    """
    mcoord = list(data.coords)[0]
    coords_ = {mcoord: new_coords,
            'y': data.coords['y'],
            'x': data.coords['x']}
    
    if chunk is None:
        z1, y1, x1 = data.chunks
        chunk_size = {'x': x1, 'y': y1, mcoord: (1)}
    else:
        chunk_size = chunk
        
    template_ = xr.DataArray(np.zeros((1, len(data.y), len(data.x))), 
                            coords=coords_,
                            dims = [mcoord, 'y', 'x']).chunk(chunk_size)
    return template_
