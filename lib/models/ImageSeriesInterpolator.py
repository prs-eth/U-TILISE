import math
from typing import Literal, Optional, Tuple

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from numba import njit, prange
from torch import Tensor, nn


class ImageSeriesInterpolator(nn.Module):
    def __init__(self, mode: Literal['last', 'next', 'closest', 'linear_interpolation']):
        """
        Trivial image time series interpolation over time.
        Note that the interpolation runs on CPU but is based on numba to speed up the computations
        (speed-up of ~600 times!).

        Args:
            mode:   str, strategy used to interpolate a spatio-temporal location (t, y, x). Choose among:
                        'last':                  Copy-paste the last observed reflectance at (y,x).
                        'next':                  Copy-past the next observed reflectance at (y,x).
                        'closest':               Copy-paste the temporally closest reflectance at (y,x);
                                                 preference for the past observation in case the last and the next
                                                 observation are temporally equally close.
                        'linear_interpolation':  Linearly interpolate between the last and the next observed
                                                 reflectance at (y,x).
        """

        super().__init__()
        self.mode = mode

    def forward(
            self, images: Tensor,
            cloud_mask: Tensor,
            days: Optional[Tensor] = None,
            return_vis_map: Optional[bool] = False
    ) -> Tensor | Tuple[Tensor, Optional[np.ndarray | Tuple[np.ndarray, np.ndarray]]]:
        """
        Args:
            images:          torch.Tensor, B x T x C x H x W, image time series to be imputed.
            cloud_mask:      torch.Tensor, B x T x 1 x H x W, associated cloud masks; 1 for occluded pixels, 0 for
                             non-occluded pixels.
            days:            torch.Tensor, (B, T), temporal sampling; number of days since the first observation in the
                             sequence. Required input if self.mode == 'linear_interpolation' or self.mode == 'closest'.
            return_vis_map:  bool, True to return the output of _find_last_visible() and/or _find_next_visible(),
                             False otherwise.

        Returns:
            inpainted:       torch.Tensor, B x T x C x H x W, imputed image time series.
            vis_maps:        np.ndarray or tuple of np.ndarray, output of _find_last_visible() and/or
                             _find_next_visible() (provided that return_vis_map == True).
        """

        if self.mode in ['linear_interpolation', 'closest']:
            assert days is not None, 'Please provide the temporal sampling information.'

        vis_maps: Optional[np.ndarray | Tuple[np.ndarray, np.ndarray]] = None

        # Convert input data to numpy arrays to enable computation speed-up via numba
        images = images.numpy()
        cloud_mask = cloud_mask.numpy()
        days = days.numpy() if days is not None else None

        if self.mode in ['last', 'next']:
            # For every occluded pixel in the image time series, find the temporally closest non-occluded pixel in the
            # past (self.mode=='last') or future (self.mode=='next')
            vis_maps = self._find_last_visible(images, cloud_mask) if self.mode == 'last' else \
                self._find_next_visible(images, cloud_mask)

            inpainted = self._inpaint_unidirectional(images, vis_maps)

        elif self.mode in ['closest', 'linear_interpolation']:
            # For every occluded pixel in the image time series, find the temporally closest non-occluded pixel in the
            # past and future
            vis_maps_last = self._find_last_visible(images, cloud_mask)
            vis_maps_next = self._find_next_visible(images, cloud_mask)

            if self.mode == 'closest':
                inpainted, vis_maps = self._inpaint_bidirectional(images, days, vis_maps_last, vis_maps_next)
            else:
                inpainted = self._linear_interpolation(images, days, vis_maps_last, vis_maps_next)
                vis_maps = (vis_maps_last, vis_maps_next)

        inpainted = torch.from_numpy(inpainted)

        if return_vis_map:
            return inpainted, vis_maps
        return inpainted

    @staticmethod
    def _find_last_visible(images: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
        """
        Computes a look-up table for pixel-wise data imputation (backward) over time.

        Strategy:
        For every spatio-temporal location (y, x, t), determine the temporal index at which (y,x) was last observed.

        Args:
            images:        np.ndarray, B x T x C x H x W, image time series.
            cloud_mask:    np.ndarray, B x T x 1 x H x W, associated cloud masks; 1 for occluded pixels, 0 for
                           non-occluded pixels.

        Returns:
            last_visible:  np.ndarray, B x T x H x W, specifies the index of the last visible observation for every
                           spatio-temporal location (y,x,t).

                           Examples:
                               [0, t, y, x] = t if pixel (y,x) has been observed (i.e., non-occluded) at time step t
                               [0, t, y, x] = np.nan if pixel (y,x) was occluded up to time step t
                               [0, t, y, x] = t2 if pixel (y,x) was last visible at time step t2 < t
        """

        # Initialization
        B, T, _, H, W = images.shape
        last_visible = np.full((B, T, H, W), np.nan, dtype=np.float32)

        # Pixel observed at time step t=0
        last_visible[:, 0, :, :][cloud_mask[:, 0, 0, :, :] == 0.] = 0

        # Iterate through the image time series (forward)
        for t in range(1, T):
            # Pixel observed at time step t
            last_visible[:, t, :, :][cloud_mask[:, t, 0, :, :] == 0.] = t

            # Pixel occluded at time step t: record at which previous time step the pixel was last visible
            last_visible[:, t, :, :][cloud_mask[:, t, 0, :, :] == 1.] = \
                last_visible[:, t-1, :, :][cloud_mask[:, t, 0, :, :] == 1.]

        return last_visible

    @staticmethod
    def _find_next_visible(images: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
        """
        Computes a look-up table for pixel-wise data imputation (forward) over time.

        Strategy:
        For every spatio-temporal location (y, x, t), determine the temporal index at which (y,x) will be observed next.

        Args:
            images:        np.ndarray, B x T x C x H x W, image time series.
            cloud_mask:    np.ndarray, B x T x 1 x H x W, associated cloud masks; 1 for occluded pixels, 0 for
                           non-occluded pixels.

        Returns:
            next_visible:  np.ndarray, B x T x H x W, specifies the index of the next visible observation for every
                           spatio-temporal location (y,x,t).

                           Examples:
                               [0, t, y, x] = t if pixel (y,x) has been observed (i.e., non-occluded) at time step t
                               [0, t, y, x] = np.nan if pixel (y,x) will be occluded for time steps >= t
                               [0, t, y, x] = t2 if pixel (y,x) will be visible again at time step t2 > t
        """

        # Initialization
        B, T, _, H, W = images.shape
        next_visible = np.full((B, T, H, W), np.nan, dtype=np.float32)

        # Pixel observed in the last image of the time series
        next_visible[:, -1, :, :][cloud_mask[:, -1, 0, :, :] == 0.] = T-1

        # Iterate through the image time series (backward)
        for t in range(T - 2, -1, -1):
            # Pixel observed at time step t
            next_visible[:, t, :, :][cloud_mask[:, t, 0, :, :] == 0.] = t

            # Pixel occluded at time step t: record at which time step the pixel is observed next
            next_visible[:, t, :, :][cloud_mask[:, t, 0, :, :] == 1.] = \
                next_visible[:, t+1, :, :][cloud_mask[:, t, 0, :, :] == 1.]

        return next_visible

    @staticmethod
    @njit(parallel=True)
    def _inpaint_unidirectional(images: np.ndarray, vis_maps: np.ndarray) -> np.ndarray:
        """
        Pixel-wise unidirectional data imputation over time (backwards or forwards).

        Args:
            images:        np.ndarray, B x T x C x H x W, image time series.
            vis_maps:      np.ndarray, B x T x H x W, look-up table for pixel-wise temporal inpainting.
                           Output of _find_last_visible() or _find_next_visible()

        Returns:
            inpainted:    np.ndarray, B x T x C x H x W, imputed image time series, where occluded pixels are
                          replaced with the last or next observed reflectance at the respective spatial location.
        """

        # Initialize imputed image time series
        inpainted = np.full(images.shape, np.nan, dtype=np.float32)
        B, T, _, H, W = images.shape

        for b in prange(B):
            for t in prange(T):
                for h in prange(H):
                    for w in prange(W):
                        if ~np.isnan(vis_maps[b, t, h, w]):
                            inpainted[b, t, :, h, w] = images[b, int(vis_maps[b, t, h, w]), :, h, w]

        return inpainted

    @staticmethod
    @njit(parallel=True)
    def _inpaint_bidirectional(
            images: np.ndarray, days: np.ndarray, vis_maps_last: np.ndarray, vis_maps_next: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pixel-wise bidirectional data imputation (closest) over time.

        Args:
            images:        np.ndarray, B x T x C x H x W, image time series.
            days:          np.ndarray, (B, T), temporal sampling; number of days since the first observation in the
                           sequence.
            vis_maps_last: np.ndarray, B x T x H x W, look-up table for pixel-wise temporal inpainting.
                           Output of _find_last_visible().
            vis_maps_next: np.ndarray, B x T x H x W, look-up table for pixel-wise temporal inpainting.
                           Output of _find_next_visible().

        Returns:
            inpainted:          np.ndarray, B x T x C x H x W, imputed image time series, where occluded pixels are
                                replaced with the temporally closest observed reflectance at the respective spatial
                                location.
            vis_maps_closest:   np.ndarray, B x T x H x W, look-up table, temporally closest observation per
                                spatio-temporal location.
        """

        B, T, H, W = vis_maps_last.shape
        vis_maps_closest = np.full((B, T, H, W), np.nan, dtype=np.float32)
        inpainted = np.full(images.shape, np.nan, dtype=np.float32)

        for b in prange(B):
            for t in prange(T):
                for h in prange(H):
                    for w in prange(W):
                        if ~np.isnan(vis_maps_last[b, t, h, w]) and ~np.isnan(vis_maps_next[b, t, h, w]):
                            t0 = int(vis_maps_last[b, t, h, w])
                            t1 = int(vis_maps_next[b, t, h, w])

                            if days[b, t] - days[b, t0] <= days[b, t1] - days[b, t]:
                                # t0 closer to t than t1
                                vis_maps_closest[b, t, h, w] = t0
                                inpainted[b, t, :, h, w] = images[b, t0, :, h, w]
                            else:
                                vis_maps_closest[b, t, h, w] = t1
                                inpainted[b, t, :, h, w] = images[b, t1, :, h, w]

                        elif ~np.isnan(vis_maps_last[b, t, h, w]):
                            vis_maps_closest[b, t, h, w] = int(vis_maps_last[b, t, h, w])
                            inpainted[b, t, :, h, w] = images[b, int(vis_maps_last[b, t, h, w]), :, h, w]

                        elif ~np.isnan(vis_maps_next[b, t, h, w]):
                            vis_maps_closest[b, t, h, w] = int(vis_maps_next[b, t, h, w])
                            inpainted[b, t, :, h, w] = images[b, int(vis_maps_next[b, t, h, w]), :, h, w]

        return inpainted, vis_maps_closest

    @staticmethod
    @njit(parallel=True)
    def _linear_interpolation(
            images: np.ndarray, days: np.ndarray, vis_maps_last: np.ndarray, vis_maps_next: np.ndarray
    ) -> np.ndarray:
        """
        Pixel-wise linear interpolation of occluded pixels over time.

        Args:
            images:        np.ndarray, B x T x C x H x W, image time series.
            days:          np.ndarray, (B, T), temporal sampling; number of days since the first observation in the
                           sequence.
            vis_maps_last: np.ndarray, B x T x H x W, look-up table for pixel-wise temporal interpolation.
                           Output of _find_last_visible().
            vis_maps_next: np.ndarray, B x T x H x W, look-up table for pixel-wise temporal interpolation.
                           Output of _find_next_visible().

        Returns:
            interpolated: np.ndarray, B x T x C x H x W, linearly interpolated image time series.
        """

        # Initialize linearly interpolated image time series
        interpolated = np.full(images.shape, np.nan, dtype=np.float32)
        B, T, _, H, W = images.shape

        for b in prange(B):
            for t in prange(T):
                for h in prange(H):
                    for w in prange(W):
                        if ~np.isnan(vis_maps_last[b, t, h, w]) and ~np.isnan(vis_maps_next[b, t, h, w]):
                            t0 = int(vis_maps_last[b, t, h, w])
                            t1 = int(vis_maps_next[b, t, h, w])

                            if t0 == t1:
                                # Pixel observed at t (i.e., non-occluded)
                                interpolated[b, t, :, h, w] = images[b, t, :, h, w]
                            else:
                                # Retrieve associated reflectances
                                refl0 = images[b, t0, :, h, w]
                                refl1 = images[b, t1, :, h, w]

                                # Linear interpolation between (t0, refl0) and (t1, refl1) by taking the temporal
                                # sampling into account
                                interpolated[b, t, :, h, w] = refl0 + (refl1 - refl0) * (
                                            (days[b, t] - days[b, t0]) / (days[b, t1] - days[b, t0]))

                        elif ~np.isnan(vis_maps_last[b, t, h, w]):
                            # Keep the last observation
                            interpolated[b, t, :, h, w] = images[b, int(vis_maps_last[b, t, h, w]), :, h, w]

                        elif ~np.isnan(vis_maps_next[b, t, h, w]):
                            # Take the next observation
                            interpolated[b, t, :, h, w] = images[b, int(vis_maps_next[b, t, h, w]), :, h, w]

        return interpolated

    @staticmethod
    def visualize_visibility_maps(
            vis_maps: np.ndarray,
            figsize: Tuple[int, int] = (18, 12),
            nrows: Optional[int] = None,
            ncols: Optional[int] = None,
            colormap: str = 'Paired',
            fontsize: int = 10
    ) -> matplotlib.figure.Figure:
        """
        Plot visibility maps (used for debugging purposes).

        Args:
            vis_maps:   np.ndarray, T x H x W, output of either _find_last_visible() or _find_next_visible().
            figsize:    (int, int), figure size.
            nrows:      int, number of rows.
            ncols:      int, number of columns.
            colormap:   str, color map.
            fontsize:   int, font size.

        Returns:
            matplotlib.figure.Figure.
        """

        maps = vis_maps.copy()
        np.nan_to_num(maps, copy=False, nan=-9999)

        # Number of time steps
        T = maps.shape[0]

        # Grid setup
        if nrows is None and ncols is None:
            ncols = min(T, 10)
            nrows = math.ceil(T / ncols)
        elif nrows is not None:
            nrows = min(T, nrows)
            ncols = math.ceil(T / nrows)
        elif ncols is not None:
            ncols = min(T, ncols)
            nrows = math.ceil(T / ncols)

        # Set up figure and image grid
        fig = plt.figure(figsize=figsize)
        grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                         nrows_ncols=(nrows, ncols),
                         axes_pad=0.15,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.15,
                         )

        # Bounds for colormap
        vmin = -1
        vmax = T - 1
        cmap = plt.get_cmap(colormap, vmax - vmin + 1)

        # Add data to image grid
        for t in range(T):
            grid[t].set_axis_off()
            h = grid[t].imshow(maps[t, :, :], cmap=cmap, vmin=vmin - 0.5, vmax=vmax + 0.5)

        cbar = grid.cbar_axes[0].colorbar(h)

        # Tell the colorbar to tick at integers
        cbar.set_ticks(np.arange(vmin, vmax + 1))

        # Colorbar tick labels
        ticklabels = [str(label) for label in np.arange(vmin, vmax + 1)]
        ticklabels[0] = 'nan'
        cbar.set_ticklabels(ticklabels, fontsize=fontsize)

        return fig
