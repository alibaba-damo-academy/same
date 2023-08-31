import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from SAMReg.tools.utils.med import get_identity

def plot_grid(x, y, ax=None, **kwargs):
    '''
    Plot grids. The bottom left corner is the origin.
    x is shown along the vertial axis in the plot.
    '''
    ax = ax or plt.gca()
    segs1 = np.stack((y.transpose(1, 0), x.transpose(1, 0)), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))

def plot_at(ax, pos, steps, phi=None, img=None):
    '''
    Plot the image and phi at position.

    Args:
        ax (Matplotlib.Figure.ax): the axes to draw the plots.
        pos (list): a list containing x, y, z.
        steps (int): steps used to sample phi.
        phi (numpy): DxWxHx3. This should be in canonical space [-1,1]. And phi aligns with pytorch grids representation where
        the vector in 4th dim contains position in H,W,D axis order.
        img (numpy): DxWxH.
    '''

    assert not (phi is None and img is None), 'phi and img cannot be None at the same time.'
    
    if img is not None:
        ax[0].imshow(img[pos[0]], origin='lower', cmap='nipy_spectral')
        ax[1].imshow(img[:, pos[1]], origin='lower', cmap='nipy_spectral')
        ax[2].imshow(img[:, :, pos[2]], origin='lower', cmap='nipy_spectral')

    if phi is not None:
        if img is not None:
            # change phi to image space
            shape = np.array(img.shape)
            phi = (phi + 1.) / 2. * (shape - 1)
        plot_grid(phi[pos[0], ::steps, ::steps, 1], phi[pos[0], ::steps, ::steps, 2], ax[0], color='pink', linewidth=0.5)
        ax[0].set_title('0th dim')
        plot_grid(phi[::steps, pos[1], ::steps, 0], phi[::steps, pos[1], ::steps, 2], ax[1], color='pink', linewidth=0.5)
        ax[1].set_title('1th dim')
        plot_grid(phi[::steps, ::steps, pos[2], 0], phi[::steps, ::steps, pos[2], 1], ax[2], color='pink', linewidth=0.5)
        ax[2].set_title('2th dim')

    return ax

def plot_comparison_at(pos, steps, phi, source, warped, target, save_to=""):
    '''
    Plot the registration result.

    Args:
        pos (list): a list containing x, y, z.
        steps (int): steps used to sample phi.
        phi (numpy): DxWxHx3. This should be in canonical space [-1,1]. The vector in 4th dim contains position in D,W,H axis order.
        source (numpy): DxWxH.
        warped (numpy): DxWxH.
        target (numpy): DxWxH.
    '''
    fig, axes = plt.subplots(4,3)

    plot_at(axes[0], pos, steps, phi, source)
    axes[0,0].set_ylabel('source')

    plot_at(axes[1], pos, steps, img=warped)
    axes[1,0].set_ylabel('warped')

    identity_field = get_identity(source.shape, in_canonical=True, inverse_coord=False)
    plot_at(axes[2], pos, steps, identity_field, warped)
    axes[2,0].set_ylabel('warped+grids')

    plot_at(axes[3], pos, steps, img=target)
    axes[3,0].set_ylabel('target')

    if save_to is not "":
        plt.savefig(save_to)
    else:
        return fig

if __name__ is '__main__':
    phi = np.random.rand(10, 20, 30, 3)
    img = np.random.rand(10, 20, 30)
    fig = plot_at((5,5,5), 1, phi, img)
    plt.savefig("./tmp/temp.png")

