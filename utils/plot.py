# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.ticker as ticker
import numpy as np
from utils.misc import to_numpy
plt.switch_backend('agg')


def plot_prediction_det(save_dir, target, prediction, epoch, index, 
                        plot_fn='contourf'):
    
    """Plot prediction for one input (`index`-th at epoch `epoch`)
    save_dir = args.pred_dir
    target = samples_target[i]
    prediction = samples_output[i]
    index = i
    plot_fn=args.plot_fn
    Args:
        save_dir: directory to save predictions
        target (np.ndarray):
        prediction (np.ndarray):
        epoch (int): which epoch
        index (int): i-th prediction
        plot_fn (str): choices=['contourf', 'imshow']
    """
    target, prediction = to_numpy(target), to_numpy(prediction)
    

    samples = np.concatenate((target, prediction, target - prediction), axis=0)
    vmin = [np.amin(samples[0]), np.amin(samples[1]),
            np.amin(samples[2])]
    vmax = [np.amax(samples[0]), np.amax(samples[1]),
            np.amax(samples[2])]

    # fig, _ = plt.subplots(3, 3, figsize=(11, 9))
    fig, _ = plt.subplots(3, 1, figsize=(11, 9))
    for j, ax in enumerate(fig.axes):
        ax.set_aspect('equal')
        ax.set_axis_off()
        if j < 3:  # j < 6:
            if plot_fn == 'contourf':
                cax = ax.contourf(samples[j], 50, cmap='jet',
                                  vmin=vmin[j % 3], vmax=vmax[j % 3])
            elif plot_fn =='imshow':
                cax = ax.imshow(samples[j], cmap='jet', origin='lower',
                                interpolation='bilinear',
                                vmin=vmin[j % 3], vmax=vmax[j % 3])   
        else:
            if plot_fn == 'contourf':
                cax = ax.contourf(samples[j], 50, cmap='jet')
            elif plot_fn =='imshow':
                cax = ax.imshow(samples[j], cmap='jet', origin='lower',
                                interpolation='bilinear')
        if plot_fn == 'contourf':
            for c in cax.collections:
                c.set_edgecolor("face")
                c.set_linewidth(0.000000000001)
        cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                            format=ticker.ScalarFormatter(useMathText=True))
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.ax.yaxis.set_offset_position('left')
        # cbar.ax.tick_params(labelsize=5)
        cbar.update_ticks()
    plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    plt.savefig(save_dir + '/pred_epoch{}_{}.png'.format(epoch, index),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_stats(save_dir, logger, x_axis):

    rmse_train = logger['rmse_train']
    rmse_test = logger['rmse_test']
    r2_train = logger['r2_train']
    r2_test = logger['r2_test']

    if 'mnlp_test' in logger.keys():
        mnlp_test = logger['mnlp_test']
        if len(mnlp_test) > 0:
            plt.figure()
            plt.plot(x_axis, mnlp_test, label="Test: {:.3f}".format(np.mean(mnlp_test[-5:])))
            plt.xlabel('Epoch')
            plt.ylabel('MNLP')
            plt.legend(loc='upper right')
            plt.savefig(save_dir + "/mnlp_test.pdf", dpi=600)
            plt.close()
            np.savetxt(save_dir + "/mnlp_test.txt", mnlp_test)
    
    if 'log_beta' in logger.keys():
        log_beta = logger['log_beta']
        if len(log_beta) > 0:
            plt.figure()
            plt.plot(x_axis, log_beta, label="Test: {:.3f}".format(np.mean(log_beta[-5:])))
            plt.xlabel('Epoch')
            plt.ylabel('Log-Beta (noise precision)')
            plt.legend(loc='upper right')
            plt.savefig(save_dir + "/log_beta.pdf", dpi=600)
            plt.close()
            np.savetxt(save_dir + "/log_beta.txt", log_beta)

    plt.figure()
    plt.plot(x_axis, r2_train, label="Train: {:.3f}".format(np.mean(r2_train[-5:])))
    plt.plot(x_axis, r2_test, label="Test: {:.3f}".format(np.mean(r2_test[-5:])))
    plt.xlabel('Epoch')
    plt.ylabel(r'$R^2$-score')
    plt.legend(loc='lower right')
    plt.savefig(save_dir + "/r2.pdf", dpi=600)
    plt.close()
    np.savetxt(save_dir + "/r2_train.txt", r2_train)
    np.savetxt(save_dir + "/r2_test.txt", r2_test)

    plt.figure()
    plt.plot(x_axis, rmse_train, label="train: {:.3f}".format(np.mean(rmse_train[-5:])))
    plt.plot(x_axis, rmse_test, label="test: {:.3f}".format(np.mean(rmse_test[-5:])))
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend(loc='upper right')
    plt.savefig(save_dir + "/rmse.pdf", dpi=600)
    plt.close()
    np.savetxt(save_dir + "/rmse_train.txt", rmse_train)
    np.savetxt(save_dir + "/rmse_test.txt", rmse_test)


