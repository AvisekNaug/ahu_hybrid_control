import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('pdf')

def rl_reward_save(train_metrics, rllogs):
    f = open(rllogs + '/Cumulative Episode Rewards.txt', 'w')
    for item in train_metrics.rewardsTrace:
        f.write("{}\n".format(item))
    f.close()

def rl_perf_save(test_perf_log, logs):

    # assert that perf metric has data from at least one episode
    assert len(test_perf_log.metrics) != 0, 'Need metric data for at least one episode'

    # performance metrics in a list where each element has
    # performance data for each episode in a dict
    perf_metric_list = test_perf_log.metrics

    # iterating through the list to save the data
    for episode_dict in perf_metric_list:
        for key, value in episode_dict.items():
            f = open(logs + '/' + key + '.txt', 'a+')
            f.writelines("%s\n" % j for j in value)
            f.close()

def rl_reward_plot(datapath, saveplotpath):

    # open file and read the content in a list
    with open(datapath, 'r') as f:
        rewardlist = [float(i.rstrip()) for i in f.readlines()]

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    # width as measured in inkscape
    width = 10.487
    height = width / 1.618
    plt.rcParams["figure.figsize"] = (width, height)

    fig, ax = plt.subplots()
    ax.plot(rewardlist)
    ax.set_ylabel('Cumulative reward per episode', fontsize=12)
    ax.set_xlabel('Episode Number', fontsize=12)
    plt.grid(which='both', linewidth=0.2)
    plt.title('Progress of cumulative reward per episode \n over number of episodes')
    # plt.show()
    fig.savefig(saveplotpath + 'Cumulative Reward.pdf',bbox_inches='tight')
    # plt.close(fig) #remove in jupyter

def rl_energy_compare(original_energy_data_path, oatpath, rl_energy_data_path, saveplotpath, period=1):
    # open file and read the content in a list
    with open(original_energy_data_path, 'r') as f:
        old_energy = [float(i.rstrip()) for i in f.readlines()]
        # open file and read the content in a list
    with open(rl_energy_data_path, 'r') as f:
        rl_energy = [float(i.rstrip()) for i in f.readlines()]
        # open file and read the content in a list
    with open(oatpath, 'r') as f:
        oat = [float(i.rstrip()) for i in f.readlines()]

        # energy savings
    energy_savings = sum([i - j for i, j in zip(old_energy, rl_energy)])

    rl_energy_pht = np.ma.masked_where(np.array(oat) > 52.0, rl_energy)
    rl_energy_rht = np.ma.masked_where(np.array(oat) <= 52.0, rl_energy)

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    # width as measured in inkscape
    width = 10.487
    height = width / 1.618
    plt.rcParams["figure.figsize"] = (width, height)

    # create the plot
    fig, ax = plt.subplots()
    ax.plot(old_energy, 'r--', label='historical setpoint based energy')
    ax.plot(rl_energy_pht, 'g--', label='controller setpoint based energy in Preheat Mode')
    ax.plot(rl_energy_rht, 'b--', label='controller setpoint based energy  in Reheat Mode')
    ax.set_title('Comparison of historical and controller \n setpoint based energy consumption')
    ax.set_xlabel('Time points at {} mins'.format(period * 5))
    ax.set_ylabel('Energy in kJ')
    ax.grid(which='both', linewidth=0.2)
    plt.text(0.95, 0.95, 'Energy Savings: {0:.2f} kJ'.format(energy_savings), fontsize=9,
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax.transAxes)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.10))
    # plt.show()
    fig.savefig(saveplotpath + 'Energy Comparison.pdf', bbox_inches='tight')
    plt.close(fig)

def oat_vs_set_point_plot(oat_data_path, pht_data_path, sat_data_path, saveplotpath, period=1):

    # open file and read the content in a list
    with open(oat_data_path, 'r') as f:
        oat = [float(i.rstrip()) for i in f.readlines()]
    # open file and read the content in a list
    with open(pht_data_path, 'r') as f:
        pht = [float(i.rstrip()) for i in f.readlines()]
    # open file and read the content in a list
    with open(sat_data_path, 'r') as f:
        sat = [float(i.rstrip()) for i in f.readlines()]

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    # width as measured in inkscape
    width = 10.487
    height = width / 1.618
    plt.rcParams["figure.figsize"] = (width, height)

    fig, ax = plt.subplots()
    # fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
    ax.plot(sat, 'g--', label='Controller Discharge Air Temperature')
    ax.plot(pht, 'k+-', label='Controller Preheat Air Temperature')
    ax.set_ylabel('Temperature in F', fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.15))
    ax.set_xlabel('Time points at {} mins'.format(period * 5), fontsize=12)
    ax1 = ax.twinx()
    ax1.plot(oat, 'r--', label='Outside Air Temperature')
    ax1.set_ylabel('Outside Air Temperature in F', fontsize=12)
    ax1.legend(loc='upper left', bbox_to_anchor=(0, -0.35))
    plt.grid(which='both', linewidth=0.2)
    plt.title('Comparison of Outside air temperature, \n controller preheat air temperature'
              ' \n and controller discharge air temperature')
    # plt.show()
    fig.savefig(saveplotpath + 'OATvsSATvsPHT' + '.pdf', bbox_inches='tight')
    plt.close(fig)

def oat_vs_rht_plot(oat_data_path, sat_data_path, saveplotpath, period=1):

    # open file and read the content in a list
    with open(oat_data_path, 'r') as f:
        oat = [float(i.rstrip()) for i in f.readlines()]
    # open file and read the content in a list
    with open(sat_data_path, 'r') as f:
        sat = [float(i.rstrip()) for i in f.readlines()]

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    # width as measured in inkscape
    width = 10.487
    height = width / 1.618
    plt.rcParams["figure.figsize"] = (width, height)

    fig, ax = plt.subplots()
    # fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
    ax.plot(sat, 'g--', label='Controller Discharge Air Temperature')
    ax.set_ylabel('Reheat Discharge Temperature in F', fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.15))
    ax.set_xlabel('Time points at {} mins'.format(period * 5), fontsize=12)
    ax1 = ax.twinx()
    ax1.plot(oat, 'r--', label='Outside Air Temperature')
    ax1.set_ylabel('Outside Air Temperature in F', fontsize=12)
    ax1.legend(loc='upper left', bbox_to_anchor=(0, -0.35))
    plt.grid(which='both', linewidth=0.2)
    plt.title('Comparison of Outside air temperature, \n controller preheat air temperature'
              ' \n and controller discharge air temperature')
    # plt.show()
    fig.savefig(saveplotpath + 'OATvsSAT' + '.pdf', bbox_inches='tight')
    plt.close(fig)

def relhumplot(rhdatapath, oat_data_path, rh_histdatapath, saveplotpath, period=1):
    # open file and read the content in a list
    with open(rhdatapath, 'r') as f:
        rh = [float(i.rstrip()) for i in f.readlines()]
        # open file and read the content in a list
    with open(rh_histdatapath, 'r') as f:
        rh_hist = [float(i.rstrip()) for i in f.readlines()]
        # open file and read the content in a list
    with open(oat_data_path, 'r') as f:
        oat = [float(i.rstrip()) for i in f.readlines()]

    rh_pht = np.ma.masked_where(np.array(oat) > 52.0, rh)
    rh_rht = np.ma.masked_where(np.array(oat) <= 52.0, rh)

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    # width as measured in inkscape
    width = 10.487
    height = width / 1.618
    plt.rcParams["figure.figsize"] = (width, height)

    # create the plot
    fig, ax = plt.subplots()
    ax.plot(rh_hist, 'r--', label='historical setpoint based relative humidity')
    ax.plot(rh_pht, 'g--', label='controller setpoint based relative humidity in Preheat Mode')
    ax.plot(rh_rht, 'b--', label='controller setpoint based relative humidity in Reheat Mode')
    ax.set_title('Comparison of historical and controller \n setpoint based relative humidity')
    ax.set_xlabel('Time points at {} mins'.format(period * 5))
    ax.set_ylabel('Relative Humidity')
    ax.grid(which='both', linewidth=0.2)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.10))
    # plt.show()
    fig.savefig(saveplotpath + 'Relative Humidity Comparison.pdf', bbox_inches='tight')
    plt.close(fig)

def oat_vs_control(splotpath, oatpath, saveplotpath, period=1):
    # open file and read the content in a list
    with open(splotpath, 'r') as f:
        splot = [float(i.rstrip()) for i in f.readlines()]
    # open file and read the content in a list
    with open(oatpath, 'r') as f:
        oat = [float(i.rstrip()) for i in f.readlines()]

    splot_low = np.ma.masked_where(np.array(oat) > 52.0, splot)
    splot_high = np.ma.masked_where((np.array(oat) <= 52.0) | (np.array(splot) >= 74.0), splot)
    splot_hhigh = np.ma.masked_where((np.array(oat) <= 52.0) | (np.array(splot) < 74.0), splot)

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    # width as measured in inkscape
    width = 10.487
    height = width / 1.618
    plt.rcParams["figure.figsize"] = (width, height)

    fig, ax = plt.subplots()
    # fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
    l1 = ax.plot(splot_low - 6 + np.random.normal(loc=0.0, scale=0.15, size=splot_low.shape), 'g--',
                 label='Controller Discharge Air Temperature in Preheat Mode')
    l2 = ax.plot(splot_high - 6 + np.random.normal(loc=0.0, scale=0.15, size=splot_high.shape), 'b--',
                 label='Controller Discharge Air Temperature in Reheat Mode')
    ax.plot(splot_hhigh - 6 + np.random.normal(loc=0.0, scale=0.09, size=splot_hhigh.shape), 'b--')
    ax.set_ylabel('Reheat Discharge Temperature in F', fontsize=12)
    # ax.legend(loc='upper left', bbox_to_anchor=(0, -0.15))
    ax.set_xlabel('Time points at {} mins'.format(period * 5), fontsize=12)
    ax1 = ax.twinx()
    l3 = ax1.plot(oat, 'r--', label='Outside Air Temperature')
    ax1.set_ylabel('Outside Air Temperature in F', fontsize=12)
    # ax1.legend(loc='upper left', bbox_to_anchor=(0, -0.25))
    ax1.axhline(y=52)
    plt.grid(which='both', linewidth=0.2)
    plt.title('Comparison of Outside air temperature, \n controller Setpoint temperature')
    # added these three lines
    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='upper left', bbox_to_anchor=(0, -0.10))
    # plt.legend(loc='upper left', bbox_to_anchor=(0, -0.10))
    # plt.show()
    fig.savefig(saveplotpath + 'OATvsController' + '.pdf', bbox_inches='tight')
    plt.close(fig)
