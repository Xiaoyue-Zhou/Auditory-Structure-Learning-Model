import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import colors

def plot_day1_byType(data, measure, title, my_color):
    trial_order = ['within_legal',
               'between_legal',
               'within_illegal',
               'between_illegal']
    
    avg_within_subj = (
        data.groupby(['subj', 'trial_type'], as_index=False)
        .agg(value = (measure, 'mean'))
    )

    dat4plot = (
        avg_within_subj.groupby('trial_type', as_index=False)
        .agg(mean_dat = ('value', 'mean'),
            sd_dat = ('value', 'std'),
            n_subj = ('value', 'count'))
    )
    dat4plot['se_dat'] = dat4plot['sd_dat'] / np.sqrt(dat4plot['n_subj'])
    dat4plot['trial_type'] = pd.Categorical(
        dat4plot['trial_type'],
        categories=trial_order,
        ordered=True
    )

    dat4plot = dat4plot.sort_values('trial_type').reset_index(drop=True)

    # plot figures
    fig, ax = plt.subplots(figsize=(3, 2.5))
    x=np.arange(len(dat4plot))
    ax.bar(
        x, 
        dat4plot['mean_dat'],
        # yerr=dat4plot['se_dat'],
        width=0.4,
        color=my_color,
        capsize=5,
        error_kw={
            "elinewidth": 1.5,
            "capthick": 1.5
        },
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Within-legal",
            "Between-legal",
            "Within-illegal",
            "Between-illegal",
        ],
        rotation=20,
        ha='right'
    )

    ax.set_xlabel("Trial type")
    ax.set_ylabel(measure)
    ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return ax

def plot_day2_byType(data_s1, data_s2, measure, ylab, title):
    trial_order = ['within_legal',
               'between_legal',
               'within_illegal',
               'between_illegal']
    palette_whole = sns.color_palette('Paired', 12)
    my_color = [palette_whole[i] for i in [0, 2, 6, 4]]
    my_color_s2 = [palette_whole[i] for i in [1, 3, 7, 5]]
    
    dat_full = pd.concat([data_s1, data_s2], ignore_index=True)

    avg_within_subj = (
        dat_full
        .groupby(['subj', 'session', 'trial_type'], as_index=False)
        .agg(value = (measure, 'mean'))
    )

    session_order = ['Day1', 'Day2']

    fig, ax = plt.subplots(
        figsize=(5, 3)
    )

    sns.barplot(
        data=avg_within_subj,
        x="trial_type",
        y="value",
        hue="session",
        order=trial_order,
        hue_order=session_order,
        estimator="mean",
        errorbar=None,
        capsize=0.08,
        dodge=True,
        ax=ax
    )

    for container, colors in zip(ax.containers, [my_color, my_color_s2]):
        for bar, color in zip(container, colors):
            bar.set_facecolor(color)
            bar.set_edgecolor('none')
    
    trial_labels = {
        "within_legal": "Within\nlegal",
        "between_legal": "Between\nlegal",
        "within_illegal": "Within\nillegal",
        "between_illegal": "Between\nillegal"
    }

    ax.set_xticklabels(
        [
            trial_labels.get(t, t)
            for t in trial_order
        ]
    )

    ax.set_xlabel("")
    ax.set_ylabel(ylab)
    ax.set_title(title)

    ax.legend(
        title=None,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5)
    )

    sns.despine(ax=ax)

    fig.tight_layout()
    return fig, ax

def plot_delta_byType(data_s1, data_s2, measure, ylab, title, colors=None):
    trial_order = ['within_legal',
               'between_legal',
               'within_illegal',
               'between_illegal']
    
    d1 = (
        data_s1
        .groupby(['subj', 'trial_type'], as_index=False)
        .agg(value = (measure, 'mean'))
    )
    d2 = (
        data_s2
        .groupby(['subj', 'trial_type'], as_index=False)
        .agg(value = (measure, 'mean'))
    )

    delta_subj = d1.merge(
        d2,
        on=['subj', 'trial_type'],
        how='inner',
        suffixes=('_s1', '_s2')
    )
    delta_subj['delta'] = (
        delta_subj['value_s2'] - delta_subj['value_s1']
    )
    delta_subj['trial_type'] = pd.Categorical(
        delta_subj['trial_type'],
        categories=trial_order,
        ordered=True
    )
    delta_subj = delta_subj.sort_values('trial_type').reset_index(drop=True)

    if colors is None:
        palette_whole = sns.color_palette('Paired', 12)
        colors = [palette_whole[i] for i in [0, 2, 6, 4]]
    
    fig, ax = plt.subplots(figsize=(3, 2.5))
    sns.barplot(data = delta_subj, 
                x='trial_type', 
                y='delta', 
                order=trial_order,
                errorbar=None,
                hue='trial_type',
                hue_order=trial_order,
                dodge=False,
                palette=colors, 
                width=0.4,
                ax=ax)
    
    ax.set_xticklabels(
        [
            "Within-legal",
            "Between-legal",
            "Within-illegal",
            "Between-illegal",
        ],
        rotation=20,
        ha='right'
    )
    ax.set_xlabel('Trial type')
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    return ax

def summarize_cross_block(data, measure, trial_order):
    # check behavioural change within session 1
    subj_blc_data = (
        data
        .groupby(['subj', 'block', 'trial_type'], 
                as_index=False)
        .agg(value = (measure, 'mean'))
    )

    group_blc_data = (
        subj_blc_data
        .groupby(['block', 'trial_type'],
                as_index=False)
        .agg(mean_val = ('value', 'mean'),
            sd_val = ('value', 'std'),
            n_subj = ('value', 'count'))
    )
    group_blc_data['se_val'] = (
        group_blc_data['sd_val'] 
        / np.sqrt(group_blc_data['n_subj'])
    )
    group_blc_data['trial_type'] = pd.Categorical(
        group_blc_data['trial_type'],
        categories=trial_order,
        ordered=True
    )
    group_blc_data = (
        group_blc_data
        .sort_values(['trial_type', 'block'])
        .reset_index(drop=True)
    )

    return subj_blc_data, group_blc_data

def plot_cross_block(group_data, 
                     ylabel, 
                     title, 
                     colors,
                     ax=None,
                     show_legend=True):
    labels = {
        "within_legal": "within-legal",
        "between_legal": "between-legal",
        "within_illegal": "within-illegal",
        "between_illegal": "between-illegal",
    }
    trial_order = ['within_legal',
               'between_legal',
               'within_illegal',
               'between_illegal']
    

    if ax is None:
        fig, ax = plt.subplots(figsize=(3,2.5))
    else:
        fig = ax.figure
    
    for iType, trial_type in enumerate(trial_order):
        condition_data = (
            group_data[group_data['trial_type'] == trial_type]
            .sort_values('block')
        )

        block_number = condition_data['block'].to_numpy() + 1
        ax.errorbar(
            block_number,
            condition_data["mean_val"],
            yerr=condition_data["se_val"],
            marker="o",
            markersize=5,
            linewidth=1.8,
            capsize=3,
            color=colors[iType],
            label=labels[trial_type],
        )

    ax.set_xticks(
        sorted(group_data['block'].unique() + 1)
    )
    ax.set_xlabel('Test block')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim((0.1, 0.9))

    if show_legend:
        ax.legend(frameon=False, 
                fontsize=6,
                loc="best")
        
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return ax