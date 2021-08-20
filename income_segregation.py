import os
import shutil

import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import palettable
import pandas as pd
import proplot as pplt
from libpysal.weights import Queen

from geosnap import datasets
from geosnap.analyze import segdyn
from segregation import singlegroup
from geosnap import datasets
from geosnap import io
from geosnap._data import data_dir

# get 
io.store_census()
if not os.path.exists(data_dir+"/acs/"):
    io.store_acs()


msas = datasets.msa_definitions()
bgs = datasets.acs(year=2018, level="bg")[["geoid", "geometry"]]
cols = ["very_low_inc", "low_inc", "med_inc", "high_inc", "very_high_inc"]


def group_incomes(df):
    #
    #
    # The proper way to to this is:
    #    - for each variable create a dict of var:upper_bound
    #    - convert each upper_bound into constant 2018 dollars
    #    - for each year, aggregate cols as best as possible to match
    #      the year's quintile breaks given at https://www.taxpolicycenter.org/statistics/household-income-quintiles
    #      maybe easier to take the average quintile breaks over the entire period

    #  variable descriptions:   https://api.census.gov/data/2016/acs/acs5/groups/B19001.html
    #  US income quintiles:    https://www.taxpolicycenter.org/statistics/household-income-quintiles

    cols = ["very_low_inc", "low_inc", "med_inc", "high_inc", "very_high_inc"]
    df = df.copy()

    df['very_low_inc'] = df['B19001_002E'] + df['B19001_003E'] + df['B19001_004E'] + df['B19001_005E']  #  $25,000
    df['low_inc'] = df['B19001_006E'] + df['B19001_007E'] + df['B19001_008E'] + df['B19001_009E'] + df['B19001_010E']  #  $50,000
    df['med_inc'] = df['B19001_011E'] + df['B19001_012E']  #  $75,000
    df['high_inc'] = df['B19001_013E'] + df['B19001_014E'] #  $125,000
    df['very_high_inc'] = df['B19001_015E'] + df['B19001_016E'] + df['B19001_017E'] #  >$125,000
    df["total"] = df["B19001_001E"]
    for col in cols:
        df["share_" + col] = df[col] / df["total"]

    df = df[cols + ["share_" + col for col in cols] + ["total", "year", "geoid"]]
    df = df.fillna(0)

    return df


def generate_dataset(msa_fips):

    dfs = []

    for year in range(2012, 2019):
        df = pd.read_parquet(f"data/acs/acs_{year}_X19_INCOME_bg.parquet")
        df["geoid"] = df.reset_index().GEOID.str[7:].values
        df = df[
            df.geoid.str[:5].isin(msas[msas["CBSA Code"] == msa_fips].stcofips.unique())
        ]
        df["year"] = year
        df = group_incomes(df)
        dfs.append(df)

    df = pd.concat(dfs)
    df = bgs.merge(df, on="geoid", how="inner")
    df = df.to_crs(df.estimate_utm_crs())
    return df


def store_data(msa):
    # only create the data if the msa directory is absent
    if not os.path.exists(f"data/{msa}/"):
        os.mkdir(f"data/{msa}/")
        df = generate_dataset(msa)
        df.to_parquet(f"data/{msa}/{msa}_income_data.parquet")
        try:
            # some metros will fail, like PR where we don't have data, or places with islands
            calc_indices(df, msa)
        except Exception as e:
            print(f"{msa} failed with {e}")
            # trash the entire metro folder if something fails
            shutil.rmtree(f"data/{msa}/")
    else:
        pass


def store_data_w_islands(msa):
    # If the dataframe is not a single connected component boundary spatial dissim will fail
    # so this function pulls out the largest component and treats that as the region

    # only create the data if the msa directory is absent
    if not os.path.exists(f"data/{msa}/"):
        os.mkdir(f"data/{msa}/")
        df = generate_dataset(msa)
        w = Queen.from_dataframe(df)
        df["labels"] = w.component_labels
        largest_component = (
            df.groupby("labels").size().sort_values(ascending=False).index[0]
        )
        df = df[df.labels == largest_component]
        df.to_parquet(f"data/{msa}/{msa}_income_data.parquet")
        try:
            # some metros will fail, like PR where we don't have data
            calc_indices(df, msa)
        except Exception as e:
            print(f"{msa} failed with {e}")
            # trash the entire metro folder if something fails
            shutil.rmtree(f"data/{msa}/")
    else:
        pass


def calc_indices(df, msa):

    multi_by_time = segdyn.multigroup_tempdyn(df, cols)
    multi_by_time.T.to_parquet(f"data/{msa}/{msa}_multigroup.parquet")

    # very high income
    single_by_time_vhi = segdyn.singlegroup_tempdyn(
        df, group_pop_var="very_high_inc", total_pop_var="total",
    )
    single_by_time_vhi.T.to_parquet(f"data/{msa}/{msa}_singlegroup_high.parquet")
    # very low income
    single_by_time_vlo = segdyn.singlegroup_tempdyn(
        df, group_pop_var="very_low_inc", total_pop_var="total",
    )
    single_by_time_vlo.T.to_parquet(f"data/{msa}/{msa}_singlegroup_low.parquet")

    # very high income
    spacetime_d_hi = segdyn.spacetime_dyn(
        df,
        singlegroup.Entropy,
        group_pop_var="very_high_inc",
        total_pop_var="total",
        distances=list(range(500, 5500, 500)),
    )
    spacetime_d_hi.columns = spacetime_d_hi.columns.values.astype(str)
    spacetime_d_hi.to_parquet(f"data/{msa}/{msa}_spacetime_entropy_high.parquet")
    # very low income
    spacetime_d_lo = segdyn.spacetime_dyn(
        df,
        singlegroup.Entropy,
        group_pop_var="very_low_inc",
        total_pop_var="total",
        distances=list(range(500, 5500, 500)),
    )
    spacetime_d_lo.columns = spacetime_d_lo.columns.values.astype(str)
    spacetime_d_lo.to_parquet(f"data/{msa}/{msa}_spacetime_entropy_low.parquet")

    # very high income
    spacetime_i_hi = segdyn.spacetime_dyn(
        df,
        singlegroup.Isolation,
        group_pop_var="very_high_inc",
        total_pop_var="total",
        distances=list(range(500, 5500, 500)),
    )
    spacetime_i_hi.columns = spacetime_i_hi.columns.values.astype(str)
    spacetime_i_hi.to_parquet(f"data/{msa}/{msa}_spacetime_isolation_high.parquet")
    # very low income
    spacetime_i_lo = segdyn.spacetime_dyn(
        df,
        singlegroup.Isolation,
        group_pop_var="very_low_inc",
        total_pop_var="total",
        distances=list(range(500, 5500, 500)),
    )
    spacetime_i_lo.columns = spacetime_i_lo.columns.values.astype(str)
    spacetime_i_lo.to_parquet(f"data/{msa}/{msa}_spacetime_isolation_low.parquet")


def plot_trend_graphs(group, msa_name, msa_fips, dpi=300):
    path = f"../figures/{msa_fips}/singlegroup/"
    if not os.path.exists(path):
        os.makedirs(path)
        try:
            df = pd.read_parquet(
                f"data/{msa_fips}/{msa_fips}_singlegroup_high.parquet"
            )
            for i, row in df.T.iterrows():

                fig, axs = plt.subplots(1, 2, figsize=(10, 4))

                df.T.loc[f"{i}"].plot(ax=axs[0])
                df.T.loc[f"{i}"].plot(kind="bar", ax=axs[1])

                fig.suptitle(f"{msa_name}\n{group} {i}")
                plt.savefig(
                    f"{path}{msa_fips}_{i.lower()}_{group.split()[0].lower()}.png",
                    dpi=dpi,
                )
                plt.close("all")
        except Exception as e:
            print(f"{msa_fips} failed with {e}")
            # trash the entire metro folder if something fails
            shutil.rmtree(path)


def plot_multiscalar_graphs(msa_name, msa_fips, dpi=300):
    path = f"../figures/{msa_fips}/multiscalar/"
    if not os.path.exists(path):
        os.makedirs(path)
        try:
            for idx in ["entropy", "isolation"]:
                fig, ax = pplt.subplots(figsize=(6, 4), axpad=0.45)

                pd.read_parquet(
                    f"/Users/knaaptime/Dropbox/projects/incseg/data/{msa_fips}/{msa_fips}_spacetime_{idx}_low.parquet"
                ).plot(
                    cmap=palettable.colorbrewer.sequential.Blues_8.mpl_colormap,
                    ax=ax,
                    legend=False,
                )
                pd.read_parquet(
                    f"/Users/knaaptime/Dropbox/projects/incseg/data/{msa_fips}/{msa_fips}_spacetime_{idx}_high.parquet"
                ).plot(
                    cmap=palettable.colorbrewer.sequential.Reds_8.mpl_colormap,
                    ax=ax,
                    legend=False,
                )

                cb1 = fig.colorbar(
                    palettable.colorbrewer.sequential.Blues_7.hex_colors,
                    values=list(range(2012, 2019, 1)),
                    formatter=fmtr,
                    tickloc="left",
                    loc="r",
                    length=0.6,
                )
                cb2 = fig.colorbar(
                    palettable.colorbrewer.sequential.Reds_7.hex_colors,
                    values=list(range(2012, 2019, 1)),
                    tickloc="right",
                    loc="r",
                    formatter=fmtr,
                    length=0.6,
                )

                cb1.set_label("Low Income", labelpad=-6, fontsize=10)
                cb2.set_label("High Income", labelpad=-6, fontsize=10)

                fig.suptitle(
                    f"{msa_name}\nMultiscalar {idx.title()} Segregation Profiles"
                )
                plt.savefig(f"{path}{msa_fips}_multiscalar_{idx}.png", dpi=dpi)
                plt.close("all")
        except Exception as e:
            print(f"{msa_fips} failed with {e}")
            # trash the entire metro folder if something fails
            shutil.rmtree(path)


def fmtr(x, pos):
    if x in [2012, 2018]:
        return str(int(x))
    return None


def gen_multi_fig(fips, profile_idx):
    df1 = pd.read_parquet(f"data/{fips}/{fips}_spacetime_{profile_idx}_high.parquet")
    df2 = pd.read_parquet(f"data/{fips}/{fips}_spacetime_{profile_idx}_low.parquet")

    for idx in ["entropy", "isolation"]:
        fig, ax = pplt.subplots(axpad=0.45)

        df1.plot(
            cmap=palettable.colorbrewer.sequential.Blues_8.mpl_colormap,
            ax=ax,
            legend=False,
        )
        df2.plot(
            cmap=palettable.colorbrewer.sequential.Reds_8.mpl_colormap,
            ax=ax,
            legend=False,
        )

        cb1 = fig.colorbar(
            palettable.colorbrewer.sequential.Blues_7.hex_colors,
            values=list(range(2012, 2019, 1)),
            formatter=fmtr,
            tickloc="left",
            loc="r",
            length=0.6,
        )
        cb2 = fig.colorbar(
            palettable.colorbrewer.sequential.Reds_7.hex_colors,
            values=list(range(2012, 2019, 1)),
            tickloc="right",
            loc="r",
            formatter=fmtr,
            length=0.6,
        )

        cb1.set_label("Low Income", labelpad=-6, fontsize=10)
        cb2.set_label("High Income", labelpad=-6, fontsize=10)

        fig.suptitle(f"Multiscalar {idx.title()} Profiles")
    return fig


def gen_single(df, index):
    df = df.copy()

    fig = df.loc[index].hvplot.bar(
        width=300, height=260, title=f"Absolute Change in {index}", grid=True
    ) + df.loc[index].to_frame().apply(lambda x: x.div(x.iloc[0]).subtract(1).mul(100)).hvplot(
        width=300, height=260, title=f"Relative (%) Change in {index}", shared_axes=False, grid=True
    )
    return hv.render(fig)


def gen_multi(df):

    fig = df.T.hvplot(title=f"Multigroup Income Segregation by Index Over Time", grid=True, height=420, width=650)
    return hv.render(fig)


def generate_delta_text(index_name, value):
    direction = 'grew' if value >0 else "fell"
    txt = f"**The {index_name} index {direction} by {np.round(abs(value),2)} percent** between the 2008-2012 and 2014-2018 ACS sampling periods  \n\n"
    return txt

def get_delta(df, index_name):
    return df.loc[index_name].to_frame().apply(lambda x: x.div(x.iloc[0]).subtract(1).mul(100)).values[6][0]

def plot_all_single(df, group):
    all_single_group = df.T.hvplot(height=560, width=650, shared_axes=False, grid=True, title=f"{group.title()} Income Segregation by Index Over Time")
    return hv.render(all_single_group)
