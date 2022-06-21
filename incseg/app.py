from pathlib import Path

import altair as alt
import contextily as ctx
import holoviews as hv
import hvplot.pandas
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static

from geosnap import datasets
from geosnap.analyze import segdyn
from geosnap.io.storage import _fips_filter
from income_segregation import gen_multi, gen_single, generate_delta_text
from incseg.income_segregation import generate_delta_text, get_delta, plot_all_single
from bokeh.plotting import figure
from bokeh.io import show
import geopandas as gpd

toner_lite = ctx.providers.Stamen.TonerLite

st.set_page_config(
    page_title="Trends in American Income Segregation",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache
def get_text():
    intro = Path("intro.md").read_text()
    dimension_text = Path("dimension_text.md").read_text()
    multiscalar_pre = Path("multiscalar_pre.md").read_text()
    multistalar_post = Path("multiscalar_post.md").read_text()
    return intro, dimension_text, multiscalar_pre, multistalar_post


intro, dimension_text, multiscalar_pre, multiscalar_post = get_text()


st.title("Exploring Recent Trends in American Income Segregation")
st.markdown(
    "Choose a metropolitan region and an income group to examine using the sidebar on the left, then scroll down to see how segregation has changed using every ACS dataset available"
)
with st.expander("More Info"):
    st.markdown(intro)
header_col1, header_col2 = st.columns(2)


@st.cache
def get_multi_data(fips):
    df = gpd.read_parquet(f"../data/{fips}/{fips}_income_data.parquet")
    cols = ["very_low_inc", "low_inc", "med_inc", "high_inc", "very_high_inc"]
    multi_by_time = segdyn.multigroup_tempdyn(df, cols,)
    return multi_by_time


@st.cache(allow_output_mutation=True)
def get_fips():
    msas = datasets.msa_definitions()
    return msas.groupby("CBSA Code").first().reset_index()


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_map_data(year):
    return datasets.acs(level="tract", year=year)


unique_fips = get_fips()


st.sidebar.write("# Select a Metropolitan Region")
metro_name = st.sidebar.selectbox("Metro Region", unique_fips["CBSA Title"])
fips = (
    unique_fips[unique_fips["CBSA Title"] == metro_name]["CBSA Code"]
    .astype(str)
    .values[0]
)

st.sidebar.write("## Select an Income Group for Singlegroup Measures")
income_group = st.sidebar.selectbox("Income Group", ["high", "low"])
st.sidebar.markdown(
    """
- High income group is >= $125,000
- Low Income group  is <= $25,000

    """
)
st.sidebar.markdown(" ")
st.sidebar.caption(
    """Each index is calculated with blockgroup-level data from 5-year Census ACS samples. 
Following Census Bureau convention, each dataset is named for the terminal year of the 5-year sample (e.g. ACS 5-year 2018 is the 2014-2018 sample).
Note that the Census Bureau discourages comparisons of ACS data with overlapping samples, so the plots
should be treated as rolling averages
"""
)
st.sidebar.write("  ")
st.sidebar.caption(
    "Powered by [`geosnap`](https://github.com/spatialucr/geosnap) and the PySAL [`segregation`](https://github.com/pysal/segregation) package. Built by [@knaaptime](https://twitter.com/knaaptime)"
)
st.sidebar.write("  ")
st.sidebar.image(
    "https://spatial.ucr.edu/images/cgs_hex_logo_v1.1.png",
    use_column_width="never",
    width=120,
    caption="Center for Geospatial Sciences @ UC Riverside",
)

with header_col1:
    st.markdown(f"## {metro_name}")
    st.subheader("Median Household Income ")

with header_col2:

    map_year = st.select_slider("Map Year", list(range(2012, 2019)), 2018)
    st.empty()

segs_single = pd.read_parquet(
    f"../data/{fips}/{fips}_singlegroup_{income_group}.parquet"
).T


@st.cache
def get_profile_data(fips, profile_idx):
    spacetime_path = f"../data/{fips}/{fips}_spacetime_{profile_idx}_high.parquet"
    spacetime_path2 = f"../data/{fips}/{fips}_spacetime_{profile_idx}_low.parquet"

    multi1 = pd.read_parquet(spacetime_path).reset_index()
    multi1 = pd.melt(multi1, id_vars=["distance"])
    multi1.columns = ["distance", "year", f"{profile_idx} Index"]

    multi2 = pd.read_parquet(spacetime_path2).reset_index()
    multi2 = pd.melt(multi2, id_vars=["distance"])
    multi2.columns = ["distance", "year", f"{profile_idx} Index"]

    return multi1, multi2


def generate_map(df):
    m = df.explore(
        "median_household_income",
        scheme="quantiles",
        k=5,
        cmap="PRGn",
        tooltip=["median_household_income"],
        tooltip_kwds={"aliases": [f"Median Household Income (ACS 5-Year {map_year})"]},
        style_kwds={"weight": 1, "stroke": False},
        legend_kwds={"scale": False},
        tiles=toner_lite,
    )
    return m


@st.cache(show_spinner=False)
def subset_tracts(tracts, fips):
    tracts = tracts.copy()

    df = _fips_filter(msa_fips=fips, data=tracts).dropna(
        subset=["median_household_income"]
    )
    return df


tracts = get_map_data(map_year)
df = subset_tracts(tracts, fips)
multi = get_multi_data(fips)

segs = (
    segs_single.T[["Gini", "Entropy", "Dissim", "Atkinson"]]
    .hvplot(title="Evenness Dimension", width=370, height=450)
    .opts(legend_position="bottom", show_grid=True)
    + segs_single.T[["AbsoluteConcentration", "RelativeConcentration", "Delta"]]
    .hvplot(title="Concentration Dimension", width=370, height=450)
    .opts(legend_position="bottom", show_grid=True)
    + segs_single.T[
        [
            "AbsoluteClustering",
            "Isolation",
            "CorrelationR",
            "Interaction",
            "SpatialProxProf",
        ]
    ]
    .hvplot(title="Exposure/Clustering Dimension", width=370, height=450)
    .opts(legend_position="bottom", show_grid=True)
)


s = hv.render(segs)
m = generate_map(df)
folium_static(m, width=1100, height=500)
st.markdown("\n\n\n")

# slight hack to get the layout a bit narrower by making col[0] 6x the size of col[1]
intro_cols = st.columns((6, 1))
with intro_cols[0]:
    # st.markdown(intro)
    pass
st.write(f"## Trends in Multigroup Measures")

multi_plots,multi_text,  = st.columns(2)

with multi_text:
    st.markdown("\n\n\n")

    multi_idx = st.selectbox("Choose a multigroup segregation index", multi.index)
    st.markdown(
        f"In {metro_name}, "
        + generate_delta_text(multi_idx, get_delta(multi, multi_idx))
    )
    st.bokeh_chart(gen_single(multi, multi_idx))

with multi_plots:
    with st.expander("Details"):
        st.markdown(
            """Multigroup indices are measured using 5 categories, loosely corresponding to the average national quintile limits for median household income over the 2008-2018 period from the [Tax Policy Center](https://www.taxpolicycenter.org/statistics/household-income-quintiles). :

- $25,000
- $50,000
- $75,000
- $125,000
- \>$125,000

See additional notes in app details at the top.

For more information on the calculation and interpretation of each index, see the [multi-group API docs](https://pysal.org/segregation/api.html#multigroup-indices) for the PySAL `segregation` package
"""
        )

    st.bokeh_chart(gen_multi(multi), use_container_width=True)
    # st.bokeh_chart(gen_single(multi, multi_idx))


st.write(f"## Trends in Singlegroup Measures ")

singlegroup_plot,singlegroup_table = st.columns(2)


with singlegroup_table:
    st.markdown(f"#### For {income_group.title()} Income Households")
with singlegroup_plot:
    with st.expander("Details"):
        st.markdown('''
Singlegroup indices focus on the top (high) and bottom (low) quintiles of the income groups described above. Use the dropdown selector in the sidebar to toggle between income groups. 

For more information on the calculation and interpretation of each index, see the [single-group API docs](https://pysal.org/segregation/api.html#multi-group-indices) for the PySAL `segregation` package
''')


first_col2,first_col1 = st.columns(2)

with first_col1:
    index = st.selectbox("Choose a singlegroup segregation index", segs_single.index)
    st.markdown(
        f"For very {income_group} income households in {metro_name}, "
        + generate_delta_text(index, get_delta(segs_single, index))
    )
    st.bokeh_chart(gen_single(segs_single, index))

with first_col2:
    st.bokeh_chart(plot_all_single(segs_single, income_group),use_container_width=True)

st.write("## Trends by Dimension")
st.markdown(dimension_text)
st.markdown("\n\n")
st.markdown(f"### Very {income_group} income segregation:")
st.bokeh_chart(s)


st.write("## Trends through Time and Space")


profile_idx = st.selectbox("Select an index to plot its multiscalar profile", ["entropy", "isolation"])
multi1, multi2 = get_profile_data(fips, profile_idx)

highlight = alt.selection(type="single", on="mouseover", fields=["year"], nearest=True)

max_val = pd.concat([multi1, multi2])[f"{profile_idx} Index"].max()

a = (
    alt.Chart(multi1)
    .mark_line()
    .encode(
        x="distance",
        y=alt.Y(f"{profile_idx} Index", scale=alt.Scale(domain=(0, max_val))),
        # y=f'{profile_idx} Index',
        color=alt.Color("year", scale=alt.Scale(scheme="reds")),
    )
)

b = (
    alt.Chart(multi2)
    .mark_line()
    .encode(
        x="distance",
        y=alt.Y(f"{profile_idx} Index", scale=alt.Scale(domain=(0, max_val))),
        color=alt.Color("year", scale=alt.Scale(scheme="blues")),
    )
)

points = (
    a.mark_circle()
    .encode(opacity=alt.value(0))
    .add_selection(highlight)
    .properties(width=550, height=450)
)

lines = a.mark_line().encode(size=alt.condition(~highlight, alt.value(1), alt.value(3)))

pointsb = (
    b.mark_circle()
    .encode(opacity=alt.value(0))
    .add_selection(highlight)
    .properties(width=550, height=450)
)

linesb = b.mark_line().encode(
    size=alt.condition(~highlight, alt.value(1), alt.value(3))
)

st.markdown(multiscalar_pre)
col1, col2 = st.columns(2)

with col1:
    st.write(f"### High Income Multiscalar {profile_idx.title()} Profile")
    st.write(points + lines)
with col2:
    st.write(f"### Low Income Multiscalar {profile_idx.title()} Profile")
    st.write(pointsb + linesb)
st.markdown(multiscalar_post)
