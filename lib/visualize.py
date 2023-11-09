from os import walk

import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool, Label, TabPanel, Tabs
from bokeh.plotting import figure, output_file, save
from bokeh.transform import dodge
from tqdm import tqdm

curdoc().theme = "dark_minimal"
common_args = {
    "height": 840,
    "width_policy": "max",
    "sizing_mode": "stretch_width",
    "hidpi": True,
    "output_backend": "webgl",
}


def calc_line_color(index, total_values):
    return (
        255,
        int((index * 255) / total_values),
        255,
    )


def draw_profit_fees(history: list[pd.DataFrame]):
    x_episodes = [str(n) for n in range(1, len(history) + 1)]
    total_profits = [ep["profit"].sum() for ep in history]
    total_fees = [ep["fees"].sum() for ep in history]

    pf_data = {
        "x_episodes": x_episodes,
        "total_profit": total_profits,
        "total_fees": total_fees,
    }
    source = ColumnDataSource(data=pf_data)

    pf_figure = figure(
        x_range=x_episodes,
        title="Total profits & fees",
        x_axis_label="Episodes",
        y_axis_label="$$$",
        **common_args,
    )

    pf_figure.vbar(
        x=dodge("x_episodes", -0.125, range=pf_figure.x_range),
        width=0.2,
        top="total_profit",
        source=source,
        color="#2e3adb",
        legend_label="total_profit",
    )

    pf_figure.vbar(
        x=dodge("x_episodes", 0.125, range=pf_figure.x_range),
        width=0.2,
        top="total_fees",
        source=source,
        color="#edc42d",
        legend_label="total_fees",
    )

    pf_figure.add_tools(
        HoverTool(
            tooltips=[
                ("EP", "@x_episodes"),
                ("Profit", "@total_profit"),
                ("Fees", "@total_fees"),
            ],
        )
    )

    pf_figure.xaxis.major_label_orientation = 1
    pf_figure.x_range.range_padding = 0.05
    pf_figure.xgrid.grid_line_color = None
    pf_figure.axis.minor_tick_line_color = None
    pf_figure.outline_line_color = None
    pf_figure.legend.location = "top_left"
    pf_figure.legend.orientation = "horizontal"

    return pf_figure


def draw_price_figure(data: pd.DataFrame, df: pd.DataFrame, name: str):
    index = data.index.values
    price_df = df[index[0] : index[-1] + 1]
    inc = price_df.close > price_df.open
    dec = price_df.open > price_df.close

    price_figure = figure(
        title=f"Price chart EP:{name}",
        x_axis_type="datetime",
        x_axis_label="Datetime",
        y_axis_label="Prices",
        **common_args,
    )

    price_figure.add_tools(
        HoverTool(
            tooltips=[("Date", "$x{%F}"), ("Price", "$y{0,0.00}")],
            formatters={"$x": "datetime"},
        )
    )

    price_figure.xaxis.major_label_orientation = 0.8

    price_figure.segment(
        price_df.index, price_df.high, price_df.index, price_df.low, color="black"
    )

    price_figure.vbar(
        price_df.index[dec],
        pd.Timedelta("1m"),
        price_df.open[dec],
        price_df.close[dec],
        color="#eb3c40",
    )
    price_figure.vbar(
        price_df.index[inc],
        pd.Timedelta("1m"),
        price_df.open[inc],
        price_df.close[inc],
        color="#49a3a3",
    )

    positions = data.set_index(price_df.index)["position"]
    position_starts = price_df[positions.diff() != 0].index
    position_ends = position_starts[1:]
    position_starts = position_starts[:-1]

    long_starts = position_starts[positions[position_starts] == 1]
    long_ends = position_ends[positions[position_starts] == 1]
    price_figure.vstrip(
        x0=long_starts,
        x1=long_ends,
        color="#24ed85",
        alpha=0.2,
    )

    short_starts = position_starts[positions[position_starts] == 2]
    short_ends = position_ends[positions[position_starts] == 2]
    price_figure.vstrip(
        x0=short_starts,
        x1=short_ends,
        color="#ed3124",
        alpha=0.2,
    )

    num_positions = (
        f" Num. Positions | Short: {len(short_starts)}  Long: {len(long_starts)} "
    )

    # Calculate position durations based on ticks -> price data has gaps
    avg_longs = round(
        (data.loc[data["position"] == 1, "position"].sum() + 1) / (len(long_starts) + 1)
    )
    avg_shorts = round(
        (data.loc[data["position"] == 2, "position"].sum() + 1)
        / (len(short_starts) + 1)
    )
    avg_durations = (
        f" Avg. Durations | Short: {avg_shorts} min.  Long: {avg_longs} min. "
    )

    # Calculate profits made per position
    total_profit_long = data.loc[
        (data["profit"] != 0) & (data["position"].shift(1) == 1), "profit"
    ].sum()
    total_profit_short = data.loc[
        (data["profit"] != 0) & (data["position"].shift(1) == 2), "profit"
    ].sum()
    total_profits = (
        f" Total Profits  | Short: {total_profit_short}  Long: {total_profit_long} "
    )

    # Calculate sma pos identity -> shift by one, action follows last sma pos
    agent_pos = data["position"][:-1].to_numpy()
    sma_pos = price_df["SMA_position"][1:].to_numpy()
    identical_count = np.sum(agent_pos == sma_pos)
    identity_percent = (identical_count / len(data["position"][:-1])) * 100
    identity = f" Identity to SMA position: {identity_percent:.2f}% "

    summary = Label(
        x=50,
        y=20,
        x_units="screen",
        y_units="screen",
        text=f"\n{num_positions}\n{avg_durations}\n{total_profits}\n{identity}\n",
        border_line_color="black",
        background_fill_color="white",
    )

    price_figure.add_layout(summary)

    return price_figure


def visualize(dir: str, prices_df: pd.DataFrame):
    output_file(filename=f"{dir}/_visualization.html", title="DQN Results")

    pb = tqdm(range(4), desc="Load episode history", position=0, leave=True)

    def update_pb(desc: str = None):
        pb.update()
        if desc:
            pb.set_description(desc)

    filenames: list[str] = next(walk(dir), (None, None, []))[2]
    csv_files = [fname for fname in filenames if fname.endswith(".csv")]
    ep_history = [pd.read_csv(f"{dir}/{file}", index_col=0) for file in csv_files]

    update_pb("Create line charts")
    simple_hover = HoverTool(
        tooltips=[("EP", "$name"), ("Timestep", "@x"), ("Value", "@y")],
    )

    balance_figure = figure(
        title="Balance per timestep",
        x_axis_label="Time Steps",
        y_axis_label="$$$",
        **common_args,
    )
    balance_figure.add_tools(simple_hover)

    for id, data in enumerate(
        tqdm(ep_history, desc="Draw lines", position=1, leave=False)
    ):
        index = data.index.values
        color = calc_line_color(id, len(ep_history))
        name = str(id + 1)

        balance_figure.line(
            index,
            data["balance"].values,
            line_width=2,
            name=name,
            color=color,
            muted_color=color,
            muted_alpha=0,
        )

    # Efficiency
    id_eps = [*enumerate(ep_history)]
    longest_eps = sorted(id_eps[:-5], key=lambda id_df: len(id_df[1]), reverse=True)[
        :10
    ]
    longest_eps.extend(id_eps[-5:])
    tabs = [
        TabPanel(
            child=draw_price_figure(data, prices_df, str(id + 1)), title=f"EP:{id+1}"
        )
        for id, data in longest_eps
    ]

    update_pb("Create profit & fees bar chart")
    pf_figure = draw_profit_fees(ep_history)

    update_pb("Build Html site")
    document = column(
        [
            Tabs(tabs=tabs, sizing_mode="stretch_width"),
            balance_figure,
            pf_figure,
        ],
        sizing_mode="stretch_width",
        width_policy="max",
    )
    save(document)
    update_pb("Visualization done!")
