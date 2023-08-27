from os import walk

import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import BoxAnnotation, HoverTool, TabPanel, Tabs
from bokeh.palettes import HighContrast3
from bokeh.plotting import figure, output_file, save
from data_processor import DataProcessor
from tqdm import tqdm


def calc_line_color(index, total_values):
    return (
        255,
        int((index * 255) / total_values),
        255,
    )


def update_pb(desc: str = None):
    pb.update()
    if desc:
        pb.set_description(desc)


curdoc().theme = "dark_minimal"

dp = DataProcessor("source.csv", 5)
common_args = {
    "height": 840,
    "width_policy": "max",
    "sizing_mode": "stretch_width",
    "hidpi": True,
    "output_backend": "webgl",
}

for dir, df in [("logs/train", dp.train_df), ("logs/eval", dp.val_df)]:
    output_file(filename=f"{dir}/visualization.html", title="DQN Results")

    pb = tqdm(range(5), desc="Load episode history", position=0, leave=True)

    filenames = next(walk(dir), (None, None, []))[2]
    json_files = [fname for fname in filenames if fname.endswith(".csv")]
    ep_history = [pd.read_csv(f"{dir}/{file}", index_col=0) for file in json_files]

    update_pb("Create line charts")
    initial_balance = ep_history[0]["balance"].iloc[0]
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

    streak_figure = figure(
        title="Streak per timestep",
        x_axis_label="Time Steps",
        y_axis_label="Streak multiplier",
        **common_args,
    )
    streak_figure.add_tools(simple_hover)

    tabs = []

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

        streak_figure.line(
            index,
            data["streak"].values,
            line_width=2,
            name=name,
            color=color,
            muted_color=color,
            muted_alpha=0,
        )

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
        position_ends = price_df[positions.diff().shift(-1) != 0].index

        price_figure.vstrip(
            x0=position_starts[positions[position_starts] == 1],
            x1=position_ends[positions[position_starts] == 1],
            color="#FF0000",
            alpha=0.2,
        )

        price_figure.vstrip(
            x0=position_starts[positions[position_starts] == 2],
            x1=position_ends[positions[position_starts] == 2],
            color="#00FF00",
            alpha=0.2,
        )

        tabs.append(TabPanel(child=price_figure, title=f"EP:{name}"))

    update_pb("Create profit & fees bar chart")
    x_episodes = [str(n) for n in range(1, len(ep_history) + 1)]
    total_profits = [ep["profit"].sum() for ep in ep_history]
    total_fees = [ep["fees"].sum() for ep in ep_history]

    pf_data = {
        "x_episodes": x_episodes,
        "total_profit": total_profits,
        "total_fees": total_fees,
    }
    pf_keys = ["total_profit", "total_fees"]
    pf_figure = figure(
        x_range=x_episodes,
        title="Total profits & fees",
        x_axis_label="Episodes",
        y_axis_label="$$$",
        **common_args,
    )
    pf_figure.vbar_stack(
        pf_keys,
        x="x_episodes",
        color=HighContrast3[:-1],
        source=pf_data,
        legend_label=pf_keys,
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

    pf_figure.x_range.range_padding = 0.05
    pf_figure.xgrid.grid_line_color = None
    pf_figure.axis.minor_tick_line_color = None
    pf_figure.outline_line_color = None
    pf_figure.legend.location = "top_left"
    pf_figure.legend.orientation = "horizontal"

    update_pb("Create streak figure")

    update_pb("Build Html site")
    document = column(
        [
            Tabs(tabs=tabs, sizing_mode="stretch_width"),
            balance_figure,
            streak_figure,
            pf_figure,
        ],
        sizing_mode="stretch_width",
        width_policy="max",
    )
    save(document)
    update_pb("Done!")
