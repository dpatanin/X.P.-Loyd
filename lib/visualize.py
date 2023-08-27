from os import walk

import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.palettes import HighContrast3
from bokeh.plotting import figure, output_file, save
from tqdm import tqdm


def get_color(index, total_values):
    return (
        255,
        int((index * 255) / total_values),
        255,
    )


def update_pb(desc: str = None):
    pb.update()
    if desc:
        pb.set_description(desc)


FULL_DATA = "https://onedrive.live.com/download?resid=2ba9b2044e887de1%21290022&authkey=!ADgq6YFliQNylSM"
DIR = "logs/train"
curdoc().theme = "dark_minimal"
output_file(filename=f"{DIR}/visualization.html", title="DQN Results")

pb = tqdm(range(5), desc="Load episode history", position=0, leave=True)
common_args = {
    "height": 840,
    "width_policy": "max",
    "sizing_mode": "stretch_width",
    "tools": "pan,wheel_zoom,box_zoom,save,reset,help, hover",
    "hidpi": True,
    "output_backend": "webgl",
}

filenames = next(walk(DIR), (None, None, []))[2]
json_files = [fname for fname in filenames if fname.endswith(".csv")]
ep_history = [pd.read_csv(f"{DIR}/{file}", index_col=0) for file in json_files]

update_pb("Create line charts")
initial_balance = ep_history[0]["balance"].iloc[0]
balance_figure = figure(
    title="Balance per timestep",
    x_axis_label="Time Steps",
    y_axis_label="$$$",
    tooltips=[("EP", "$name"), ("Timestep", "@x"), ("Value", "@y")],
    **common_args,
)

streak_figure = figure(
    title="Streak per timestep",
    x_axis_label="Time Steps",
    y_axis_label="Streak multiplier",
    tooltips=[("EP", "$name"), ("Timestep", "@x"), ("Value", "@y")],
    **common_args,
)

for id, data in enumerate(ep_history):
    index = data.index.values
    color = get_color(id, len(ep_history))
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
    tooltips=[
        ("EP", "@x_episodes"),
        ("Profit", "@total_profit"),
        ("Fees", "@total_fees"),
    ],
    **common_args,
)
pf_figure.vbar_stack(
    pf_keys,
    x="x_episodes",
    color=HighContrast3[:-1],
    source=pf_data,
    legend_label=pf_keys,
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
    [balance_figure, streak_figure, pf_figure],
    sizing_mode="stretch_width",
    width_policy="max",
)
save(document)
update_pb("Done!")
