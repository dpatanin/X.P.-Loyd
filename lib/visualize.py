import json
from os import walk

from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.palettes import HighContrast3
from bokeh.plotting import figure, show
from tqdm import tqdm

curdoc().theme = "dark_minimal"
WIDTH = 1600
HEIGHT = 840

pb = tqdm(range(7), desc="Load episode history", position=0, leave=True)


def get_color(index, total_values):
    return (
        255,
        int((id * 255) / total_values),
        255,
        0.3 + ((index / (total_values - 1)) * 0.7),
    )


def update_pb(desc: str = None):
    pb.update()
    if desc:
        pb.set_description(desc)


# filenames = next(walk("logs/episode-history"), (None, None, []))[2]
filenames = ["0-100000.json"]
ep_history = []
for file in tqdm(filenames, desc="Load files", position=1, leave=False):
    with open(f"logs/episode-history/{file}") as f:
        ep_history.extend(json.load(f))

update_pb("Create timestep figures")
keys = ep_history[0].keys()
ts_progress = {key: [ep[key] for ep in ep_history] for key in keys}
ts_figures = [
    figure(
        title=f"{key} per timestep",
        x_axis_label="Time Steps",
        y_axis_label=key,
        height=HEIGHT,
        width_policy="max",
        sizing_mode="stretch_width",
    )
    for key in keys
]


update_pb("Create timestep lines")
for nk, key in enumerate(tqdm(keys, desc="Figures", position=1, leave=False)):
    for id, data in enumerate(
        tqdm(ts_progress[key], desc="Lines", position=2, leave=False)
    ):
        ts_figures[nk].line(
            list(range(1, len(data) + 1)),
            data,
            legend_label=f"EP {id+1}",
            line_width=2,
            color=get_color(id, len(ts_progress[key])),
        )

update_pb("Read profits, fees & checkpoints")
x_episodes = [str(n) for n in range(1, len(ep_history) + 1)]
total_profits = [ep["total_profit"][-1] for ep in ep_history]
total_fees = [ep["total_fees"][-1] for ep in ep_history]
checkpoints = [ep["checkpoint"][-1] for ep in ep_history]

update_pb("Create profits-fees figure")
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
    toolbar_location=None,
    tools="hover",
    tooltips="$name @x_episodes: @$name",
    height=HEIGHT,
    width_policy="max",
    sizing_mode="stretch_width",
)
pf_figure.vbar_stack(
    pf_keys,
    x="x_episodes",
    color=HighContrast3[:-1],
    source=pf_data,
    legend_label=pf_keys,
)
pf_figure.x_range.range_padding = 0.1
pf_figure.xgrid.grid_line_color = None
pf_figure.axis.minor_tick_line_color = None
pf_figure.outline_line_color = None
pf_figure.legend.location = "top_left"
pf_figure.legend.orientation = "horizontal"

update_pb("Create checkpoints figure")
c_figure = figure(
    x_range=x_episodes,
    title="Final checkpoints per episode",
    x_axis_label="Episodes",
    y_axis_label="Checkpoints",
    height=HEIGHT,
    width_policy="max",
    sizing_mode="stretch_width",
)
c_figure.vbar(
    x=x_episodes,
    top=checkpoints,
    line_width=2,
    color=(240, 0, 255),
)
c_figure.y_range.start = 0
c_figure.x_range.range_padding = 0.1
c_figure.xgrid.grid_line_color = None
c_figure.axis.minor_tick_line_color = None
c_figure.outline_line_color = None

update_pb("Build Html site")
show(
    column(
        [*ts_figures, pf_figure, c_figure],
        sizing_mode="stretch_width",
        width_policy="max",
    )
)
update_pb("Done!")
