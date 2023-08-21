import json
from os import walk

from bokeh.core.properties import Int
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import LayoutDOM, Legend, LegendItem
from bokeh.palettes import HighContrast3
from bokeh.plotting import figure, show
from tqdm import tqdm

curdoc().theme = "dark_minimal"


class BelowLegend(LayoutDOM):
    __implementation__ = "BelowLegend.ts"

    width = Int(default=100)
    height = Int(default=100)


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


common_args = {
    "height": 840,
    "width_policy": "max",
    "sizing_mode": "stretch_width",
    "tools": "pan,wheel_zoom,box_zoom,save,reset,help, hover",
    "hidpi": True,
    "output_backend": "webgl",
}

pb = tqdm(range(7), desc="Load episode history", position=0, leave=True)

# filenames = next(walk("logs/episode-history"), (None, None, []))[2]
filenames = ["0-100000.json"]
ep_history = []
for file in filenames:
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
        **common_args,
    )
    for key in keys
]


update_pb("Create timestep lines")
for nk, key in enumerate(keys):
    ts_f = ts_figures[nk]
    ts_p = ts_progress[key]
    legend_it = [
        LegendItem(
            label=f"EP {id+1}",
            renderers=[
                ts_f.line(
                    list(range(1, len(data) + 1)),
                    data,
                    line_width=2,
                    color=get_color(id, len(ts_p)),
                    muted_color=get_color(id, len(ts_p)),
                    muted_alpha=0,
                )
            ],
            index=id,
        )
        for id, data in enumerate(ts_p)
    ]

    legend = Legend(items=legend_it, orientation="horizontal")
    legend.click_policy = "mute"
    ts_f.add_layout(legend, "below")

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
    **common_args,
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
    **common_args,
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
