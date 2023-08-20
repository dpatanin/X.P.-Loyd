import json
from os import walk

from bokeh.layouts import column
from bokeh.palettes import HighContrast3
from bokeh.plotting import figure, show


def get_a(index, total_values):
    return 0.2 + ((index / (total_values - 1)) * 0.8)


# filenames = next(walk("logs/episode-history"), (None, None, []))[2]
filenames = ["0-100000.json"]
ep_history = []
for file in filenames:
    with open(f"logs/episode-history/{file}") as f:
        ep_history.extend(json.load(f))

keys = ep_history[0].keys()

time_step_progress = {key: [] for key in keys}
for ep in ep_history:
    for key in keys:
        time_step_progress[key].append(ep[key])

time_step_figures = []
for key in keys:
    p = figure(
        title=f"{key} per timestep",
        x_axis_label="Time Steps",
        y_axis_label=key,
    )
    for id, data in enumerate(time_step_progress[key]):
        p.line(
            list(range(len(data))),
            data,
            legend_label=f"EP {id+1}",
            line_width=2,
            color=(240, 248, 255, get_a(id, len(time_step_progress[key]))),
        )
    time_step_figures.append(p)

x_episodes = [f"EP{e+1}" for e in range(len(ep_history))]
episodic_summary = {"total_profit": [], "total_fees": [], "checkpoint": []}
for ep in ep_history:
    for key in episodic_summary:
        episodic_summary[key].append(ep[key][-1])

profit_fees_data = {
    "x_episodes": x_episodes,
    "total_profit": episodic_summary["total_profit"],
    "total_fees": episodic_summary["total_fees"],
}
profit_fees_keys = ["total_profit", "total_fees"]
profit_fees = figure(
    x_range=x_episodes,
    title="Total profits & fees",
    tools="hover",
    tooltips="$name @x_episodes: @$name",
)
profit_fees.vbar_stack(
    profit_fees_keys,
    x=x_episodes,
    color=HighContrast3,
    source=profit_fees_data,
    legend_label=profit_fees_keys,
)

profit_fees.y_range.start = 0
profit_fees.x_range.range_padding = 0.1
profit_fees.xgrid.grid_line_color = None
profit_fees.axis.minor_tick_line_color = None
profit_fees.outline_line_color = None
profit_fees.legend.location = "top_left"
profit_fees.legend.orientation = "horizontal"

show(column(time_step_figures))
