from bokeh.core.property.numeric import Size
from bokeh.models.callbacks import Callback
from bokeh.plotting import figure, output_file, show
import numpy as np
from bokeh.models import CustomJS, RadioGroup, Button, Span, Dropdown, renderers
from bokeh.models import Button
from bokeh.models import Legend
from bokeh.layouts import grid, row, column, layout, gridplot
from bokeh.layouts import column
from bokeh.layouts import row
from bokeh.io import curdoc
from bokeh.models import DataTable, TableColumn, PointDrawTool, ColumnDataSource, BoxEditTool
from bokeh.events import DoubleTap, Tap, MouseLeave
from bokeh.models.widgets import (Panel, Tabs, DataTable, TableColumn,
                                  Paragraph, Slider, Div, Button, Select)
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool, TapTool, BoxSelectTool
from scipy.signal import zpk2ss, ss2zpk, tf2zpk, zpk2tf
from numpy import linspace, logspace
from numpy import asarray, tan, array, pi, arange, cos, log10, unwrap, angle
from matplotlib.pyplot import axvline, axhline
from scipy.signal import freqz
from bokeh.events import MouseMove, Tap, MouseLeave
from scipy import signal


output_file("GUI.html")

Choosen = "red"
FilterZerosList = []
FilterPolesList = []
ZerosList = []
PolesList = []
zeros_coef = [1]
poles_coef = [1]
zero_filter_pos = []
pole_filter_pos = []
conj = False
TOOLS = "tap"
bound = 10

# Draw Zero/pole plot
plot = figure(title="Pole-Zero Plot: Click for pole & Double Click for zero", x_axis_label="Real",
              y_axis_label="Imaginary", plot_width=400, plot_height=400, )  # sets size and makes it square
#curdoc().theme = 'dark_minimal'
ax = plot.axis
theta = np.linspace(-np.pi, np.pi, 201)
plot.line(np.sin(theta), np.cos(theta), color='red', line_width=1)

vline = Span(location=0, dimension='height', line_color='black', line_width=1)
hline = Span(location=0, dimension='width', line_color='black', line_width=1)

plot.renderers.extend([vline, hline])

# Draw Mag response
MagnitudeGraph = figure(title="Magnitude Response ", x_axis_label="Frequency",
                        y_axis_label="Amplitude", plot_width=400, plot_height=300)

PhaseGraph = figure(title="Phase Response ", x_axis_label="Frequency",
                    y_axis_label="Phase", plot_width=400, plot_height=300)
w = np.linspace(0, np.pi, 200)    # for evauluating H(w)
z = np.exp(1j*w)
f = np.linspace(0, 180, 200)         # for ploting H(w)
H = np.polyval(zeros_coef, z) / np.polyval(poles_coef, z)
phase = np.unwrap(np.angle(H))
#w_full, h_full = signal.freqz(b, a, whole=True)
#w, h = signal.freqz(b, a)


MagnitudeGraph.line(f, abs(H), line_width=2)
PhaseGraph.line(f, phase, line_width=2)

ClearPoles = Button(label="Clear Poles", button_type="success")
ClearZeros = Button(label="Clear Zeros", button_type="success")
ClearAll = Button(label="Clear All", button_type="success")
Apply = Button(label="Apply", button_type="success")
Apply1 = Button(label="Apply", button_type="success")
ClearCustom = Button(label="Clear Filter", button_type="success")
source = ColumnDataSource(data=dict(x=[], y=[]))

zero_conj = ColumnDataSource(data=dict(x=[], y=[]))
# plot.circle(source=source,x='x',y='y')

source1 = ColumnDataSource(data=dict(x=[], y=[]))
source2 = ColumnDataSource(data=dict(x=[], y=[]))
source3 = ColumnDataSource(data=dict(x=[], y=[]))

pole_conj = ColumnDataSource(data=dict(x=[], y=[]))
# plot.cross(source=source1,x='x',y='y')
rendererzero = plot.circle(x='x', y='y', source=source, size=10)
rendererpole = plot.cross(x='x', y='y', source=source1, size=10)
rendererzero_conj = plot.circle(x='x', y='y', source=zero_conj, size=10)
rendererpole_conj = plot.cross(x='x', y='y', source=pole_conj, size=10)
draw_tool = PointDrawTool(renderers=[
                          rendererzero, rendererpole, rendererzero_conj, rendererpole_conj], empty_value='black')
box_select = BoxSelectTool()
plot.add_tools(box_select)
plot.add_tools(draw_tool)
#plot.toolbar.active_tap = draw_tool
allPass_fig1 = figure(title='Filters', tools="tap,reset",
                      x_range=[-2.5, 2.5], y_range=[-2.5, 2.5],
                      plot_height=250, plot_width=250, toolbar_location=None)
filter1_phase = figure(title='Phase', tools="tap,reset",
                       plot_height=250, plot_width=350, toolbar_location=None)
filter1_phase.line(f, phase, line_width=2)
filter1_circle = allPass_fig1.circle(
    0, 0, radius=1, line_color='gold', fill_alpha=0, line_width=2)
filter1poles = ColumnDataSource(data=dict(x=[0.3, 1], y=[0.6, 1]))
filter1zeros = ColumnDataSource(data=dict(x=[], y=[]))
filter2poles = ColumnDataSource(data=dict(x=[0.36], y=[0.7]))
filter2zeros = ColumnDataSource(
    data=dict(x=[0.36/0.36*0.36+0.7*0.7], y=[0.7/0.36*0.36+0.7*0.7]))
source5 = ColumnDataSource(
    data=dict(x_of_poles_2=[], y_of_poles_2=[], x_of_zeros_2=[], y_of_zeros_2=[]))
#source3 = ColumnDataSource(data=dict(x_of_poles_3=[], y_of_poles_3=[]))


renderer_5 = allPass_fig1.cross(
    x='x_of_poles_2', y='y_of_poles_2', source=source5, color='blue', size=10)
renderer_5 = allPass_fig1.circle(
    x='x_of_zeros_2', y='y_of_zeros_2', source=source5, color='blue', size=10)

custom_zeros_source = ColumnDataSource(data=dict(x=[], y=[]))
custom_poles_source = ColumnDataSource(data=dict(x=[], y=[]))


custom_fig = figure(title='Customize Your Filter',
                    tools="tap",
                    plot_height=400, plot_width=400,
                    x_range=[-2.5, 2.5], y_range=[-2.5, 2.5],
                    toolbar_location='above')

custom_phase_fig = figure(title='Phase Response',
                          tools="tap,reset",
                          plot_height=400, plot_width=400,
                          toolbar_location=None)

custom_phase_fig.line(f, phase, line_width=2)


custom_unit_circle = custom_fig.circle(0, 0, radius=1,
                                       line_color='gold', fill_alpha=0, line_width=2)

custom_zeros = custom_fig.circle('x', 'y', source=custom_zeros_source,
                                 line_color='blue', size=10, fill_alpha=0, line_width=2)

custom_poles = custom_fig.cross('x', 'y', source=custom_poles_source,
                                line_color='blue', size=10, line_width=3)

renderer_3 = custom_phase_fig.cross(
    x='x_of_poles_3', y='y_of_poles_3', source=source3, color='blue', size=10)
menu = Select(options=['None', 'Filter1', 'Filter2', 'Filter3',
              'All', 'added custum'], value='None', title='Filters')


def Filters(attr, old, new):
    if menu.value == "None":
        source5.data['x_of_poles_2'].clear()
        source5.data['y_of_poles_2'].clear()
        source5.data['x_of_zeros_2'].clear()
        source5.data['y_of_zeros_2'].clear()
        new_data_3 = {'x_of_poles_2': source5.data['x_of_poles_2'], 'y_of_poles_2': source5.data['y_of_poles_2'],
                      'x_of_zeros_2': source5.data['x_of_zeros_2'], 'y_of_zeros_2': source5.data['y_of_zeros_2'], }
        source5.data = new_data_3
    elif menu.value == "Filter1":
        source5.data = dict(x_of_poles_2=[1, 1, -2], y_of_poles_2=[1, 1.3, -1], x_of_zeros_2=[
                            1/1+1, 1/1.3*1.3+1, -2/4+1], y_of_zeros_2=[1/2, 1.3/1.3*1.3+1, -1/5])
        new_data_4 = {'x_of_poles_2': source5.data['x_of_poles_2'], 'y_of_poles_2': source5.data['y_of_poles_2'],
                      'x_of_zeros_2': source5.data['x_of_zeros_2'], 'y_of_zeros_2': source5.data['y_of_zeros_2'], }
        source5.data = new_data_4
       # Plot_Filter_response([source5.data['x_of_poles_2']+1j*source5.data['y_of_poles_2'],source5.data['x_of_poles_2']-1j*source5.data['y_of_poles_2']])
    elif menu.value == "Filter2":
        source5.data = dict(x_of_poles_2=[-1, -1, 2], y_of_poles_2=[-1, -1.3, 1],
                            x_of_zeros_2=[-1/2, -1/1.3*1.3+1, 2/5], y_of_zeros_2=[-1/2, -1.3/1.3*1.3+1, 1/5])
        new_data_5 = {'x_of_poles_2': source5.data['x_of_poles_2'], 'y_of_poles_2': source5.data['y_of_poles_2'],
                      'x_of_zeros_2': source5.data['x_of_zeros_2'], 'y_of_zeros_2': source5.data['y_of_zeros_2'], }
        source5.data = new_data_5
    elif menu.value == "Filter3":
        source5.data = dict(x_of_poles_2=[-2, -3, 0], y_of_poles_2=[-1.5, 2, 1.5],
                            x_of_zeros_2=[-2/7, -3/12, 0], y_of_zeros_2=[-1.5/7, 2/12, 1.5/3])
        new_data_6 = {'x_of_poles_2': source5.data['x_of_poles_2'], 'y_of_poles_2': source5.data['y_of_poles_2'],
                      'x_of_zeros_2': source5.data['x_of_zeros_2'], 'y_of_zeros_2': source5.data['y_of_zeros_2'], }
        source5.data = new_data_6
    elif menu.value == "All":
        source5.data = dict(x_of_poles_2=[1, 1, -2, -1, -1, 2, -2, -3, 0], y_of_poles_2=[1, 1.3, -1, -1, -1.3, 1, -1.5, 2, 1.5], x_of_zeros_2=[
                            1/2, 1/1.3*1.3+1, -2/5, -1/2, -1/1.3*1.3+1, 2/7, -2/8, -3/12, 0], y_of_zeros_2=[1/2, 1.3/1.3*1.3+1, -1/5, -1/2, -1.3/1.3*1.3+1, 1/7, -1.5/8, 2/12, 1.5/3])
        new_data_6 = {'x_of_poles_2': source5.data['x_of_poles_2'], 'y_of_poles_2': source5.data['y_of_poles_2'],
                      'x_of_zeros_2': source5.data['x_of_zeros_2'], 'y_of_zeros_2': source5.data['y_of_zeros_2'], }
        source5.data = new_data_6
    elif menu.value == "added custum":
        source5.data = dict(x_of_poles_2=source3.data['x'], y_of_poles_2=source3.data['y'],
                            x_of_zeros_2=source2.data['x'], y_of_zeros_2=source2.data['y'])
        new_data_7 = {'x_of_poles_2': source5.data['x_of_poles_2'], 'y_of_poles_2': source5.data['y_of_poles_2'],
                      'x_of_zeros_2': source5.data['x_of_zeros_2'], 'y_of_zeros_2': source5.data['y_of_zeros_2'], }
        source5.data = new_data_7


menu.on_change('value', Filters)


def addFilter():

    DrawFreqResponse()
    plot.cross(x='x_of_poles_2', y='y_of_poles_2',
               source=source5, color='blue', size=10)
    plot.circle(x='x_of_zeros_2', y='y_of_zeros_2',
                source=source5, color='blue', size=10)


Apply.on_click(addFilter)

dummy = False


def addCustomFilter():
    global dummy
    dummy = True
    DrawFreqResponse()
    plot.circle(x='x', y='y', source=source2, color='blue', size=10)
    plot.cross(x='x', y='y', source=source3, color='blue', size=10)


Apply1.on_click(addCustomFilter)


LABELS = ["Disable Conjegate", "Enable Conjegate"]
radio_group = RadioGroup(labels=LABELS, active=0)
#radio_group.js_on_click('active', callback2)
radio_group.js_on_click(CustomJS(code="""
    console.log('radio_group: active=' + this.active, this.toString())
"""))


def AddZeros(event):

    Coords = (event.x, event.y)
    ZerosList.append(Coords)
    source.data = dict(x=[i[0] for i in ZerosList], y=[i[1]
                       for i in ZerosList])
    print(ZerosList)


def AddPoles(event):
    Coor = (event.x, event.y)
    PolesList.append(Coor)
    source1.data = dict(x=[i[0] for i in PolesList],
                        y=[i[1] for i in PolesList])
    print(PolesList)


def AddfilterZeros(event):

    Coo = (event.x, event.y)
    FilterZerosList.append(Coo)
    source2.data = dict(x=[i[0] for i in FilterZerosList], y=[
                        i[1] for i in FilterZerosList])
    print(FilterZerosList)


def AddfilterPoles(event):
    Co = (event.x, event.y)
    FilterPolesList.append(Co)
    source3.data = dict(x=[i[0] for i in FilterPolesList], y=[
                        i[1] for i in FilterPolesList])
    print(FilterPolesList)


def Clear_Zeros(event):
    global source
    global ZerosList
    ZerosList.clear()
    source.data = {k: [] for k in source.data}
    zero_conj.data = {k: [] for k in zero_conj.data}


ClearZeros.on_click(Clear_Zeros)


def Clear_Poles(event):
    global source1
    global PolesList
    PolesList.clear()
    source1.data = {k: [] for k in source1.data}
    pole_conj.data = {k: [] for k in pole_conj.data}


ClearPoles.on_click(Clear_Poles)


def Clear_All(event):
    global source
    global ZerosList
    ZerosList.clear()
    source.data = {k: [] for k in source.data}
    zero_conj.data = {k: [] for k in zero_conj.data}
    global source1
    global PolesList
    PolesList.clear()
    source1.data = {k: [] for k in source1.data}
    pole_conj.data = {k: [] for k in pole_conj.data}


ClearAll.on_click(Clear_All)


def Radio_check(active):
    if active == 1:
        global conj
        if conj == False:
            conj = True
        else:
            conj = False

    if active == 0:
        conj = False


def clearCustomFilter():

    global custom_poles_source, custom_zeros_source, dummy

    if dummy == True:

        source2.data.clear()
        FilterZerosList.clear()
        FilterPolesList.clear()

        source3.data.clear()

    dummy = False


ClearCustom.on_click(clearCustomFilter)


radio_group.on_click(Radio_check)
plot.on_event(DoubleTap, AddZeros)
plot.on_event(Tap, AddPoles)
#####################################################
custom_fig.on_event(Tap, AddfilterPoles)
custom_fig.on_event(DoubleTap, AddfilterZeros)
######################################################
rendererzero1 = custom_fig.circle(x='x', y='y', source=source2, size=10)
rendererpole1 = custom_fig.cross(x='x', y='y', source=source3, size=10)
draw_tool_1 = PointDrawTool(renderers=[rendererzero1], empty_value="red")
draw_tool_2 = PointDrawTool(renderers=[rendererpole1], empty_value="red")
custom_fig.add_tools(draw_tool_1, draw_tool_2)
custom_fig.toolbar.active_tap = draw_tool_1
custom_fig.toolbar.active_tap = draw_tool_2


def DrawFreqResponse():

    MagnitudeGraph.renderers = []
    PhaseGraph.renderers = []

    zero_coef = zeros_coef
    pole_coef = poles_coef
    if type(zeros_coef) == float:
        zero_coef = [zeros_coef]

    if type(poles_coef) == float:
        pole_coef = [poles_coef]

    w = np.linspace(0, np.pi, 200)    # for evauluating H(w)
    z = np.exp(1j*w)
    f = np.linspace(0, 180, 200)         # for ploting H(w)
    H = np.polyval(zero_coef, z) / np.polyval(pole_coef, z)
    phase = np.unwrap(np.angle(H))

    MagnitudeGraph.line(f, abs(H), line_width=2)
    PhaseGraph.line(f, phase, line_width=2)


def DrawFilterPhase():
    filter1_phase.renderers = []
    zero_coef = zeros_coef
    pole_coef = poles_coef
    if type(zeros_coef) == float:
        zero_coef = [zeros_coef]

    if type(poles_coef) == float:
        pole_coef = [poles_coef]
    w = np.linspace(0, np.pi, 200)    # for evauluating H(w)
    z = np.exp(1j*w)
    f = np.linspace(0, 180, 200)         # for ploting H(w)
    H = np.polyval(zero_coef, z) / np.polyval(pole_coef, z)
    phase = np.unwrap(np.angle(H))
    filter1_phase.line(f, phase, line_width=2)


def Drawphase():

    custom_phase_fig.renderers = []

    zero_coef = zeros_coef
    pole_coef = poles_coef
    if type(zeros_coef) == float:
        zero_coef = [zeros_coef]

    if type(poles_coef) == float:
        pole_coef = [poles_coef]

    w = np.linspace(0, np.pi, 200)    # for evauluating H(w)
    z = np.exp(1j*w)
    f = np.linspace(0, 180, 200)         # for ploting H(w)
    H = np.polyval(zero_coef, z) / np.polyval(pole_coef, z)
    phase = np.unwrap(np.angle(H))

    custom_phase_fig.line(f, phase, line_width=2)


def Filter_Coefs():
    global zeros_coef, poles_coef
    poles_pos_fil = []
    zeros_pos_fil = []
    for i in range(len(source5.data['x_of_poles_2'])):
        poles_pos_fil.append(
            source5.data['x_of_poles_2'][i]+1j*source5.data['y_of_poles_2'][i])
        zeros_pos_fil.append(
            source5.data['x_of_zeros_2'][i]+1j*source5.data['y_of_zeros_2'][i])
    poles_coef = np.poly(poles_pos_fil)
    zeros_coef = np.poly(zeros_pos_fil)
    DrawFilterPhase()


def Cus_Filter_Coefs():

    global zeros_coef, poles_coef
    zeros_pos1 = []
    poles_pos1 = []

    for i in range(len(source2.data['x'])):
        zeros_pos1.append(source2.data['x'][i]+1j*source2.data['y'][i])

    for i in range(len(source3.data['x'])):
        poles_pos1.append(source3.data['x'][i]+1j*source3.data['y'][i])

    poles_coef = np.poly(poles_pos1)
    zeros_coef = np.poly(zeros_pos1)

    Drawphase()


"""
    global zeros_coef , poles_coef
    poles_pos_fil = []
    zeros_pos_fil = []
    for i in range(len(source3.data['x_of_poles_2'])):
         poles_pos_fil.append(source3.data['x_of_poles_3'][i]+1j*source3.data['y_of_poles_3'][i])
         #zeros_pos_fil.append(source5.data['x_of_poles_2'][i]+1j*source5.data['y_of_poles_2'][i])
    poles_coef = np.poly(poles_pos_fil)
    #zeros_coef = np.poly( zeros_pos_fil)
    Drawphase()
"""


def Set_Coefs():
    global zeros_coef, poles_coef
    zeros_pos = []
    poles_pos = []
    poles_pos_conj = []
    zeros_pos_conj = []
    poles_pos_fil = []
    zeros_pos_fil = []
    for i in range(len(source.data['x'])):
        zeros_pos.append(source.data['x'][i]+1j*source.data['y'][i])
        if conj:
            zero_conj.data['x'] = source.data['x']
            zero_conj.data['y'] = -1*np.array(source.data['y'])
            zeros_pos_conj.append(source.data['x'][i]-1j*source.data['y'][i])

    for i in range(len(source1.data['x'])):
        poles_pos.append(source1.data['x'][i]+1j*source1.data['y'][i])
        if conj:
            pole_conj.data['x'] = source1.data['x']
            pole_conj.data['y'] = -1*np.array(source1.data['y'])
            poles_pos_conj.append(source1.data['x'][i]-1j*source1.data['y'][i])

    poles_coef = np.poly(poles_pos)
    zeros_coef = np.poly(zeros_pos)

    DrawFreqResponse()


plot.on_event(Tap, Set_Coefs)
plot.on_event(MouseLeave, Set_Coefs)
allPass_fig1.on_event(Tap, Filter_Coefs)
allPass_fig1.on_event(MouseLeave, Filter_Coefs)
custom_fig.on_event(Tap, Cus_Filter_Coefs)
custom_fig.on_event(MouseLeave, Cus_Filter_Coefs)
rendererzero1 = custom_fig.circle(x='x', y='y', source=source2, size=10)
rendererpole1 = custom_fig.cross(x='x', y='y', source=source3, size=10)


organize_layout1 = column(radio_group, ClearPoles, ClearZeros, ClearAll)
organize_layout2 = column(plot)
Layout = row(organize_layout1, organize_layout2, MagnitudeGraph, PhaseGraph)
Header2 = Div(text='<h1 style="color: black">All Pass Filters Library</h1>')
Layout2 = row(menu)
Layout3 = row(allPass_fig1, filter1_phase)
Layout4 = row(Apply)
Header3 = Div(text='<h1 style="color: black">Customize Your Filter</h1>')
Layout5 = row(custom_fig, custom_phase_fig)
Layout6 = row(Apply1, ClearCustom)


x = layout([
    [Layout], [Header2], [Layout2], [Layout3], [
        Layout4], [Header3], [Layout5], [Layout6]

])

# show(Layout)
curdoc().add_root(x)
# Open the command prompt and navigate into the folder where the code file is located using cd <folder>
# to move into a folder, for instance, cd Documents, and cd ../ to move out of a folder.
# Once in the same folder as the .py file, run bokeh serve --show example.py to open the graph in
# localhost.
# To exit the server, press Ctrl + C in command prompt. It may take a few seconds to shut down.
