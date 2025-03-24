
import numpy as np
import pandas as pd
import holoviews as hv
import panel as pn
import time
import panel.widgets as pnw
from bokeh.settings import settings
settings.log_level('error')
import logging
logging.getLogger('bokeh').setLevel(logging.ERROR)
from holoviews.streams import Selection1D,BoundsXY,Lasso
from sklearn.decomposition import PCA as sklearn_PCA
hv.extension('bokeh', width=100)
hv.renderer('bokeh').theme = 'dark_minimal'
pn.config.theme = 'dark'
from PySide6.QtWidgets import QMainWindow, QComboBox
from PySide6.QtWebEngineWidgets import QWebEngineView
from shapely.geometry import Point, Polygon
test_dataFrame = pd.DataFrame()
if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    app = QApplication([])   
    test_dataFrame = pd.read_csv(r"C:\Users\titouanp\Downloads\fusion-statistics_density.csv")
dataset = hv.Dataset([])
x_select = pnw.Select(name='X-Axis', value='None', options=['None'],width=150)
y_select = pnw.Select(name='Y-Axis', value='None', options=['None'],width=150)
size_select = pnw.Select(name='Size', value='None', options=['None'] + ['None'],width=150)
group_select = pnw.Select(name='Group-By', value='None', options=['None'],width=150)
aggregate_select = pnw.Select(name='Aggregate-By', value='None', options=['None'],width=150)
filter_out_selection = pnw.Button(name= "Remove outside Selection", disabled = True)
filter_in_selection = pnw.Button(name= "Remove inside Selection", disabled = True)
display_mode = pnw.ToggleGroup(name='display', options=['Scatter', 'Hist', 'PCA'], behavior="radio")

hv_layout = pn.Column(
    pn.Row(pn.layout.HSpacer(),display_mode,pn.layout.HSpacer()),
    "",
    pn.Row(pn.layout.HSpacer(),x_select, y_select, size_select,pn.layout.HSpacer()),
    pn.Row(pn.layout.HSpacer(),group_select, aggregate_select,  pn.layout.HSpacer()),
    pn.Row(pn.layout.HSpacer(),filter_out_selection,filter_in_selection,pn.layout.HSpacer())
    )
hv_layout.servable()
timer = 0
point_opts = dict(cmap='Category10',color = "lightblue", responsive=True, line_color='black',tools=['box_select','lasso_select','hover'],active_tools=[], size=5)
points = hv.Points([]).opts(**point_opts)
selection =Selection1D(source=points)
cross_selector_feature_PCA = pn.widgets.CrossSelector(name='PCA Feature', value=[], 
    options=[],sizing_mode='stretch_both')
pca_name = pn.widgets.TextInput(name = "PCA feature prefix", value="PCA",width = 150)
pca_axis = pn.widgets.IntInput(name='Axis to generate', value=3, step=1, start=1, end=10,width = 150)
pca_button = pn.widgets.Button(name = 'Create PCA Features', align=("end","end"))
save_pca_button = pn.widgets.Button(name = 'Save PCA Features', align=("end","end"))
pca_layout = pn.Column(cross_selector_feature_PCA,pn.Row(pn.layout.HSpacer(),pca_name,pca_axis,pca_button,save_pca_button,pn.layout.HSpacer()))

def selection_event(event):
    select_exist = False
    if event.new:
        if len(event.new)>0:
            select_exist = True
    for el in hv_layout[4]:
        el.disabled = not select_exist

def box_wrapper(selection:Selection1D):
    def box_callback(event):
        if event.new:
            bou = event.new
    return box_callback
def lasso_wrapper(selection:Selection1D):
    def lasso_callback(event):
        if event.new is not None:
            geo = event.new
    return lasso_callback

def filter_selection(event):
    global points

    if len(selection.index)==0:return []
    ag = [el.value for el in [group_select,aggregate_select] if el.value !="None"] if aggregate_select.value !="None" else []
    inside = event.obj.name == 'Remove inside Selection'
    selected_indices = index_from_selection(inside=inside,ag=ag)
    update_dataset(dataset.data.loc[selected_indices])
    points = points.clone(data = dataset.aggregate(ag,np.mean) if len(ag)>0 else dataset)
    selection.reset()
    return selected_indices

def index_from_selection(inside=False, ag=[]):
    if len(ag)>0:
        ag_dataset = dataset.aggregate(ag,np.mean)
        selected_indices = ag_dataset.data.index[selection.index]
        if inside:
            selected_indices = list(set(ag_dataset.data.index) - set(selected_indices))
        new_index = True
        for el in ag:
            grp_to_keep = pd.Index(ag_dataset.data.loc[selected_indices][el])
            new_index*=dataset.data[el].isin(grp_to_keep)
        selected_indices = new_index[new_index].index 
    else:
        selected_indices = dataset.data.index[selection.index]
        if inside:
            selected_indices = list(set(dataset.data.index) - set(selected_indices))
    return selected_indices

def update_source():
    for el in [selection,bounds,lasso]:
        el.source = points

def update_button_visibility(f):
    def wrapper(*args,**kwargs):
        ret = f(*args,**kwargs)
        p,h = display_mode.value == "PCA",display_mode.value == "Hist"
        s = not p and not h
        for lay in [hv_layout[2],hv_layout[3],hv_layout[4]]:
                for el in lay:
                    if type(el) is pn.layout.HSpacer: continue
                    show =  s or (h and el in [x_select, group_select,aggregate_select])
                    el.visible = show
        return ret
    return wrapper




def xy_changed(*args,**kwargs):
    global points
    x,y = x_select.value,y_select.value
    if not x in points.data.columns or not y in points.data.columns: return
    points = points.clone(kdims=[x, y],label="%s vs %s" % (x.title(), y.title()))

def update_vdims():
    global points
    vdims = [el.value for el in [group_select,size_select,aggregate_select] if el.value !="None"]
    points = points.clone(vdims=vdims)

def size_changed(*args,**kwargs):
    update_vdims()
    size = size_select.value 
    s = hv.dim(size).norm()*18+2  if size != 'None' else 5
    point_opts["size"] = s

def group_changed(*args,**kwargs):
    update_vdims()
    group = group_select.value
    if group != 'None':
        point_opts["cmap"] = 'Category10'
        point_opts["color"] = group
    else:
        point_opts["cmap"] = 'rainbow'
        point_opts["color"] = "lightblue"

def aggregator_changed(*args,**kwargs):
    global points
    ag = [el.value for el in [group_select,aggregate_select] if el.value !="None"]
    vdims = [el.value for el in [group_select,size_select,aggregate_select] if el.value !="None"]
    points = points.clone(data=dataset if aggregate_select.value =="None" else dataset.aggregate(ag,np.mean), vdims=vdims)

def watch_selector():
    x_select.param.watch(xy_changed,"value")
    y_select.param.watch(xy_changed,"value")
    size_select.param.watch(size_changed,"value")
    group_select.param.watch(group_changed,"value")
    aggregate_select.param.watch(aggregator_changed,"value")
    filter_in_selection.param.watch(filter_selection,"value")
    filter_out_selection.param.watch(filter_selection,"value")
    selection.param.watch(selection_event,"index")

watch_selector()
bounds = BoundsXY(source=points)
bounds.param.watch(box_wrapper(selection),'bounds')

lasso = Lasso(source=points)
lasso.param.watch(lasso_wrapper(selection),'geometry')

def get_dataframe():
    return test_dataFrame

def update_source_dataframe(df:pd.DataFrame):
    test_dataFrame = dataset.data

def run_PCA(*args,**kwargs):
    global points
    features = cross_selector_feature_PCA.value
    name = pca_name.value
    data, comp, ratio = PCA(dataset.data[features], n=min(pca_axis.value,10,len(features)))
    data.columns = [name+'-'+el for el in data.columns]
    update_dataset(pd.concat([dataset.data,data],axis=1))
    update_selector()
    ag = [el.value for el in [group_select,aggregate_select] if el.value !="None"]
    points=points.clone(data=dataset if aggregate_select.value =="None" else dataset.aggregate(ag,np.mean),kdims=[x_select.value, y_select.value])
    update_source_dataframe(dataset.data)
    if len(data.columns)>1:
        x_select.value = data.columns[0]
        y_select.value = data.columns[1]
        display_mode.value = "Scatter"

def save_fusion_dataframe_with_pca():
    print(' I ma in save fusion dataframe :')
    # test_dataFrame.to_csv('fusion-statistics.csv')

def save_PCA(*args, **kwargs):
    save_fusion_dataframe_with_pca()

pca_button.on_click(run_PCA)
save_pca_button.on_click(save_PCA)


def PCA(data, n=3):
    """Performe PCA on data

    Args:
        data (pandas.Dataframe): each row is a data point, each column a descriptor
        n (int, optional): number of axes for the pca.

    Returns:
        tuple: 
            - pandas.Dataframe : datapoints projected in the PCA 
            - pandas.Dataframe : descriptor correlation with the PCA axes
            - pandas.Dataframe : Explained variance ratio of each PCA axes
    """
    normalized_data=(data-data.mean())/data.std()
    normalized_data[normalized_data.isna()]=0
    pca = sklearn_PCA(n_components=n)
    pca.fit(normalized_data)
    out = pca.transform(normalized_data)    
    out = pd.DataFrame(out, index=data.index, columns=["Ax%d"%(i+1)for i in range(n)])
    out = 2*((out-out.min())/(out.max()-out.min())-0.5)
    comp = pd.concat([normalized_data,out],axis=1).corr()[out.columns].loc[normalized_data.columns].T

    return out, comp, pca.explained_variance_ratio_
def update_dataset(dataframe:pd.DataFrame):
    global dataset
    kdims = [el for el in ['file', 'name_split_1', 'name_split_2', 'name_split_3', 'cell_Id', 'corresponding_organoid'] if el in  dataframe.columns]
    if 'Unnamed: 0' in dataframe.columns: 
        dataframe.drop(columns=['Unnamed: 0'], inplace=True)
    for col in kdims:
        dataframe[col] = dataframe[col].astype(str)
    vdims = [col for col in dataframe.columns if col not in kdims and not dataframe[col].dtype == object]
    dataset = hv.Dataset(dataframe, kdims, vdims)

def update_selector():
    for i,el in enumerate([x_select,y_select,size_select]):
        el.options = (["None"] if i>1 else [])+[str(el) for el in dataset.vdims]
        el.value = el.value if el.value in el.options else el.options[min(i,len(el.options)-1)]
    for i,el in enumerate([group_select,aggregate_select]):
        el.options = ['None'] + [str(el) for el in dataset.kdims]
        el.value = el.value if el.value in el.options else el.options[0]


@pn.depends(x_select.param.value, y_select.param.value, group_select.param.value,aggregate_select.param.value , size_select.param.value, filter_out_selection.param.value,filter_in_selection.param.value, display_mode.param.value)
@update_button_visibility
def create_figure(*args):
    global points
    update_source()
    points = points.opts(**point_opts)
    if display_mode.value == "Hist":
        hist = hv.operation.histogram(points,dimension=x_select.value,groupby= [group_select.value] if group_select.value != "None" else None, normed=True).opts(responsive=True).opts(hv.opts.Histogram(alpha=0.2,color=hv.Cycle('Category10')))
        curv =  (hv.Curve(hist, label='') if type(hist) is hv.Histogram else  hv.NdOverlay ({k:hv.Curve(v).opts(hv.opts.Curve(color=hv.Cycle('Category10'))) for k,v in hist.items()},kdims= group_select.value)  )
        hist =  hist * curv
        hist.label = label="Histogram of %s" %x_select.value.title()
        return hist
    elif display_mode.value == "PCA":
        
        cross_selector_feature_PCA.options = x_select.values
        cross_selector_feature_PCA.value = [x_select.values[:4]]
        return pca_layout
    
    return points


def advanced_dataset_scater_plot():
    global points
    main_dataframe = get_dataframe()
    if main_dataframe.empty: return None
    
    update_dataset(main_dataframe)
    update_selector()
    ag = [el.value for el in [group_select,aggregate_select] if el.value !="None"]
    points=points.clone(data=dataset if aggregate_select.value =="None" else dataset.aggregate(ag,np.mean),kdims=[x_select.value, y_select.value])

    
    hv_layout[1] = create_figure
    return hv_layout

class PannelWindow(QMainWindow):
    def __init__(self):
        super(PannelWindow, self).__init__()
        self.plot_widget = QWebEngineView()
        self.plot_widget.setMinimumSize(500,500)
        self.setCentralWidget(self.plot_widget)
        self.server_thread = pn.serve(pannels,port=5006,address="localhost", show=False, threaded=True)
    def closeEvent(self, event):
        self.server_thread.stop()



pannels = pannels = {'/%s-%d'%(f.__name__,i): lambda f=f,kwargs=kwargs: f(**kwargs) for i,(f,kwargs) in enumerate([
    (advanced_dataset_scater_plot,{}),
    ])}
 

if __name__ == '__main__':
    window = PannelWindow()
    window.show()
    window.plot_widget.load("http://localhost:5006%s"%next(iter(pannels)))
    app.exec()
    window.server_thread.stop()