import src.features_global as glob
import pickle
from src.external import *
from typing import Tuple
from src.features_stat import ScatterPlot, PCA, LDA,LDAScore, tTest, SSMD, mannWhitney, benjaminiHochberg, benjaminiHochbergAdjust, summaryView, thresh2Val, spiderplotMulty,summaryCorrelationReduction, autoAnalysis, autoReplicateRadarEvent,compareDensityPlot, crossTests,BoxPlot1D,ViolinPlot,BarPlot, Descriptor, DESCRIPTOR_TYPE
from src.features_messages import FILTERMSG, ComponentAnalysisOptions, PCAMODE,PCATYPE
from src.features_popups import FolderImportPopup, AddFilterOptions, AddAggregationOptions
#region helper functions

def get_safe_data():
    return glob.CONTEXT.data_manager.get_safe_data()
def get_main_windows():
    return glob.CONTEXT.main_windows

class ChoseFromChoiceListPopup(QDialog):
    def __init__(self,choices=[]):
        super().__init__()
        v_layout = QVBoxLayout()
        v_layout.addWidget(QLabel("Chose Index Columns"))
        self.choice_list = QtWidgets.QListWidget(self)
        for choice in choices:
            it = QListWidgetItem(str(choice))
            it.choice = choice
            self.choice_list.addItem(it)
        self.choice_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        v_layout.addWidget(self.choice_list)        
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        v_layout.addWidget(self.buttonBox)
        self.setLayout(v_layout)

def choseFromChoiceList(list_choice):
    out = []
    popup = ChoseFromChoiceListPopup(list_choice)
    popup.show()
    popup.setWindowModality(Qt.WindowModality.WindowModal)
    if popup.exec():
        for el in qt_list_get_all(popup.choice_list):
            if el.isSelected():
                out.append(el.choice)
        return out
def qt_list_get_all(listwidget):
    return [listwidget.item(i) for i in range(listwidget.count())]    
def with_style(func):
    def inner(*arg,**kwargs):
        matplotlib.use("qtagg" if get_main_windows().displayINTRadioButton.isChecked() else "agg")
        with plt.style.context(get_main_windows().get_style_label()):
            return func(*arg,**kwargs)
    return inner

#endregion


class MainApp:
    def __init__(self):
        self.folder_root = Path(r"./statics")
        matplotlib.style.use(['fast'])

#region inport
    def load_csv(self, file_name = None, index_cols = None):
        if not file_name:
            open_file_name = QFileDialog.getOpenFileNames(get_main_windows(), " File dialog ", str(self.folder_root), filter="CSV (*.csv)")
            if (len(open_file_name[0])<1) : return
            file_name = open_file_name[0][0]
        glob.CONTEXT.data_manager.set_data (pd.read_csv(Path(file_name),encoding="latin",encoding_errors="replace"))
        if index_cols is None:
            index_cols = [glob.CONTEXT.data_manager.features[0]]
        self.chose_meta_data(index_cols = index_cols )

    def load_from_folder(self,index_cols = None, import_options=None):
        if not import_options:
            popup = FolderImportPopup()
            # popup.show()
            popup.setModal(True)
            # popup.setWindowModality(Qt.WindowModality.WindowModal)
            if not popup.exec():
                return
            import_options = popup.returnOptions()
        glob_cmd = '/'.join(["*" for i in range(import_options["depth"])]+[import_options["file_name"]])
        paths = list(Path(import_options["folder_root"]).glob(glob_cmd))
        data_list = []
        file_name_range = range(0,-1)
        subfolder_name_range = range(0,-1)
        for path in paths:
            try:
                local_data =  pd.read_csv(path,encoding="latin",encoding_errors="replace")
                if import_options["add_fileName"]:
                    local_data["file_name"] = path.stem[file_name_range]
                if import_options["add_subfolder"]:
                    cols = list(local_data.columns)
                    new_cols = []
                    for i in range(import_options["depth"]):
                        local_data["subfolder_name_%d"%(i+1)] = path.parents[-i].stem[subfolder_name_range]
                        new_cols.append("subfolder_name_%d"%(i+1))
                    local_data = local_data[new_cols+cols]
                data_list.append(local_data)
            except:
                print("import of %s failed"%str(path))
        data = pd.concat(data_list,axis='index',join='inner',ignore_index=True)
        glob.CONTEXT.data_manager.set_data (data)
        if index_cols is None:
            index_cols = [glob.CONTEXT.data_manager.features[0]]
        self.chose_meta_data(index_cols = index_cols )
        
    def load_project(self, file_name=None):
        if not file_name:
            file_name = QFileDialog.getOpenFileName(get_main_windows(), " File dialog ", str(self.folder_root))
            if file_name=='' or file_name[0]=='':
                return
            file_name = file_name[0]
        try:
            glob.CONTEXT.data_manager.as_copy(pickle.load(open(Path(file_name),"rb")))
        except:
            button = QMessageBox.critical(None,"Error", "failed loading %s"%str(file_name),buttons=QMessageBox.StandardButton.Discard,defaultButton=QMessageBox.StandardButton.Discard)
            return        
        self.chose_meta_data(index_cols = glob.CONTEXT.data_manager.get_index_cols() )
    
    def save_project(self, file_name=None):
        if not file_name:
            file_name = QFileDialog.getSaveFileName(get_main_windows(), " File dialog ", str(self.folder_root))
            if file_name=='' or file_name[0]=='':
                return
            file_name = file_name[0]
        try:
            pickle.dump(glob.CONTEXT.data_manager.dump_view(),open(Path(file_name),"wb"))
        except:
            button = QMessageBox.critical(None,"Error", "failed saving %s"%str(file_name),buttons=QMessageBox.StandardButton.Discard,defaultButton=QMessageBox.StandardButton.Discard)
    
    def export_csv(self, file_name=None):
        if not file_name:
            file_name = QFileDialog.getSaveFileName(get_main_windows(), " File dialog ", str(self.folder_root), filter="CSV (*.csv)")
            if file_name=='' or file_name[0]=='':
                return
            file_name = file_name[0]
        try:
            glob.CONTEXT.data_manager.data.reset_index().to_csv(Path(file_name),index=False)
        except:
            button = QMessageBox.critical(None,"Error", "failed saving %s"%str(file_name),buttons=QMessageBox.StandardButton.Discard,defaultButton=QMessageBox.StandardButton.Discard)
#endregion

#region Data & Index managment
    def remove_selected_groupe(self):
        selection = get_main_windows().get_selected_group()
        glob.CONTEXT.data_manager.drop_data_with_index(selection)
        self.update_group_list()

    def keep_only_selected_groupe(self):
        selection = get_main_windows().get_selected_group()
        inverse = glob.CONTEXT.data_manager.get_unique_index().difference(selection)
        glob.CONTEXT.data_manager.drop_data_with_index(inverse)
        self.update_group_list()

    def undo_last_groupe_filter(self):
        glob.CONTEXT.data_manager.undrop_data(n=1)
        self.update_group_list()

    def reset_group_filter(self):
        glob.CONTEXT.data_manager.undrop_data(n=-1)
        self.update_group_list()

    def chose_meta_data(self, index_cols = None):
        if not index_cols:
            index_cols = choseFromChoiceList(glob.CONTEXT.data_manager.features)
        if index_cols is None:
            return
        glob.CONTEXT.data_manager.set_index(index_cols)
        self.update_group_list()
        self.update_descriptor_list()
        self.update_features()

    def update_features(self):
        get_main_windows().update_feature_grid()
        get_main_windows().update_bag_list()
    
    def feature_grid_item_clicked(self,i,j):
        it = get_main_windows().tableWidget.item(i,j)
        was_toogled = it.is_toogled
                
        feature = glob.CONTEXT.data_manager.features[j]
        tab =glob.CONTEXT.data_manager.descriptors

        if i == 0:
            if was_toogled and len(glob.CONTEXT.data_manager.index)==1:
                return #should rise warning
            if was_toogled and feature in glob.CONTEXT.data_manager.index:
                glob.CONTEXT.data_manager.index.remove(feature)
            elif not was_toogled:
                glob.CONTEXT.data_manager.index.add(feature)
            
            glob.CONTEXT.data_manager.set_index(glob.CONTEXT.data_manager.get_index_cols())
            self.update_group_list()
            get_main_windows().update_bag_list()
        elif i==1:
            if was_toogled or not glob.CONTEXT.data_manager.feature_to_numeric(feature):
                glob.CONTEXT.data_manager.feature_to_category(feature)
        elif i==2:
            tab[feature].ignore = not was_toogled
        else:
            return
        self.update_descriptor_list()
        get_main_windows().update_feature_grid_item(i,j,it)

    def add_feature_gate(self):
        popup = AddFilterOptions(glob.CONTEXT.data_manager.descriptors)
        # popup.show()
        # popup.setWindowModality(Qt.WindowModality.WindowModal)
        popup.setModal(True)
        if not popup.exec():
            return
        filter_querry = popup.results()
        if len(filter_querry)< 1:
            return
        glob.CONTEXT.data_manager.drop_data_with_querry(" & ".join(filter_querry))
        self.update_group_list()
        
    def data_group_item_clicked(self, row:int, col:int):
        if not row in glob.CONTEXT.data_manager.controls:
            glob.CONTEXT.data_manager.controls.add(row)
            color = QColor(0,0,0,60)
            get_main_windows().dataGroupWidget.verticalHeaderItem(row).setText("Control")
        else:
            glob.CONTEXT.data_manager.controls.remove(row)
            color = QColor(0,0,0,15)
            get_main_windows().dataGroupWidget.verticalHeaderItem(row).setText("Test")

        for col in  range(get_main_windows().dataGroupWidget.columnCount()):
            it = get_main_windows().dataGroupWidget.item(row,col)
            it.setBackground(color)
    
    def update_group_list(self):
        index = glob.CONTEXT.data_manager.data.index
        is_multi_index = isinstance(index, pd.MultiIndex)
        unique_index =  index.unique()
        if len(unique_index)>glob.MAX_INDEX_SIZE:
            unique_index = unique_index[:glob.MAX_INDEX_SIZE]
        n = 1 if not is_multi_index else len(index.levels)
        get_main_windows().dataGroupWidget.clear()
        get_main_windows().dataGroupWidget.setColumnCount(n)
        # get_main_windows().dataGroupWidget.verticalHeader().setVisible(False)
        get_main_windows().dataGroupWidget.setRowCount(len(unique_index))
        get_main_windows().dataGroupWidget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        idx_names = ([str(el) for el in index.names]if is_multi_index else [index.name])
        get_main_windows().dataGroupWidget.setHorizontalHeaderLabels(idx_names)
        get_main_windows().dataGroupWidget.setVerticalHeaderLabels(["Test" for _ in range(len(unique_index))])
        get_main_windows().dataGroupWidget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        get_main_windows().dataGroupWidget.verticalHeader().setSectionsClickable(True)
        get_main_windows().dataGroupWidget.verticalHeader().sectionHandleDoubleClicked.connect(self.data_group_item_clicked)
        for i,idx in enumerate(unique_index):
            
            for j, name in enumerate(idx_names):
                it = QTableWidgetItem()
                it.setText(str(idx[j]) if is_multi_index else str(idx))
                it.setBackground(QColor(0,0,0,15))
                get_main_windows().dataGroupWidget.setItem(i,j,it)


    def show_count_popup(self):
        counts = glob.CONTEXT.data_manager.get_index_counts()
        counts.name = "Counts"

        if get_main_windows().displayCSVRadioButton.isChecked():
            if type(counts.index[0]) is tuple: counts.index = pd.MultiIndex.from_tuples(counts.index, names=glob.CONTEXT.data_manager.get_unique_index().names)
            else: counts.index.name = glob.CONTEXT.data_manager.get_unique_index().name
            self.display_plot_results(pd.DataFrame(counts))
            return
        msg = counts.to_string().replace('nan','').replace("("," ").replace(")"," ").replace(",", " ")
        dlg = QMessageBox()
        dlg.setWindowTitle("Counts")
        dlg.setText(msg)
        dlg.exec()     

    def set_aggregators(self):
        popup = AddAggregationOptions(glob.CONTEXT.data_manager.descriptors, glob.CONTEXT.data_manager.aggregators)
        popup.setModal(True)
        if not popup.exec():
            return
        glob.CONTEXT.data_manager.aggregators = popup.results()

#endregion

#region Descriptor Managment
    def update_descriptor_list(self):
        descriptor_list = glob.CONTEXT.data_manager.get_valid_descriptor_names(clust_orderd=False)
        get_main_windows().descriptorListWidget.clear()
        for desc_name in descriptor_list:
            text = str(glob.CONTEXT.data_manager.descriptors[desc_name])
            item = QListWidgetItem(text)
            item.descriptor_name= desc_name
            get_main_windows().descriptorListWidget.addItem(item)

    def filter_descriptor(self,filter_msg):
        selection = get_main_windows().get_selected_descriptors()
        if filter_msg==FILTERMSG.RESET:
            for desc in glob.CONTEXT.data_manager.descriptors.values():
                desc.setIgnore(False)
        elif filter_msg==FILTERMSG.EXCLUDE:
            for desc_name in selection:
                glob.CONTEXT.data_manager.descriptors[desc_name].setIgnore(True)
        elif filter_msg==FILTERMSG.KEEP_ONLY:
                for desc in glob.CONTEXT.data_manager.descriptors.values():
                    desc.setIgnore(desc.name not in selection)
        self.update_descriptor_list()
        self.update_features()
        
    def descriptor_corrleations(self,correlation_param=None, inplace_filtering = True):
        thresh,show,post_filtering,correlation_dist = (get_main_windows().get_correlation_params() if correlation_param is None else correlation_param)
        correlation_distances = glob.CONTEXT.data_manager.find_descriptor_corrleations(thresh,correlation_dist,not show)
        if show:
            self.display_plot_results()
            plt.figure("Correlation Map")
            plt.imshow(correlation_distances);plt.yticks(range(len(correlation_distances.index)),correlation_distances.index,fontsize=5)
            self.display_plot_results(correlation_distances)
        filtering=None
        if post_filtering:
            filtering = glob.CONTEXT.data_manager.descriptor_corrleations_filtering(inplace_filtering)
        self.update_descriptor_list()
        self.update_features()
        return filtering
    
    def reset_descriptor_corrleations(self):
        glob.CONTEXT.data_manager.reset_descriptor_corrleations()
        self.update_descriptor_list()


#region single descriptor plot
    def get_single_data_col(self):
        selected = get_main_windows().get_selected_descriptors()
        if len(selected)==0:return 
        else: selected == selected[0]
        desc_col = get_safe_data().loc[:,selected].copy()
        group_keys = get_main_windows().get_selected_group()
        if len(group_keys)>0:
            desc_col = desc_col.loc[group_keys]
        return desc_col
    @with_style
    def bar_plot(self):
        desc_col = self.get_single_data_col()
        if desc_col is None: return
        fig, data = BarPlot(desc_col,title="Bar plot %s"%desc_col.columns[0],color = get_main_windows().get_plot_color())
        self.display_plot_results(data)

    @with_style
    def box_plot(self):
        desc_col = self.get_single_data_col()
        if desc_col is None: return
        fig,data = BoxPlot1D(desc_col,title="Box plot %s"%desc_col.columns[0],color = get_main_windows().get_plot_color())
        self.display_plot_results(data)

    @with_style
    def violin_plot(self):
        desc_col = self.get_single_data_col()
        if desc_col is None: return
        fig, data = ViolinPlot(desc_col,title="Violin plot %s"%desc_col.columns[0],color = get_main_windows().get_plot_color())
        self.display_plot_results(data)


#region scatter, pca & lda
    @with_style
    def scatter_plot(self):
        selected = get_main_windows().get_selected_descriptors()
        data_2_plot = get_safe_data().loc[:,selected]
        group_keys = get_main_windows().get_selected_group()
        if len(group_keys)>0:
            data_2_plot = data_2_plot.loc[group_keys]
        fig = ScatterPlot(data_2_plot,vectors=None,point=None,  title="Scatter Plot", score=None)
        self.display_plot_results(data_2_plot)
    @with_style
    def density_plot(self):
        selected = get_main_windows().get_selected_descriptors()[:2]
        group_keys = get_main_windows().get_selected_group()
        if len(group_keys)<1:
            group_keys=None
        data_2_plot = get_safe_data().loc[:,selected]
        if len(selected)==2:
            fig = compareDensityPlot(data_2_plot[selected[0]],data_2_plot[selected[1]],modes=["style", "alpha","fill"],indexGroupKeys=group_keys,n=200,proportions=[0.1,0.5,0.9])
            self.display_plot_results(data_2_plot)
        elif len(selected)==1:
            self.violin_plot()

    @staticmethod
    
    def solve_ca_options(options:ComponentAnalysisOptions)->Tuple[bool,list,list,list,list,int,int,PCAMODE]: 
        is_lda=options.pca_type is PCATYPE.LDA
        descs = glob.CONTEXT.data_manager.get_valid_descriptor_names() if options.descs is None else options.descs
        ref = glob.CONTEXT.data_manager.get_control_index() if options.ref is None else options.ref
        trial = get_main_windows().get_selected_group() if options.trial is None else options.trial
        axes= get_main_windows().get_axis(len(descs)) if options.axes is None else options.axes
        param =  get_main_windows().get_lda_and_pca_params() if options.ca_param is None else options.ca_param

        max_points = param.max_points
        number_of_correlation_to_show =param.number_of_correlation_to_show
        mode =param.mode
        return is_lda,descs,ref,trial,axes,param.max_points,param.number_of_correlation_to_show,param.mode
    
    def componnent_analysis_plot(self, options:ComponentAnalysisOptions):
        is_lda,descs,ref,trial,axes,max_points,number_of_correlation_to_show,mode = self.solve_ca_options(options)
        descriptors = glob.CONTEXT.data_manager.descriptors
        showtodesc = mode.is_vectorial_mode()
        data_2_plot = get_safe_data()

        data, comp, ratio = (LDA if is_lda else PCA)(data_2_plot, n=min(10,len(data_2_plot.index.unique())-1 if is_lda else 10,len(data_2_plot.columns)))
        vectors = None
        if is_lda:
            score = LDAScore(data_2_plot, n=min(10,len(data_2_plot.index.unique())-1,len(data_2_plot.columns)))
        if len(descs)>0 and number_of_correlation_to_show==0:
            comp = comp[descs]
        if len(list(set(trial+ref)))>0:
            data = data.loc[list(set(trial+ref))]

        cluster_list = [descriptors[desc].clustId(replace_C0=None) for desc in comp.T.index]
        descs_comp = list(comp.T.index)
        comp_reduced = summaryCorrelationReduction(descriptors,comp.T).T
        col = comp_reduced.columns
        if showtodesc:
            data = comp.T
            if mode is PCAMODE.CLUSTER:
                data["Cluster"] = cluster_list
                data.set_index('Cluster', append=True, inplace=True)
                data = data.droplevel(0)
            elif mode is PCAMODE.TARGET:
                data["Object"] = [descriptors[desc].origine for desc in data.index]
                data.set_index('Object', append=True, inplace=True)
                data = data.droplevel(0)
            else:
                data.index = data.index.set_names(["Descriptor"])
            
        arg_sort_comp = np.argsort(abs(comp_reduced),axis=1)
        
        correlations = [[
            "%s (%.2f)"%(col[el[-1-j]].replace("_"," "), comp_reduced.iloc[i,el[-1-j]]) 
            for j in range(number_of_correlation_to_show)]
            for i, el  in enumerate(arg_sort_comp.values)]


        data = data[axes]

        if showtodesc and number_of_correlation_to_show>0:
            arg_sort_comp = arg_sort_comp.loc[axes]
            cor = list(set(sum( [[
            col[el[-1-j]] 
            for j in range(number_of_correlation_to_show)]
            for i, el  in enumerate(arg_sort_comp.values)],[])))
            clust = [c in cor or d in descs for c, d in zip(cluster_list,descs_comp)]
            data = data[clust]
        elif mode is PCAMODE.AUTO and len(descs)+number_of_correlation_to_show >0:
            arg_sort_comp = arg_sort_comp.loc[axes]
            cor = list(set(sum( [[
            col[el[-1-j]] 
            for j in range(number_of_correlation_to_show)]
            for i, el  in enumerate(arg_sort_comp.values)],[])))
            clust = [c in cor or d in descs for c, d in zip(cluster_list,descs_comp)]
            vectors = comp.T[axes].loc[descs] if number_of_correlation_to_show==0 and len(descs)>0 else comp.T[axes][clust]
            
        if len(data)>max_points and not showtodesc:
            data = data.groupby(data.index).apply(lambda x: x.sample(min(max(int(max_points/len(data.index.unique())),1),len(x)), ))
            data = data.droplevel(0)
        data.columns = ["\n".join(
            ["%s "%ax +("" if is_lda else"(%.1f %%) "%(ratio[int(ax.replace("Ax",""))-1]*100))] + ["%s"%(correlations[int(ax.replace("Ax",""))-1][j]) for j in range(number_of_correlation_to_show) ]
            ) for i, ax in enumerate(axes)]
        if vectors is not None:
            vectors.columns = data.columns
        ScatterPlot(
            data=data if not showtodesc else None,
            vectors=vectors if mode is PCAMODE.AUTO else data if showtodesc else None,
            point = None,
            title=("PCA Scatter Plot" if not showtodesc else "PCA Feature Corelation Plot").replace("PCA","LDA" if is_lda else "PCA"),
            score = score if is_lda else None)
        self.display_plot_results(data)


#region model classif:


    def fit_current_model(self):
        X = get_safe_data().dropna()
        model = get_main_windows().get_model()
        model.fit_pipe(X, get_main_windows().get_classification_bags())

    def pred_curent_model(self):
        X = get_safe_data()
        model = get_main_windows().get_model()
        Z:pd.DataFrame = model.pred_pipe(X, get_main_windows().get_classification_bags())
        name  = get_main_windows().get_curent_model_name()
        Z.columns = ["%s__%s"%(name, str(col).replace("'","").replace(",","").replace("(","").replace(")","")) for col in Z.columns]
        for col in Z.columns:
            if col.replace(name,"").replace(' ','').replace('_','').replace('-','') =='':continue
            glob.CONTEXT.data_manager.add_column(Z[col])
        self.update_descriptor_list()
        self.update_features()
    def test_curent_model(self):
        X = get_safe_data()
        model = get_main_windows().get_model()
        scores = model.test(X, get_main_windows().get_classification_bags())
        get_main_windows().toogle_cv_scors(scores)

    def cross_validate_curent_model(self):
        X = get_safe_data()
        model = get_main_windows().get_model()
        scores = model.cross_validate(X, get_main_windows().get_classification_bags())
        get_main_windows().toogle_cv_scors(scores)

    def optim_curent_model(self):
        X = get_safe_data()
        w = get_main_windows()
        model = get_main_windows().get_model()
        scores = model.optimize_params(X, n_iter = w.get_optim_iter(),bag_names=w.get_classification_bags())
        w.toogle_cv_scors(scores)
        w.show_model_param()





#endregion
#region test plot
    @with_style
    def cross_test_plot(self, *args, test_param = None):
        desc_col = self.get_single_data_col()
        is_ssmd, is_pval, is_uval, hide_low, _, thresh_low = get_main_windows().get_test_params() if test_param is None else test_param
        type_test = "ssmd" if is_ssmd else "p_val" if is_pval else "u_val" 
        data = crossTests(desc_col,type_test = type_test, hide_low=hide_low,thresh_low=thresh_low)
        self.display_plot_results(data)
    @with_style
    def summary_plot(self, descs=None, ref=None, trial=None, test_param=None):
        """ Lanch a summary event: one or severall selected Trial condition (ie, one of glob.CONTEXT.data_managerata.index.unique() / unique metadata combination)
            are tested against a reference condition.
            Options are read from glob.CONTEXT.main_windowsindow.  Test include SSMD, Student and Mann Whitney. Test is performed for all non filtered descriptors.
            Option include BH correction, thresholding and sinificativity threshold value adjustment.
            Depending on diplay options, a plot/image/csv is shown to user

        """
        descs = glob.CONTEXT.data_manager.get_valid_descriptor_names() if descs is None else descs
        if len(descs)==0: return 
        ref = glob.CONTEXT.data_manager.get_control_index() if ref is None else ref
        if ref == []: return 
        trial = get_main_windows().get_selected_group() if trial is None else trial
        
        ref_data = get_safe_data()[descs].loc[ref]
        trial_data = get_safe_data()[descs].loc[trial] if len(trial)>0 else get_safe_data()[descs]

        is_ssmd, is_pval, is_uval, hide_low, correct_bh, thresh = get_main_windows().get_test_params() if test_param is None else test_param
        correct_bh = correct_bh and not is_ssmd

        if is_pval:
            summary = trial_data.groupby(trial_data.index.names).apply(tTest,ref_data)
            title = "P-Values Summary  Using T-test %s"%("" if not correct_bh else " and Benjamini Hochberg correction")
        elif is_uval:
            summary = trial_data.groupby(trial_data.index.names).apply(mannWhitney,ref_data)
            title = "U-Values Summary Using Mann-Whitney test %s"%("" if not correct_bh else "and Benjamini Hochberg correction")
        elif is_ssmd:
            summary = trial_data.groupby(trial_data.index.names).apply(SSMD,ref_data)
            title = "SSMD Summary"
            thresh = 1
            is_ssmd = True
        else: assert(False)
        summary.index = summary.index.droplevel(-1)
        if hide_low:            
            if correct_bh and not is_ssmd:
                pvals = summary
                correction = pvals.groupby(pvals.index.names).apply(benjaminiHochberg,thresh)
            elif not is_ssmd:
                summary[summary>thresh] = np.nan
            else:
                summary[abs(summary) < thresh] = np.nan
        if correct_bh:
                pvals = summary
                summary = pvals.groupby(pvals.index.names).apply(benjaminiHochbergAdjust,thresh)#
                if hide_low:
                    summary[~correction] = np.nan

        summary = summaryCorrelationReduction(glob.CONTEXT.data_manager.descriptors,summary.T,group_by_object = False,reduction_type="absMax" if is_ssmd else "absMin").T
        summaryView(summary, title)
        self.display_plot_results(summary)
    @with_style
    def radar_plot(self, descs=None, ref=None, trial=None, radar_param=None):
        """ Lanch a Radar PLot event: one or severall selected Trial condition (ie, one of glob.CONTEXT.data_managerata.index.unique() / unique metadata combination)
            are tested against a reference condition.
            Two test are done parallely: SSMD  and Mann Whitney. Test is performed for all non filtered descriptors.
            Options are read from glob.CONTEXT.main_windowsindow.  Option include SSMD lim (in order to zoom in/out) and sinificativity threshold value adjustment.
            Depending on diplay options, a plot/image/csv is shown to user

        """
        descs = glob.CONTEXT.data_manager.get_valid_descriptor_names() if descs is None else descs
        if len(descs)==0: return 
        ref = glob.CONTEXT.data_manager.get_control_index() if ref is None else ref
        if ref == []: return 
        trial = get_main_windows().get_selected_group() if trial is None else trial
        ssmd_radar_limit, sort_radar_values, thresh =  get_main_windows().get_radar_params() if radar_param is None else radar_param
        safe_data = get_safe_data()
        ref_data = safe_data[descs].loc[ref].copy()
        trial_data = safe_data[descs].loc[trial] if len(trial)>0 else safe_data[descs]

        summary = trial_data.groupby(trial_data.index.names).apply(SSMD,ref_data).T
        summary=summary.droplevel(-1, axis=1)
        thresh = float(thresh)
        sinif = trial_data.groupby(trial_data.index.names).apply(mannWhitney,ref_data)
        sinif = sinif.droplevel(-1,axis=0)
        sinif = sinif.groupby(sinif.index.names).apply(benjaminiHochbergAdjust,thresh).T
        sinif.values[sinif.values>1]=1
        if len(ref)==1:
            control_name = " ".join(map(str,ref[0])) if type(ref[0]) is tuple else str(ref[0])
        else:
            control_name = "Composit Control"       


        summary = summaryCorrelationReduction(glob.CONTEXT.data_manager.descriptors,summary,group_by_object = True)
        sinif = summaryCorrelationReduction(glob.CONTEXT.data_manager.descriptors,sinif,"absMin",group_by_object = True)

        sinif.values[...] = thresh2Val(sinif.values,thresh_dict = {1:3,thresh:15,thresh/5:25,thresh/50:40})
        spiderplotMulty(summary,sinif, control_name,shiny=True,lim=(-ssmd_radar_limit,ssmd_radar_limit),thresh=thresh, sort_val=sort_radar_values)
        self.display_plot_results(summary)
#endregion

    def automatic_analysis(self, descriptors=None, ref=None, trial=None, control=None, testing=False):
        control = choseFromChoiceList(glob.CONTEXT.data_manager.get_unique_index()) if not control else control
        ref = glob.CONTEXT.data_manager.get_control_index() if not ref else ref
        if ref == []: return 
        trial = get_main_windows().get_selected_group() if not trial else trial
        param = (0.4,False,True,"square")
        filtered_descriptors = self.descriptor_corrleations(correlation_param=param,inplace_filtering=False) if not descriptors else descriptors
        if filtered_descriptors ==[]:return 
        pdf_file = autoAnalysis(get_safe_data(),ref,control,trial,filtered_descriptors)
        if not testing:
            handle = webbrowser.get()
            handle.open(pdf_file)
        if not descriptors:
            self.reset_descriptor_corrleations()

    def replicate_radar(self, descriptors=None, ref=None, trial=None, control=None,radar_param=None, testing=False):
        control = choseFromChoiceList(glob.CONTEXT.data_manager.get_unique_index()) if not control else control
        ref = glob.CONTEXT.data_manager.get_control_index() if not ref else ref
        if ref == []: return 
        trial = get_main_windows().get_selected_group() if not trial else trial
        param = (0.4,False,True,"square")
        valid_descriptors = glob.CONTEXT.data_manager.get_valid_descriptor() if not descriptors else descriptors
        if valid_descriptors ==[]:return 
        ssmd_radar_limit, sort_radar_values, thresh =  get_main_windows().get_radar_params() if radar_param is None else radar_param
        pdf_file = autoReplicateRadarEvent(get_safe_data(),ref,control,trial,valid_descriptors,ssmd_radar_limit,thresh)
        if not testing:
            handle = webbrowser.get()
            handle.open(pdf_file)

    def display_plot_results(self, data = None):
        """called after a plot event read display option an window and produced:
            -an interactiv plot
            -a PNG
            -a PDF
            -a csv 
            the 3 last option create a temporary file oppended with user browser

        Args:
            data (pandas.DataFrame, optional): Only usefull for the csv option, to be converted in a .csv tempfile . Defaults to None.
        """
        if get_main_windows().displayINTRadioButton.isChecked():
            #plt.get_current_fig_manager().window.wm_geometry("+20+20")
            if len(plt.get_fignums())>0: plt.gcf().show()
        else:
            handle = webbrowser.get()
            prefix = plt.gcf().canvas.manager.get_window_title().replace(" ","_").replace("(","").replace(")","")+"_"
            if get_main_windows().displayPNGRadioButton.isChecked():
                
                temp = tempfile.mkstemp(suffix='.png',prefix=prefix)
                plt.savefig(temp[1],format="png", dpi=400)

            elif get_main_windows().displayPDFRadioButton.isChecked():
                temp = tempfile.mkstemp(suffix='.pdf',prefix=prefix)
                plt.savefig(temp[1],format="pdf")

            elif get_main_windows().displayCSVRadioButton.isChecked() and data is not None:
                if "-log10" in prefix:
                    prefix=prefix.replace("_-log10_"," ")
                    data = 10**(-data)
                temp = tempfile.mkstemp(suffix='.csv',prefix=prefix)
                sep = "\t"
                data.to_csv(temp[1],sep=sep)
            else:
                return
            handle.open(temp[1])
            plt.close(plt.gcf().number)