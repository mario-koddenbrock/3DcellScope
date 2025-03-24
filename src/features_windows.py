
from src.external import *
from src.features_UI import Ui_MainWindow
from src.features_stat import DESCRIPTOR_TYPE
from src.features_messages import FILTERMSG, ComponentAnalysisParameters, ComponentAnalysisOptions, PCATYPE
import src.features_global as glob
from src.features_popups import FolderImportPopup
from src.featuresML import RandomForest,MLP,DecisionTree,SVM, MODEL_LIB, PREPROC, Classifier


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        super(Ui_MainWindow,self).__init__()
        
        self.setupUi(self)
        my_icon = QIcon()
        my_icon.addFile(str(Path('icon.png')))
        self.setWindowIcon(my_icon)
        self.setWindowTitle('FeaturesScore')
        # self.connect_components()
        self.init_style_labels()
        self.init_model_fame()

    def connect_components(self):
        #self.loadDataButton.clicked.connect(glob.CONTEXT.app_manager.load_csv)
        self.actionLoad_CSV.triggered.connect(glob.CONTEXT.app_manager.load_csv)
        self.actionLoad_Multiple.triggered.connect(lambda:glob.CONTEXT.app_manager.load_from_folder(None,None))
        self.actionLoad_Project.triggered.connect(glob.CONTEXT.app_manager.load_project)
        self.actionSave_Project.triggered.connect(glob.CONTEXT.app_manager.save_project)
        self.actionExport_CSV.triggered.connect(glob.CONTEXT.app_manager.export_csv)
        #self.actionAggregate_Data.triggered.connect(glob.CONTEXT.app_manager.set_aggregators)
        self.aggregateDataButton.clicked.connect(glob.CONTEXT.app_manager.set_aggregators)
        self.filterGroupButtonRemove.clicked.connect(glob.CONTEXT.app_manager.remove_selected_groupe)
        self.filterGroupButtonKeepOnly.clicked.connect(glob.CONTEXT.app_manager.keep_only_selected_groupe)
        self.filterGroupButtonBack.clicked.connect(glob.CONTEXT.app_manager.undo_last_groupe_filter)
        self.filterGroupButtonReset.clicked.connect(glob.CONTEXT.app_manager.reset_group_filter)
        self.crossTestingButton.clicked.connect(glob.CONTEXT.app_manager.cross_test_plot)
        self.barPlotButton.clicked.connect(glob.CONTEXT.app_manager.bar_plot)
        self.boxPlotButton.clicked.connect(glob.CONTEXT.app_manager.box_plot)
        #self.choseMetadataButton.clicked.connect(glob.CONTEXT.app_manager.chose_meta_data)
        self.scatterPlotButton.clicked.connect(glob.CONTEXT.app_manager.scatter_plot)
        self.densityPlotButton.clicked.connect(glob.CONTEXT.app_manager.density_plot)
        self.pcaButton.clicked.connect(lambda: glob.CONTEXT.app_manager.componnent_analysis_plot(ComponentAnalysisOptions(PCATYPE.PCA)))
        self.ldaButton.clicked.connect(lambda: glob.CONTEXT.app_manager.componnent_analysis_plot(ComponentAnalysisOptions(PCATYPE.LDA)))
        self.summaryPlottButton.clicked.connect(glob.CONTEXT.app_manager.summary_plot)
        self.radarPlotButton.clicked.connect(glob.CONTEXT.app_manager.radar_plot)
        self.countsButton.clicked.connect(glob.CONTEXT.app_manager.show_count_popup)
        self.excludeSelectedDescriptorButton.clicked.connect(lambda: glob.CONTEXT.app_manager.filter_descriptor(FILTERMSG.EXCLUDE))
        self.keepOnlySelectedDescriptorButton.clicked.connect(lambda: glob.CONTEXT.app_manager.filter_descriptor(FILTERMSG.KEEP_ONLY))
        self.resetDescriptorFilterButton.clicked.connect(lambda: glob.CONTEXT.app_manager.filter_descriptor(FILTERMSG.RESET))
        self.ApllyDescriptorCorrelationButton.clicked.connect(lambda:glob.CONTEXT.app_manager.descriptor_corrleations(None))
        self.resetDescriptorCorelationButton.clicked.connect(glob.CONTEXT.app_manager.reset_descriptor_corrleations)
        self.automaticAnalysisButton.clicked.connect(glob.CONTEXT.app_manager.automatic_analysis)
        self.replicateRadarButton.clicked.connect(glob.CONTEXT.app_manager.replicate_radar)
        self.tableWidget.cellActivated.connect(glob.CONTEXT.app_manager.feature_grid_item_clicked)
        self.hideIgnoredFeaturesCheckBox.stateChanged.connect(self.update_feature_grid_hiden_state)
        self.featureGateButton.clicked.connect(glob.CONTEXT.app_manager.add_feature_gate)
        self.dataGroupWidget.cellDoubleClicked.connect(glob.CONTEXT.app_manager.data_group_item_clicked)
        self.dataGroupWidget.verticalHeader().sectionHandleDoubleClicked.connect(glob.CONTEXT.app_manager.data_group_item_clicked)
        self.modelComboBox.activated.connect(self.show_model_param)
        self.modelComboBox.editTextChanged.connect(self.rename_model)
        self.toogleEditModelButton.clicked.connect(self.toogle_edit_model)
        self.addModelButton.clicked.connect(self.add_model)
        self.modelFitButton.clicked.connect(glob.CONTEXT.app_manager.fit_current_model)
        self.modelPredictButton.clicked.connect(glob.CONTEXT.app_manager.pred_curent_model)
        self.crossValidateButton.clicked.connect(glob.CONTEXT.app_manager.cross_validate_curent_model)
        self.classifOptimButton.clicked.connect(glob.CONTEXT.app_manager.optim_curent_model)
        self.testClassifButton.clicked.connect(glob.CONTEXT.app_manager.test_curent_model)
        self.preprocChoiceList.itemClicked.connect(self.update_model_preproc)

#region GetFunction
    def get_hide_ignored_features(self):
        return self.hideIgnoredFeaturesCheckBox.isChecked()
    def get_axis(self,max_val:int):
        axis = []
        for spin in [self.ax1SpinBox,self.ax2SpinBox,self.ax3SpinBox]:
            v = spin.value()
            if v>0 and v not in axis and v<=max_val:
                axis.append("Ax%d"%spin.value())
        return axis
    
    def get_selected_controls(self):
        #deprecated use data_manager.get control
        return [
            self.controlListWidget.item(i).my_index
            for i in range(self.controlListWidget.count())
            if self.controlListWidget.item(i).isSelected()
            ]
    def get_selected_tests(self):
        #deprecated use get_selected_group
        return [
            self.testListWidget.item(i).my_index
            for i in range(self.testListWidget.count())
            if self.testListWidget.item(i).isSelected()
            ]
    def get_selected_group(self):
        index = glob.CONTEXT.data_manager.get_unique_index()
        selected = self.dataGroupWidget.selectedIndexes()
        rows = set([s.row() for s in selected])
        return [index[r] for r in rows]
    def get_plot_color(self):
        return self.plotColorComboBox.currentText()
    def get_style_label(self):
        return self.stylePlotComboBox.currentText().lower()

    def get_selected_descriptors(self):
        listwidget = self.descriptorListWidget
        return [listwidget.item(i).descriptor_name for i in range(listwidget.count()) if listwidget.item(i).isSelected()]
    def get_lda_and_pca_params(self):
        return ComponentAnalysisParameters(
            int(self.maxPointLineEdit.text()),
            self.brestCorrelationSpinBox.value(),
            self.pcaModeComboBox.currentText()
        )
    def get_test_params(self):
        return(
            self.ssmdRadioButton.isChecked(),
            self.pValueRadioButton.isChecked(),
            self.uValueRadioButton.isChecked(),
            self.maskSummaryCheckBox.isChecked(),
            self.correctionSummaryCheckBox.isChecked(),
            min(max(0,float(self.significativityTresholdLineEdit.text())),0.9),
        )
    def get_radar_params(self):
        return(
            self.ssmdAxisLimitSpinBox.value(),
            self.sortRadarCheckBox.isChecked(),
            min(max(0,float(self.significativityTresholdLineEdit.text())),0.9)
        )
    def get_correlation_params(self):
        return(
            self.distanceThresholdCorrelationSpinBox.value(),
            self.showCorrelationCheckBox.isChecked(),
            self.filterDescriptorWithCorrelationCheckBox.isChecked(),
            "square" if self.squareDistCorrelationRadioButton.isChecked() else (
            "abs" if self.absDistCorrelationRadioButton.isChecked() else(
            "default"))
        )
#endregion

    def update_feature_grid(self):
        hide_ignored_columns = self.get_hide_ignored_features()
        features = glob.CONTEXT.data_manager.features
        self.tableWidget.setColumnCount(len(features))
        self.tableWidget.setGridStyle(Qt.PenStyle.NoPen)
        self.tableWidget.setRowCount(3)
        self.tableWidget.setHorizontalHeaderLabels([str(el) for el in features])
        self.tableWidget.setVerticalHeaderLabels(["Index","Numeric","Ignored"])
        self.tableWidget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        for j, feature in enumerate(glob.CONTEXT.data_manager.features):
            tab = glob.CONTEXT.data_manager.descriptors
            desc = tab[feature]
            for i in range(3):
                it = QTableWidgetItem()
                self.update_feature_grid_item(i,j, it, desc, hide_ignored_columns)
                self.tableWidget.setItem(i,j,it)

    def update_feature_grid_item(self, i:int,j:int, it, descriptor = None, hide_ignored_columns = None):
        descriptor = glob.CONTEXT.data_manager.get_descriptor(j) if descriptor is None else descriptor
        it.setBackground(QColor(0,0,0,15))
        it.is_toogled = False
        if ((i==0 and descriptor.name in glob.CONTEXT.data_manager.index)
        or (i==1 and descriptor.type == DESCRIPTOR_TYPE.NUMERIC)
        or (i==2 and descriptor.ignore)):
            it.setBackground(QColor(0,0,0,60))
            it.is_toogled = True
        if i==2:
            hide_ignored_columns = self.get_hide_ignored_features() if hide_ignored_columns is None else hide_ignored_columns
            self.tableWidget.setColumnHidden(j,hide_ignored_columns & descriptor.ignore )

    def update_feature_grid_hiden_state(self):
        hide_ignored_columns = self.get_hide_ignored_features()
        features = glob.CONTEXT.data_manager.features
        for j, feature_name in enumerate(glob.CONTEXT.data_manager.features):
            descriptor = glob.CONTEXT.data_manager.get_descriptor(feature_name)
            self.tableWidget.setColumnHidden(j,hide_ignored_columns & descriptor.ignore )

    def init_style_labels(self):
        style_list = sorted(
        style.capitalize() for style in plt.style.available
        if style != 'classic' and not style.startswith('_') and (style[0].lower()==style[0]))
        self.stylePlotComboBox.addItems(style_list)

    def init_model_fame(self):
        self.update_model_list()
        self.preprocChoiceList.clear()
        self.preprocChoiceList.addItems([k for k in PREPROC.keys()])
        self.show_model_param()


    def update_model_list(self):
        if len(MODEL_LIB)> self.modelComboBox.count():
            for k in MODEL_LIB.keys():
                if self.modelComboBox.findText(k)==-1:
                    self.modelComboBox.addItems([k])


    def update_bag_list(self):
        index = glob.CONTEXT.data_manager.data.index
        names = [str(el) for el in index.names]if isinstance(index, pd.MultiIndex) else [index.name]
        self.bagChoiceList.clear()
        self.bagChoiceList.addItems(list(names))
    def update_model_preproc(self):
        m = self.get_model()
        m.transforms = [key for i, key in zip(range(self.preprocChoiceList.count()), PREPROC.keys()) if self.preprocChoiceList.item(i).isSelected()]

    def toogle_edit_model(self):
        if self.classifOptWidget.isEnabled():
            self.classifOptWidget.setEnabled(False)
            self.toogleEditModelButton.setText('Edit')
            self.update_model_param()
        else:
            self.classifOptWidget.setEnabled(True)
            self.toogleEditModelButton.setText('Save')

    def toogle_cv_scors(self,scores:dict = None):
        self.cvScorsComboBox.clear()
        if scores is None: 
            self.cvScorsComboBox.setEnabled(False)
        else:
            self.cvScorsComboBox.setEnabled(True)
            text_items = ["%s : %.3f"%(k, np.mean(v)) for k,v in scores.items()]
            self.cvScorsComboBox.addItems(text_items)
            


    def get_curent_model_name(self):
        return self.modelComboBox.currentText()
    def get_model(self)->Classifier:
        return  MODEL_LIB[self.modelComboBox.currentText()]
    
    def get_classification_bags(self)->list:
        return [el.text() for el in self.bagChoiceList.selectedItems()]
    
    def get_optim_iter(self):
        return self.classifOptimIterSpinBox.value()

    def set_curent_model(self, model):
        MODEL_LIB[self.modelComboBox.currentText()]=model
    
    def add_model(self):
        name_root = self.modelTypeComboBox.currentText()
        model = {
            'Decision Tree':DecisionTree,
            'SVM':SVM,
            'Random Forest': RandomForest,
            'MLP': MLP
            }[name_root]()
        i=1
        global MODEL_LIB
        while "%s%d"%(name_root,i) in MODEL_LIB:i+=1
        name = "%s%d"%(name_root,i)
        MODEL_LIB[name]=model
        self.update_model_list()
        self.modelComboBox.setCurrentIndex(self.modelComboBox.findText(name))
        self.show_model_param()

    def rename_model(self, new_name):
        old_text = self.modelComboBox.itemText(self.modelComboBox.currentIndex())
        MODEL_LIB[new_name] = MODEL_LIB.pop(old_text)
        self.modelComboBox.setItemText(self.modelComboBox.currentIndex(),new_name)


    def update_model_param(self):
        model = self.get_model()
        simplified_args = model.simplified_args
        args = {}
        for i,(k,opts) in enumerate(simplified_args.items()):
            widg = self.__getattribute__("ClassifOpt%dWidg"%(i+1))
            widg.show()
            children = widg.children()
            lab_first = 1* (type(children[0]) is QLabel)
            lab:QLabel = children[(1+lab_first)%2]
            cmb:QComboBox = children[lab_first]
            args[lab.text()] = cmb.currentText()
        model.__init__(args)
        self.set_curent_model(model)


    def show_model_param(self):
        model = self.get_model()
        args = model.features_kwarg_str
        simplified_args = model.simplified_args
        for i,(k,opts) in enumerate(simplified_args.items()):
            widg = self.__getattribute__("ClassifOpt%dWidg"%(i+1))
            widg.show()
            children = widg.children()
            lab_first = 1* (type(children[0]) is QLabel)
            lab:QLabel = children[(1+lab_first)%2]
            cmb:QComboBox = children[lab_first]
            lab.setText(k)
            cmb.clear()
            cmb.addItems(opts)
            arg = args[k]
            if arg not in opts: cmb.addItems([arg])
            cmb.setCurrentIndex(cmb.findText(arg))

        for i in range(len(simplified_args),6):
            widg = self.__getattribute__("ClassifOpt%dWidg"%(i+1))
            widg.hide()
        for i in range(self.preprocChoiceList.count()):
            it = self.preprocChoiceList.item(i)
            it.setSelected(it.text() in model.transforms)
