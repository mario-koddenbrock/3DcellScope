import sys
import warnings
original_stdout = sys.stdout
original_warning_settings = warnings.filters[:]
warnings.simplefilter("ignore")
from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt
from PySide6 import QtCore
import unittest
import  OS_Manager 
from OS_Popups import Ui_Dialog
from pathlib import Path
from unittest.mock import patch
from unittest.mock import MagicMock
from tifffile import imread
import inspect
import time
import shutil
import pandas as pd
import pytest
warnings.filters = original_warning_settings
test_im_fold = Path("images")
class TestMyApplication(unittest.TestCase):    
    @classmethod
    def setUpClass(self):
        self.app = OS_Manager.app
        self.app.setPalette(OS_Manager.create_dark_palette())
        self.app.setStyle("Fusion")
        self.main_window = OS_Manager.my_window
        OS_Manager.connections(self.main_window)
        OS_Manager.my_window.VolumeMinLineEdit.setText("640")
        OS_Manager.my_window.VolumeMaxLineEdit.setText("3000")
        self.test_im_fold = test_im_fold
        self.time_recorder = []

    def simple_func_timer(self, func, name, *args,**kwargs):
        tic = time.time()
        ret = func(*args,**kwargs)
        toc = time.time() - tic
        self.time_recorder.append({"name":name, "time":toc})
        return ret   
    @classmethod
    def tearDownClass(self):
        OS_Manager.my_window.setFocus()
        OS_Manager.my_window.show()
        QTest.qWait(3000)
        for item in self.test_im_fold.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
        pd.set_option('display.float_format', '{:.1f}s'.format)
        print(pd.DataFrame(self.time_recorder))
        pd.reset_option('display.float_format')
        

    def test_application_started(self):
        """ Check if the main window or widget is created successfully
            Start the application event loop and  verify that the application is running
        """
        self.assertIsNotNone(self.main_window)

        self.main_window.show()
        exposed = QTest.qWaitForWindowExposed(self.main_window,1000)
        
        self.assertTrue(exposed, "Application failed to start in time")

    def test_image_folder_open(self):
        """Check if the test folder images are present and try to load it
        """
        self.assertTrue(self.test_im_fold.exists(), "Testing image folder not found")
        self.assertTrue(len(list(self.test_im_fold.glob("*.tif")))>0, "Testing image folder is empty")
        OS_Manager.QFileDialog.getExistingDirectory = MagicMock(return_value="images")
        self.simple_func_timer(OS_Manager.openFolder,"Oppening image folder")
        QTest.qWait(100)
        self.assertTrue(len(OS_Manager.fileInfos) >0, "no image where loaded")
        OS_Manager.my_window.FileListWidget.setCurrentRow(OS_Manager.my_window.FileListWidget.row(OS_Manager.my_window.FileListWidget.findItems("ZeroG-Breast-Cancer-Organoid",Qt.MatchFlag.MatchExactly)[0]),QtCore.QItemSelectionModel.SelectionFlag.SelectCurrent)
        OS_Manager.imageSelected(OS_Manager.my_window.FileListWidget.item(OS_Manager.my_window.FileListWidget.currentRow()))
        QTest.qWait(100)

    def test_image_im_displayed(self):
        """Check if the first image of the test folder is correctly diplayed
        """
        self.assertGreater(len(OS_Manager.my_window.image_ax.images),0,"no image displayed")
        im_displayed = OS_Manager.my_window.image_ax.images[0].get_array().data
        im2_display = imread(Path("images/ZeroG-Breast-Cancer-Organoid_C4.tif"))
        self.assertTrue(im_displayed.shape[:2] == im2_display[0].shape[:2], "image is not of correct shape")
    
    def test_auto_resize(self):
        """ test the auto computaiton of images voxel size for nuclei computation
        """
        popup = Ui_Dialog((OS_Manager.my_window.DeltaXLineEdit.text(), OS_Manager.my_window.DeltaYLineEdit.text(), OS_Manager.my_window.DeltaZLineEdit.text()), OS_Manager.my_window.DefaultModelComboBox.currentText())
        popup.ComputeValues()
        outputResize = popup.outputResize
        OS_Manager.my_window.ResampleXLineEdit.setText(str(outputResize[0]))
        OS_Manager.my_window.ResampleYLineEdit.setText(str(outputResize[1]))
        OS_Manager.my_window.ResampleZLineEdit.setText(str(outputResize[2]))

    def test_correct_bleaching(self):
        """ test the correct bleaching images preprocessing
        """
        OS_Manager.my_window.MethodPreprocessingComboBox.setCurrentIndex( OS_Manager.my_window.MethodPreprocessingComboBox.findText("correct bleaching"))
        OS_Manager.my_window.AddPreprocessingButton.click()
        QTest.qWait(10)
        OS_Manager.my_window.PreprocessingCheckBox.setChecked(True)
        QTest.qWait(100)

    def test_simple_nuclei_process(self):
        """ test a simple nuclei process 
        """
        z = OS_Manager.my_window.ImageViewerSlider.value()
        self.simple_func_timer(OS_Manager.processFile, "Nuclei process")
        OS_Manager.my_window.ImageViewerSlider.setValue(z)
        OS_Manager.my_window.MaskVisualisationComboBox.setCurrentIndex(OS_Manager.my_window.MaskVisualisationComboBox.findText("Nuclei"))
        QTest.qWait(100)
        self.assertIsNotNone(OS_Manager.dictPredicted["images_paths"]['mask_nuclei'],"no nuclei mask generated")
        self.assertTrue(Path(OS_Manager.dictPredicted["images_paths"]['mask_nuclei']).exists(),"no nuclei mask file saved")
        self.assertIsNotNone(OS_Manager.dictLayers["nucleiContourOverlay"],"no nuclei contour generated")

    def test_nuclei_organoid_process(self):
        """ test an organoid + nuclei process 
        """
        z = OS_Manager.my_window.ImageViewerSlider.value()
        QTest.qWait(100)
        OS_Manager.my_window.DetectOrganoidCheckBox.setChecked(True)
        self.simple_func_timer(OS_Manager.processFile, "Nuclei and Organoid process")
        OS_Manager.my_window.ImageViewerSlider.setValue(z)
        OS_Manager.my_window.MaskVisualisationComboBox.setCurrentIndex(OS_Manager.my_window.MaskVisualisationComboBox.findText("Organoid"))
        QTest.qWait(100)
        self.assertIsNotNone(OS_Manager.dictPredicted["images_paths"]['mask_nuclei'],"no nuclei mask generated")
        self.assertTrue(Path(OS_Manager.dictPredicted["images_paths"]['mask_nuclei']).exists(),"no nuclei mask file saved")
        self.assertIsNotNone(OS_Manager.dictPredicted["images_paths"]['mask_organo'],"no oganoid mask generated")
        self.assertTrue(Path(OS_Manager.dictPredicted["images_paths"]['mask_organo']).exists(),"no oganoid mask file saved")
        self.assertIsNotNone(OS_Manager.dictLayers["nucleiContourOverlay"],"no nuclei contour generated")
        self.assertIsNotNone(OS_Manager.dictLayers["organoidContourOverlay"]," no oganoid contour generated")

    def test_nuclei_organoid_cell_process(self):
        """ test an organoid + nuclei + cell process 
        """
        z = OS_Manager.my_window.ImageViewerSlider.value()
        OS_Manager.my_window.DetectOrganoidCheckBox.setChecked(True)
        OS_Manager.my_window.DetectCellsCheckBox.setChecked(True)
        self.simple_func_timer(OS_Manager.processFile, "Nuclei, Organoid and Cell process")
        OS_Manager.my_window.ImageViewerSlider.setValue(z)
        OS_Manager.my_window.MaskVisualisationComboBox.setCurrentIndex(OS_Manager.my_window.MaskVisualisationComboBox.findText("Cells"))
        QTest.qWait(100)
        self.assertIsNotNone(OS_Manager.dictPredicted["images_paths"]['mask_nuclei'],"no nuclei mask generated")
        self.assertTrue(Path(OS_Manager.dictPredicted["images_paths"]['mask_nuclei']).exists(),"no nuclei mask file saved")
        self.assertIsNotNone(OS_Manager.dictPredicted["images_paths"]['mask_organo'],"no oganoid mask generated")
        self.assertTrue(Path(OS_Manager.dictPredicted["images_paths"]['mask_organo']).exists(),"no oganoid mask file saved")
        self.assertIsNotNone(OS_Manager.dictPredicted["images_paths"]['mask_cell'],"no cell mask generated")
        self.assertTrue(Path(OS_Manager.dictPredicted["images_paths"]['mask_cell']).exists(),"no cell mask file saved")
        self.assertIsNotNone(OS_Manager.dictLayers["nucleiContourOverlay"],"no nuclei contour generated")
        self.assertIsNotNone(OS_Manager.dictLayers["organoidContourOverlay"]," no oganoid contour generated")
        self.assertIsNotNone(OS_Manager.dictLayers["cellContourOverlay"]," no cell contour generated")        

def simple_process_with_each_file():
    params = []
    def g(i,fi):
        def f():
            OS_Manager.my_window.FileListWidget.setCurrentRow(i) 
            OS_Manager.my_window.FileListWidget.itemClicked.emit(OS_Manager.my_window.FileListWidget.item(i))
            QTest.qWait(100)
            OS_Manager.my_window.ProcessFileButton.clicked.emit()
            QTest.qWait(100)
            assert OS_Manager.currentMask is not None
        return f
    for i, fi in enumerate(Path(test_im_fold).glob("*.tif")):
        params.append({"f":g(i,fi), "info":"simple test with file %s"%fi.stem})
    return params

def advanced_multiparam_process_with_multichanel_file():

    def g(preprocess, organo_method, cell_method):
        def f():
            OS_Manager.my_window.PreprocessingListWidget.clear()
            for i in range(max(1, OS_Manager.fileInfos[OS_Manager.my_window.FileListWidget.currentRow()].channel)):
                if  OS_Manager.fileInfos[OS_Manager.my_window.FileListWidget.currentRow()].channel >0:
                    OS_Manager.my_window.ChannelPreprocessingComboBox.setCurrentIndex(i)
                OS_Manager.my_window.MethodPreprocessingComboBox.setCurrentIndex(OS_Manager.my_window.MethodPreprocessingComboBox.findText(preprocess))
                OS_Manager.my_window.AddPreprocessingButton.clicked.emit()
            OS_Manager.my_window.DetectOrganoidCheckBox.setChecked(True)
            OS_Manager.my_window.MethodOrganoidComboBox.setCurrentIndex(OS_Manager.my_window.MethodOrganoidComboBox.findText(organo_method))
            OS_Manager.my_window.DetectCellsCheckBox.setChecked(True)
            OS_Manager.my_window.MethodCellsComboBox.setCurrentIndex(OS_Manager.my_window.MethodCellsComboBox.findText(cell_method))

            QTest.qWait(100)
            OS_Manager.my_window.ProcessFileButton.clicked.emit()
            QTest.qWait(100)
            for mask in ["Nuclei", "Cells","Organoid"]:
                OS_Manager.my_window.MaskVisualisationComboBox.setCurrentIndex(OS_Manager.my_window.MaskVisualisationComboBox.findText(mask))
                QTest.qWait(100)
                assert OS_Manager.currentMask is not None, "no mask produced, likeky due to process error"
                assert OS_Manager.currentMask.sum()>0, "mask produced is empty likely due to process error"

        return f
    params = []
    for prep in ["correct bleaching"]:
        for organo_met in ["Otsu Threshold", "Dynamic Range Threshold"]:
            for cell_met in ["Watershed intensity channel","Watershed distance map"]:
                params.append({"f":g(prep, organo_met, cell_met), "info":"advanced_multiparam_process with" + " ".join([prep, organo_met, cell_met])})
    return params



def generate_process_file_param():
    params = []
    params +=simple_process_with_each_file()
    params.append({"f":lambda:OS_Manager.my_window.FileListWidget.itemClicked.emit(OS_Manager.my_window.FileListWidget.findItems("ZeroG-Breast-Cancer-Organoid",Qt.MatchFlag.MatchExactly)[0]), "info":"not a test: image on first file"}) 
    params+=advanced_multiparam_process_with_multichanel_file()
    return params

@pytest.mark.parametrize("prep",generate_process_file_param())    
def test_multiparameter_pytest_only(prep):
    prep.pop("f")()
    

def order_sort(test_case, methode_name_1, methode_name_2):
    methode1 = getattr(TestMyApplication, methode_name_1, None)
    methode2 = getattr(TestMyApplication, methode_name_2, None)
    return  inspect.getsourcelines(methode1)[1] - inspect.getsourcelines(methode2)[1]

unittest.TestLoader.sortTestMethodsUsing = order_sort

if __name__ == '__main__':
    unittest.main()