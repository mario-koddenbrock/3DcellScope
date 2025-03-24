
from enum import Enum
import src.features_windows as features_windows
import src.features_app as features_app
import src.features_data as features_data

MAX_INDEX_SIZE = 100
class GlobalContext():
    def __init__(self) -> None:
        self.app_manager:features_app.MainApp = None
        self.data_manager:features_data.DataManager = None
        self.main_windows:features_windows.MainWindow = None

CONTEXT:GlobalContext =  GlobalContext()

def initialyse_context():
    global CONTEXT
    CONTEXT.app_manager = features_app.MainApp()
    CONTEXT.data_manager = features_data.DataManager()
    CONTEXT.main_windows = features_windows.MainWindow()
