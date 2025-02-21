"""General Configuration File"""
import pandas as pd
from datetime import datetime


SAMPLE_RATE = 0.10

DT_TODAY = pd.Timestamp.today().date()
TODAY = DT_TODAY.strftime('%Y-%m-%d')