
# Downloads the SMD Hourly Data from 2011 to 2017
# from https://www.iso-ne.com/isoexpress/web/reports
# /load-and-demand/-/tree/zone-info

import os
import sys
from urllib.request import urlretrieve

def download_data():
    
    data_dir = "energy_load/data/GEFCom2017-D"
    # Make raw data directory if it doesn't already exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Check if data has already been downloaded and stop if so
    for i in range(5):
        year = str(2011+i)
        fname = year + "_smd_hourly.xls"
        fpath = data_dir + "/" + fname
        if os.path.exists(fpath):
            sys.exit("Data file " + fpath + " already exists.")

    # Download the data
    urls = [
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/znl_info/hourly/2011_smd_hourly.xls",
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/znl_info/hourly/2012_smd_hourly.xls",
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/znl_info/hourly/2013_smd_hourly.xls",
        "https://www.iso-ne.com/static-assets/documents/2015/05/2014_smd_hourly.xls",
        "https://www.iso-ne.com/static-assets/documents/2015/02/smd_hourly.xls"
    ]

    for i, url in enumerate(urls):
        year = str(2011+i)
        fname = year + "_smd_hourly.xls"
        fpath = data_dir + "/" + fname
        print('Downloading', url)
        f, _ = urlretrieve(url, fpath)
        print('Downloaded to', fpath)


if __name__=="__main__":
    download_data()
