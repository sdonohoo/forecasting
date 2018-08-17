
# Downloads the SMD Hourly Data from 2011 to 2017
# from https://www.iso-ne.com/isoexpress/web/reports
# /load-and-demand/-/tree/zone-info

import os
import sys
from urllib.request import urlretrieve

def download_data():
    
    data_dir = os.path.join("energy_load", "GEFCom2017-D_Prob_MT_hourly", "data")

    # Download the data
    urls = [
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/znl_info/hourly/2011_smd_hourly.xls",
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/znl_info/hourly/2012_smd_hourly.xls",
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/znl_info/hourly/2013_smd_hourly.xls",
        "https://www.iso-ne.com/static-assets/documents/2015/05/2014_smd_hourly.xls",
        "https://www.iso-ne.com/static-assets/documents/2015/02/smd_hourly.xls",
        "https://www.iso-ne.com/static-assets/documents/2016/02/smd_hourly.xls",
        "https://www.iso-ne.com/static-assets/documents/2017/02/2017_smd_hourly.xlsx"
    ]

    for url in urls:
        
        url_tokens = url.split('/')
        fname = url_tokens[-1]
    
        # add prefix to 2015 and 2016 file names
        year_month = url_tokens[-3:-1]
        year_month = "/".join(year_month)
        print(year_month)
        if year_month in ["2015/02", "2016/02"]:
            fname = url_tokens[-3] + "_" + fname

        fpath = os.path.join(data_dir, fname)

        # Check if file already exists and skip if so
        if os.path.exists(fpath):
            print(fpath + " already exists")
            continue
        print('Downloading', url)
        f, _ = urlretrieve(url, fpath)
        print('Downloaded to', fpath)


if __name__=="__main__":
    download_data()
