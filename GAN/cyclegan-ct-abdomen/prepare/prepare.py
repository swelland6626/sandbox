import os
import sys
import glob
import pydicom
import pandas as pd


def main(root_folder,csv_file):
    mypath = os.path.join(root_folder,'**','*.dcm')
    dcm_file_list =[path for path in glob.glob(mypath,recursive=True)]
    print(len(dcm_file_list))
    mylist = []
    for dcm_file in dcm_file_list:
        ds = pydicom.dcmread(dcm_file,stop_before_pixels=True)        
        dcm_dict = {
            # 'series_description':ds.SeriesDescription,
            'dcm_file':dcm_file,
            'patient-id':ds.PatientID,
            'study_instance_uid':ds.StudyInstanceUID,
            'study_date':ds.StudyDate,
            'series_instance_uid':ds.SeriesInstanceUID,            
        }
        mylist.append(dcm_dict)

    # pd.DataFrame(mylist).to_csv('data.csv',index=False)
    pd.DataFrame(mylist).to_csv('data_circles.csv',index=False)


if __name__ == "__main__":

    root_folder = sys.argv[1]
    # sys.argv[1] looks like --> python prepare.py /radraid/swelland/images/DRO-Toolkit
    # It is best practice not to hardcode so that the script is the most generalizable possible.

    csv_file = 'data.csv'
    # csv_file = 'data_circles.csv'
    main(root_folder,csv_file)

'''
cd prepare
python prepare.py /radraid/pteng/tmp/c4kc-kits
'''
