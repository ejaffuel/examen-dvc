import requests
import os
import logging
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config_manager import ConfigurationManager
from src.data.check_structure import check_existing_folder

config_manager = ConfigurationManager()
data_import_raw_config = config_manager.get_data_import_raw_config()

def import_raw_data(input_url,
                    output_folderpath, 
                    output_filename
                    ):
    '''import from input_url in output_folderpath/output_filename'''
    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath)
    # download all the files
    output_file = os.path.join(output_folderpath, output_filename)
    print(f'downloading {input_url} as {os.path.basename(output_file)}')
    response = requests.get(input_url)
    if response.status_code == 200:
        # Process the response content as needed
        content = response.text
        text_file = open(output_file, "wb")
        text_file.write(content.encode('utf-8'))
        text_file.close()
    else:
        print(f'Error accessing the object {input_url}:', response.status_code)
                
def main(input_url,
         output_folderpath, 
         output_filename
        ):
    """ Upload data from AWS s3 in ./data/raw_data
    """
    import_raw_data(input_url, output_folderpath, output_filename)
    logger = logging.getLogger(__name__)
    logger.info('making raw data set')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    input_url = data_import_raw_config.input_url
    output_folderpath = Path(data_import_raw_config.output_filepath).parent
    output_filename = Path(data_import_raw_config.output_filepath).name
    main(input_url, output_folderpath, output_filename)