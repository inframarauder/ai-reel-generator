'''
utility methods for various
file operations
'''
import os

def pre_flight_checks(app_config):
    '''
    pre-flight checks on the inputs
    before running the main code
    '''
    # pre-flight checks
    input_path = app_config.video_input_folder
    output_path = app_config.video_output_folder

    # read access is required to input_path
    if not os.access(input_path, os.R_OK):
        print(f"Lacking read access to video input folder: {input_path}")
        exit(1)
    
    # read and write access is required to output_path
    if not os.access(output_path, os.R_OK) and os.access(output_path, os.W_OK):
        print(f"Lacking read or write access to output folder: {output_path}")
        exit(1)
    return

def get_input_videos_list(input_folder):
    '''
    Get list of all supported video
    files in the folder
    '''
    supported_extensions = ["mp4"]
    return [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(tuple(ext for ext in supported_extensions))
    ]