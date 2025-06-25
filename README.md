# ai-reel-generator
A python-based tool to generate short video content (like Instagram reels or YouTube Shorts). 

## How it works
- First, we take the following inputs -
    - `input_folder` -  a folder containing several videos (ideally dumped from a GoPro or similar camera) 
    - `scene_prompt` - a prompt describing the kind of scenes you want to extract from the input videos
    - `audio_path` - path to an audio file to be put as background music in the final reel video
    - `output_folder` - output folder to render the generated reel.
    
- With these inputs, the app searches the `input_folder` for parts of the videos that match the scene described in the `scene_prompt` and cuts out the clips with the highest matches (using cosine similarity)

- Finally, the app 
    - combines the extracted clips and the provided audio 
    - syncs the cuts to the beats of the audio 
    - renders the final reel at the given output folder

## Usage

1. Clone this repository - 
    ```
    git clone https://github.com/inframarauder/ai-reel-generator.git
    ```
2. Create a python virtual environment inside the repo -
    ```
    cd ai-reel-generator
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3. Install the requirements - 
    ```
    pip install -r requirements.txt
    ```
4. Run the application -
    ```
    python index.py \               
            -i "<path-to-input-videos-folder>" \
            -a "<path-to-audio>" \
            -s "<scene-prompt>" \
            -o "<output-folder>"
    ```

## Configurations
There are some default settings configured in `configs/settings.py` which you may edit as per your needs.