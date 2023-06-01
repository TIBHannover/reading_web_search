# Installation
´´´
conda create -n eye_tracking_features python==3.10 --file requirements.txt
conda activate eye_tracking_features
´´´

# Input Data
## Eye Tracking Data
Put Eye-Tracking-Data for each participant in folder data/raw_fixations/uid.tsv, i.e.,
- data
  - raw_fixations
    - 001.tsv
    - 002.tsv
    - ...
Required fields are the following:
- uid *(user id)*
- timestamp *(timestamp in the session in ms, i.e., first entry is 0.0)*
- duration *(duration of the fixation in ms)*
- url *(visited url)*
- mean_x *(mean x-coordinate for left and right eye on the screen)*
- mean_y *(mean y-coordinate for left and right eye on the screen)*

# Words Map Data
Put the information about words on the webpages in data/words_map.tsv
- data
  - words_map.tsv
Required fields are the following:
- uid *(user id)*
- url
- word *(text at this position)*
- x *(x-coordinate of the top left corner of the bounding box)*
- y *(y-coordinate of the top left corner of the bounding box)*
- width *(width of the bounding box in pixels)*
- height *(height of the bounding box in pixels)*