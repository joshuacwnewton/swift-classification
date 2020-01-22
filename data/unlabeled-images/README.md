### Purpose

This directory is to be accessed by the `label-images.ipynb` Jupyter notebook. That notebook provides tools for labeling images placed in this directory, and moving them to a new location.

### Usage

Place images in this directory with filenames in the form of:

`"<video-filename>"_<frame-number>_<seg-#>_<total-seg-#>.png`

Where each attribute is defined as follows:

* `"<video-filename>"`: Necessary for balancing video sources when sampling for eventual datasets. Quotes are helpful for parsing with Regex, as video filenames can contain `_` characters.
* `<frame-number>`: Necessary for uniquely identifying the segment image. This also provides a means of balancing time periods when sampling for eventual datasets.
* `<seg-#>`: Represents an integer label for segments on a per-frame basis. Necessary for uniquely identifying the segment image. Segments are numbered from their position in the frame, top-down, starting with 1.
* `<total-seg-#>`: Represents the total number of segments for the frame containing this segment. "Nice to have" for understanding frame context.

Examples of this labeling scheme include:

* `"ch04_20180519194457"_3550_1_5.png`
* `"ch02_20160513200533"_54219_3_3.png`
* `"NPD_572_CHSW_2019_July_21"_1939_2_3.png`

