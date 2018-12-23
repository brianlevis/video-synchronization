# CS 194-26 Final Project
## Video Synchronization
### Overview
This was made as the final project for my [Computational Photography](https://inst.eecs.berkeley.edu/~cs194-26/fa18/) class, and development is ongoing.

My goal was to modify the frames of a video such that they appeared somehow synchronized or reacting to a song or other audio. Applications include syncing dance-heavy music videos to different songs, or making it look like a person or animal is dancing to a beat.

Peak matching audio strength with optical flow magnitude yielded cool results, but I have started to implement everything from [Abe Davis's Visual Rhythm Project](http://abedavis.com/visualbeat/), which matches audio onset envelopes with video impact envelopes, taking tempo into account.

See a demo [here](https://brianlevis.com/cs194-26/final/).
### Directory Structure
```
### Code Files
main.py             # Select file names and methods
peak.py             # Simple peak detection functions
media
---- media.py       # Interfaces and content processing code
sychronize/
---- rhythm.py      # Identify potential matching points
---- synch.py       # Merge media

### Required Media Folders
input_files/
---- audio/
---- video/
output_files/
```
