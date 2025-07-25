# seedtrack3d
track seeds in 3d

# Installation

```pip install git+https://github.com/SheffieldMLtracking/seedtrack3d.git```

# Usage

The commandline tool's help:
```usage: seedtrack3d [-h] [--calibration CALIBRATION]
                   [--calibrationN CALIBRATIONN] [--images IMAGES]
                   --calsquarewidth CALSQUAREWIDTH --hfov HFOV [--skip SKIP]
                   [--output_filename OUTPUT_FILENAME]
                   [--recompute_calibration_cache] [--save_debug_images]
                   [--threshold THRESHOLD]

Using a series of videos (saved as tiff files) and a stereo-mirror system,
reconstruct 3d path of seed being dropped.

options:
  -h, --help            show this help message and exit
  --calibration CALIBRATION
                        Path to calibration images
  --calibrationN CALIBRATIONN
                        How many calibration images to use
  --images IMAGES       Path to seed drop images
  --calsquarewidth CALSQUAREWIDTH
                        Size of the calibration square, in metres (e.g. 0.175)
  --hfov HFOV           The horizontal field of view, in degrees (e.g. 19.2)
  --skip SKIP           Only use every nth image (only useful for speed during
                        testing)
  --output_filename OUTPUT_FILENAME
                        Filename to write 3d coordinates in. Default is to the
                        last folder in path.
  --recompute_calibration_cache
                        Recompute cache
  --save_debug_images   Save a pdf in the images folder to help detect errors.
  --threshold THRESHOLD
                        Brightness threshold for seed detection (default = 7)
                        -- decrease to be more sensitivitive, increase to be
                        less sensitive
```

Example usage:

`seedtrack3d --calibration /home/mike/Documents/Research/seed_data/calibration --images /home/mike/Documents/Research/seed_data/250305/caryopsis_t1/ --calibrationN 12 --calsquarewidth 0.175 --hfov 19.2`

This uses the images in `/home/mike/Documents/Research/seed_data/calibration` to calibrate the "cameras", then finds the seed in the folder "/home/mike/Documents/Research/seed_data/250305/caryopsis_t1/". It uses a default threshold of '7' for this (how much the seeds have to change the pixel brightness compared to the background). Saves to default (gets name from images folder, in this case 'caryopsis_t1'). Note that I've made the size of the calibration square and the Horizontal FOV mandatory settings so people don't forget to change them if these values do get altered.

In this case it saves a file that looks like this:
```142.00000,2.28055,-0.02679,-0.23343
143.00000,2.28141,-0.02712,-0.20389
144.00000,2.28223,-0.02719,-0.17377
145.00000,2.28313,-0.02755,-0.14367
146.00000,2.28402,-0.02791,-0.11318
147.00000,2.28519,-0.02829,-0.08214
148.00000,2.28623,-0.02898,-0.05110
149.00000,2.28741,-0.02938,-0.01968
150.00000,2.28823,-0.02946,0.01163
151.00000,2.28930,-0.02956,0.04330
152.00000,2.28975,-0.02933,0.07567
153.00000,2.28993,-0.02974,0.10803
154.00000,2.29013,-0.03016,0.14050
155.00000,2.29042,-0.03088,0.17310
156.00000,2.29078,-0.03165,0.20605
157.00000,2.29086,-0.03240,0.23913
```

the four columns are: Time (in image index -- I don't know the frame rate), x, y, z (in metres).
z isn't necessarily vertical (depends on camera alignments etc).
