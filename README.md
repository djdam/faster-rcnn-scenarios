# faster-rcnn-scenarios
Tool for rapidly creating "alternating optimization" scenarios for py-faster-rcnn.

Requirements:
-
- Set environment variable CAFFE_ROOT to point at your local caffe installation (with py-faster-rcnn integration)
- Set environment variable FASTER_RCNN_ROOT to point to your local py-Faster-rcnn folder

Example:
-
Take a look at examples/different_scales.py which contains a simple set of scenarios training with different anchor scales. Please first change the train/test imdbs before running the script. The example will generate 3 scenario configurations in the examples/scenarios folder, plus a "run_all.sh" script. Run the latter to start training and testing all scenarios at once.

Logging goes to scenarios/[scenario name]/logs and output (final caffemodel) goes to scenarios/[scenario name]/output.

Enjoy!


