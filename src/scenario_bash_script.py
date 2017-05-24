SCRIPT="""
#!/bin/bash
# Generate train/test script for scenario "{scenario}" using the faster-rcnn "alternating optimization" method

set -x
set -e

rm -f $CAFFE_ROOT/data/cache/*.pkl
rm -f {scenarios_dir}/{scenario}/output/*.pkl

DIR=`pwd`

function quit {{
   cd $DIR
   exit 0
}}

export PYTHONUNBUFFERED="True"

TRAIN_IMDB={train_imdb}
TEST_IMDB={test_imdb}

cd {py_faster_rcnn}

mkdir -p {scenarios_dir}/{scenario}/logs >/dev/null
LOG="{scenarios_dir}/{scenario}/logs/log.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time {train_script} {scenario_file} || quit

time ./tools/test_net.py --gpu {gpu_id} \\
  --def {testproto} \\
  --net {net_final_path} \\
  --imdb {test_imdb} \\
  --cfg {config_path}  || quit

chmod u+x {plot_script}
{plot_script} $LOG {scenarios_dir}/{scenario}/output/results.png || true

MEAN_AP=`grep "Mean AP = " ${{LOG}} | awk '{{print $3}}'`

echo "{scenario} finished with mAP=$MEAN_AP" >> {scenarios_dir}/status.txt
quit
"""