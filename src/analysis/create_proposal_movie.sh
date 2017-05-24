#!/bin/bash
export OUT_DIR=/home/dennis/workspace/faster-rcnn-scenarios/src/analysis/output
TEMP_DIR=$OUT_DIR/temp_proposal_movie
mkdir $TEMP_DIR

function create_frame {
    PREFIX=`seq -f "%04g" $1`
    FILEMASK=$OUT_DIR/$PREFIX\*
    echo $FILEMASK
    #if $(ls $FILEMASK); then
    #    echo "file $i exists"
    #fi
}
END=250

exists() { [[ -f $1 ]]; }

for PREFIX in $(seq -f "%04g" 2000); do
    FILE_PATH_START=$OUT_DIR/$PREFIX
    if exists $FILE_PATH_START*; then
        montage $FILE_PATH_START* -tile 2x1 -geometry +0+0 $TEMP_DIR/$PREFIX.png
    else
        break
    fi
done


convert -delay 50 -loop 0 $TEMP_DIR/*.png $TEMP_DIR/animation.gif
ffmpeg -f gif -i $TEMP_DIR/animation.gif $TEMP_DIR/rpn_proposal.mp4
#rm -rf $TEMP_DIR
