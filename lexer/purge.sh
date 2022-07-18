SCRIPT_DIR=$(dirname $(realpath $0))

DIR=$SCRIPT_DIR
if [ $# -gt 0 ]; then
  DIR=$1
fi

$SCRIPT_DIR/clean.sh $DIR
find $DIR -name "*.log" -type f -delete