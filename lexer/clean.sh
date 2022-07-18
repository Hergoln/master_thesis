DIR="."
if [ $# -gt 0 ]; then
  DIR=$1
fi

make clean -C $DIR
find $DIR -name "*.parsed" -type f -delete